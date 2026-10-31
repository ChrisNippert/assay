// Package pulls runs model downloads in the background so they survive
// page changes, and tracks their progress.
package pulls

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"log/slog"
	"sort"
	"sync"
	"time"

	"assay/internal/ollama"
)

// Puller is the part of the Ollama client the manager needs.
type Puller interface {
	Pull(ctx context.Context, model string, fn func(ollama.PullProgress) error) error
	Copy(ctx context.Context, src, dst string) error
}

// Job states.
const (
	Running  = "running"
	Done     = "done"
	Failed   = "failed"
	Canceled = "canceled"
)

// Job is a snapshot of one download.
type Job struct {
	ID          string    `json:"id"`
	Model       string    `json:"model"`
	Alias       string    `json:"alias,omitempty"`
	State       string    `json:"state"`
	Status      string    `json:"status"` // Ollama's latest status line
	Completed   int64     `json:"completed"`
	Total       int64     `json:"total"`
	BytesPerSec float64   `json:"bytesPerSec"`
	Error       string    `json:"error,omitempty"`
	Warning     string    `json:"warning,omitempty"`
	Started     time.Time `json:"started"`
	Finished    time.Time `json:"finished,omitzero"`
}

type job struct {
	Job
	cancel   context.CancelFunc
	layers   map[string][2]int64 // digest -> completed, total
	lastAt   time.Time
	lastSeen int64
}

// Manager tracks downloads.
type Manager struct {
	ollama Puller
	log    *slog.Logger
	mu     sync.Mutex
	jobs   map[string]*job
}

// New returns a manager that pulls through p.
func New(p Puller, log *slog.Logger) *Manager {
	if log == nil {
		log = slog.Default()
	}
	return &Manager{ollama: p, log: log, jobs: map[string]*job{}}
}

// keepFinished is how long finished jobs stay in the list.
const keepFinished = 10 * time.Minute

// Start begins pulling model, then names it alias if alias is non-empty.
// If the same model is already downloading, that job is returned instead.
func (m *Manager) Start(model, alias string) Job {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, j := range m.jobs {
		if j.Model == model && j.State == Running {
			return j.Job
		}
	}
	var b [6]byte
	rand.Read(b[:])
	ctx, cancel := context.WithCancel(context.Background())
	j := &job{
		Job:    Job{ID: hex.EncodeToString(b[:]), Model: model, Alias: alias, State: Running, Status: "starting", Started: time.Now()},
		cancel: cancel,
		layers: map[string][2]int64{},
	}
	m.jobs[j.ID] = j
	go m.run(ctx, j)
	return j.Job
}

func (m *Manager) run(ctx context.Context, j *job) {
	err := m.ollama.Pull(ctx, j.Model, func(p ollama.PullProgress) error {
		m.update(j, p)
		return nil
	})
	var warning string
	if err == nil && j.Alias != "" {
		if cerr := m.ollama.Copy(context.Background(), j.Model, j.Alias); cerr != nil {
			warning = "downloaded, but couldn't add the short name: " + cerr.Error()
		}
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	j.Finished = time.Now()
	j.BytesPerSec = 0
	switch {
	case errors.Is(err, context.Canceled):
		j.State, j.Status = Canceled, "canceled"
	case err != nil:
		j.State, j.Status, j.Error = Failed, "failed", err.Error()
		m.log.Warn("pull failed", "model", j.Model, "err", err)
	default:
		j.State, j.Status, j.Warning = Done, "done", warning
		j.Completed = j.Total
	}
}

func (m *Manager) update(j *job, p ollama.PullProgress) {
	m.mu.Lock()
	defer m.mu.Unlock()
	j.Status = p.Status
	if p.Digest != "" && p.Total > 0 {
		j.layers[p.Digest] = [2]int64{p.Completed, p.Total}
		var done, total int64
		for _, l := range j.layers {
			done += l[0]
			total += l[1]
		}
		j.Completed, j.Total = done, total
	}
	// Smooth the speed so it doesn't jump around with every chunk.
	now := time.Now()
	if j.lastAt.IsZero() {
		j.lastAt, j.lastSeen = now, j.Completed
	} else if dt := now.Sub(j.lastAt).Seconds(); dt >= 0.5 {
		rate := float64(j.Completed-j.lastSeen) / dt
		if j.BytesPerSec == 0 {
			j.BytesPerSec = rate
		} else {
			j.BytesPerSec = 0.7*j.BytesPerSec + 0.3*rate
		}
		j.lastAt, j.lastSeen = now, j.Completed
	}
}

// Cancel stops a running download. Ollama keeps the partial data, so
// starting the same pull again resumes it.
func (m *Manager) Cancel(id string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	j, ok := m.jobs[id]
	if !ok || j.State != Running {
		return false
	}
	j.cancel()
	return true
}

// List returns running jobs and recently finished ones, newest first.
func (m *Manager) List() []Job {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]Job, 0, len(m.jobs))
	for id, j := range m.jobs {
		if j.State != Running && time.Since(j.Finished) > keepFinished {
			delete(m.jobs, id)
			continue
		}
		out = append(out, j.Job)
	}
	sort.Slice(out, func(i, k int) bool { return out[i].Started.After(out[k].Started) })
	return out
}

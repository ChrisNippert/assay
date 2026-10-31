package server

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"sync"
	"time"

	"assay/internal/bench"
	"assay/internal/hw"
	"assay/internal/ollama"
)

// Measure-all runs the speed test on several models one after another in
// the background, so leaving the Library doesn't stop it. It reuses
// runBench, so a number here is the same as one from Compare.

// Item states.
const (
	itemPending = "pending"
	itemRunning = "running"
	itemDone    = "done"
	itemSkipped = "skipped"
	itemFailed  = "failed"
)

type measureItem struct {
	Name   string  `json:"name"`
	Title  string  `json:"title"`
	State  string  `json:"state"`
	Reason string  `json:"reason,omitempty"`
	GenTPS float64 `json:"genTps,omitempty"`
}

// measureJob is the current (or last) measure-all run.
type measureJob struct {
	mu      sync.Mutex
	running bool
	Items   []measureItem
	Waiting string // why it's paused, e.g. a reply being written
	// KeptPinned names the pinned models left loaded; empty when it
	// evicted everything.
	KeptPinned []string
	Cancelled  bool
	Finished   time.Time
	cancel     context.CancelFunc
}

type measureState struct {
	Running    bool          `json:"running"`
	Items      []measureItem `json:"items"`
	Done       int           `json:"done"` // finished, skipped or failed
	Total      int           `json:"total"`
	Current    string        `json:"current"` // title of the one being measured
	Waiting    string        `json:"waiting,omitempty"`
	KeptPinned []string      `json:"keptPinned,omitempty"`
	Cancelled  bool          `json:"cancelled,omitempty"`
}

func (j *measureJob) state() measureState {
	j.mu.Lock()
	defer j.mu.Unlock()
	st := measureState{Running: j.running, Items: slices.Clone(j.Items), Total: len(j.Items), Waiting: j.Waiting, KeptPinned: j.KeptPinned, Cancelled: j.Cancelled}
	if st.Items == nil {
		st.Items = []measureItem{}
	}
	for _, it := range j.Items {
		switch it.State {
		case itemRunning:
			st.Current = it.Title
		case itemDone, itemSkipped, itemFailed:
			st.Done++
		}
	}
	return st
}

func (j *measureJob) set(i int, f func(*measureItem)) {
	j.mu.Lock()
	f(&j.Items[i])
	j.mu.Unlock()
}

// measureRequest starts a run. No names means every installed model that
// hasn't been tested. EvictPinned says it may unload pinned models too.
type measureRequest struct {
	Names       []string `json:"names"`
	EvictPinned bool     `json:"evictPinned"`
}

func (s *Server) handleMeasureStart(w http.ResponseWriter, r *http.Request) {
	var req measureRequest
	if !readJSON(w, r, &req) {
		return
	}
	st, err := s.startMeasure(r.Context(), req)
	if err != nil {
		var ce *conflictError
		if errors.As(err, &ce) {
			writeError(w, http.StatusConflict, err.Error())
			return
		}
		writeError(w, ollamaStatus(err), err.Error())
		return
	}
	writeJSON(w, http.StatusAccepted, st)
}

func (s *Server) handleMeasureState(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	writeJSON(w, http.StatusOK, s.measure.state())
}

func (s *Server) handleMeasureCancel(w http.ResponseWriter, r *http.Request) {
	s.measure.mu.Lock()
	if s.measure.running && s.measure.cancel != nil {
		s.measure.cancel()
	}
	s.measure.mu.Unlock()
	writeJSON(w, http.StatusOK, s.measure.state())
}

type conflictError struct{ msg string }

func (e *conflictError) Error() string { return e.msg }

// measureBudget is the memory left for a model while the pinned ones stay
// loaded.
func measureBudget(mem hw.Memory, pinned []loadedModel) hw.Memory {
	for _, m := range pinned {
		mem.VRAM = max(0, mem.VRAM-m.SizeVRAM)
		mem.RAM = max(0, mem.RAM-(m.Size-m.SizeVRAM))
		if n := len(mem.Cards); n > 0 && mem.VRAM == 0 {
			mem.Cards = nil
		}
	}
	return mem
}

// planMeasure decides, for each named model, whether it can be measured
// next to the pinned ones that stay loaded.
func (s *Server) planMeasure(models []ollama.Model, names []string, pinned []loadedModel, evict bool) []measureItem {
	groups := groupModels(models)
	budget := s.memory
	if !evict {
		budget = measureBudget(s.memory, pinned)
	}
	var items []measureItem
	for _, name := range names {
		gi := slices.IndexFunc(groups, func(g modelGroup) bool { return g.Has(name) })
		if gi < 0 {
			continue
		}
		g := groups[gi]
		title, _ := prettyModel(g.Primary.Name)
		if g.HFRepo != "" && g.PrimaryOwner() != "" {
			title, _ = prettyModel(g.ShortName())
		}
		it := measureItem{Name: g.Primary.Name, Title: title, State: itemPending}
		switch {
		case slices.ContainsFunc(items, func(o measureItem) bool { return o.Name == it.Name }):
			continue
		case isEmbeddingName(g.Primary.Name):
			it.State, it.Reason = itemSkipped, "It's an embedding model; there's no text to write."
		case !evict && slices.ContainsFunc(pinned, func(m loadedModel) bool { return g.Has(m.Name) }):
			it.State, it.Reason = itemSkipped, "It's loaded and pinned, and a fresh load is what gets timed. Eject it first."
		case !evict && len(pinned) > 0 && hw.Classify(hw.Estimate(g.Primary.Size, parseParams(g.Primary.Details.ParameterSize), fitCtx, false), budget).Level == hw.FitNone:
			it.State, it.Reason = itemSkipped, "Doesn't fit in memory next to "+pinnedTitles(pinned)+", which is pinned."
		}
		items = append(items, it)
	}
	return items
}

func pinnedTitles(pinned []loadedModel) string {
	var t []string
	for _, m := range pinned {
		title, _ := prettyModel(m.Name)
		t = append(t, title)
	}
	return strings.Join(t, ", ")
}

func (s *Server) startMeasure(ctx context.Context, req measureRequest) (measureState, error) {
	j := &s.measure
	j.mu.Lock()
	if j.running {
		j.mu.Unlock()
		return measureState{}, &conflictError{"a measurement is already running"}
	}
	j.mu.Unlock()
	if !benchMu.TryLock() {
		return measureState{}, &conflictError{"another speed test is running"}
	}
	benchMu.Unlock()
	if n := len(s.runningChats()); n > 0 {
		return measureState{}, &conflictError{"a reply is being written; measuring would unload its model. Try again when it's done."}
	}
	models, err := s.ollama.List(ctx)
	if err != nil {
		return measureState{}, err
	}
	names := req.Names
	if len(names) == 0 {
		for _, g := range groupModels(models) {
			if _, ok := s.bench.Get(g.Primary.Digest); !ok && !isEmbeddingName(g.Primary.Name) {
				names = append(names, g.Primary.Name)
			}
		}
	}
	loaded, _ := s.loaded(ctx)
	var pinned []loadedModel
	for _, m := range loaded {
		if m.Pinned {
			pinned = append(pinned, m)
		}
	}
	keep := len(pinned) > 0 && !req.EvictPinned
	items := s.planMeasure(models, names, pinned, !keep)
	if len(items) == 0 {
		return measureState{}, &conflictError{"nothing to measure"}
	}
	// What was loaded (and not pinned) comes back afterwards.
	var restore []string
	for _, m := range loaded {
		if !m.Pinned && !m.Loading {
			restore = append(restore, m.Name)
		}
	}
	jctx, cancel := context.WithCancel(context.Background())
	j.mu.Lock()
	j.running, j.Cancelled, j.Waiting, j.Items, j.cancel = true, false, "", items, cancel
	j.KeptPinned = nil
	if keep {
		for _, m := range pinned {
			j.KeptPinned = append(j.KeptPinned, m.Name)
		}
	}
	j.mu.Unlock()
	go s.runMeasure(jctx, keep, restore)
	return j.state(), nil
}

// runMeasure goes through the planned items one at a time.
func (s *Server) runMeasure(ctx context.Context, keepPinned bool, restore []string) {
	j := &s.measure
	defer func() {
		j.mu.Lock()
		j.running, j.Waiting, j.Finished = false, "", time.Now()
		j.cancelled(ctx)
		j.mu.Unlock()
	}()
	measured := false
	for i := range j.Items {
		if ctx.Err() != nil {
			break
		}
		if j.Items[i].State != itemPending {
			continue
		}
		name := j.Items[i].Name
		if err := s.waitForGPU(ctx); err != nil {
			break
		}
		j.set(i, func(it *measureItem) { it.State = itemRunning })
		res, err := s.benchOne(ctx, name, keepPinned)
		measured = true
		switch {
		case ctx.Err() != nil:
			j.set(i, func(it *measureItem) { it.State = itemPending })
		case err != nil:
			s.log.Warn("measure all", "model", name, "err", err)
			j.set(i, func(it *measureItem) { it.State, it.Reason = itemFailed, err.Error() })
		default:
			j.set(i, func(it *measureItem) { it.State, it.GenTPS = itemDone, res.GenTPS })
		}
	}
	if !measured {
		return
	}
	// Leave memory as it was: the last model measured goes, what was loaded
	// before comes back.
	rctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	s.unloadFor(rctx, keepPinned)
	for _, n := range restore {
		if _, err := s.loadAuto(rctx, n, "5m"); err != nil {
			s.log.Warn("measure all: couldn't reload", "model", n, "err", err)
		}
	}
}

func (j *measureJob) cancelled(ctx context.Context) {
	if ctx.Err() != nil {
		j.Cancelled = true
	}
}

// waitForGPU holds the job back while a chat reply is being written (the
// test would unload its model) or another speed test runs.
func (s *Server) waitForGPU(ctx context.Context) error {
	for {
		busy := ""
		if len(s.runningChats()) > 0 {
			busy = "Waiting for a reply to finish"
		} else if !benchMu.TryLock() {
			busy = "Waiting for another speed test"
		} else {
			benchMu.Unlock()
		}
		s.measure.mu.Lock()
		s.measure.Waiting = busy
		s.measure.mu.Unlock()
		if busy == "" {
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(time.Second):
		}
	}
}

// benchOne measures and saves one model, holding the speed-test lock.
func (s *Server) benchOne(ctx context.Context, name string, keepPinned bool) (bench.Result, error) {
	if !benchMu.TryLock() {
		return bench.Result{}, errors.New("another speed test is running")
	}
	defer benchMu.Unlock()
	run := s.benchFn
	if run == nil {
		run = s.runBench
	}
	res, err := run(ctx, name, keepPinned)
	if err != nil {
		return res, err
	}
	if err := s.bench.Put(res); err != nil {
		return res, fmt.Errorf("couldn't save the result: %w", err)
	}
	return res, nil
}

// unloadFor unloads models before a test: all of them, or all but the
// pinned ones.
func (s *Server) unloadFor(ctx context.Context, keepPinned bool) error {
	if !keepPinned {
		return s.unloadAll(ctx)
	}
	for range 60 {
		loaded, err := s.loaded(ctx)
		if err != nil {
			return err
		}
		left := 0
		for _, m := range loaded {
			if m.Pinned {
				continue
			}
			left++
			s.ollama.Unload(ctx, m.Name)
		}
		if left == 0 {
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(500 * time.Millisecond):
		}
	}
	return errors.New("the other models didn't unload")
}

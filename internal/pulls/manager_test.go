package pulls

import (
	"context"
	"errors"
	"testing"
	"time"

	"assay/internal/ollama"
)

type fakePuller struct {
	block  chan struct{}
	err    error
	copied [][2]string
}

func (f *fakePuller) Pull(ctx context.Context, model string, fn func(ollama.PullProgress) error) error {
	fn(ollama.PullProgress{Status: "pulling manifest"})
	fn(ollama.PullProgress{Status: "pulling a", Digest: "sha256:a", Total: 100, Completed: 50})
	fn(ollama.PullProgress{Status: "pulling b", Digest: "sha256:b", Total: 10, Completed: 10})
	if f.block != nil {
		select {
		case <-f.block:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return f.err
}

func (f *fakePuller) Copy(ctx context.Context, src, dst string) error {
	f.copied = append(f.copied, [2]string{src, dst})
	return nil
}

func waitFor(t *testing.T, m *Manager, id, state string) Job {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		for _, j := range m.List() {
			if j.ID == id && j.State == state {
				return j
			}
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("job %s never reached %s: %+v", id, state, m.List())
	return Job{}
}

func TestPullThenAlias(t *testing.T) {
	f := &fakePuller{}
	m := New(f, nil)
	j := m.Start("hf.co/x/y:Q4_K_M", "y:q4_k_m")
	done := waitFor(t, m, j.ID, Done)
	if done.Total != 110 || done.Completed != 110 {
		t.Errorf("progress = %d/%d", done.Completed, done.Total)
	}
	if len(f.copied) != 1 || f.copied[0][1] != "y:q4_k_m" {
		t.Errorf("copies = %v", f.copied)
	}
}

func TestProgressAndCancel(t *testing.T) {
	f := &fakePuller{block: make(chan struct{})}
	m := New(f, nil)
	j := m.Start("hf.co/x/y:Q8_0", "")
	running := waitFor(t, m, j.ID, Running)
	for running.Total != 110 {
		time.Sleep(5 * time.Millisecond)
		running = m.List()[0]
	}
	if running.Completed != 60 {
		t.Errorf("completed = %d, want 60", running.Completed)
	}
	if again := m.Start("hf.co/x/y:Q8_0", ""); again.ID != j.ID {
		t.Error("starting the same model twice made a second job")
	}
	if !m.Cancel(j.ID) {
		t.Fatal("cancel returned false")
	}
	waitFor(t, m, j.ID, Canceled)
}

func TestFailure(t *testing.T) {
	m := New(&fakePuller{err: errors.New("file does not exist")}, nil)
	j := m.Start("hf.co/x/nope:Q4_K_M", "")
	failed := waitFor(t, m, j.ID, Failed)
	if failed.Error != "file does not exist" {
		t.Errorf("error = %q", failed.Error)
	}
}

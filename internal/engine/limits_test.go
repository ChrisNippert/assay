package engine

import (
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"testing"

	"ollamui/internal/hw"
	"ollamui/internal/limits"
)

func flagVal(a []string, flag string) string {
	if i := slices.Index(a, flag); i >= 0 && i+1 < len(a) {
		return a[i+1]
	}
	return ""
}

// limitEngine is speedEngine with a 200 MiB MTP head on disk, a limits
// store, and fake live GPU numbers (12 GiB total, 2 GiB used).
func limitEngine(t *testing.T, l limits.Limits) (*Engine, *Model) {
	t.Helper()
	e, m := speedEngine(t)
	f := e.store.Blob(m.MTP)
	if err := os.MkdirAll(filepath.Dir(f), 0o755); err != nil {
		t.Fatal(err)
	}
	fh, err := os.Create(f)
	if err != nil {
		t.Fatal(err)
	}
	fh.Truncate(200 << 20)
	fh.Close()
	st, _ := limits.Open("")
	if err := st.Set(l); err != nil {
		t.Fatal(err)
	}
	e.Limits = st
	e.Live = func() hw.Live { return hw.Live{VRAMTotal: 12 << 30, VRAMUsed: 2 << 30} }
	return e, m
}

func TestNoLimitsChangesNothing(t *testing.T) {
	e, m := limitEngine(t, limits.Limits{})
	a := argsFor(e, m, noSpec)
	for _, bad := range []string{"--fit-target", "--prio", "--spec-draft-ngl", "-ub"} {
		if slices.Contains(a, bad) {
			t.Errorf("unexpected %s in %v", bad, a)
		}
	}
}

func TestVRAMLimitMargin(t *testing.T) {
	// 10 GiB free, limit 3 GiB, nothing else loaded: leave 7 GiB free.
	e, m := limitEngine(t, limits.Limits{MaxVRAMGB: 3})
	if got := flagVal(argsFor(e, m, noSpec), "--fit-target"); got != "7168" {
		t.Errorf("fit-target = %q, want 7168", got)
	}
	// Another loaded model already holds 1 GiB: only 2 GiB for this one.
	e.procs = map[string]*proc{"other": {model: &Model{Name: "other"}, vram: 1 << 30}}
	if got := flagVal(argsFor(e, m, noSpec), "--fit-target"); got != "8192" {
		t.Errorf("with another model: %q, want 8192", got)
	}
	// A limit above what is free keeps the default margin.
	e.procs = nil
	e.Limits.Set(limits.Limits{MaxVRAMGB: 11})
	if got := flagVal(argsFor(e, m, noSpec), "--fit-target"); got != "1024" {
		t.Errorf("big limit: %q, want 1024", got)
	}
	// Hand-set layers: fit is off, nothing to pass.
	e.Limits.Set(limits.Limits{MaxVRAMGB: 3})
	if a := argsFor(e, m, map[string]any{"num_gpu": 10.0}); slices.Contains(a, "--fit-target") {
		t.Errorf("fit-target with hand-set layers: %v", a)
	}
	// Unknown VRAM: skipped.
	e.Live = func() hw.Live { return hw.Live{} }
	if a := argsFor(e, m, noSpec); slices.Contains(a, "--fit-target") {
		t.Errorf("fit-target without VRAM info: %v", a)
	}
}

func TestDraftGPUArgs(t *testing.T) {
	e, m := limitEngine(t, limits.Limits{})
	a := argsFor(e, m, map[string]any{"draft_gpu": "always"})
	if flagVal(a, "--spec-draft-ngl") != "all" {
		t.Errorf("no --spec-draft-ngl all: %v", a)
	}
	// 1024 + 200 MiB head + 1024 for its buffers.
	if got := flagVal(a, "--fit-target"); got != "2248" {
		t.Errorf("fit-target = %q, want 2248", got)
	}
	// Combined with a VRAM limit of 3 GiB (7168 margin).
	e.Limits.Set(limits.Limits{MaxVRAMGB: 3})
	if got := flagVal(argsFor(e, m, map[string]any{"draft_gpu": "always"}), "--fit-target"); got != "8392" {
		t.Errorf("combined = %q, want 8392", got)
	}
	// Hand-set layers: the flag stays, fit-target doesn't.
	a = argsFor(e, m, map[string]any{"draft_gpu": "always", "num_gpu": 5.0})
	if flagVal(a, "--spec-draft-ngl") != "all" || slices.Contains(a, "--fit-target") {
		t.Errorf("hand-set layers: %v", a)
	}
	// No draft in use (speed-up off): nothing.
	e.Limits.Set(limits.Limits{})
	a = argsFor(e, m, map[string]any{"draft_gpu": "always", "speedup": "off", "spec_ngram": "off"})
	if slices.Contains(a, "--spec-draft-ngl") || slices.Contains(a, "--fit-target") {
		t.Errorf("no draft: %v", a)
	}
}

func TestGentleArgs(t *testing.T) {
	e, m := limitEngine(t, limits.Limits{Gentle: true})
	a := argsFor(e, m, nil)
	if flagVal(a, "--prio") != "-1" || flagVal(a, "-ub") != "256" {
		t.Errorf("gentle flags: %v", a)
	}
	if got, want := flagVal(a, "-t"), strconv.Itoa(max(1, runtime.NumCPU()/2)); got != want {
		t.Errorf("threads = %q, want %q", got, want)
	}
	// The model's own thread count wins.
	if got := flagVal(argsFor(e, m, map[string]any{"num_thread": 3.0}), "-t"); got != "3" {
		t.Errorf("own threads = %q", got)
	}
}

func TestMMProjCPUArg(t *testing.T) {
	e, m := limitEngine(t, limits.Limits{})
	vm := *m
	vm.Projector = "proj.gguf"
	a := strings.Join(argsFor(e, &vm, nil), " ")
	if !strings.Contains(a, "--mmproj") || strings.Contains(a, "--no-mmproj-offload") {
		t.Errorf("default: %s", a)
	}
	a = strings.Join(argsFor(e, &vm, map[string]any{"mmproj_cpu": "on"}), " ")
	if !strings.Contains(a, "--no-mmproj-offload") {
		t.Errorf("mmproj_cpu on: %s", a)
	}
	// Without a projector there is nothing to keep in RAM.
	if a := strings.Join(argsFor(e, m, map[string]any{"mmproj_cpu": "on"}), " "); strings.Contains(a, "--no-mmproj-offload") {
		t.Errorf("no projector: %s", a)
	}
}

func TestVisionOffArg(t *testing.T) {
	e, m := limitEngine(t, limits.Limits{})
	vm := *m
	vm.Projector = "proj.gguf"
	if a := strings.Join(argsFor(e, &vm, map[string]any{"vision": "off", "mmproj_cpu": "on"}), " "); strings.Contains(a, "mmproj") {
		t.Errorf("vision off still loads the projector: %s", a)
	}
}

func TestDraftRoomWithoutDraftGPU(t *testing.T) {
	e, m := limitEngine(t, limits.Limits{})
	// The MTP head is used on Auto: its room is kept even without "always".
	a := argsFor(e, m, nil)
	if got := flagVal(a, "--fit-target"); got != "2248" {
		t.Errorf("fit-target = %q, want 2248 (%v)", got, a)
	}
	if flagVal(a, "--spec-draft-ngl") != "" {
		t.Errorf("draft pinned without asking: %v", a)
	}
}

// noSpec turns speculative decoding off, so a test sees only the limits.
var noSpec = map[string]any{"speedup": "off", "spec_ngram": "off"}

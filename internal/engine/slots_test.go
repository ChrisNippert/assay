package engine

import (
	"strings"
	"testing"
)

func TestSlotsAndKeepWith(t *testing.T) {
	st, err := OpenStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	e := New(st, Options{})
	m := &Model{Name: "a:b", Files: []string{"sha256-abc.gguf"}}
	if a := strings.Join(e.args(m, loadOptsFrom(nil), 1), " "); !strings.Contains(a, "-np 1") || strings.Contains(a, "-kvu") {
		t.Errorf("one slot by default: %s", a)
	}
	o := loadOptsFrom(map[string]any{"num_parallel": 4.0, "num_ctx": 32768})
	if a := strings.Join(e.args(m, o, 1), " "); !strings.Contains(a, "-np 4 -kvu") || !strings.Contains(a, "-c 32768") {
		t.Errorf("four slots sharing the context: %s", a)
	}
	if o.sameLoad(loadOptsFrom(map[string]any{"num_ctx": 32768})) {
		t.Error("a change of slots needs a restart")
	}
	// keep_with names only installed models, by their full names.
	if got := e.keepWith(map[string]any{"keep_with": "nope, other"}); len(got) != 0 {
		t.Errorf("keep_with %v", got)
	}
}

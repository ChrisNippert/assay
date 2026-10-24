package engine

import (
	"context"
	"os"
	"slices"
	"strings"
	"testing"

	"ollamui/internal/ollama"
)

// specFlags returns the --spec-type value and -md file of the arguments.
func specFlags(a []string) (types, md string) {
	for i, s := range a {
		switch s {
		case "--spec-type":
			types = a[i+1]
		case "-md":
			md = a[i+1]
		}
	}
	return
}

func speedEngine(t *testing.T) (*Engine, *Model) {
	t.Helper()
	st, err := OpenStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	e := New(st, Options{})
	for _, m := range []*Model{
		{Name: "big", Files: []string{"big.gguf"}, MTP: "big-mtp.gguf", Repo: "o/Big-GGUF", Size: 100},
		{Name: "plain", Files: []string{"plain.gguf"}, Repo: "o/Plain-GGUF", Size: 100},
		{Name: "small", Files: []string{"small.gguf"}, Size: 10},
		{Name: "spark", Files: []string{"spark.gguf"}, Draft: "dspark", DraftFor: "Org/Big", Size: 5},
		{Name: "other-spark", Files: []string{"os.gguf"}, Draft: "eagle3", DraftFor: "Org/Else", Size: 5},
		{Name: "named", Files: []string{"n.gguf"}, Draft: "eagle3", Repo: "x/Big-EAGLE3-Speculator-GGUF", Size: 5},
		{Name: "unnamed", Files: []string{"u.gguf"}, Draft: "dflash", Repo: "x/Other-DFlash-GGUF", Size: 5},
	} {
		if err := st.Put(m); err != nil {
			t.Fatal(err)
		}
	}
	m, _ := st.Get("big")
	return e, m
}

func argsFor(e *Engine, m *Model, opts map[string]any) []string {
	o := planSpeed(m, loadOptsFrom(m.options(opts)))
	return e.args(m, o, 1234)
}

func TestSpeedAuto(t *testing.T) {
	e, big := speedEngine(t)
	plain, _ := e.store.Get("plain")

	types, md := specFlags(argsFor(e, big, nil))
	if types != "draft-mtp,ngram-mod" || !strings.HasSuffix(md, "big-mtp.gguf") {
		t.Errorf("auto with a head: %q, %q", types, md)
	}
	// No head: only the n-gram guesser, which needs no model.
	types, md = specFlags(argsFor(e, plain, nil))
	if types != "ngram-mod" || md != "" {
		t.Errorf("auto without a head: %q, %q", types, md)
	}
}

func TestSpeedChoices(t *testing.T) {
	e, big := speedEngine(t)
	for _, c := range []struct {
		name string
		opts map[string]any
		want string
	}{
		{"off", map[string]any{"speedup": "off"}, ""},
		{"off keeps an asked-for n-gram", map[string]any{"speedup": "off", "spec_ngram": "ngram-cache"}, "ngram-cache"},
		{"mtp only", map[string]any{"speedup": "mtp", "spec_ngram": "off"}, "draft-mtp"},
		{"n-gram off under auto", map[string]any{"spec_ngram": "off"}, "draft-mtp"},
		{"n-gram type", map[string]any{"spec_ngram": "ngram-simple"}, "draft-mtp,ngram-simple"},
		{"n-gram on under off", map[string]any{"speedup": "off", "spec_ngram": "on"}, "ngram-mod"},
		{"a chosen draft beats the head", map[string]any{"draft_model": "small", "spec_ngram": "off"}, "draft-simple"},
		{"mtp beats a chosen draft", map[string]any{"speedup": "mtp", "draft_model": "small", "spec_ngram": "off"}, "draft-mtp"},
		{"off ignores a chosen draft", map[string]any{"speedup": "off", "draft_model": "small"}, ""},
		{"dspark draft", map[string]any{"draft_model": "spark", "spec_ngram": "off"}, "draft-dspark"},
		{"draft and n-gram", map[string]any{"draft_model": "small"}, "draft-simple,ngram-mod"},
	} {
		types, _ := specFlags(argsFor(e, big, c.opts))
		if types != c.want {
			t.Errorf("%s: --spec-type %q, want %q", c.name, types, c.want)
		}
	}
	a := argsFor(e, big, map[string]any{"draft_max": int64(6)})
	if i := slices.Index(a, "--spec-draft-n-max"); i < 0 || a[i+1] != "6" {
		t.Errorf("draft length lost: %v", a)
	}
	// Without a draft the length means nothing.
	plain, _ := e.store.Get("plain")
	if a := argsFor(e, plain, map[string]any{"draft_max": int64(6)}); slices.Contains(a, "--spec-draft-n-max") {
		t.Errorf("draft length without a draft: %v", a)
	}
}

func TestSpeedEmbeddingGetsNone(t *testing.T) {
	e, big := speedEngine(t)
	o := loadOptsFrom(big.options(nil))
	o.Embedding = true
	if a := e.args(big, planSpeed(big, o), 1); slices.Contains(a, "--spec-type") || slices.Contains(a, "-md") {
		t.Errorf("embedding model speculates: %v", a)
	}
}

func TestDraftChoices(t *testing.T) {
	e, big := speedEngine(t)
	got, err := e.Drafts("big")
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, d := range got {
		names = append(names, d.Name)
	}
	// "spark" was made for Org/Big (matches o/Big-GGUF); "other-spark" for another model.
	if !slices.Contains(names, "spark") || slices.Contains(names, "other-spark") || slices.Contains(names, "big") {
		t.Errorf("drafts for big = %v", names)
	}
	if !slices.Contains(names, "named") || slices.Contains(names, "unnamed") {
		t.Errorf("a draft with no stated base is matched by name: %v", names)
	}
	if err := e.checkDraft(big, "other-spark"); err == nil {
		t.Error("accepted a draft made for another model")
	}
	if err := e.checkDraft(big, "spark"); err != nil {
		t.Error(err)
	}
}

func TestPullBringsMTPHead(t *testing.T) {
	h := &fakeHF{files: map[string][]byte{
		"M-UD-Q4_K_XL.gguf": []byte("weights"),
		"mtp-M.gguf":        []byte("the mtp head"),
	}}
	e := testEngine(t, h)
	if err := e.Pull(context.Background(), "hf.co/o/M-GGUF:UD-Q4_K_XL", func(ollama.PullProgress) error { return nil }); err != nil {
		t.Fatal(err)
	}
	m, err := e.store.Get("hf.co/o/M-GGUF:UD-Q4_K_XL")
	if err != nil {
		t.Fatal(err)
	}
	if b, _ := os.ReadFile(e.store.Blob(m.MTP)); string(b) != "the mtp head" {
		t.Errorf("MTP head = %q (%q)", b, m.MTP)
	}
	if m.Size != int64(len("weights")+len("the mtp head")) {
		t.Errorf("size = %d", m.Size)
	}
	if err := e.Delete(context.Background(), m.Name); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(e.store.Blob(m.MTP)); !os.IsNotExist(err) {
		t.Errorf("head left behind: %v", err)
	}
}

func TestPullDraftMarksModel(t *testing.T) {
	h := &fakeHF{files: map[string][]byte{
		"M-Q4_K_M.gguf":      []byte("weights"),
		"dspark-M-Q8_0.gguf": []byte("a draft"),
		"mtp-M.gguf":         []byte("the mtp head"),
	}}
	e := testEngine(t, h)
	if err := e.Pull(context.Background(), "hf.co/o/M-GGUF:DSpark-Q8_0", func(ollama.PullProgress) error { return nil }); err != nil {
		t.Fatal(err)
	}
	m, err := e.store.Get("hf.co/o/M-GGUF:DSpark-Q8_0")
	if err != nil {
		t.Fatal(err)
	}
	if m.Draft != "dspark" || m.DraftFor != "o/M-GGUF" || m.MTP != "" {
		t.Errorf("draft = %+v", m)
	}
	list, _ := e.List(context.Background())
	if len(list) != 0 {
		t.Errorf("a draft can't chat, so it isn't a model to pick: %v", list)
	}
}

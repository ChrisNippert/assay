package engine

import (
	"bytes"
	"context"
	"encoding/binary"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"assay/internal/presets"
)

// ggufHeader is a GGUF file with only string metadata: enough for the
// engine to read its architecture and type.
func ggufHeader(kv ...string) []byte {
	var b bytes.Buffer
	w := func(v any) { binary.Write(&b, binary.LittleEndian, v) }
	str := func(s string) { w(uint64(len(s))); b.WriteString(s) }
	b.WriteString("GGUF")
	w(uint32(3))
	w(uint64(0))
	w(uint64(len(kv) / 2))
	for i := 0; i < len(kv); i += 2 {
		str(kv[i])
		w(uint32(8)) // string
		str(kv[i+1])
	}
	return b.Bytes()
}

// loraSource writes a LoRA adapter for arch into dir and returns its path.
func loraSource(t *testing.T, dir, name, arch string) string {
	t.Helper()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, ggufHeader("general.architecture", arch, "general.type", "adapter", "adapter.type", "lora"), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// loraStore is a store with one model (base:1, qwen35) and an engine on it.
func loraStore(t *testing.T) (*Engine, *Store, string) {
	t.Helper()
	dir := t.TempDir()
	st, err := OpenStore(filepath.Join(dir, "store"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(st.Blob("base.gguf"), ggufHeader("general.architecture", "qwen35"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := st.Put(&Model{Name: "base:1", Files: []string{"base.gguf"}, Source: "hf"}); err != nil {
		t.Fatal(err)
	}
	return New(st, Options{}), st, filepath.Join(dir, "src")
}

func TestAddAdapter(t *testing.T) {
	e, st, src := loraStore(t)
	if err := os.Mkdir(src, 0o755); err != nil {
		t.Fatal(err)
	}
	file := loraSource(t, src, "tuned.gguf", "qwen35")

	a, err := e.AddAdapter(file, "")
	if err != nil {
		t.Fatal(err)
	}
	if a.Name != "tuned" || a.Arch != "qwen35" {
		t.Fatalf("added %+v", a)
	}
	m, err := st.Get("tuned")
	if err != nil {
		t.Fatal(err)
	}
	// Same disk: a hard link, so the original is still there and untouched.
	if m.Link != "hard" || m.Adapter == nil || m.Adapter.Arch != "qwen35" {
		t.Errorf("manifest = %+v", m)
	}
	orig, _ := os.Stat(file)
	dst, _ := os.Stat(st.Blob(m.Files[0]))
	if !os.SameFile(orig, dst) {
		t.Error("not a hard link of the original")
	}

	// Adapters are listed apart from models, and the models list is unchanged.
	if got := e.Adapters(); len(got) != 1 || got[0].Name != "tuned" {
		t.Errorf("adapters = %+v", got)
	}
	if list := st.List(); len(list) != 1 || list[0].Name != "base:1" {
		t.Errorf("models = %v", list)
	}

	// Names are unique across models and adapters; only .gguf files that
	// are LoRA adapters are taken.
	if _, err := e.AddAdapter(file, "base:1"); err == nil {
		t.Error("a name the library has was accepted")
	}
	if _, err := e.AddAdapter(file, "tuned"); err == nil {
		t.Error("a duplicate name was accepted")
	}
	plain := filepath.Join(src, "plain.gguf")
	os.WriteFile(plain, ggufHeader("general.architecture", "qwen35"), 0o644)
	if _, err := e.AddAdapter(plain, ""); err == nil {
		t.Error("a model file was accepted as an adapter")
	}
	if _, err := e.AddAdapter("relative.gguf", ""); err == nil {
		t.Error("a relative path was accepted")
	}
	if _, err := e.AddAdapter(filepath.Join(src, "missing.gguf"), ""); err == nil {
		t.Error("a missing file was accepted")
	}

	// Removing the adapter forgets it and leaves the original alone.
	if err := st.Delete("tuned"); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(file); err != nil {
		t.Errorf("the original was removed: %v", err)
	}
	if len(e.Adapters()) != 0 {
		t.Error("the adapter is still listed")
	}
}

func TestAddAdapterInPlace(t *testing.T) {
	e, st, src := loraStore(t)
	os.Mkdir(src, 0o755)
	file := loraSource(t, src, "far.gguf", "qwen35")
	// As if the file were on another disk: the link fails.
	old := hardLink
	hardLink = func(string, string) error { return &os.LinkError{Op: "link", Err: os.ErrPermission} }
	defer func() { hardLink = old }()

	if _, err := e.AddAdapter(file, "far"); err != nil {
		t.Fatal(err)
	}
	m, _ := st.Get("far")
	if m.Link != "in place" || len(m.Files) != 1 || m.Files[0] != file || m.Origin != file {
		t.Errorf("in-place manifest = %+v", m)
	}
	if _, err := os.Stat(file); err != nil {
		t.Fatal(err)
	}
}

func TestAdaptersForArchitecture(t *testing.T) {
	e, st, src := loraStore(t)
	os.Mkdir(src, 0o755)
	if _, err := e.AddAdapter(loraSource(t, src, "for-qwen.gguf", "qwen35"), ""); err != nil {
		t.Fatal(err)
	}
	if _, err := e.AddAdapter(loraSource(t, src, "for-llama.gguf", "llama"), ""); err != nil {
		t.Fatal(err)
	}
	got, err := e.AdaptersFor("base:1")
	if err != nil || len(got) != 1 || got[0].Name != "for-qwen" {
		t.Fatalf("compatible = %+v, %v", got, err)
	}
	if got, _ := e.AdaptersFor("for-qwen"); got != nil {
		t.Errorf("an adapter offers adapters: %+v", got)
	}

	// The adapters take no part in listing, drafting or Show, and can't load.
	ctx := context.Background()
	if _, _, err := e.ensure(ctx, "for-qwen", nil); err == nil || !strings.Contains(err.Error(), "not a model") {
		t.Errorf("ensure on an adapter: %v", err)
	}
	if _, err := e.Show(ctx, "for-qwen"); err == nil {
		t.Error("Show answered for an adapter")
	}
	if ds, _ := e.Drafts("base:1"); len(ds) != 0 {
		t.Errorf("drafts = %+v", ds)
	}

	// checkLoRA: the right architecture passes; others are named.
	base, _ := st.Get("base:1")
	if err := e.checkLoRA(base, []presets.LoRA{{Name: "for-qwen", Scale: 1}}); err != nil {
		t.Errorf("compatible adapter refused: %v", err)
	}
	if err := e.checkLoRA(base, []presets.LoRA{{Name: "for-llama", Scale: 1}}); err == nil || !strings.Contains(err.Error(), "for llama") {
		t.Errorf("wrong architecture: %v", err)
	}
	if err := e.checkLoRA(base, []presets.LoRA{{Name: "base:1", Scale: 1}}); err == nil {
		t.Error("a model was accepted as an adapter")
	}
	if err := e.checkLoRA(base, []presets.LoRA{{Name: "nope", Scale: 1}}); err == nil || !strings.Contains(err.Error(), "isn't installed") {
		t.Errorf("not installed: %v", err)
	}
	if err := e.checkLoRA(base, nil); err != nil {
		t.Errorf("no adapters: %v", err)
	}
}

func TestLoRAArgs(t *testing.T) {
	e, st, src := loraStore(t)
	os.Mkdir(src, 0o755)
	if _, err := e.AddAdapter(loraSource(t, src, "a.gguf", "qwen35"), "a"); err != nil {
		t.Fatal(err)
	}
	if _, err := e.AddAdapter(loraSource(t, src, "b.gguf", "qwen35"), "b"); err != nil {
		t.Fatal(err)
	}
	base, _ := st.Get("base:1")
	o := loadOptsFrom(map[string]any{"lora": []any{
		map[string]any{"name": "b", "scale": 0.5},
		map[string]any{"name": "a", "scale": 1.0},
	}})
	if !reflect.DeepEqual(o.LoRA, []presets.LoRA{{Name: "b", Scale: 0.5}, {Name: "a", Scale: 1}}) {
		t.Fatalf("loadOptsFrom = %+v", o.LoRA)
	}
	a := strings.Join(e.args(base, o, 1), " ")
	// Adapters load unapplied, in the order given; their scales come next.
	bi := strings.Index(a, "--lora "+st.Blob(mustGet(t, st, "b").Files[0]))
	ai := strings.Index(a, "--lora "+st.Blob(mustGet(t, st, "a").Files[0]))
	if bi < 0 || ai < 0 || bi > ai || !strings.Contains(a, "--lora-init-without-apply") {
		t.Errorf("args = %s", a)
	}
	if a := strings.Join(e.args(base, loadOptsFrom(nil), 1), " "); strings.Contains(a, "--lora") {
		t.Errorf("args without adapters = %s", a)
	}
}

func mustGet(t *testing.T, st *Store, name string) *Model {
	t.Helper()
	m, err := st.Get(name)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func TestLoRALoadIdentity(t *testing.T) {
	ab := func(scale float64) []presets.LoRA { return []presets.LoRA{{Name: "a", Scale: scale}} }
	base := loadOpts{GPULayers: -1, LoRA: ab(1)}
	cases := []struct {
		name         string
		want         loadOpts
		same, scales bool
	}{
		{"identical", loadOpts{GPULayers: -1, LoRA: ab(1)}, true, true},
		{"scale only", loadOpts{GPULayers: -1, LoRA: ab(0.5)}, false, true},
		{"adapter removed", loadOpts{GPULayers: -1}, false, false},
		{"adapter renamed", loadOpts{GPULayers: -1, LoRA: []presets.LoRA{{Name: "b", Scale: 1}}}, false, false},
		{"other option", loadOpts{GPULayers: 10, LoRA: ab(1)}, false, false},
	}
	for _, c := range cases {
		if got := base.sameLoad(c.want); got != c.same {
			t.Errorf("%s: sameLoad = %v", c.name, got)
		}
		if got := base.sameNames(c.want); got != c.scales {
			t.Errorf("%s: sameNames = %v", c.name, got)
		}
	}
}

func TestSetScalesLive(t *testing.T) {
	var got string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/lora-adapters" || r.Method != http.MethodPost {
			http.NotFound(w, r)
			return
		}
		b, _ := io.ReadAll(r.Body)
		got = string(b)
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()
	e := New(nil, Options{})
	p := &proc{base: srv.URL}
	uses := []presets.LoRA{{Name: "a", Scale: 0.5}, {Name: "b", Scale: 2}}
	if err := e.setScales(context.Background(), p, uses); err != nil {
		t.Fatal(err)
	}
	if got != `[{"id":0,"scale":0.5},{"id":1,"scale":2}]` {
		t.Errorf("sent %s", got)
	}

	failing := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "no", http.StatusBadRequest)
	}))
	defer failing.Close()
	if err := e.setScales(context.Background(), &proc{base: failing.URL}, uses); err == nil {
		t.Error("a refused scale change reported success")
	}
}

func TestFoundAdapters(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "tuned.gguf"), ggufHeader("general.architecture", "qwen35", "general.type", "adapter", "adapter.type", "lora"), 0o644)
	os.WriteFile(filepath.Join(dir, "model.gguf"), ggufHeader("general.architecture", "qwen35"), 0o644)

	f := &Finder{}
	models := scanFolder(dir)
	f.markAdapters(models)
	var adapters, plain int
	for _, c := range models {
		switch {
		case c.Adapter && c.Arch == "qwen35" && strings.HasPrefix(c.Name, "tuned"):
			adapters++
		case !c.Adapter && strings.HasPrefix(c.Name, "model"):
			plain++
		}
	}
	if adapters != 1 || plain != 1 {
		t.Fatalf("candidates = %+v", models)
	}
	// The header is remembered while the file doesn't change.
	if len(f.kinds) != 2 {
		t.Errorf("kinds = %+v", f.kinds)
	}

	// Import all takes the model only; the adapter is picked by name.
	sources := []Source{{ID: "folder", Dir: dir, Models: models}}
	all, _ := pickFound(sources, "folder", dir, nil)
	if len(all) != 1 || all[0].Adapter {
		t.Errorf("import all = %+v", all)
	}
	named, err := pickFound(sources, "folder", dir, []string{"tuned"})
	if err != nil || len(named) != 1 || !named[0].Adapter {
		t.Fatalf("named = %+v, %v", named, err)
	}

	st, _ := OpenStore(t.TempDir())
	if n, err := st.ImportFound("folder", named); err != nil || n != 1 {
		t.Fatalf("import = %d, %v", n, err)
	}
	m, err := st.Get("tuned")
	if err != nil || m.Adapter == nil || m.Adapter.Arch != "qwen35" {
		t.Fatalf("imported adapter = %+v, %v", m, err)
	}
	if len(st.List()) != 0 {
		t.Error("an adapter showed up as a model")
	}
}

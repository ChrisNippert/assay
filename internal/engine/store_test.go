package engine

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"ollamui/internal/ollama"
)

// fakeOllama lays out an Ollama models folder the way 0.40 does: v1
// manifests for hf.co names, manifests-v2 symlinks for short names.
func fakeOllama(t *testing.T) string {
	dir := t.TempDir()
	write := func(path, data string) {
		os.MkdirAll(filepath.Dir(filepath.Join(dir, path)), 0o755)
		if err := os.WriteFile(filepath.Join(dir, path), []byte(data), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("blobs/sha256-aaa", "weights")
	write("blobs/sha256-ppp", "projector")
	write("blobs/sha256-par", `{"temperature":0.6,"stop":["<|im_end|>"]}`)
	man := `{"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:aaa","size":7},` +
		`{"mediaType":"application/vnd.ollama.image.projector","digest":"sha256:ppp","size":9},` +
		`{"mediaType":"application/vnd.ollama.image.params","digest":"sha256:par","size":40}]}`
	write("manifests/hf.co/unsloth/Qwen3-8B-GGUF/Q4_K_M", man)
	write("blobs/sha256-man2", man)
	os.MkdirAll(filepath.Join(dir, "manifests-v2/ollama.com/library/qwen3-8b"), 0o755)
	if err := os.Symlink("../../../../blobs/sha256-man2", filepath.Join(dir, "manifests-v2/ollama.com/library/qwen3-8b/q4_k_m")); err != nil {
		t.Fatal(err)
	}
	return dir
}

func TestImportOllama(t *testing.T) {
	src := fakeOllama(t)
	s, err := OpenStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	n, err := s.ImportOllama(src)
	if err != nil || n != 1 {
		t.Fatalf("import: %d %v", n, err)
	}
	m, err := s.Get("qwen3-8b:q4_k_m")
	if err != nil {
		t.Fatal(err)
	}
	if m.Name != "hf.co/unsloth/Qwen3-8B-GGUF:Q4_K_M" || len(m.Aliases) != 1 || m.Repo != "unsloth/Qwen3-8B-GGUF" || m.Quant != "Q4_K_M" {
		t.Errorf("model = %+v", m)
	}
	if m.Projector == "" || m.Size != 16 || m.Params["temperature"] != 0.6 || m.Digest != "aaa" {
		t.Errorf("files/params = %+v", m)
	}
	if b, _ := os.ReadFile(s.Blob(m.Files[0])); string(b) != "weights" {
		t.Errorf("blob not linked: %q", b)
	}
	if n, _ := s.ImportOllama(src); n != 0 {
		t.Errorf("second import added %d", n)
	}

	// Deleting an alias keeps the model; deleting the model removes files.
	if err := s.Delete("qwen3-8b:q4_k_m"); err != nil {
		t.Fatal(err)
	}
	if m, err := s.Get("hf.co/unsloth/Qwen3-8B-GGUF:Q4_K_M"); err != nil || len(m.Aliases) != 0 {
		t.Fatalf("after alias delete: %+v %v", m, err)
	}
	blob := s.Blob(m.Files[0])
	if err := s.Delete("hf.co/unsloth/Qwen3-8B-GGUF:Q4_K_M"); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(blob); !os.IsNotExist(err) {
		t.Error("blob left behind")
	}
	if _, err := os.Stat(filepath.Join(src, "blobs/sha256-aaa")); err != nil {
		t.Error("deleting here must not touch Ollama's copy")
	}
}

func TestOllamaName(t *testing.T) {
	for in, want := range map[string]string{
		"hf.co/unsloth/Qwen3-8B-GGUF/Q8_0":        "hf.co/unsloth/Qwen3-8B-GGUF:Q8_0",
		"registry.ollama.ai/library/qwen3/8b":     "qwen3:8b",
		"ollama.com/library/gemma-4-12b/q4":       "gemma-4-12b:q4",
		"registry.ollama.ai/someone/thing/latest": "someone/thing:latest",
		"x/y": "",
	} {
		if got := ollamaName(in); got != want {
			t.Errorf("ollamaName(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestGfxOverride(t *testing.T) {
	for v, want := range map[string]string{"100301": "10.3.0", "100300": "", "110002": "11.0.0", "110000": "", "110001": "", "90012": "", "0": ""} {
		if got := gfxOverride(v); got != want {
			t.Errorf("gfxOverride(%s) = %q, want %q", v, got, want)
		}
	}
}

func TestModelSettingsLayering(t *testing.T) {
	st, err := OpenStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	e := New(st, Options{CacheType: "q8_0", FlashAttn: "on"})
	m := &Model{Name: "x:1", Files: []string{"a.gguf"},
		Params:   map[string]any{"temperature": 0.6, "top_k": 20},
		Settings: map[string]any{"temperature": 0.9, "cache_type": "f16", "flash_attn": "off", "num_ctx": 8192.0, "keep_alive": "-1"}}
	if err := st.Put(m); err != nil {
		t.Fatal(err)
	}
	m, _ = st.Get("x:1")
	o := loadOptsFrom(m.options(map[string]any{"num_ctx": 4096}))
	if o.Ctx != 4096 || o.CacheType != "f16" || o.FlashAttn != "off" {
		t.Errorf("load opts = %+v", o)
	}
	a := strings.Join(e.args(m, o, 1), " ")
	if !strings.Contains(a, "-c 4096") || !strings.Contains(a, "-fa off") || strings.Contains(a, "-ctk") {
		t.Errorf("args = %s", a)
	}
	// A quantized cache turns flash attention back on.
	a = strings.Join(e.args(m, loadOpts{GPULayers: -1, CacheType: "q4_0", FlashAttn: "off"}, 1), " ")
	if !strings.Contains(a, "-fa on") || !strings.Contains(a, "-ctk q4_0") {
		t.Errorf("args = %s", a)
	}
	b := chatBody(m, ollama.ChatRequest{Options: map[string]any{"top_k": 5}})
	if b["temperature"] != 0.9 || b["top_k"] != 5 {
		t.Errorf("body = %v", b)
	}
	if ka := modelKeepAlive(m, nil); ka >= 0 {
		t.Errorf("keep alive = %v, want pinned", ka)
	}
	if err := e.SetSettings("x:1", nil); err != nil {
		t.Fatal(err)
	}
	if s, _ := e.Settings("x:1"); s != nil {
		t.Errorf("settings left: %v", s)
	}
}

func TestReplyFormatFields(t *testing.T) {
	cases := []struct {
		opts map[string]any
		want string
	}{
		{map[string]any{"reply_format": "text"}, ""},
		{map[string]any{"reply_format": "json"}, `{"response_format":{"type":"json_object"}}`},
		{map[string]any{"reply_format": "schema", "reply_schema": `{"type":"object"}`}, `{"response_format":{"json_schema":{"schema":{"type":"object"}},"type":"json_schema"}}`},
		{map[string]any{"reply_format": "schema"}, ""},
		{map[string]any{"reply_format": "grammar", "reply_grammar": `root ::= "a"`}, `{"grammar":"root ::= \"a\""}`},
		{map[string]any{"reply_schema": `{"type":"object"}`}, ""},
	}
	for _, c := range cases {
		got := replyFields(c.opts)
		if c.want == "" {
			if got != nil {
				t.Errorf("%v: got %v", c.opts, got)
			}
			continue
		}
		if b, _ := json.Marshal(got); string(b) != c.want {
			t.Errorf("%v: got %s, want %s", c.opts, b, c.want)
		}
	}
	m := &Model{Name: "x:1", Files: []string{"a.gguf"}}
	b := chatBody(m, ollama.ChatRequest{Options: map[string]any{"reply_format": "json", "temperature": 0.1}})
	if rf, _ := b["response_format"].(map[string]any); rf["type"] != "json_object" || b["temperature"] != 0.1 {
		t.Errorf("body = %v", b)
	}
	if _, ok := b["reply_format"]; ok {
		t.Error("the setting itself leaked into the body")
	}
}

func TestDraftAndThreadArgs(t *testing.T) {
	st, err := OpenStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	e := New(st, Options{CacheType: "q8_0", FlashAttn: "on"})
	for _, m := range []*Model{{Name: "big:1", Files: []string{"big.gguf"}}, {Name: "small:1", Files: []string{"small.gguf"}}} {
		if err := st.Put(m); err != nil {
			t.Fatal(err)
		}
	}
	m, _ := st.Get("big:1")
	o := loadOptsFrom(map[string]any{"draft_model": "small:1", "draft_max": 8.0, "num_thread": 6.0})
	a := strings.Join(e.args(m, o, 1), " ")
	for _, want := range []string{"-md " + st.Blob("small.gguf"), "--spec-draft-n-max 8", "-t 6"} {
		if !strings.Contains(a, want) {
			t.Errorf("args lack %q: %s", want, a)
		}
	}
	if a := strings.Join(e.args(m, loadOptsFrom(nil), 1), " "); strings.Contains(a, "-md") || strings.Contains(a, " -t ") {
		t.Errorf("args = %s", a)
	}
}

func TestExpertPlacementArgs(t *testing.T) {
	st, err := OpenStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	e := New(st, Options{CacheType: "f16", FlashAttn: "on"})
	m := &Model{Name: "moe:1", Files: []string{"moe.gguf"}}
	args := func(o map[string]any) string { return strings.Join(e.args(m, loadOptsFrom(o), 1), " ") }

	// Auto passes none of it: llama.cpp fits the layers and reads the file.
	if a := args(nil); strings.Contains(a, "-ngl") || strings.Contains(a, "cpu-moe") || strings.Contains(a, "load-mode") {
		t.Errorf("auto args = %s", a)
	}
	// All experts on the CPU, with everything else on the GPU.
	if a := args(map[string]any{"n_cpu_moe": int64(-1), "load_mode": "none"}); !strings.Contains(a, "-ngl all") || !strings.Contains(a, " --cpu-moe ") || !strings.Contains(a, "--load-mode none") {
		t.Errorf("all experts args = %s", a)
	}
	// The first N layers' experts; 0 keeps them all on the GPU.
	if a := args(map[string]any{"n_cpu_moe": 14.0, "load_mode": "mlock"}); !strings.Contains(a, "-ngl all --n-cpu-moe 14 --load-mode mlock") {
		t.Errorf("layer count args = %s", a)
	}
	if a := args(map[string]any{"n_cpu_moe": 0.0}); !strings.Contains(a, "--n-cpu-moe 0") {
		t.Errorf("zero args = %s", a)
	}
	// A GPU layer count is still passed as it was set.
	if a := args(map[string]any{"num_gpu": 20.0, "n_cpu_moe": 4.0}); !strings.Contains(a, "-ngl 20 --n-cpu-moe 4") {
		t.Errorf("both args = %s", a)
	}
	// Values the presets don't allow are ignored.
	if o := loadOptsFrom(map[string]any{"load_mode": "dio", "n_cpu_moe": -5.0}); o.LoadMode != "" || o.CPUMoE != cpuMoEUnset {
		t.Errorf("bad values loaded as %+v", o)
	}
}

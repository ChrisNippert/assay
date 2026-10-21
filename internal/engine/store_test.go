package engine

import (
	"os"
	"path/filepath"
	"testing"
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

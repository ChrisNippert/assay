package engine

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestReadableName(t *testing.T) {
	for in, want := range map[string]string{
		"hf.co/unsloth/Qwen3.5-2B-GGUF:Q4_K_M": "unsloth-Qwen3.5-2B-GGUF-Q4_K_M",
		"qwen3.5-9b:q8_0":                      "qwen3.5-9b-q8_0",
		"my model (v2)":                        "my-model-v2",
	} {
		if got := readableName(in); got != want {
			t.Errorf("readableName(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestByNameLinks(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("no symlinks on Windows")
	}
	dir := t.TempDir()
	s, err := OpenStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	os.WriteFile(filepath.Join(dir, "blobs", "sha256-a.gguf"), []byte("a"), 0o644)
	os.WriteFile(filepath.Join(dir, "blobs", "sha256-p.gguf"), []byte("p"), 0o644)
	if err := s.Put(&Model{Name: "hf.co/o/R-GGUF:Q4_K_M", Files: []string{"sha256-a.gguf"}, Projector: "sha256-p.gguf"}); err != nil {
		t.Fatal(err)
	}
	links := filepath.Join(dir, "by-name")
	for _, n := range []string{"o-R-GGUF-Q4_K_M.gguf", "o-R-GGUF-Q4_K_M.mmproj.gguf"} {
		if b, err := os.ReadFile(filepath.Join(links, n)); err != nil || len(b) != 1 {
			t.Errorf("%s: %v", n, err)
		}
	}
	// A file of the user's own is left alone; a stale link goes.
	os.WriteFile(filepath.Join(links, "notes.txt"), []byte("mine"), 0o644)
	os.Symlink("../blobs/gone.gguf", filepath.Join(links, "old.gguf"))
	if err := s.Delete("hf.co/o/R-GGUF:Q4_K_M"); err != nil {
		t.Fatal(err)
	}
	ents, _ := os.ReadDir(links)
	if len(ents) != 1 || ents[0].Name() != "notes.txt" {
		t.Errorf("after delete: %v", ents)
	}
	clearByName(dir)
	if _, err := os.Stat(filepath.Join(links, "notes.txt")); err != nil {
		t.Errorf("clearByName removed a user file: %v", err)
	}
}

func TestSplitModelLinks(t *testing.T) {
	got := byNameLinks([]*Model{{Name: "big", Files: []string{"sha256-1.gguf", "/abs/2.gguf"}}})
	if got["big-00001-of-00002.gguf"] != filepath.Join("..", "blobs", "sha256-1.gguf") || got["big-00002-of-00002.gguf"] != "/abs/2.gguf" {
		t.Errorf("links: %v", got)
	}
}

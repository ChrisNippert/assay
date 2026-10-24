package engine

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestSetServer(t *testing.T) {
	dir := t.TempDir()
	e := New(nil, Options{Server: filepath.Join(dir, "a", "llama-server")})
	if e.ServerPath() != filepath.Join(dir, "a", "llama-server") {
		t.Fatalf("path %q", e.ServerPath())
	}
	os.MkdirAll(filepath.Join(dir, "b"), 0o755)
	os.WriteFile(filepath.Join(dir, "b", "libggml-vulkan.so"), nil, 0o644)
	e.SetServer(filepath.Join(dir, "b", "llama-server"))
	if e.ServerPath() != filepath.Join(dir, "b", "llama-server") || e.Backend() != "Vulkan" {
		t.Errorf("after SetServer: %q %q", e.ServerPath(), e.Backend())
	}
	e.SetServer("")
	if _, err := e.start(context.Background(), &Model{}, loadOpts{}); err == nil {
		t.Error("started without a server")
	}
}

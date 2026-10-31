package main

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestRenamedMovesOldFolder(t *testing.T) {
	parent := t.TempDir()
	old := filepath.Join(parent, "ollamui")
	os.MkdirAll(filepath.Join(old, "chats"), 0o755)
	os.WriteFile(filepath.Join(old, "chats", "a.json"), []byte("{}"), 0o644)
	dir := renamed(parent)
	if dir != filepath.Join(parent, "assay") {
		t.Fatalf("dir = %s", dir)
	}
	if _, err := os.Stat(filepath.Join(dir, "chats", "a.json")); err != nil {
		t.Errorf("not moved: %v", err)
	}
	if runtime.GOOS != "windows" {
		if _, err := os.Stat(filepath.Join(old, "chats", "a.json")); err != nil {
			t.Errorf("old path doesn't lead to the data: %v", err)
		}
	}
	// A second start leaves things as they are.
	if renamed(parent) != dir {
		t.Error("second call changed the folder")
	}
}

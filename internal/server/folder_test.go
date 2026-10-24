package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"ollamui/internal/chats"
	"ollamui/internal/engine"
	"ollamui/internal/ollama"
	"ollamui/internal/presets"
	"ollamui/internal/pulls"
)

type blockingPuller struct{ release chan struct{} }

func (b blockingPuller) Pull(ctx context.Context, model string, fn func(ollama.PullProgress) error) error {
	select {
	case <-b.release:
	case <-ctx.Done():
	}
	return nil
}
func (b blockingPuller) Copy(ctx context.Context, src, dst string) error { return nil }

func TestModelsFolderMove(t *testing.T) {
	base := t.TempDir()
	store, err := engine.OpenStore(filepath.Join(base, "models"))
	if err != nil {
		t.Fatal(err)
	}
	os.WriteFile(store.Blob("a.gguf"), []byte("weights"), 0o644)
	store.Put(&engine.Model{Name: "a:1", Files: []string{"a.gguf"}, Source: "hf"})
	eng := engine.New(store, engine.Options{})
	eng.Folder = engine.FolderSetting{File: filepath.Join(base, "config", "models.json")}

	bp := blockingPuller{release: make(chan struct{})}
	chatStore, _ := chats.Open(t.TempDir())
	presetStore, _ := presets.Open(t.TempDir())
	s, err := New(Options{Ollama: eng, Chats: chatStore, Presets: presetStore, Pulls: pulls.New(bp, nil)})
	if err != nil {
		t.Fatal(err)
	}
	do := func(method, path, body string) *httptest.ResponseRecorder {
		r := httptest.NewRequest(method, "http://localhost"+path, strings.NewReader(body))
		r.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		s.ServeHTTP(w, r)
		return w
	}
	dst := filepath.Join(base, "elsewhere")

	// A download in progress blocks the move.
	s.pulls.Start("some/model", "")
	if w := do("POST", "/api/models/folder", `{"path":"`+dst+`"}`); w.Code != http.StatusConflict {
		t.Fatalf("move during a download: %d %s", w.Code, w.Body)
	}
	close(bp.release)
	for i := 0; i < 100 && s.pullsRunning(); i++ {
		time.Sleep(10 * time.Millisecond)
	}

	if w := do("POST", "/api/models/folder/check", `{"path":"`+store.Dir()+`/blobs"}`); w.Code != http.StatusBadRequest {
		t.Errorf("target inside the store: %d %s", w.Code, w.Body)
	}
	if w := do("POST", "/api/models/folder", `{"path":"`+dst+`"}`); w.Code != http.StatusAccepted {
		t.Fatalf("move: %d %s", w.Code, w.Body)
	}
	var info struct {
		Dir  string
		Move struct{ State string }
	}
	for i := 0; i < 200; i++ {
		json.Unmarshal(do("GET", "/api/models/folder", "").Body.Bytes(), &info)
		if info.Move.State != "running" {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if info.Move.State != "done" || info.Dir != dst {
		t.Fatalf("after move: %+v", info)
	}
	if b, err := os.ReadFile(store.Blob("a.gguf")); err != nil || string(b) != "weights" {
		t.Errorf("blob after move: %q %v", b, err)
	}
	if got := engine.LoadFolder(eng.Folder.File); got != dst {
		t.Errorf("saved folder = %q", got)
	}
}

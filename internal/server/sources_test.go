package server

import (
	"encoding/json"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"assay/internal/chats"
	"assay/internal/engine"
	"assay/internal/presets"
)

func TestFoundFolderImport(t *testing.T) {
	base := t.TempDir()
	store, err := engine.OpenStore(filepath.Join(base, "models"))
	if err != nil {
		t.Fatal(err)
	}
	eng := engine.New(store, engine.Options{})
	eng.Finder = engine.NewFinder(filepath.Join(base, "config", "sources.json"))
	chatStore, _ := chats.Open(t.TempDir())
	presetStore, _ := presets.Open(t.TempDir())
	s, err := New(Options{Ollama: eng, Chats: chatStore, Presets: presetStore})
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
	folder := filepath.Join(base, "mine")
	os.MkdirAll(folder, 0o755)
	os.WriteFile(filepath.Join(folder, "tiny-Q4_K_M.gguf"), []byte("weights"), 0o644)

	if w := do("POST", "/api/models/sources/folder", `{"path":"relative"}`); w.Code != 400 {
		t.Errorf("relative path: %d", w.Code)
	}
	if w := do("POST", "/api/models/sources/folder", `{"path":"`+folder+`"}`); w.Code != 204 {
		t.Fatalf("add: %d %s", w.Code, w.Body)
	}
	var list struct {
		Sources []struct {
			ID, Dir string
			Models  []struct{ Name string }
		}
	}
	json.Unmarshal(do("GET", "/api/models/sources", "").Body.Bytes(), &list)
	var found bool
	for _, src := range list.Sources {
		if src.ID == "folder" && src.Dir == folder && len(src.Models) == 1 && src.Models[0].Name == "tiny:Q4_K_M" {
			found = true
		}
	}
	if !found {
		t.Fatalf("sources = %s", do("GET", "/api/models/sources", "").Body)
	}
	if w := do("POST", "/api/models/sources/import", `{"source":"folder","dir":"`+folder+`","names":["nope"]}`); w.Code != 400 {
		t.Errorf("unknown name: %d", w.Code)
	}
	if w := do("POST", "/api/models/sources/import", `{"source":"folder","dir":"`+folder+`","names":["tiny:Q4_K_M"]}`); w.Code != 200 {
		t.Fatalf("import: %d %s", w.Code, w.Body)
	}
	if m, err := store.Get("tiny:q4_k_m"); err != nil || m.Source != "folder" {
		t.Errorf("model = %+v, %v", m, err)
	}
	if page := do("GET", "/library", "").Body.String(); !strings.Contains(page, "linked from a folder") {
		t.Error("the library doesn't say where the model is linked from")
	}
	if w := do("DELETE", "/api/models/sources/folder", `{"path":"`+folder+`"}`); w.Code != 204 {
		t.Errorf("remove: %d", w.Code)
	}
	if _, err := os.Stat(filepath.Join(folder, "tiny-Q4_K_M.gguf")); err != nil {
		t.Error("the folder's file is gone")
	}
}

package scores

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

const fixture = `{"data":[
 {"id":"qwen/qwen3.5-9b","hugging_face_id":"Qwen/Qwen3.5-9B","benchmarks":{"artificial_analysis":{"intelligence_index":32.5}}},
 {"id":"qwen/qwen3.5-9b:free","hugging_face_id":"Qwen/Qwen3.5-9B","benchmarks":{"artificial_analysis":{"intelligence_index":32.5}}},
 {"id":"meta-llama/llama-3.3-70b-instruct","hugging_face_id":"","benchmarks":{"artificial_analysis":{"intelligence_index":14.5}}},
 {"id":"a/twin","hugging_face_id":null,"benchmarks":{"artificial_analysis":{"intelligence_index":10}}},
 {"id":"b/twin","hugging_face_id":null,"benchmarks":{"artificial_analysis":{"intelligence_index":20}}},
 {"id":"x/unrated","hugging_face_id":"x/Unrated","benchmarks":{"artificial_analysis":{"intelligence_index":null}}},
 {"id":"x/nobench","hugging_face_id":"x/NoBench","benchmarks":null}
]}`

func serve(t *testing.T, status *int) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if *status != 200 {
			http.Error(w, "nope", *status)
			return
		}
		w.Write([]byte(fixture))
	}))
	t.Cleanup(srv.Close)
	return srv
}

func TestRefreshAndLookup(t *testing.T) {
	status := 200
	srv := serve(t, &status)
	path := filepath.Join(t.TempDir(), "sub", "scores.json")
	s := Open(srv.URL, path, nil)
	if !s.Stale() {
		t.Error("empty store should be stale")
	}
	if _, ok := s.Lookup("Qwen/Qwen3.5-9B"); ok {
		t.Error("found a score before any refresh")
	}
	if err := s.Refresh(context.Background()); err != nil {
		t.Fatal(err)
	}
	if s.Stale() {
		t.Error("fresh store reports stale")
	}
	cases := []struct {
		id   string
		want float64
		ok   bool
	}{
		{"Qwen/Qwen3.5-9B", 32.5, true},
		{"qwen/qwen3.5-9b", 32.5, true},
		{"someone/Qwen3.5-9B", 32.5, true},               // name-only fallback
		{"bartowski/Llama-3.3-70B-Instruct", 14.5, true}, // via OpenRouter id
		{"foo/twin", 0, false},                           // ambiguous name
		{"x/Unrated", 0, false},
		{"x/NoBench", 0, false},
		{"nobody/nothing", 0, false},
	}
	for _, c := range cases {
		e, ok := s.Lookup(c.id)
		if ok != c.ok || e.Score != c.want {
			t.Errorf("Lookup(%q) = %v, %v; want %v, %v", c.id, e.Score, ok, c.want, c.ok)
		}
	}

	// The cache survives a restart, and a failing refresh keeps serving it.
	status = 500
	s2 := Open(srv.URL, path, nil)
	if s2.Stale() {
		t.Error("reloaded cache should be fresh")
	}
	if err := s2.Refresh(context.Background()); err == nil {
		t.Error("expected refresh error")
	}
	if e, ok := s2.Lookup("Qwen/Qwen3.5-9B"); !ok || e.Score != 32.5 {
		t.Errorf("stale data lost after failed refresh: %v %v", e, ok)
	}
}

func TestOfflineAndDamagedCache(t *testing.T) {
	path := filepath.Join(t.TempDir(), "scores.json")
	os.WriteFile(path, []byte("{not json"), 0o644)
	s := Open("http://127.0.0.1:1", path, nil)
	if err := s.Refresh(context.Background()); err == nil {
		t.Error("expected error when offline")
	}
	if _, ok := s.Lookup("a/b"); ok {
		t.Error("unexpected score")
	}
	var nilStore *Store
	if _, ok := nilStore.Lookup("a/b"); ok {
		t.Error("nil store returned a score")
	}
}

func TestBaseModel(t *testing.T) {
	tags := []string{"gguf", "base_model:Qwen/Qwen3-8B", "base_model:quantized:Qwen/Qwen3-8B"}
	if got := BaseModel(tags); got != "Qwen/Qwen3-8B" {
		t.Errorf("got %q", got)
	}
	if got := BaseModel([]string{"base_model:finetune:Qwen/Qwen3-8B", "base_model:Qwen/Qwen3-8B"}); got != "" {
		t.Errorf("finetune inherited a base: %q", got)
	}
}

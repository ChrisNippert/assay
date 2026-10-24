package engine

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// fakeProc puts a stand-in llama-server (an httptest server) in front of a
// model, so Proxy finds it already running.
func fakeProc(t *testing.T, h http.HandlerFunc, settings map[string]any) (*Engine, *proc) {
	t.Helper()
	store, err := OpenStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	m := &Model{Name: "hf.co/o/r:Q4", Aliases: []string{"short:q4"}, Files: []string{"x"}, Params: map[string]any{"temperature": 0.6, "top_k": 20.0}, Settings: settings}
	if err := store.Put(m); err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewServer(h)
	t.Cleanup(ts.Close)
	e := New(store, Options{})
	m, _ = store.Get(m.Name)
	p := &proc{model: m, opts: loadOptsFrom(m.options(nil)), base: ts.URL, done: make(chan struct{})}
	e.procs[m.Name] = p
	return e, p
}

func TestProxyFillsDefaultsAndRewritesModel(t *testing.T) {
	var got map[string]any
	e, p := fakeProc(t, func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&got)
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "data: {}\n\ndata: [DONE]\n\n")
	}, nil)
	body := `{"model":"short:q4","stream":true,"temperature":0.1,"tools":[{"type":"function","function":{"name":"f"}}],"messages":[]}`
	res, err := e.Proxy(context.Background(), "short:q4", map[string]any{"top_p": 0.5, "num_predict": 64.0}, "/v1/chat/completions", []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	p.mu.Lock()
	busy := p.busy
	p.mu.Unlock()
	if busy != 1 {
		t.Errorf("busy = %d while the body is open", busy)
	}
	b, _ := io.ReadAll(res.Body)
	res.Body.Close()
	res.Body.Close()
	if string(b) != "data: {}\n\ndata: [DONE]\n\n" {
		t.Errorf("body = %q", b)
	}
	if got["model"] != "hf.co/o/r:Q4" {
		t.Errorf("model = %v", got["model"])
	}
	if got["temperature"] != 0.1 || got["top_k"] != 20.0 || got["top_p"] != 0.5 || got["max_tokens"] != 64.0 {
		t.Errorf("sampling = %v", got)
	}
	if got["tools"] == nil || got["stream"] != true {
		t.Errorf("tools/stream not passed: %v", got)
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.busy != 0 || p.expires.Before(time.Now()) {
		t.Errorf("busy %d, expires %v", p.busy, p.expires)
	}
}

// A preset's reply format goes to llama-server only when the body asks for
// no format of its own.
func TestProxyReplyFormatFromPreset(t *testing.T) {
	var got map[string]any
	e, _ := fakeProc(t, func(w http.ResponseWriter, r *http.Request) {
		got = nil
		json.NewDecoder(r.Body).Decode(&got)
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{}`)
	}, nil)
	preset := map[string]any{"reply_format": "json"}
	send := func(body string) map[string]any {
		t.Helper()
		res, err := e.Proxy(context.Background(), "short:q4", preset, "/v1/chat/completions", []byte(body))
		if err != nil {
			t.Fatal(err)
		}
		res.Body.Close()
		return got
	}
	if rf, _ := send(`{"messages":[]}`)["response_format"].(map[string]any); rf["type"] != "json_object" {
		t.Errorf("preset format not applied: %v", got)
	}
	if rf, _ := send(`{"messages":[],"response_format":{"type":"text"}}`)["response_format"].(map[string]any); rf["type"] != "text" {
		t.Errorf("the body's own format was replaced: %v", got)
	}
	if got := send(`{"messages":[],"grammar":"root ::= \"a\""}`); got["response_format"] != nil || got["grammar"] != `root ::= "a"` {
		t.Errorf("grammar and format both set: %v", got)
	}
}

func TestProxyErrors(t *testing.T) {
	e, _ := fakeProc(t, func(w http.ResponseWriter, r *http.Request) {}, nil)
	_, err := e.Proxy(context.Background(), "nope", nil, "/v1/chat/completions", []byte(`{}`))
	if err == nil || err.Error() == "" {
		t.Fatal("want not found")
	}
	if _, err := e.Proxy(context.Background(), "short:q4", nil, "/v1/embeddings", []byte(`{}`)); err == nil {
		t.Error("a chat model must not embed")
	}
	if _, err := e.Proxy(context.Background(), "short:q4", nil, "/v1/chat/completions", []byte(`{`)); err == nil {
		t.Error("bad JSON must fail")
	}
}

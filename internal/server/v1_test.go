package server

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"assay/internal/ollama"
	"assay/internal/presets"
)

// v1Backend adds a recording Proxy to the fake Ollama backend.
type v1Backend struct {
	backend
	model   string
	options map[string]any
	path    string
	body    map[string]any
}

func (b *v1Backend) IsEmbedding(name string) bool { return name == "qwen3:8b-embed" }

func (b *v1Backend) Proxy(ctx context.Context, model string, options map[string]any, path string, body []byte) (*http.Response, error) {
	if model == "nope" {
		return nil, &ollama.APIError{StatusCode: 404, Message: "model nope not found"}
	}
	b.model, b.options, b.path = model, options, path
	json.Unmarshal(body, &b.body)
	rec := httptest.NewRecorder()
	rec.Header().Set("Content-Type", "text/event-stream")
	io.WriteString(rec, "data: {\"choices\":[]}\n\ndata: [DONE]\n\n")
	return rec.Result(), nil
}

func v1Request(s *Server, method, path, body string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(method, "http://localhost"+path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	s.ServeHTTP(rec, req)
	return rec
}

func TestV1Models(t *testing.T) {
	s, _, _ := newTestServer(t)
	s.ollama = &v1Backend{backend: s.ollama}
	if err := s.presets.Save(&presets.Preset{Name: "coder", Base: "qwen3:8b"}); err != nil {
		t.Fatal(err)
	}
	rec := v1Request(s, "GET", "/v1/models", "")
	var out struct {
		Object string
		Data   []struct{ ID string }
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil || out.Object != "list" {
		t.Fatalf("%d %s", rec.Code, rec.Body)
	}
	if len(out.Data) != 2 || out.Data[0].ID != "qwen3:8b" || out.Data[1].ID != "coder" {
		t.Errorf("data = %+v", out.Data)
	}
}

func TestV1ChatPreset(t *testing.T) {
	s, _, _ := newTestServer(t)
	b := &v1Backend{backend: s.ollama}
	s.ollama = b
	think := false
	s.presets.Save(&presets.Preset{Name: "coder", Base: "qwen3:8b", System: "Be terse.", Think: &think, Options: map[string]any{"temperature": 0.2}})

	rec := v1Request(s, "POST", "/v1/chat/completions", `{"model":"coder","stream":true,"tools":[{"type":"function"}],"messages":[{"role":"user","content":"hi"}]}`)
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), "[DONE]") || rec.Header().Get("Content-Type") != "text/event-stream" {
		t.Fatalf("%d %v %s", rec.Code, rec.Header(), rec.Body)
	}
	if b.model != "qwen3:8b" || b.options["temperature"] != 0.2 || b.path != "/v1/chat/completions" {
		t.Errorf("proxy got %s %v %s", b.model, b.options, b.path)
	}
	msgs := b.body["messages"].([]any)
	if len(msgs) != 2 || msgs[0].(map[string]any)["content"] != "Be terse." || b.body["tools"] == nil || b.body["reasoning_budget"] != 0.0 {
		t.Errorf("body = %v", b.body)
	}

	// The request's own system message wins over the preset's.
	v1Request(s, "POST", "/v1/chat/completions", `{"model":"coder","messages":[{"role":"system","content":"mine"}]}`)
	if msgs := b.body["messages"].([]any); len(msgs) != 1 {
		t.Errorf("messages = %v", msgs)
	}
}

func TestV1Errors(t *testing.T) {
	s, _, _ := newTestServer(t)
	rec := v1Request(s, "GET", "/v1/models", "")
	if rec.Code != http.StatusNotImplemented || !strings.Contains(rec.Body.String(), `"error":{"code":null`) {
		t.Errorf("ollama engine: %d %s", rec.Code, rec.Body)
	}
	s.ollama = &v1Backend{backend: s.ollama}
	rec = v1Request(s, "POST", "/v1/chat/completions", `{"model":"nope","messages":[]}`)
	if rec.Code != 404 || !strings.Contains(rec.Body.String(), `"type":"not_found_error"`) {
		t.Errorf("unknown model: %d %s", rec.Code, rec.Body)
	}
	if rec = v1Request(s, "POST", "/v1/chat/completions", `{"messages":[]}`); rec.Code != 400 {
		t.Errorf("no model: %d", rec.Code)
	}
}

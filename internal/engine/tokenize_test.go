package engine

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"ollamui/internal/ollama"
)

// TestTokenizeAndSummaryRequest runs Tokenize and a non-thinking chat against
// a fake llama-server that a model already "runs" in.
func TestTokenizeAndSummaryRequest(t *testing.T) {
	var body map[string]any
	var tokenized string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/tokenize":
			var in struct {
				Content string `json:"content"`
			}
			json.NewDecoder(r.Body).Decode(&in)
			tokenized = in.Content
			json.NewEncoder(w).Encode(map[string]any{"tokens": []int{1, 2, 3, 4}})
		case "/v1/chat/completions":
			json.NewDecoder(r.Body).Decode(&body)
			w.Header().Set("Content-Type", "text/event-stream")
			io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"Bis\"}}]}\n\n")
			io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"cuit\"},\"finish_reason\":\"stop\"}]}\n\n")
			io.WriteString(w, "data: [DONE]\n\n")
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	e, st, _ := loraStore(t)
	m, err := st.Get("base:1")
	if err != nil {
		t.Fatal(err)
	}
	// The options ensure works out for a dense model, so the running server
	// counts as the same load and isn't started again.
	o := loadOptsFrom(m.options(nil))
	o.CPUMoE, o.LoadMode = cpuMoEUnset, ""
	o = planSpeed(m, o)
	e.procs[m.Name] = &proc{model: m, asked: "base:1", opts: o, base: srv.URL, done: make(chan struct{}), ctx: 4096}

	ctx := context.Background()
	n, ctxLen, err := e.Tokenize(ctx, "base:1", nil, "hello there")
	if err != nil || n != 4 || ctxLen != 4096 || tokenized != "hello there" {
		t.Fatalf("Tokenize = %d, %d, %v (sent %q)", n, ctxLen, err, tokenized)
	}

	var got strings.Builder
	err = e.Chat(ctx, ollama.ChatRequest{
		Model:    "base:1",
		Messages: []ollama.Message{{Role: "user", Content: "Summarize."}},
		Options:  map[string]any{"temperature": 0.2, "num_predict": 512},
		Think:    false,
	}, func(r ollama.ChatResponse) error {
		got.WriteString(r.Message.Content)
		return nil
	})
	if err != nil || got.String() != "Biscuit" {
		t.Fatalf("Chat = %q, %v", got.String(), err)
	}
	if kw, _ := body["chat_template_kwargs"].(map[string]any); kw["enable_thinking"] != false {
		t.Errorf("chat_template_kwargs = %v, want thinking off", body["chat_template_kwargs"])
	}
	if body["reasoning_budget"] != float64(0) || body["max_tokens"] != float64(512) || body["temperature"] != 0.2 {
		t.Errorf("request = %v", body)
	}
}

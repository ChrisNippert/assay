package ollama

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func newTestClient(t *testing.T, h http.HandlerFunc) *Client {
	t.Helper()
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	c, err := New(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func TestList(t *testing.T) {
	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/tags" {
			t.Errorf("path = %s", r.URL.Path)
		}
		io.WriteString(w, `{"models":[{"name":"qwen3:8b","size":5200000000,"details":{"family":"qwen3","parameter_size":"8.2B","quantization_level":"Q4_K_M"}}]}`)
	})
	models, err := c.List(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 1 || models[0].Name != "qwen3:8b" || models[0].Details.QuantizationLevel != "Q4_K_M" {
		t.Fatalf("got %+v", models)
	}
}

func TestChatStream(t *testing.T) {
	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		var req ChatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatal(err)
		}
		if req.Stream == nil || !*req.Stream {
			t.Error("stream not requested")
		}
		io.WriteString(w, `{"message":{"role":"assistant","content":"","thinking":"hmm"}}
{"message":{"role":"assistant","content":"Hel"}}
{"message":{"role":"assistant","content":"lo"}}
{"message":{"role":"assistant","content":""},"done":true,"eval_count":2,"eval_duration":1000000000}
`)
	})
	var content, thinking strings.Builder
	var last ChatResponse
	err := c.Chat(context.Background(), ChatRequest{Model: "m", Messages: []Message{{Role: "user", Content: "hi"}}}, func(r ChatResponse) error {
		content.WriteString(r.Message.Content)
		thinking.WriteString(r.Message.Thinking)
		last = r
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if content.String() != "Hello" || thinking.String() != "hmm" {
		t.Fatalf("content=%q thinking=%q", content.String(), thinking.String())
	}
	if !last.Done || last.EvalCount != 2 {
		t.Fatalf("last = %+v", last)
	}
}

func TestStreamErrorLine(t *testing.T) {
	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"status":"pulling manifest"}
{"error":"pull model manifest: file does not exist"}
`)
	})
	var n int
	err := c.Pull(context.Background(), "hf.co/nope/nope:Q4_K_M", func(PullProgress) error { n++; return nil })
	if err == nil || !strings.Contains(err.Error(), "file does not exist") {
		t.Fatalf("err = %v", err)
	}
	if n != 1 {
		t.Fatalf("progress callbacks = %d, want 1", n)
	}
}

func TestErrorStatus(t *testing.T) {
	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		io.WriteString(w, `{"error":"model 'x' not found"}`)
	})
	err := c.Delete(context.Background(), "x")
	if !IsNotFound(err) || !strings.Contains(err.Error(), "model 'x' not found") {
		t.Fatalf("err = %v", err)
	}
}

func TestLoadUnload(t *testing.T) {
	var bodies []map[string]any
	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/generate" {
			t.Errorf("path = %s", r.URL.Path)
		}
		var b map[string]any
		json.NewDecoder(r.Body).Decode(&b)
		bodies = append(bodies, b)
		io.WriteString(w, `{"done":true}`)
	})
	if err := c.Load(context.Background(), "m", "-1", nil); err != nil {
		t.Fatal(err)
	}
	if err := c.Unload(context.Background(), "m"); err != nil {
		t.Fatal(err)
	}
	if bodies[0]["keep_alive"] != float64(-1) {
		t.Errorf("load keep_alive = %#v", bodies[0]["keep_alive"])
	}
	if _, ok := bodies[0]["prompt"]; ok {
		t.Error("load should not send a prompt")
	}
	if bodies[1]["keep_alive"] != float64(0) {
		t.Errorf("unload keep_alive = %#v", bodies[1]["keep_alive"])
	}
}

func TestParseParameters(t *testing.T) {
	got := ParseParameters("temperature                    0.6\nstop                           \"<|im_start|>\"\nstop                           \"<|im_end|>\"\ntop_k                          20\n")
	if got["temperature"] != 0.6 || got["top_k"] != 20.0 {
		t.Errorf("numbers: %#v", got)
	}
	stop, _ := got["stop"].([]string)
	if len(stop) != 2 || stop[1] != "<|im_end|>" {
		t.Errorf("stop: %#v", got["stop"])
	}
}

func TestResolveHost(t *testing.T) {
	cases := map[string]string{
		"":                     DefaultURL,
		"0.0.0.0":              "http://127.0.0.1:11434",
		"0.0.0.0:9999":         "http://127.0.0.1:9999",
		"gpu-box:11434":        "http://gpu-box:11434",
		"https://ollama.lan":   "https://ollama.lan:11434",
		"http://10.0.0.5:8080": "http://10.0.0.5:8080",
	}
	for in, want := range cases {
		if got := ResolveHost(in); got != want {
			t.Errorf("ResolveHost(%q) = %q, want %q", in, got, want)
		}
	}
}

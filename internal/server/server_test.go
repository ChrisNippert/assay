package server

import (
	"bufio"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"ollamui/internal/chats"
	"ollamui/internal/hf"
	"ollamui/internal/hw"
	"ollamui/internal/ollama"
	"ollamui/internal/presets"
	"ollamui/internal/pulls"
)

// fakeOllama answers the handful of endpoints the server uses.
type fakeOllama struct {
	mu         sync.Mutex
	loaded     map[string]time.Time // name -> expires_at
	chatReqs   []ollama.ChatRequest
	createReqs []ollama.CreateRequest
	deleted    []string
	reply      string // NDJSON body for /api/chat
}

func (f *fakeOllama) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()
	switch r.URL.Path {
	case "/api/tags":
		io.WriteString(w, `{"models":[{"name":"qwen3:8b","size":5000000000,"details":{"family":"qwen3","parameter_size":"8.2B","quantization_level":"Q4_K_M"}}]}`)
	case "/api/ps":
		var out struct {
			Models []map[string]any `json:"models"`
		}
		for name, exp := range f.loaded {
			out.Models = append(out.Models, map[string]any{"name": name, "size": 6e9, "size_vram": 6e9, "expires_at": exp})
		}
		json.NewEncoder(w).Encode(out)
	case "/api/generate":
		var req struct {
			Model     string `json:"model"`
			KeepAlive any    `json:"keep_alive"`
		}
		json.NewDecoder(r.Body).Decode(&req)
		if req.KeepAlive == float64(0) {
			delete(f.loaded, req.Model)
		} else {
			f.loaded[req.Model] = time.Now().Add(200 * 365 * 24 * time.Hour)
		}
		io.WriteString(w, `{"done":true}`)
	case "/api/show":
		var req struct{ Model string }
		json.NewDecoder(r.Body).Decode(&req)
		if req.Model != "qwen3:8b" {
			w.WriteHeader(http.StatusNotFound)
			io.WriteString(w, `{"error":"model not found"}`)
			return
		}
		io.WriteString(w, `{"parameters":"temperature 0.6\ntop_k 20\nstop \"<|im_end|>\"","model_info":{"qwen3.context_length":40960},"capabilities":["completion","thinking"]}`)
	case "/api/create":
		var req ollama.CreateRequest
		json.NewDecoder(r.Body).Decode(&req)
		f.createReqs = append(f.createReqs, req)
		io.WriteString(w, `{"status":"success"}`+"\n")
	case "/api/delete":
		var req struct{ Model string }
		json.NewDecoder(r.Body).Decode(&req)
		f.deleted = append(f.deleted, req.Model)
	case "/api/chat":
		var req ollama.ChatRequest
		json.NewDecoder(r.Body).Decode(&req)
		f.chatReqs = append(f.chatReqs, req)
		io.WriteString(w, f.reply)
	default:
		http.NotFound(w, r)
	}
}

func newTestServer(t *testing.T) (*Server, *fakeOllama, *chats.Store) {
	t.Helper()
	fake := &fakeOllama{loaded: map[string]time.Time{}}
	ts := httptest.NewServer(fake)
	t.Cleanup(ts.Close)
	client, err := ollama.New(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	store, err := chats.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	presetStore, err := presets.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	hfSrv := httptest.NewServer(http.HandlerFunc(fakeHF))
	t.Cleanup(hfSrv.Close)
	s, err := New(Options{
		Ollama: client, Chats: store, Presets: presetStore,
		HF: hf.New(hfSrv.URL, ""), Pulls: pulls.New(client, nil),
		Memory: hw.Memory{GPU: "Test GPU", VRAM: 12e9, RAM: 48e9},
	})
	if err != nil {
		t.Fatal(err)
	}
	return s, fake, store
}

func fakeHF(w http.ResponseWriter, r *http.Request) {
	switch r.URL.Path {
	case "/api/models":
		io.WriteString(w, `[
			{"id":"unsloth/Qwen3-8B-GGUF","downloads":10,"gated":false,"gguf":{"total":8190735360,"architecture":"qwen3","context_length":40960}},
			{"id":"unsloth/Qwen3-30B-A3B-GGUF","downloads":5,"gated":false,"gguf":{"total":30500000000,"architecture":"qwen3moe","context_length":262144}}]`)
	case "/api/models/unsloth/Qwen3-8B-GGUF":
		io.WriteString(w, `{"id":"unsloth/Qwen3-8B-GGUF","gated":false,"gguf":{"total":8190735360,"architecture":"qwen3","context_length":40960}}`)
	case "/api/models/unsloth/Qwen3-8B-GGUF/tree/main":
		io.WriteString(w, `[{"type":"file","path":"Qwen3-8B-Q4_K_M.gguf","size":5027784512},{"type":"file","path":"Qwen3-8B-BF16.gguf","size":16388044384}]`)
	case "/unsloth/Qwen3-8B-GGUF/resolve/main/params":
		io.WriteString(w, `{"temperature":0.6}`)
	default:
		http.NotFound(w, r)
	}
}

func do(s *Server, method, path, body string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	req.Host = "localhost:8484"
	if body != "" {
		req.Header.Set("Content-Type", "application/json")
	}
	rec := httptest.NewRecorder()
	s.ServeHTTP(rec, req)
	return rec
}

func TestChatStreamsAndSaves(t *testing.T) {
	s, fake, store := newTestServer(t)
	fake.reply = `{"message":{"role":"assistant","thinking":"let me think"}}
{"message":{"role":"assistant","content":"Hello "}}
{"message":{"role":"assistant","content":"there"}}
{"message":{"role":"assistant","content":""},"done":true,"eval_count":2,"eval_duration":500000000}
`
	rec := do(s, "POST", "/api/chat", `{"model":"qwen3:8b","content":"hi"}`)
	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}

	var events []event
	sc := bufio.NewScanner(rec.Body)
	for sc.Scan() {
		var e event
		if err := json.Unmarshal(sc.Bytes(), &e); err != nil {
			t.Fatalf("bad line %q: %v", sc.Text(), err)
		}
		events = append(events, e)
	}
	if len(events) != 5 || events[0].Type != "chat" || events[1].Type != "thinking" || events[4].Type != "done" {
		t.Fatalf("events = %+v", events)
	}
	if got := events[4].Stats.TokensPerSec; got != 4 {
		t.Errorf("tokens/sec = %v, want 4", got)
	}

	c, err := store.Get(events[0].ID)
	if err != nil {
		t.Fatal(err)
	}
	if c.Title != "hi" || len(c.Messages) != 2 {
		t.Fatalf("chat = %+v", c)
	}
	a := c.Messages[1]
	if a.Role != "assistant" || a.Content != "Hello there" || a.Thinking != "let me think" {
		t.Fatalf("assistant message = %+v", a)
	}

	// A follow-up sends the full history, without old thinking.
	fake.reply = `{"message":{"role":"assistant","content":"ok"},"done":true}` + "\n"
	rec = do(s, "POST", "/api/chat", `{"chatId":"`+c.ID+`","model":"qwen3:8b","content":"again"}`)
	if rec.Code != 200 {
		t.Fatalf("status %d", rec.Code)
	}
	last := fake.chatReqs[len(fake.chatReqs)-1]
	if len(last.Messages) != 3 || last.Messages[1].Thinking != "" {
		t.Fatalf("history sent = %+v", last.Messages)
	}
}

func TestChatErrorEvent(t *testing.T) {
	s, fake, _ := newTestServer(t)
	fake.reply = `{"error":"model requires more system memory"}` + "\n"
	rec := do(s, "POST", "/api/chat", `{"model":"qwen3:8b","content":"hi"}`)
	if !strings.Contains(rec.Body.String(), `"type":"error"`) || !strings.Contains(rec.Body.String(), "more system memory") {
		t.Fatalf("body = %s", rec.Body)
	}
}

func TestLoadEjectAndPinnedKeepAlive(t *testing.T) {
	s, fake, _ := newTestServer(t)
	if rec := do(s, "POST", "/api/models/load", `{"name":"qwen3:8b"}`); rec.Code != 204 {
		t.Fatalf("load: %d %s", rec.Code, rec.Body)
	}
	rec := do(s, "GET", "/api/loaded", "")
	var loaded []loadedModel
	json.Unmarshal(rec.Body.Bytes(), &loaded)
	if len(loaded) != 1 || !loaded[0].Pinned {
		t.Fatalf("loaded = %+v", loaded)
	}

	// Chatting with a pinned model keeps it pinned.
	fake.reply = `{"message":{"role":"assistant","content":"ok"},"done":true}` + "\n"
	do(s, "POST", "/api/chat", `{"model":"qwen3:8b","content":"hi"}`)
	if ka := fake.chatReqs[0].KeepAlive; ka != float64(-1) {
		t.Errorf("keep_alive = %#v, want -1", ka)
	}

	if rec := do(s, "POST", "/api/models/eject", `{"name":"qwen3:8b"}`); rec.Code != 204 {
		t.Fatalf("eject: %d", rec.Code)
	}
	rec = do(s, "GET", "/api/loaded", "")
	if strings.TrimSpace(rec.Body.String()) != "[]" {
		t.Fatalf("after eject: %s", rec.Body)
	}
}

func TestPagesRender(t *testing.T) {
	s, _, _ := newTestServer(t)
	for path, want := range map[string]string{
		"/chat":                           "qwen3:8b",
		"/library":                        "qwen3:8b",
		"/discover":                       "Search GGUF models",
		"/presets":                        "No presets yet",
		"/discover/unsloth/Qwen3-8B-GGUF": "Pick a quant",
	} {
		rec := do(s, "GET", path, "")
		if rec.Code != 200 || !strings.Contains(rec.Body.String(), want) {
			t.Errorf("%s: status %d, body missing %q", path, rec.Code, want)
		}
	}
}

func TestHostGuard(t *testing.T) {
	s, _, _ := newTestServer(t)
	for host, want := range map[string]int{
		"localhost:8484":     200,
		"127.0.0.1:8484":     200,
		"[::1]:8484":         200,
		"100.64.0.7:8484":    200,
		"evil.example:8484":  403,
		"attacker.localhost": 403,
	} {
		req := httptest.NewRequest("GET", "/library", nil)
		req.Host = host
		rec := httptest.NewRecorder()
		s.ServeHTTP(rec, req)
		if rec.Code != want {
			t.Errorf("Host %s: status %d, want %d", host, rec.Code, want)
		}
	}
}

func TestRejectsCrossOriginPost(t *testing.T) {
	s, _, _ := newTestServer(t)
	req := httptest.NewRequest("POST", "/api/models/delete", strings.NewReader(`{"name":"qwen3:8b"}`))
	req.Host = "localhost:8484"
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Sec-Fetch-Site", "cross-site")
	req.Header.Set("Origin", "https://evil.example")
	rec := httptest.NewRecorder()
	s.ServeHTTP(rec, req)
	if rec.Code != http.StatusForbidden {
		t.Fatalf("status %d, want 403", rec.Code)
	}
}

func TestSearchAddsFitAndMoE(t *testing.T) {
	s, _, _ := newTestServer(t)
	rec := do(s, "GET", "/api/hf/search?q=qwen3&author=unsloth", "")
	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	var out struct{ Results []searchResult }
	json.Unmarshal(rec.Body.Bytes(), &out)
	if len(out.Results) != 2 {
		t.Fatalf("results = %+v", out.Results)
	}
	small, big := out.Results[0], out.Results[1]
	if small.MoE || !big.MoE {
		t.Errorf("moe flags: %v %v", small.MoE, big.MoE)
	}
	if small.Fit.Level != hw.FitGPU || big.Fit.Level == hw.FitGPU {
		t.Errorf("fits: %s %s", small.Fit.Level, big.Fit.Level)
	}
}

func TestModelPageMarksBestQuant(t *testing.T) {
	s, _, _ := newTestServer(t)
	body := do(s, "GET", "/discover/unsloth/Qwen3-8B-GGUF", "").Body.String()
	if !strings.Contains(body, `data-pull="hf.co/unsloth/Qwen3-8B-GGUF:Q4_K_M"`) {
		t.Error("Q4_K_M download button missing")
	}
	if !strings.Contains(body, "quant-row best") || strings.Count(body, "quant-row best") != 1 {
		t.Error("expected exactly one best quant")
	}
	if !strings.Contains(body, "Recommended settings") {
		t.Error("recommended params missing")
	}
}

func TestModelPagePartial(t *testing.T) {
	s, _, _ := newTestServer(t)
	rec := do(s, "GET", "/discover/unsloth/Qwen3-8B-GGUF?partial=1&ctx=16384", "")
	if rec.Code != 200 {
		t.Fatalf("status %d", rec.Code)
	}
	body := rec.Body.String()
	if !strings.Contains(body, `class="quant-row`) || !strings.Contains(body, `data-pull="hf.co/unsloth/Qwen3-8B-GGUF:Q4_K_M"`) {
		t.Error("quant row missing from partial")
	}
	for _, bad := range []string{"<html", `class="sidebar"`, `class="crumbs`, "onchange="} {
		if strings.Contains(body, bad) {
			t.Errorf("partial should not contain %q", bad)
		}
	}
	if !strings.Contains(body, "data-close-panel") || !strings.Contains(body, `<option value="16384" selected>`) {
		t.Error("expected close button and ctx selection")
	}
	if rec := do(s, "GET", "/discover/unsloth/Nope-GGUF?partial=1", ""); rec.Code != 404 || strings.Contains(rec.Body.String(), "<html") {
		t.Errorf("missing repo partial: %d", rec.Code)
	}
}

func TestPresetPublishAndChat(t *testing.T) {
	s, fake, store := newTestServer(t)
	rec := do(s, "PUT", "/api/presets/coder", `{"base":"qwen3:8b","system":"Be terse.","published":true,"options":{"temperature":0.3,"num_ctx":16384}}`)
	if rec.Code != 200 {
		t.Fatalf("save: %d %s", rec.Code, rec.Body)
	}
	if len(fake.createReqs) != 1 {
		t.Fatalf("create calls = %d", len(fake.createReqs))
	}
	cr := fake.createReqs[0]
	if cr.Model != "ollamui/coder" || cr.From != "qwen3:8b" || cr.System != "Be terse." || cr.Parameters["temperature"] != 0.3 {
		t.Errorf("create = %+v", cr)
	}

	// The chat's own tweak wins over the preset; the preset fills the rest.
	fake.reply = `{"message":{"role":"assistant","content":"ok"},"done":true}` + "\n"
	rec = do(s, "POST", "/api/chat", `{"preset":"coder","content":"hi","options":{"temperature":0.9}}`)
	if rec.Code != 200 {
		t.Fatalf("chat: %d %s", rec.Code, rec.Body)
	}
	req := fake.chatReqs[0]
	if req.Model != "qwen3:8b" || req.Options["temperature"] != 0.9 || req.Options["num_ctx"] != float64(16384) {
		t.Errorf("chat request = %+v", req)
	}
	if len(req.Messages) != 2 || req.Messages[0].Role != "system" || req.Messages[0].Content != "Be terse." {
		t.Errorf("messages = %+v", req.Messages)
	}
	list, _ := store.List()
	c, _ := store.Get(list[0].ID)
	if c.Preset != "coder" || c.Options["temperature"] != 0.9 || c.Options["num_ctx"] != nil {
		t.Errorf("stored chat should keep only its own tweaks: %+v", c)
	}

	// Unpublishing removes the Ollama model.
	rec = do(s, "PUT", "/api/presets/coder", `{"base":"qwen3:8b","published":false}`)
	if rec.Code != 200 || len(fake.deleted) != 1 || fake.deleted[0] != "ollamui/coder" {
		t.Fatalf("unpublish: %d %v", rec.Code, fake.deleted)
	}
}

func TestPresetValidation(t *testing.T) {
	s, _, _ := newTestServer(t)
	cases := map[string]string{
		"/api/presets/Bad%20Name": `{"base":"qwen3:8b"}`,
		"/api/presets/x":          `{"base":"not-installed:1b"}`,
	}
	for path, body := range cases {
		if rec := do(s, "PUT", path, body); rec.Code != 400 {
			t.Errorf("%s: status %d, want 400", path, rec.Code)
		}
	}
	if rec := do(s, "PUT", "/api/presets/x", `{"base":"qwen3:8b","options":{"temprature":1}}`); rec.Code != 400 {
		t.Errorf("typo option accepted: %d", rec.Code)
	}
}

func TestModelDefaults(t *testing.T) {
	s, _, _ := newTestServer(t)
	rec := do(s, "GET", "/api/models/defaults?name=qwen3:8b", "")
	var d modelDefaults
	json.Unmarshal(rec.Body.Bytes(), &d)
	if d.Options["temperature"] != 0.6 || d.ContextLength != 40960 || !d.Thinking {
		t.Errorf("defaults = %+v", d)
	}
}

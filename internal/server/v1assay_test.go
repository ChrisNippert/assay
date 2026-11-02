package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

// v1API plays an API model: it calls the client's tool once, then answers
// with what the tool said; it records the bodies it was sent.
type v1API struct {
	mu     sync.Mutex
	bodies []map[string]any
}

func (f *v1API) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/models" {
		fmt.Fprint(w, `{"data":[{"id":"m","supported_parameters":["tools"]}]}`)
		return
	}
	var body map[string]any
	json.NewDecoder(r.Body).Decode(&body)
	f.mu.Lock()
	f.bodies = append(f.bodies, body)
	f.mu.Unlock()
	msgs := body["messages"].([]any)
	last := msgs[len(msgs)-1].(map[string]any)
	delta := map[string]any{}
	switch {
	case body["tools"] == nil:
		delta["content"] = "plain answer"
	case last["role"] == "tool":
		delta["content"] = "the file says " + last["content"].(string)
	default:
		delta["reasoning_content"] = "let me look"
		delta["tool_calls"] = []any{map[string]any{"index": 0, "id": "t1", "function": map[string]any{"name": "read_file", "arguments": `{"path":"a.go"}`}}}
	}
	b, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{"delta": delta}}})
	fmt.Fprintf(w, "data: %s\n\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":40,\"completion_tokens\":7,\"cost\":0.002}}\n\ndata: [DONE]\n\n", b)
}

// sseChunks reads a streamed answer's chunks.
func sseChunks(t *testing.T, body string) []map[string]any {
	var out []map[string]any
	if !strings.HasSuffix(body, "data: [DONE]\n\n") {
		t.Fatalf("no [DONE]: %s", body)
	}
	for _, ev := range strings.Split(strings.TrimSpace(body), "\n\n") {
		data := strings.TrimPrefix(ev, "data: ")
		if data == "[DONE]" {
			continue
		}
		var c map[string]any
		if err := json.Unmarshal([]byte(data), &c); err != nil {
			t.Fatalf("%v: %s", err, data)
		}
		out = append(out, c)
	}
	return out
}

func TestV1APIModelAndTeam(t *testing.T) {
	s, fake, _ := newTestServer(t)
	fake.tags = threeModels
	api := &v1API{}
	ts := httptest.NewServer(api)
	defer ts.Close()
	do(s, "PUT", "/api/remote/endpoints/new", `{"name":"Fake","url":"`+ts.URL+`"}`)
	do(s, "PUT", "/api/remote/models/new", `{"endpoint":"fake","model":"m","title":"M"}`)
	do(s, "PUT", "/api/teams/new", `{"title":"Duo","strategy":"pipeline","lead":"api:m","worker":"api:m","size":2}`)

	// Listed even on the Ollama engine, which has no /v1 of its own.
	rec := v1Request(s, "GET", "/v1/models", "")
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), `"id":"api:m"`) || !strings.Contains(rec.Body.String(), `"id":"team:duo"`) {
		t.Fatalf("models: %d %s", rec.Code, rec.Body)
	}

	// An API model calls the editor's tool, streamed in OpenAI's shape.
	tools := `[{"type":"function","function":{"name":"read_file","description":"Read a file","parameters":{"type":"object"}}}]`
	rec = v1Request(s, "POST", "/v1/chat/completions", `{"model":"api:m","stream":true,"stream_options":{"include_usage":true},"tools":`+tools+`,"max_tokens":500,"messages":[{"role":"developer","content":"Be brief."},{"role":"user","content":[{"type":"text","text":"what's in a.go?"}]}]}`)
	if rec.Code != 200 || rec.Header().Get("Content-Type") != "text/event-stream" {
		t.Fatalf("%d %s", rec.Code, rec.Body)
	}
	cs := sseChunks(t, rec.Body.String())
	first := cs[0]["choices"].([]any)[0].(map[string]any)["delta"].(map[string]any)
	end := cs[len(cs)-2]["choices"].([]any)[0].(map[string]any)
	calls, _ := end["delta"].(map[string]any)["tool_calls"].([]any)
	usage, _ := cs[len(cs)-1]["usage"].(map[string]any)
	if first["role"] != "assistant" || first["reasoning_content"] != "let me look" || end["finish_reason"] != "tool_calls" || len(calls) != 1 || usage["cost"] != 0.002 {
		t.Fatalf("chunks %v", cs)
	}
	call := calls[0].(map[string]any)
	if call["id"] != "t1" || call["function"].(map[string]any)["arguments"] != `{"path":"a.go"}` {
		t.Errorf("call %v", call)
	}
	api.mu.Lock()
	sent := api.bodies[0]
	api.mu.Unlock()
	if sent["tools"] == nil || sent["max_tokens"] != 500.0 || sent["messages"].([]any)[0].(map[string]any)["role"] != "system" {
		t.Errorf("sent %v", sent)
	}

	// The tool's result goes back, without streaming.
	rec = v1Request(s, "POST", "/v1/chat/completions", `{"model":"api:m","tools":`+tools+`,"messages":[{"role":"user","content":"what's in a.go?"},{"role":"assistant","content":null,"tool_calls":[{"id":"t1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"a.go\"}"}}]},{"role":"tool","tool_call_id":"t1","content":"package a"}]}`)
	var whole struct {
		Object  string
		Choices []struct {
			Message      struct{ Content string }
			FinishReason string `json:"finish_reason"`
		}
	}
	if json.Unmarshal(rec.Body.Bytes(), &whole); whole.Object != "chat.completion" || whole.Choices[0].Message.Content != "the file says package a" || whole.Choices[0].FinishReason != "stop" {
		t.Fatalf("%d %s", rec.Code, rec.Body)
	}
	api.mu.Lock()
	sentMsgs := api.bodies[len(api.bodies)-1]["messages"].([]any)
	api.mu.Unlock()
	if a := sentMsgs[1].(map[string]any); a["tool_calls"] == nil || sentMsgs[2].(map[string]any)["tool_call_id"] != "t1" {
		t.Errorf("history sent %v", sentMsgs)
	}

	// A team answers plain chat: its work as reasoning, the answer as content.
	rec = v1Request(s, "POST", "/v1/chat/completions", `{"model":"team:duo","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	var reasoning, content string
	for _, c := range sseChunks(t, rec.Body.String()) {
		for _, ch := range c["choices"].([]any) {
			d := ch.(map[string]any)["delta"].(map[string]any)
			r, _ := d["reasoning_content"].(string)
			x, _ := d["content"].(string)
			reasoning, content = reasoning+r, content+x
		}
	}
	if !strings.Contains(reasoning, "## Plan (M)") || content != "plain answer" {
		t.Errorf("reasoning %q content %q", reasoning, content)
	}

	// Unknown names still go to the engine.
	if rec = v1Request(s, "POST", "/v1/chat/completions", `{"model":"api:nope","messages":[]}`); rec.Code != http.StatusNotImplemented {
		t.Errorf("unknown: %d %s", rec.Code, rec.Body)
	}
}

// editorAPI plays a team in an editor: the lead hands out a task, then
// reads a file with the editor's tool, then answers with what it read;
// helpers answer their task.
type editorAPI struct {
	mu   sync.Mutex
	lead []map[string]any // the lead's requests
}

func (f *editorAPI) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/models" {
		fmt.Fprint(w, `{"data":[{"id":"m","supported_parameters":["tools"]}]}`)
		return
	}
	var body map[string]any
	json.NewDecoder(r.Body).Decode(&body)
	msgs := body["messages"].([]any)
	sys := msgs[0].(map[string]any)["content"].(string)
	delta := map[string]any{}
	if strings.Contains(sys, "You're a helper") {
		delta["content"] = "result for " + msgs[len(msgs)-1].(map[string]any)["content"].(string)
	} else {
		f.mu.Lock()
		f.lead = append(f.lead, body)
		f.mu.Unlock()
		helped, read := false, ""
		for _, m := range msgs {
			m := m.(map[string]any)
			if m["role"] == "tool" && strings.Contains(m["content"].(string), "<result") {
				helped = true
			}
			if m["role"] == "tool" && m["tool_call_id"] == "r1" {
				read = m["content"].(string)
			}
		}
		switch {
		case read != "":
			delta["content"] = "done: " + read
		case helped:
			delta["content"] = "Reading the file."
			delta["tool_calls"] = []any{map[string]any{"index": 0, "id": "r1", "function": map[string]any{"name": "read_file", "arguments": `{"path":"a.go"}`}}}
		default:
			delta["tool_calls"] = []any{map[string]any{"index": 0, "id": "s1", "function": map[string]any{"name": "spawn_agents", "arguments": `{"tasks":["draft a test"]}`}}}
		}
	}
	b, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{"delta": delta}}})
	fmt.Fprintf(w, "data: %s\n\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":40,\"completion_tokens\":5}}\n\ndata: [DONE]\n\n", b)
}

func TestV1TeamEditorTools(t *testing.T) {
	s, fake, _ := newTestServer(t)
	fake.tags = threeModels
	api := &editorAPI{}
	ts := httptest.NewServer(api)
	defer ts.Close()
	do(s, "PUT", "/api/remote/endpoints/new", `{"name":"Fake","url":"`+ts.URL+`"}`)
	do(s, "PUT", "/api/remote/models/new", `{"endpoint":"fake","model":"m","title":"M"}`)
	do(s, "PUT", "/api/teams/new", `{"title":"Ed","strategy":"agent","lead":"api:m","worker":"api:m","budget":100000,"minutes":5}`)
	tools := `[{"type":"function","function":{"name":"read_file","description":"Read a file","parameters":{"type":"object"}}}]`

	// The lead's helper runs inside the request; only the editor's call
	// comes out.
	rec := v1Request(s, "POST", "/v1/chat/completions", `{"model":"team:ed","stream":true,"tools":`+tools+`,"messages":[{"role":"system","content":"You are in Zed."},{"role":"user","content":"test a.go"}]}`)
	var reasoning, content string
	var calls []any
	var finish any
	for _, c := range sseChunks(t, rec.Body.String()) {
		if c["error"] != nil {
			t.Fatal(c["error"])
		}
		for _, ch := range c["choices"].([]any) {
			ch := ch.(map[string]any)
			d := ch["delta"].(map[string]any)
			r, _ := d["reasoning_content"].(string)
			x, _ := d["content"].(string)
			reasoning, content = reasoning+r, content+x
			if tc, ok := d["tool_calls"].([]any); ok {
				calls = append(calls, tc...)
			}
			if ch["finish_reason"] != nil {
				finish = ch["finish_reason"]
			}
		}
	}
	if finish != "tool_calls" || len(calls) != 1 || calls[0].(map[string]any)["id"] != "r1" || content != "Reading the file." || !strings.Contains(reasoning, "result for draft a test") {
		t.Fatalf("finish %v calls %v content %q reasoning %q", finish, calls, content, reasoning)
	}
	api.mu.Lock()
	first := api.lead[0]
	api.mu.Unlock()
	sys := first["messages"].([]any)[0].(map[string]any)["content"].(string)
	if !strings.Contains(sys, "spawn_agents") || !strings.Contains(sys, "You are in Zed.") || len(first["tools"].([]any)) != 2 {
		t.Errorf("the lead's first request: %v", first)
	}

	// The editor sends back the call and its result: the lead sees its
	// helper's work again too.
	rec = v1Request(s, "POST", "/v1/chat/completions", `{"model":"team:ed","tools":`+tools+`,"messages":[{"role":"system","content":"You are in Zed."},{"role":"user","content":"test a.go"},{"role":"assistant","content":"Reading the file.","tool_calls":[{"id":"r1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"a.go\"}"}}]},{"role":"tool","tool_call_id":"r1","content":"package a"}]}`)
	var whole struct {
		Choices []struct {
			Message      struct{ Content string }
			FinishReason string `json:"finish_reason"`
		}
	}
	if json.Unmarshal(rec.Body.Bytes(), &whole); len(whole.Choices) != 1 || whole.Choices[0].Message.Content != "done: package a" || whole.Choices[0].FinishReason != "stop" {
		t.Fatalf("%d %s", rec.Code, rec.Body)
	}
	api.mu.Lock()
	lastLead := api.lead[len(api.lead)-1]
	api.mu.Unlock()
	if b, _ := json.Marshal(lastLead["messages"]); !strings.Contains(string(b), "result for draft a test") {
		t.Errorf("the lead lost its helper's work: %s", b)
	}

	// A lead that can't call tools can't work in an editor.
	if rec := do(s, "PUT", "/api/teams/new", `{"title":"Local","strategy":"agent","lead":"small:q4","worker":"api:m","budget":100000,"minutes":5}`); rec.Code != 200 {
		t.Fatalf("%d %s", rec.Code, rec.Body)
	}
	if rec = v1Request(s, "POST", "/v1/chat/completions", `{"model":"team:local","tools":`+tools+`,"messages":[{"role":"user","content":"hi"}]}`); rec.Code == 200 || !strings.Contains(rec.Body.String(), "can't call tools") {
		t.Errorf("local lead: %d %s", rec.Code, rec.Body)
	}
}

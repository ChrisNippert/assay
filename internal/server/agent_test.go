package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"assay/internal/evals"
)

// agentAPI plays a tool-calling lead and its helpers. The lead spawns two
// helpers, then (once it has results) a third, then answers; each call
// reports `tokens` tokens. It records what each call was sent.
type agentAPI struct {
	mu     sync.Mutex
	tokens int
	bodies []map[string]any
}

func (f *agentAPI) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/models" {
		fmt.Fprint(w, `{"data":[{"id":"m","supported_parameters":["tools"]}]}`)
		return
	}
	var body map[string]any
	json.NewDecoder(r.Body).Decode(&body)
	f.mu.Lock()
	f.bodies = append(f.bodies, body)
	tokens := f.tokens
	f.mu.Unlock()
	msgs := body["messages"].([]any)
	sys := msgs[0].(map[string]any)["content"].(string)
	results := 0
	for _, m := range msgs {
		if m.(map[string]any)["role"] == "tool" {
			results++
		}
	}
	delta := map[string]any{}
	switch {
	case strings.Contains(sys, "You're a helper"):
		delta["content"] = "result for " + msgs[len(msgs)-1].(map[string]any)["content"].(string)
	case body["tools"] != nil && results == 0:
		delta["tool_calls"] = []any{map[string]any{"index": 0, "id": "c1", "function": map[string]any{"name": "spawn_agents", "arguments": `{"tasks":["compute 17*20","compute 17*3"]}`}}}
	case body["tools"] != nil && results == 1:
		delta["tool_calls"] = []any{map[string]any{"index": 0, "id": "c2", "function": map[string]any{"name": "spawn_agents", "arguments": `{"tasks":["check 340+51"]}`}}}
	default:
		delta["content"] = "Answer: 391"
	}
	b, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{"delta": delta}}})
	fmt.Fprintf(w, "data: %s\n\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":50,\"completion_tokens\":%d}}\n\ndata: [DONE]\n\n", b, tokens)
}

func TestAgentTeam(t *testing.T) {
	s, fake, _ := newTestServer(t)
	fake.tags = threeModels
	lib, _ := evals.OpenLibrary(t.TempDir())
	s.evalLib, s.evalRes = lib, evals.OpenResults(t.TempDir())
	lib.Save("one", []byte(`name = "One"
check = "number"
[[questions]]
id = "a"
prompt = "What is 17 × 23?"
answer = "391"
`))
	api := &agentAPI{tokens: 10}
	ts := httptest.NewServer(api)
	defer ts.Close()
	do(s, "PUT", "/api/remote/endpoints/new", `{"name":"Fake","url":"`+ts.URL+`"}`)
	do(s, "PUT", "/api/remote/models/new", `{"endpoint":"fake","model":"m","title":"M"}`)
	if rec := do(s, "PUT", "/api/teams/new", `{"title":"Agent","strategy":"agent","lead":"api:m","worker":"api:m","budget":100000,"minutes":5}`); rec.Code != 200 {
		t.Fatalf("team: %d %s", rec.Code, rec.Body)
	}

	// In chat: two rounds of helpers, then the answer.
	rec := do(s, "POST", "/api/chat", `{"model":"team:agent","content":"What is 17 × 23?"}`)
	var content string
	var done map[string]any
	for _, line := range strings.Split(strings.TrimSpace(rec.Body.String()), "\n") {
		var ev map[string]any
		json.Unmarshal([]byte(line), &ev)
		switch ev["type"] {
		case "content":
			content += ev["text"].(string)
		case "done":
			done = ev["stats"].(map[string]any)
		case "error":
			t.Fatal(ev["message"])
		}
	}
	steps, _ := done["team"].([]any)
	// Round 1, 2 helpers, round 2, 1 helper, answer.
	if content != "Answer: 391" || len(steps) != 6 || steps[5].(map[string]any)["step"] != "Answer" || steps[0].(map[string]any)["ctx"] != float64(60) {
		t.Fatalf("content %q steps %v", content, steps)
	}
	api.mu.Lock()
	var helper, lastLead map[string]any
	for _, b := range api.bodies {
		sys := b["messages"].([]any)[0].(map[string]any)["content"].(string)
		if strings.Contains(sys, "You're a helper") {
			helper = b
		} else {
			lastLead = b
		}
	}
	api.mu.Unlock()
	if n := len(helper["messages"].([]any)); n != 2 || helper["tools"] != nil {
		t.Errorf("a helper should see only its task: %v", helper)
	}
	msgs := lastLead["messages"].([]any)
	tool := msgs[len(msgs)-1].(map[string]any)
	if tool["role"] != "tool" || tool["tool_call_id"] != "c2" || !strings.Contains(tool["content"].(string), "result for check 340+51") || !strings.Contains(tool["content"].(string), "[Budget: ") {
		t.Errorf("the lead's last tool result: %v", tool)
	}

	// A small budget: after the first round it must answer, without tools.
	api.mu.Lock()
	api.tokens, api.bodies = 600, nil
	api.mu.Unlock()
	do(s, "PUT", "/api/teams/agent", `{"title":"Agent","strategy":"agent","lead":"api:m","worker":"api:m","budget":1000,"minutes":5}`)
	if _, err := s.startEvalLevels(context.Background(), evalRunRequest{Models: []string{"team:agent"}, Sets: []string{"one"}}); err != nil {
		t.Fatal(err)
	}
	if it := waitEvals(t, s).Items[0]; it.State != itemDone || it.Correct != 1 {
		t.Fatalf("item %+v", it)
	}
	api.mu.Lock()
	defer api.mu.Unlock()
	last := api.bodies[len(api.bodies)-1]
	lm := last["messages"].([]any)
	if last["tools"] != nil || !strings.Contains(lm[len(lm)-1].(map[string]any)["content"].(string), "token budget is used up") || len(api.bodies) != 4 {
		t.Errorf("calls %d; the last: %v", len(api.bodies), last)
	}
}

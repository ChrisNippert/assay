package server

import (
	"bufio"
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	"assay/internal/chats"
	"assay/internal/tools"
)

func events(t *testing.T, rec *httptest.ResponseRecorder) []event {
	t.Helper()
	var out []event
	sc := bufio.NewScanner(rec.Body)
	for sc.Scan() {
		var e event
		if err := json.Unmarshal(sc.Bytes(), &e); err != nil {
			t.Fatalf("bad line %q: %v", sc.Text(), err)
		}
		out = append(out, e)
	}
	return out
}

func types(evs []event) string {
	var out []string
	for _, e := range evs {
		out = append(out, e.Type)
	}
	return strings.Join(out, ",")
}

const callCalc = `{"message":{"role":"assistant","thinking":"I should calculate."}}
{"message":{"role":"assistant","content":"","tool_calls":[{"id":"c1","function":{"name":"calculator","arguments":{"expression":"6*7"}}}]},"done":true,"eval_count":10,"eval_duration":1000000000}
`

func TestToolLoop(t *testing.T) {
	s, fake, store := newTestServer(t)
	fake.caps = `"completion","tools"`
	fake.replies = []string{callCalc}
	fake.reply = `{"message":{"role":"assistant","thinking":"Now I know."}}
{"message":{"role":"assistant","content":"It is 42."}}
{"message":{"role":"assistant"},"done":true,"eval_count":5,"eval_duration":1000000000}
`
	rec := do(s, "POST", "/api/chat", `{"model":"qwen3:8b","content":"what is 6*7?"}`)
	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	evs := events(t, rec)
	if got := types(evs); got != "chat,thinking,tool,toolresult,thinking,content,done" {
		t.Fatalf("events: %s", got)
	}
	if evs[2].Call.Name != "calculator" || evs[3].Call.Label != "Calculated 6*7 = 42" || evs[3].Call.Result != "42" {
		t.Errorf("call events: %+v / %+v", evs[2].Call, evs[3].Call)
	}
	if st := evs[6].Stats; st.OutputTokens != 15 {
		t.Errorf("stats should add up the steps: %+v", st)
	}

	// The second request carries the call and its result, and both ask with tools.
	if len(fake.chatReqs) != 2 {
		t.Fatalf("%d model requests", len(fake.chatReqs))
	}
	first, second := fake.chatReqs[0], fake.chatReqs[1]
	if len(first.Tools) != 4 || first.Tools[0].Function.Name != "fetch_url" { // web search is off by default
		t.Errorf("tools sent: %+v", first.Tools)
	}
	if sys := first.Messages[0]; sys.Role != "system" || !strings.Contains(sys.Content, "You can use tools") {
		t.Errorf("system = %+v", sys)
	}
	m := second.Messages
	if len(m) != 4 || m[2].Role != "assistant" || len(m[2].ToolCalls) != 1 || m[2].ToolCalls[0].ID != "c1" || m[2].Thinking != "I should calculate." ||
		m[3].Role != "tool" || m[3].Content != "42" || m[3].ToolCallID != "c1" {
		t.Errorf("second request messages: %+v", m)
	}

	// Saved: one assistant message with the call and only the final text.
	c, _ := store.Get(evs[0].ID)
	if len(c.Messages) != 2 {
		t.Fatalf("messages: %+v", c.Messages)
	}
	a := c.Messages[1]
	if a.Content != "It is 42." || a.Thinking != "Now I know." || len(a.Calls) != 1 {
		t.Fatalf("assistant: %+v", a)
	}
	if k := a.Calls[0]; k.Name != "calculator" || k.Args != `{"expression":"6*7"}` || k.Result != "42" || k.Think != "I should calculate." || k.Step != 0 {
		t.Errorf("call: %+v", k)
	}

	// A follow-up replays the call in the history.
	fake.reply = `{"message":{"role":"assistant","content":"ok"},"done":true}` + "\n"
	rec = do(s, "POST", "/api/chat", `{"chatId":"`+c.ID+`","model":"qwen3:8b","content":"and double it?"}`)
	if rec.Code != 200 {
		t.Fatalf("status %d", rec.Code)
	}
	h := fake.chatReqs[len(fake.chatReqs)-1].Messages
	var roles []string
	for _, x := range h {
		roles = append(roles, x.Role)
	}
	if strings.Join(roles, ",") != "system,user,assistant,tool,assistant,user" || h[4].Content != "It is 42." {
		t.Errorf("history roles: %v", roles)
	}
}

func TestToolsOffOrUnsupported(t *testing.T) {
	s, fake, _ := newTestServer(t)
	fake.reply = `{"message":{"role":"assistant","content":"hi"},"done":true}` + "\n"
	// The default fake model has no "tools" capability.
	do(s, "POST", "/api/chat", `{"model":"qwen3:8b","content":"hi"}`)
	if r := fake.chatReqs[0]; len(r.Tools) != 0 || strings.Contains(r.Messages[0].Content, "tools") {
		t.Errorf("tools sent to a model without them: %+v", r)
	}
	fake.caps = `"completion","tools"`
	do(s, "POST", "/api/chat", `{"model":"qwen3:8b","content":"hi","tools":false}`)
	if r := fake.chatReqs[1]; len(r.Tools) != 0 {
		t.Errorf("tools sent although switched off: %+v", r.Tools)
	}
	// The preset's switch is used when the chat has none.
	if rec := do(s, "PUT", "/api/presets/notools", `{"base":"qwen3:8b","tools":false}`); rec.Code != 200 {
		t.Fatalf("preset: %d %s", rec.Code, rec.Body)
	}
	do(s, "POST", "/api/chat", `{"preset":"notools","content":"hi"}`)
	if r := fake.chatReqs[2]; len(r.Tools) != 0 {
		t.Errorf("preset's tools:false ignored: %+v", r.Tools)
	}
	do(s, "POST", "/api/chat", `{"preset":"notools","content":"hi","tools":true}`)
	if r := fake.chatReqs[3]; len(r.Tools) == 0 {
		t.Error("the chat's switch should beat the preset's")
	}
	// Settings can turn a tool off.
	if rec := do(s, "PUT", "/api/tools/settings", `{"search":"duckduckgo","off":["calculator","fetch_url"]}`); rec.Code != 200 {
		t.Fatalf("settings: %d %s", rec.Code, rec.Body)
	}
	do(s, "POST", "/api/chat", `{"model":"qwen3:8b","content":"hi"}`)
	var names []string
	for _, x := range fake.chatReqs[4].Tools {
		names = append(names, x.Function.Name)
	}
	if strings.Join(names, ",") != "web_search,search_chats,current_time" {
		t.Errorf("tools = %v", names)
	}
	if rec := do(s, "PUT", "/api/tools/settings", `{"search":"bing"}`); rec.Code != 400 {
		t.Errorf("bad backend: %d", rec.Code)
	}
}

func TestToolStepLimit(t *testing.T) {
	s, fake, store := newTestServer(t)
	fake.caps = `"completion","tools"`
	for range maxToolSteps {
		fake.replies = append(fake.replies, `{"message":{"role":"assistant","tool_calls":[{"function":{"name":"current_time","arguments":{}}},{"function":{"name":"nonesuch","arguments":"oops"}}]},"done":true}`+"\n")
	}
	fake.reply = `{"message":{"role":"assistant","content":"Giving up."},"done":true}` + "\n"
	rec := do(s, "POST", "/api/chat", `{"model":"qwen3:8b","content":"loop"}`)
	evs := events(t, rec)
	if evs[len(evs)-1].Type != "done" {
		t.Fatalf("events: %s", types(evs))
	}
	if len(fake.chatReqs) != maxToolSteps+1 {
		t.Fatalf("%d requests, want %d", len(fake.chatReqs), maxToolSteps+1)
	}
	if last := fake.chatReqs[maxToolSteps]; len(last.Tools) != 0 {
		t.Error("the last request should come without tools")
	}
	c, _ := store.Get(evs[0].ID)
	a := c.Messages[1]
	if a.Content != "Giving up." || len(a.Calls) != 2*maxToolSteps {
		t.Fatalf("assistant: %d calls, %q", len(a.Calls), a.Content)
	}
	if k := a.Calls[1]; !k.Error || !strings.Contains(k.Result, "no tool called nonesuch") || a.Calls[2].Step != 1 || a.Calls[2].ID == a.Calls[0].ID {
		t.Errorf("calls: %+v / %+v", a.Calls[1], a.Calls[2])
	}
}

func TestToolMessagesRoundTrip(t *testing.T) {
	m := chats.Message{Role: "assistant", Content: "final", Calls: []chats.ToolCall{
		{Step: 0, ID: "a", Name: "calculator", Args: `{"expression":"1+1"}`, Result: "2", Said: "ok", Think: "hm"},
		{Step: 0, ID: "b", Name: "current_time", Args: `{}`, Result: "noon"},
		{Step: 1, ID: "c", Name: "calculator", Args: `not json`, Result: "Error"},
	}}
	got := toOllama([]chats.Message{m})
	var roles []string
	for _, x := range got {
		roles = append(roles, x.Role)
	}
	if strings.Join(roles, ",") != "assistant,tool,tool,assistant,tool,assistant" || len(got[0].ToolCalls) != 2 || got[0].Content != "ok" || got[0].Thinking != "hm" {
		t.Fatalf("%v %+v", roles, got[0])
	}
	if string(got[3].ToolCalls[0].Function.Arguments) != `"not json"` || got[5].Content != "final" {
		t.Errorf("%+v", got[3])
	}
	_ = tools.Catalog
}

func TestStoppedReplyStaysInContext(t *testing.T) {
	msgs := toOllama([]chats.Message{
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "Half an ans", Stopped: true},
		{Role: "user", Content: "go on"},
		{Role: "assistant", Thinking: "let me think", Stopped: true},
	})
	if got := msgs[1].Content; !strings.HasPrefix(got, "Half an ans") || !strings.Contains(got, "stopped") {
		t.Fatalf("stopped reply = %q", got)
	}
	if got := msgs[3].Content; !strings.Contains(got, "let me think") {
		t.Fatalf("thinking-only stopped reply = %q", got)
	}
}

func TestLoopCheck(t *testing.T) {
	l := newLoopCheck()
	line := "Actually, I think the correct approach is to use the lens equation.\n"
	for i := 0; i < loopRepeats-1; i++ {
		if l.add("Some new thought number " + strings.Repeat("x", i) + " that goes on for a while.\n") {
			t.Fatal("distinct lines taken for a loop")
		}
		// Split mid-line, as a stream does.
		if l.add(line[:20]) || l.add(line[20:]) {
			t.Fatalf("loop found after %d repeats", i+1)
		}
	}
	if !l.add(line) {
		t.Fatal("loop not found")
	}
	short := newLoopCheck()
	for i := 0; i < 50; i++ {
		if short.add("Wait.\n") {
			t.Fatal("short line taken for a loop")
		}
	}
}

package server

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"assay/internal/chats"
	"assay/internal/ollama"
	"assay/internal/tools"
)

func sevenMessages() []chats.Message {
	var out []chats.Message
	for i := 1; i <= 7; i++ {
		role := "user"
		if i%2 == 0 {
			role = "assistant"
		}
		out = append(out, chats.Message{Role: role, Content: "m" + string(rune('0'+i))})
	}
	return out
}

func TestTrimmedMessages(t *testing.T) {
	msgs := sevenMessages()
	msgs[2].Thinking = "hmm"
	msgs[2].Calls = []chats.ToolCall{{Name: "calculator", Result: "4"}}
	before := make([]chats.Message, len(msgs))
	copy(before, msgs)

	trims := []chats.Trim{
		{ID: "a", Messages: []int{0, 2, 5}, Note: "first note"},
		{ID: "b", Messages: []int{2, 99, -1}, Note: "later note"},
	}
	out := trimmedMessages(msgs, trims)

	if out[0].Content != "[trimmed: first note]" || out[0].Role != "user" {
		t.Errorf("message 0: %+v", out[0])
	}
	if out[2].Content != "[trimmed: later note]" || out[2].Role != "user" {
		t.Errorf("message 2 should take the later note, keep its role: %+v", out[2])
	}
	if out[2].Thinking != "" || out[2].Calls != nil {
		t.Errorf("a trimmed message keeps no thinking or tool calls: %+v", out[2])
	}
	// The last 4 (indexes 3..6) are never trimmed, even when listed.
	if out[5].Content != "m6" {
		t.Errorf("message 5 is among the last 4 and must stay: %+v", out[5])
	}
	if out[1].Content != "m2" {
		t.Errorf("untrimmed message changed: %+v", out[1])
	}
	for i := range msgs {
		if msgs[i].Content != before[i].Content || msgs[i].Thinking != before[i].Thinking || len(msgs[i].Calls) != len(before[i].Calls) {
			t.Errorf("saved message %d was changed", i)
		}
	}

	// With no trims the messages come back as they are.
	if got := trimmedMessages(msgs, nil); &got[0] != &msgs[0] {
		t.Errorf("no trims should return the same messages")
	}
}

// TestTrimsUndoAndDrop covers undo, and the trims of messages an edit cuts.
func TestTrimsUndoAndDrop(t *testing.T) {
	c := &chats.Chat{Trims: []chats.Trim{
		{ID: "t1", Messages: []int{0, 3}, Note: "a"},
		{ID: "t2", Messages: []int{1}, Note: "b"},
	}}
	c.DropTrimsFrom(1) // an edit from message 1 cuts it and everything after
	if len(c.Trims) != 1 || c.Trims[0].ID != "t1" || len(c.Trims[0].Messages) != 1 || c.Trims[0].Messages[0] != 0 {
		t.Errorf("after DropTrimsFrom(1): %+v", c.Trims)
	}
	if !c.UndoTrim("t1") || c.UndoTrim("t1") || len(c.Trims) != 0 {
		t.Errorf("undo: %+v", c.Trims)
	}
}

func TestForgetTool(t *testing.T) {
	run := func(c *chats.Chat, args string) tools.Result {
		return tools.Run(context.Background(), []tools.Tool{forgetTool(c)}, "forget", json.RawMessage(args))
	}
	fresh := func() *chats.Chat { return &chats.Chat{Messages: sevenMessages()} }

	c := fresh()
	r := run(c, `{"message_ids":[2,1,2],"note":"the first draft"}`)
	if r.Err || r.Label != "trimmed 2 messages" || len(c.Trims) != 1 {
		t.Fatalf("ids: %+v, trims %+v", r, c.Trims)
	}
	if got := c.Trims[0].Messages; len(got) != 2 || got[0] != 0 || got[1] != 1 {
		t.Errorf("numbers are 1-based and deduplicated: %v", got)
	}
	if c.Trims[0].Note != "the first draft" || c.Trims[0].ID == "" {
		t.Errorf("trim: %+v", c.Trims[0])
	}

	c = fresh()
	if r := run(c, `{"from":2,"to":3,"note":"range"}`); r.Err || len(c.Trims) != 1 || len(c.Trims[0].Messages) != 2 {
		t.Errorf("range: %+v %+v", r, c.Trims)
	}
	c = fresh()
	if r := run(c, `{"from":1,"note":"one"}`); r.Err || len(c.Trims[0].Messages) != 1 {
		t.Errorf("from alone is one message: %+v", r)
	}

	// The last 4 and numbers past the end are left out, and reported.
	c = fresh()
	if r := run(c, `{"message_ids":[4,1,9],"note":"mixed"}`); r.Err || len(c.Trims[0].Messages) != 1 ||
		!strings.Contains(r.Text, "4 is one of the last 4") || !strings.Contains(r.Text, "9 is not a message") {
		t.Errorf("partial: %+v %+v", r, c.Trims)
	}
	c = fresh()
	if r := run(c, `{"message_ids":[5,6,7,8],"note":"none"}`); !r.Err || len(c.Trims) != 0 {
		t.Errorf("nothing left to trim should fail: %+v", r)
	}
	c = fresh()
	if r := run(c, `{"from":1,"to":1000,"note":"all"}`); r.Err || len(c.Trims[0].Messages) != 3 {
		t.Errorf("a range past the end keeps the messages that can go: %+v %+v", r, c.Trims)
	}

	for _, bad := range []string{
		`{"message_ids":[1]}`,
		`{"message_ids":[1],"note":"  "}`,
		`{"note":"x"}`,
		`{"from":3,"to":2,"note":"x"}`,
		`not json`,
	} {
		c = fresh()
		if r := run(c, bad); !r.Err || len(c.Trims) != 0 {
			t.Errorf("%s should fail: %+v", bad, r)
		}
	}
}

// TestForgetTurn runs a reply that trims, then checks the next prompt carries
// the note, the saved chat is unchanged, and undo brings the message back.
func TestForgetTurn(t *testing.T) {
	s, fake, store := newTestServer(t)
	fake.caps = `"completion","tools"`
	c := chats.New("qwen3:8b")
	c.Messages = sevenMessages()
	if err := store.Save(c); err != nil {
		t.Fatal(err)
	}

	// Off: the tool isn't offered, and the reply goes ahead.
	fake.reply = `{"message":{"role":"assistant","content":"ok"},"done":true}` + "\n"
	do(s, "POST", "/api/chat", `{"model":"qwen3:8b","chatId":"`+c.ID+`","content":"first"}`)
	for _, x := range fake.chatReqs[len(fake.chatReqs)-1].Tools {
		if x.Function.Name == "forget" {
			t.Fatalf("forget offered with the switch off")
		}
	}

	// On: the model trims message 1, and the call records the trim.
	fake.replies = []string{`{"message":{"role":"assistant","tool_calls":[{"id":"f1","function":{"name":"forget","arguments":{"message_ids":[1],"note":"old chat"}}}]},"done":true}` + "\n"}
	fake.reply = `{"message":{"role":"assistant","content":"Noted."},"done":true}` + "\n"
	rec := do(s, "POST", "/api/chat", `{"model":"qwen3:8b","chatId":"`+c.ID+`","content":"second","forget":true}`)
	var call *chats.ToolCall
	for _, e := range events(t, rec) {
		if e.Type == "toolresult" {
			call = e.Call
		}
	}
	if call == nil || call.Trim == "" || call.Label != "trimmed 1 message" {
		t.Fatalf("forget call: %+v", call)
	}
	saved, err := store.Get(c.ID)
	if err != nil {
		t.Fatal(err)
	}
	if len(saved.Trims) != 1 || saved.Messages[0].Content != "m1" {
		t.Fatalf("saved chat: trims %+v, first message %q", saved.Trims, saved.Messages[0].Content)
	}

	// The next prompt carries the note in place of the message.
	do(s, "POST", "/api/chat", `{"model":"qwen3:8b","chatId":"`+c.ID+`","content":"third"}`)
	if !hasContent(fake.chatReqs[len(fake.chatReqs)-1].Messages, "[trimmed: old chat]") || hasContent(fake.chatReqs[len(fake.chatReqs)-1].Messages, "m1") {
		t.Errorf("next prompt should carry the note, not the message")
	}

	// Undo puts it back.
	if rec := do(s, "DELETE", "/api/chats/"+c.ID+"/trims/"+call.Trim, ""); rec.Code != 204 {
		t.Fatalf("undo: status %d: %s", rec.Code, rec.Body)
	}
	do(s, "POST", "/api/chat", `{"model":"qwen3:8b","chatId":"`+c.ID+`","content":"fourth"}`)
	if !hasContent(fake.chatReqs[len(fake.chatReqs)-1].Messages, "m1") {
		t.Errorf("undo should put the message back into the prompt")
	}
	if rec := do(s, "DELETE", "/api/chats/"+c.ID+"/trims/"+call.Trim, ""); rec.Code != 404 {
		t.Errorf("a second undo should be not found, got %d", rec.Code)
	}
}

func hasContent(msgs []ollama.Message, want string) bool {
	for _, m := range msgs {
		if m.Content == want {
			return true
		}
	}
	return false
}

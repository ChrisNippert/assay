package server

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"assay/internal/chats"
	"assay/internal/tools"
)

func asst(content string, calls ...chats.ToolCall) chats.Message {
	return chats.Message{Role: "assistant", Content: content, Calls: calls}
}

func TestApplyEdit(t *testing.T) {
	code := "Here:\n```go\nfunc a() int {\n\treturn 1\n}\n```\nDone."
	tests := []struct {
		name    string
		msgs    []chats.Message
		find    string
		replace string
		want    string // the updated block, when no error
		err     string // a substring of the error, when one is expected
	}{
		{name: "single match", msgs: []chats.Message{asst(code)}, find: "return 1", replace: "return 2",
			want: "```go\nfunc a() int {\n\treturn 2\n}\n```"},
		{name: "multi-line find", msgs: []chats.Message{asst(code)}, find: "func a() int {\n\treturn 1", replace: "func a() int {\n\treturn 3",
			want: "```go\nfunc a() int {\n\treturn 3\n}\n```"},
		{name: "empty replace deletes", msgs: []chats.Message{asst(code)}, find: "\treturn 1\n", replace: "",
			want: "```go\nfunc a() int {\n}\n```"},
		{name: "not found", msgs: []chats.Message{asst(code)}, find: "return 9", replace: "x", err: "no code block"},
		{name: "empty find", msgs: []chats.Message{asst(code)}, find: "", replace: "x", err: "find is empty"},
		{name: "found twice in one block", msgs: []chats.Message{asst("```\nx\nx\n```")}, find: "x", replace: "y", err: "occurs 2 times"},
		{name: "text outside blocks is not a block", msgs: []chats.Message{asst("return 1 is prose")}, find: "return 1", replace: "x", err: "no code block"},
		{name: "unclosed block is ignored", msgs: []chats.Message{asst("```go\nreturn 1\n")}, find: "return 1", replace: "x", err: "no code block"},
		{name: "latest block containing find wins", msgs: []chats.Message{
			asst("```\nv := 1\n```"), asst("```\nv := 1\n```\n```\nw := 2\n```")},
			find: "v := 1", replace: "v := 5", want: "```\nv := 5\n```"},
		{name: "older block used when newer lacks find", msgs: []chats.Message{
			asst("```\nold()\n```"), asst("```\nnew()\n```")},
			find: "old()", replace: "older()", want: "```\nolder()\n```"},
		{name: "edit from an earlier edit is the latest", msgs: []chats.Message{
			asst("```\nx := 1\n```"),
			asst("Fixed.", chats.ToolCall{Name: "edit_code", Result: "Edited. The updated block:\n```\nx := 2\n```"})},
			find: "x := 2", replace: "x := 3", want: "```\nx := 3\n```"},
		{name: "failed edit is not a block", msgs: []chats.Message{
			asst("```\nx := 1\n```"),
			asst("", chats.ToolCall{Name: "edit_code", Error: true, Result: "Error: broken\n```\nx := 2\n```"})},
			find: "x := 2", replace: "x := 3", err: "no code block"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			before := make([]chats.Message, len(tt.msgs))
			copy(before, tt.msgs)
			got, err := applyEdit(tt.msgs, tt.find, tt.replace)
			if tt.err != "" {
				if err == nil || !strings.Contains(err.Error(), tt.err) {
					t.Fatalf("error %v, want one containing %q", err, tt.err)
				}
			} else if err != nil {
				t.Fatalf("unexpected error: %v", err)
			} else if got != tt.want {
				t.Errorf("block:\n%s\nwant:\n%s", got, tt.want)
			}
			for i := range before {
				if tt.msgs[i].Content != before[i].Content {
					t.Errorf("message %d was changed", i)
				}
			}
		})
	}
}

func TestEditCodeTool(t *testing.T) {
	c := &chats.Chat{Messages: []chats.Message{{Role: "user", Content: "write it"}, asst("```\nfoo()\n```")}}
	tl := editCodeTool(c)
	if tl.Name != "edit_code" || len(tl.Parameters["required"].([]string)) != 2 {
		t.Fatalf("schema: %+v", tl)
	}
	run := func(args string) tools.Result {
		return tools.Run(context.Background(), []tools.Tool{tl}, "edit_code", json.RawMessage(args))
	}
	if r := run(`{"find":"foo","replace":"bar"}`); r.Err || !strings.Contains(r.Text, "```\nbar()\n```") {
		t.Errorf("success: %+v", r)
	}
	if r := run(`{"find":"foo"}`); !r.Err || !strings.Contains(r.Text, "both required") {
		t.Errorf("missing replace: %+v", r)
	}
	if r := run(`not json`); !r.Err {
		t.Errorf("bad json: %+v", r)
	}
	if r := run(`{"find":"zzz","replace":"q"}`); !r.Err || !strings.HasPrefix(r.Text, "Error: ") {
		t.Errorf("not found: %+v", r)
	}
	if c.Messages[1].Content != "```\nfoo()\n```" {
		t.Errorf("the saved message changed: %q", c.Messages[1].Content)
	}
}

// TestEditCodeTurn checks the tool is offered only when the chat turns it on,
// and that the updated block comes back to the chat.
func TestEditCodeTurn(t *testing.T) {
	s, fake, _ := newTestServer(t)
	fake.caps = `"completion","tools"`
	fake.reply = `{"message":{"role":"assistant","content":"` + "```go\\nfunc a() {}\\n```" + `"},"done":true}` + "\n"
	rec := do(s, "POST", "/api/chat", `{"model":"qwen3:8b","content":"write a()"}`)
	id := events(t, rec)[0].ID

	// Off by default: no edit_code in the request.
	fake.reply = `{"message":{"role":"assistant","content":"ok"},"done":true}` + "\n"
	do(s, "POST", "/api/chat", `{"model":"qwen3:8b","chatId":"`+id+`","content":"again"}`)
	for _, x := range fake.chatReqs[len(fake.chatReqs)-1].Tools {
		if x.Function.Name == "edit_code" {
			t.Fatalf("edit_code offered with the switch off")
		}
	}

	// On: the model calls it, and the updated block is in the reply's call.
	fake.replies = []string{`{"message":{"role":"assistant","tool_calls":[{"id":"e1","function":{"name":"edit_code","arguments":{"find":"a()","replace":"b()"}}}]},"done":true}` + "\n"}
	fake.reply = `{"message":{"role":"assistant","content":"Renamed."},"done":true}` + "\n"
	rec = do(s, "POST", "/api/chat", `{"model":"qwen3:8b","chatId":"`+id+`","content":"rename it","editCode":true}`)
	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	found := false
	for _, x := range fake.chatReqs[len(fake.chatReqs)-2].Tools {
		found = found || x.Function.Name == "edit_code"
	}
	if !found {
		t.Errorf("edit_code not offered with the switch on")
	}
	var res string
	for _, e := range events(t, rec) {
		if e.Type == "toolresult" {
			res = e.Call.Result
		}
	}
	if !strings.Contains(res, "```go\nfunc b() {}\n```") {
		t.Errorf("tool result: %q", res)
	}
}

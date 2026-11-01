package server

import (
	"context"
	"encoding/json"
	"fmt"
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
		{name: "wrong indentation still matches the line", msgs: []chats.Message{asst(code)}, find: "    return 1", replace: "    return 4",
			want: "```go\nfunc a() int {\n\treturn 4\n}\n```"},
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
			e, err := applyEdit(tt.msgs, nil, tt.find, tt.replace)
			got := e.block
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
	tl := editCodeTool(c, nil)
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
	if m := fake.chatReqs[len(fake.chatReqs)-1].Messages; strings.Contains(m[0].Content, editSystem) {
		t.Errorf("edit instruction sent with the switch off")
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
	if m := fake.chatReqs[len(fake.chatReqs)-2].Messages; m[0].Role != "system" || !strings.Contains(m[0].Content, editSystem) {
		t.Errorf("no edit instruction in the system message: %+v", m[0])
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

// Edits made earlier in the same reply aren't saved yet, but later edits
// in that reply build on them.
func TestEditCodeStacksInOneReply(t *testing.T) {
	c := &chats.Chat{Messages: []chats.Message{asst("```cpp\nstruct Node {\n    Node* next;\n    Node(int v) {\n        next = nullptr;\n    }\n};\n```")}}
	tool := editCodeTool(c, nil)
	r1 := tool.Run(context.Background(), json.RawMessage(`{"find":"    Node* next;","replace":"    Node* next;\n    Node* prev;"}`))
	r2 := tool.Run(context.Background(), json.RawMessage(`{"find":"        next = nullptr;","replace":"        next = nullptr;\n        prev = nullptr;"}`))
	if r1.Err || r2.Err || !strings.Contains(r2.Text, "Node* prev;") || !strings.Contains(r2.Text, "prev = nullptr;") {
		t.Fatalf("second edit lost the first:\n%s\n%s", r1.Text, r2.Text)
	}
	// Adding prev again now finds the edited block, where it would be a duplicate the model can see.
	if r3 := tool.Run(context.Background(), json.RawMessage(`{"find":"    Node* prev;","replace":"    Node* prev; // back"}`)); r3.Err || strings.Count(r3.Text, "Node* prev") != 1 {
		t.Fatalf("third edit: %s", r3.Text)
	}
}

func TestTextToolCalls(t *testing.T) {
	text := "Sure.\n<|tool_call>call:edit_code{find:<|\"|>    Node* next;<|\"|>,replace:<|\"|>    Node* next;\n    Node* prev;<|\"|>}<tool_call|>"
	calls, rest := textToolCalls(text)
	if len(calls) != 1 || calls[0].Function.Name != "edit_code" || rest != "Sure." {
		t.Fatalf("calls %+v rest %q", calls, rest)
	}
	var a struct{ Find, Replace string }
	if err := json.Unmarshal(calls[0].Function.Arguments, &a); err != nil || a.Find != "    Node* next;" || a.Replace != "    Node* next;\n    Node* prev;" {
		t.Fatalf("args %s (%v)", calls[0].Function.Arguments, err)
	}
	calls, _ = textToolCalls("<|tool_call>call:web_search{query:<|\"|>go 1.27<|\"|>,count:3,safe:true}<tool_call|>")
	if len(calls) != 1 || string(calls[0].Function.Arguments) != `{"count":3,"query":"go 1.27","safe":true}` {
		t.Fatalf("%+v", calls)
	}
	if calls, rest := textToolCalls("no calls <|tool_call> broken"); len(calls) != 0 || rest != "no calls <|tool_call> broken" {
		t.Fatalf("%+v %q", calls, rest)
	}
}

// A call left in the reply text still runs, and the markup isn't saved.
func TestEditCodeTextCallRuns(t *testing.T) {
	s, fake, _ := newTestServer(t)
	fake.caps = `"completion","tools"`
	fake.reply = `{"message":{"role":"assistant","content":"` + "```go\\nfunc a() {}\\n```" + `"},"done":true}` + "\n"
	id := events(t, do(s, "POST", "/api/chat", `{"model":"qwen3:8b","content":"write a()"}`))[0].ID
	fake.replies = []string{`{"message":{"role":"assistant","content":"<|tool_call>call:edit_code{find:<|\"|>a()<|\"|>,replace:<|\"|>b()<|\"|>}<tool_call|>"},"done":true}` + "\n"}
	fake.reply = `{"message":{"role":"assistant","content":"Renamed."},"done":true}` + "\n"
	rec := do(s, "POST", "/api/chat", `{"model":"qwen3:8b","chatId":"`+id+`","content":"rename it","editCode":true}`)
	var res, shown string
	for _, e := range events(t, rec) {
		if e.Type == "toolresult" {
			res = e.Call.Result
		}
		if e.Type == "content" {
			shown += e.Text
		}
	}
	if !strings.Contains(res, "func b() {}") || strings.Contains(shown, "tool_call") {
		t.Fatalf("result %q shown %q", res, shown)
	}
}

// A long block sends the model only the lines around the change; the whole
// block is kept for the user and for the next edit.
func TestEditCodeLongBlock(t *testing.T) {
	var b strings.Builder
	b.WriteString("```go\n")
	for i := range 200 {
		fmt.Fprintf(&b, "x%d := %d\n", i, i)
	}
	b.WriteString("```")
	c := &chats.Chat{Messages: []chats.Message{asst(b.String())}}
	tool := editCodeTool(c, nil)
	r := tool.Run(context.Background(), json.RawMessage(`{"find":"x100 := 100","replace":"x100 := -1"}`))
	if r.Err || !strings.Contains(r.Text, "Lines 98–104 of 200") || strings.Contains(r.Text, "x0 := 0") || !strings.Contains(r.Text, "x100 := -1") {
		t.Fatalf("model got: %s", r.Text)
	}
	if !strings.Contains(r.Full, "x0 := 0") || !strings.Contains(r.Full, "x199 := 199") || !strings.Contains(r.Full, "x100 := -1") {
		t.Fatalf("full block missing lines")
	}
	if r2 := tool.Run(context.Background(), json.RawMessage(`{"find":"x0 := 0","replace":"x0 := 9"}`)); r2.Err || !strings.Contains(r2.Full, "x100 := -1") || !strings.Contains(r2.Full, "x0 := 9") {
		t.Fatalf("second edit lost the first")
	}
}

// An attached file can be edited, and the call is labelled with its name.
func TestEditCodeAttachedFile(t *testing.T) {
	c := &chats.Chat{Messages: []chats.Message{{Role: "user", Content: "fix this", Attachments: []chats.Attachment{{Name: "list.cpp", Kind: "doc", File: "f1"}}}}}
	file := func(a chats.Attachment) (string, bool) { return "struct Node {\n    Node* next;\n};\n", a.File == "f1" }
	r := editCodeTool(c, file).Run(context.Background(), json.RawMessage(`{"find":"    Node* next;","replace":"    Node* next;\n    Node* prev;"}`))
	if r.Err || r.Label != "Edited list.cpp" || !strings.Contains(r.Full, "```cpp\nstruct Node {\n    Node* next;\n    Node* prev;\n};\n```") {
		t.Fatalf("%+v", r)
	}
	// The saved call keeps the name for the next reply's edits.
	c.Messages = append(c.Messages, asst("Added prev.", chats.ToolCall{Name: "edit_code", Label: r.Label, Result: r.Text, Block: r.Full}))
	if r2 := editCodeTool(c, file).Run(context.Background(), json.RawMessage(`{"find":"    Node* prev;","replace":"    Node* prev = nullptr;"}`)); r2.Err || r2.Label != "Edited list.cpp" || !strings.Contains(r2.Full, "prev = nullptr") {
		t.Fatalf("%+v", r2)
	}
}

// A multi-line find with the wrong indentation (Gemma copied a maze with 4
// spaces where the file had 12) still matches, and the replacement takes
// the file's indentation.
func TestEditCodeLooseIndent(t *testing.T) {
	code := "```html\n<script>\n        const maze = [\n            [1,1,1],\n            [1,0,1],\n        ];\n</script>\n```"
	c := &chats.Chat{Messages: []chats.Message{asst(code)}}
	r := editCodeTool(c, nil).Run(context.Background(), json.RawMessage(`{"find":"const maze = [\n    [1,1,1],\n    [1,0,1],","replace":"const maze = [\n    [1,1,1],\n    [1,0,0],\n    [1,1,1],"}`))
	want := "        const maze = [\n            [1,1,1],\n            [1,0,0],\n            [1,1,1],\n        ];"
	if r.Err || !strings.Contains(r.Full, want) {
		t.Fatalf("%s\n%s", r.Text, r.Full)
	}
	// A real mismatch says which line differs.
	r = editCodeTool(c, nil).Run(context.Background(), json.RawMessage(`{"find":"const maze = [\n    [1,1,1],\n    [1,9,1],","replace":"x"}`))
	if !r.Err || !strings.Contains(r.Text, `line 3 differs: the block has "[1,0,1],"`) {
		t.Fatalf("%s", r.Text)
	}
}

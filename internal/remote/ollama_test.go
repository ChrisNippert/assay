package remote

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"assay/internal/ollama"
)

func TestChatOllama(t *testing.T) {
	var got map[string]any
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&got)
		for _, line := range []string{
			`{"choices":[{"delta":{"reasoning":"hmm"}}]}`,
			`{"choices":[{"delta":{"content":"Let me look."}}]}`,
			`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"web_search","arguments":"{\"q\":"}}]}}]}`,
			`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"go\"}"}}]},"finish_reason":"tool_calls"}]}`,
			`{"choices":[],"usage":{"prompt_tokens":30,"completion_tokens":12,"cost":0.01}}`,
			`[DONE]`,
		} {
			fmt.Fprintf(w, "data: %s\n\n", line)
		}
	}))
	defer ts.Close()
	ep := Endpoint{ID: "x", Name: "X", URL: ts.URL + "/v1"}
	m := Model{ID: "m", Model: "vendor/m", System: "Be brief."}
	var call ollama.ToolCall
	call.ID = "c0"
	call.Function.Name = "web_search"
	call.Function.Arguments = json.RawMessage(`{"q":"x"}`)
	req := ollama.ChatRequest{
		Model: "api:m",
		Messages: []ollama.Message{
			{Role: "system", Content: "Chat rules."},
			{Role: "user", Content: "look", Images: []string{"AAAA"}},
			{Role: "assistant", ToolCalls: []ollama.ToolCall{call}},
			{Role: "tool", Content: "results", ToolCallID: "c0"},
		},
		Options: map[string]any{"temperature": 0.7, "num_predict": 500, "num_ctx": 8192, "reply_format": "json"},
		Think:   false,
	}
	var text, think strings.Builder
	var final ollama.ChatResponse
	var cost float64
	err := ChatOllama(context.Background(), ep, m, req, func(r ollama.ChatResponse) error {
		text.WriteString(r.Message.Content)
		think.WriteString(r.Message.Thinking)
		if r.Done {
			final = r
		}
		return nil
	}, &cost)
	if err != nil {
		t.Fatal(err)
	}
	if text.String() != "Let me look." || think.String() != "hmm" || cost != 0.01 || final.EvalCount != 12 || final.PromptEvalCount != 30 || final.DoneReason != "stop" {
		t.Errorf("reply %q %q cost %v final %+v", text.String(), think.String(), cost, final)
	}
	if c := final.Message.ToolCalls; len(c) != 1 || c[0].ID != "c1" || string(c[0].Function.Arguments) != `{"q":"go"}` {
		t.Errorf("tool calls %+v", c)
	}
	msgs := got["messages"].([]any)
	first := msgs[0].(map[string]any)
	if len(msgs) != 4 || first["role"] != "system" || first["content"] != "Be brief.\n\nChat rules." {
		t.Fatalf("messages %v", msgs)
	}
	if parts, ok := msgs[1].(map[string]any)["content"].([]any); !ok || len(parts) != 2 {
		t.Errorf("image message %v", msgs[1])
	}
	if tc := msgs[2].(map[string]any)["tool_calls"].([]any); tc[0].(map[string]any)["function"].(map[string]any)["arguments"] != `{"q":"x"}` {
		t.Errorf("tool call message %v", msgs[2])
	}
	if msgs[3].(map[string]any)["tool_call_id"] != "c0" {
		t.Errorf("tool result %v", msgs[3])
	}
	if got["temperature"] != 0.7 || got["max_tokens"] != float64(500) || got["num_ctx"] != nil || got["response_format"] == nil || got["reasoning_effort"] != nil {
		t.Errorf("body %v", got)
	}
}

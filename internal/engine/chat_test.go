package engine

import (
	"encoding/json"
	"testing"

	"assay/internal/ollama"
)

func TestToolCallWire(t *testing.T) {
	// Ollama's shape (arguments an object) goes out as OpenAI's (a string).
	var c ollama.ToolCall
	c.ID, c.Function.Name, c.Function.Arguments = "c1", "calculator", json.RawMessage(`{"expression":"1+1"}`)
	w := toWire([]ollama.ToolCall{c})
	if len(w) != 1 || w[0].Type != "function" || w[0].Function.Arguments != `{"expression":"1+1"}` {
		t.Fatalf("%+v", w)
	}
	b, _ := json.Marshal(chatMessage{Role: "assistant", Content: "", ToolCalls: w})
	var back map[string]any
	json.Unmarshal(b, &back)
	if args := back["tool_calls"].([]any)[0].(map[string]any)["function"].(map[string]any)["arguments"]; args != `{"expression":"1+1"}` {
		t.Errorf("arguments on the wire: %#v", args)
	}

	// Streamed pieces are joined by the caller; whole calls convert back,
	// with unusable arguments kept as a string and none as {}.
	back2 := fromWire([]wireCall{
		{ID: "a", Function: struct {
			Name      string `json:"name,omitempty"`
			Arguments string `json:"arguments"`
		}{"f", `{"x":1}`}},
		{Function: struct {
			Name      string `json:"name,omitempty"`
			Arguments string `json:"arguments"`
		}{"g", `{"x":`}},
		{Function: struct {
			Name      string `json:"name,omitempty"`
			Arguments string `json:"arguments"`
		}{"h", ``}},
	})
	if string(back2[0].Function.Arguments) != `{"x":1}` || string(back2[1].Function.Arguments) != `"{\"x\":"` || string(back2[2].Function.Arguments) != `{}` {
		t.Errorf("%s | %s | %s", back2[0].Function.Arguments, back2[1].Function.Arguments, back2[2].Function.Arguments)
	}
}

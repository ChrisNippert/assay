package remote

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"time"

	"assay/internal/ollama"
)

// ChatOllama runs a chat request in Ollama's shape (what the rest of Assay
// speaks) against an API model: messages with images and tool calls, tools,
// sampling options, thinking and the reply format. It streams pieces to fn
// and ends with a Done chunk carrying the tool calls and token counts, like
// the local engine. The cost, when the API reports it, goes to cost.
func ChatOllama(ctx context.Context, ep Endpoint, m Model, req ollama.ChatRequest, fn func(ollama.ChatResponse) error, cost *float64) error {
	start := time.Now()
	msgs := make([]map[string]any, 0, len(req.Messages)+1)
	if m.System != "" {
		msgs = append(msgs, map[string]any{"role": "system", "content": m.System})
	}
	for _, x := range req.Messages {
		w := map[string]any{"role": x.Role, "content": x.Content}
		if len(x.Images) > 0 {
			parts := []map[string]any{{"type": "text", "text": x.Content}}
			for _, img := range x.Images {
				parts = append(parts, map[string]any{"type": "image_url", "image_url": map[string]string{"url": "data:image/png;base64," + img}})
			}
			w["content"] = parts
		}
		if len(x.ToolCalls) > 0 {
			w["tool_calls"] = toWire(x.ToolCalls)
		}
		if x.ToolCallID != "" {
			w["tool_call_id"] = x.ToolCallID
		}
		msgs = append(msgs, w)
	}
	// The model's own system prompt and the chat's become one message.
	if len(msgs) > 1 && m.System != "" && msgs[1]["role"] == "system" {
		msgs[0]["content"] = m.System + "\n\n" + msgs[1]["content"].(string)
		msgs = append(msgs[:1], msgs[2:]...)
	}
	body := map[string]any{"model": m.Model, "messages": msgs, "stream": true, "stream_options": map[string]any{"include_usage": true}}
	if len(req.Tools) > 0 {
		body["tools"] = req.Tools
	}
	for k, f := range map[string]string{"temperature": "temperature", "top_p": "top_p", "top_k": "top_k", "min_p": "min_p", "seed": "seed", "stop": "stop", "presence_penalty": "presence_penalty", "frequency_penalty": "frequency_penalty", "repeat_penalty": "repetition_penalty"} {
		if v, ok := req.Options[k]; ok {
			body[f] = v
		}
	}
	if n, ok := req.Options["num_predict"]; ok {
		if v, _ := toInt(n); v > 0 {
			body["max_tokens"] = v
		}
	}
	switch mode, _ := req.Options["reply_format"].(string); mode {
	case "json":
		body["response_format"] = map[string]any{"type": "json_object"}
	case "schema":
		if s, _ := req.Options["reply_schema"].(string); json.Valid([]byte(s)) {
			body["response_format"] = map[string]any{"type": "json_schema", "json_schema": map[string]any{"name": "reply", "schema": json.RawMessage(s)}}
		}
	}
	effort := m.Reasoning
	switch t := req.Think.(type) {
	case bool:
		if !t {
			effort = "off"
		}
	case string:
		effort = t
	}
	if ep.OpenRouter() {
		body["usage"] = map[string]any{"include": true}
		switch effort {
		case "off":
			body["reasoning"] = map[string]any{"enabled": false}
		case "low", "medium", "high":
			body["reasoning"] = map[string]any{"effort": effort}
		}
	} else if effort != "" && effort != "off" {
		body["reasoning_effort"] = effort
	}

	b, _ := json.Marshal(body)
	hr, err := http.NewRequestWithContext(ctx, http.MethodPost, ep.URL+"/chat/completions", bytes.NewReader(b))
	if err != nil {
		return err
	}
	header(hr, ep)
	hr.Header.Set("Content-Type", "application/json")
	resp, err := HTTP.Do(hr)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return &ollama.APIError{StatusCode: resp.StatusCode, Message: apiError(resp)}
	}
	final := ollama.ChatResponse{Model: req.Model, Done: true, DoneReason: "stop"}
	var first time.Time
	var calls []wireCall
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 64<<10), 8<<20)
	for sc.Scan() {
		line, ok := strings.CutPrefix(strings.TrimSpace(sc.Text()), "data:")
		if !ok {
			continue
		}
		line = strings.TrimSpace(line)
		if line == "[DONE]" {
			break
		}
		var c struct {
			Choices []struct {
				Delta struct {
					Content          string     `json:"content"`
					Reasoning        string     `json:"reasoning"`
					ReasoningContent string     `json:"reasoning_content"`
					ToolCalls        []wireCall `json:"tool_calls"`
				} `json:"delta"`
				FinishReason string `json:"finish_reason"`
			} `json:"choices"`
			Usage *struct {
				PromptTokens     int     `json:"prompt_tokens"`
				CompletionTokens int     `json:"completion_tokens"`
				Cost             float64 `json:"cost"`
			} `json:"usage"`
			Error *struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		if json.Unmarshal([]byte(line), &c) != nil {
			continue
		}
		if c.Error != nil {
			return errors.New(c.Error.Message)
		}
		if c.Usage != nil {
			final.PromptEvalCount, final.EvalCount = c.Usage.PromptTokens, c.Usage.CompletionTokens
			if cost != nil {
				*cost += c.Usage.Cost
			}
		}
		for _, ch := range c.Choices {
			if ch.FinishReason != "" {
				final.DoneReason = ch.FinishReason
				if final.DoneReason == "tool_calls" {
					final.DoneReason = "stop"
				}
			}
			for _, tc := range ch.Delta.ToolCalls {
				for len(calls) <= tc.Index {
					calls = append(calls, wireCall{})
				}
				w := &calls[tc.Index]
				if tc.ID != "" {
					w.ID = tc.ID
				}
				w.Function.Name += tc.Function.Name
				w.Function.Arguments += tc.Function.Arguments
			}
			r := ch.Delta.Reasoning + ch.Delta.ReasoningContent
			if ch.Delta.Content == "" && r == "" {
				continue
			}
			if first.IsZero() {
				first = time.Now()
			}
			out := ollama.ChatResponse{Model: req.Model, CreatedAt: time.Now(), Message: ollama.Message{Role: "assistant", Content: ch.Delta.Content, Thinking: r}}
			if err := fn(out); err != nil {
				return err
			}
		}
	}
	if err := sc.Err(); err != nil {
		return err
	}
	now := time.Now()
	if first.IsZero() {
		first = now
	}
	// No local timings: the prompt took until the first piece arrived, the
	// reply the rest.
	final.CreatedAt, final.TotalDuration = now, int64(now.Sub(start))
	final.PromptEvalDuration, final.EvalDuration = int64(first.Sub(start)), int64(now.Sub(first))
	final.Message = ollama.Message{Role: "assistant", ToolCalls: fromWire(calls)}
	return fn(final)
}

func toInt(v any) (int, bool) {
	switch n := v.(type) {
	case int:
		return n, true
	case int64:
		return int(n), true
	case float64:
		return int(n), true
	}
	return 0, false
}

// wireCall is a tool call in OpenAI's shape, arguments as a JSON string.
type wireCall struct {
	Index    int    `json:"index"`
	ID       string `json:"id,omitempty"`
	Type     string `json:"type,omitempty"`
	Function struct {
		Name      string `json:"name,omitempty"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

func toWire(calls []ollama.ToolCall) []wireCall {
	var out []wireCall
	for i, c := range calls {
		w := wireCall{Index: i, ID: c.ID, Type: "function"}
		w.Function.Name = c.Function.Name
		w.Function.Arguments = string(c.Function.Arguments)
		if w.Function.Arguments == "" {
			w.Function.Arguments = "{}"
		}
		out = append(out, w)
	}
	return out
}

// fromWire keeps arguments that aren't JSON as a string, so the tool loop
// can tell the model.
func fromWire(calls []wireCall) []ollama.ToolCall {
	var out []ollama.ToolCall
	for _, w := range calls {
		var c ollama.ToolCall
		c.ID = w.ID
		c.Function.Name = w.Function.Name
		args := strings.TrimSpace(w.Function.Arguments)
		if args == "" {
			args = "{}"
		}
		if json.Valid([]byte(args)) {
			c.Function.Arguments = json.RawMessage(args)
		} else {
			c.Function.Arguments, _ = json.Marshal(args)
		}
		out = append(out, c)
	}
	return out
}

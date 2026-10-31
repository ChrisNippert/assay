package engine

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"assay/internal/ollama"
)

// sampling maps Ollama option names to llama-server's request fields.
var sampling = map[string]string{
	"temperature": "temperature", "top_p": "top_p", "top_k": "top_k", "min_p": "min_p",
	"repeat_penalty": "repeat_penalty", "repeat_last_n": "repeat_last_n",
	"presence_penalty": "presence_penalty", "frequency_penalty": "frequency_penalty",
	"seed": "seed", "stop": "stop", "typical_p": "typical_p", "tfs_z": "tfs_z",
	"mirostat": "mirostat", "mirostat_tau": "mirostat_tau", "mirostat_eta": "mirostat_eta",
}

type chatMessage struct {
	Role             string     `json:"role"`
	Content          any        `json:"content"`
	ReasoningContent string     `json:"reasoning_content,omitempty"`
	ToolCalls        []wireCall `json:"tool_calls,omitempty"`
	ToolCallID       string     `json:"tool_call_id,omitempty"`
}

// wireCall is a tool call in OpenAI's shape, where the arguments are a
// JSON string.
type wireCall struct {
	Index    int    `json:"index"`
	ID       string `json:"id,omitempty"`
	Type     string `json:"type,omitempty"`
	Function struct {
		Name      string `json:"name,omitempty"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// toWire converts calls from Ollama's shape (arguments as an object).
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

// fromWire is the other way: arguments that aren't a JSON object (a model
// that wrote garbage) are kept as a string so the loop can tell the model.
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

type contentPart struct {
	Type     string            `json:"type"`
	Text     string            `json:"text,omitempty"`
	ImageURL map[string]string `json:"image_url,omitempty"`
}

// chatBody builds llama-server's /v1/chat/completions request. Model
// defaults (from the model's parameters) come first, the request's
// options override them.
func chatBody(m *Model, req ollama.ChatRequest) map[string]any {
	msgs := make([]chatMessage, 0, len(req.Messages))
	for _, x := range req.Messages {
		if len(x.Images) == 0 {
			msgs = append(msgs, chatMessage{Role: x.Role, Content: x.Content, ReasoningContent: x.Thinking, ToolCalls: toWire(x.ToolCalls), ToolCallID: x.ToolCallID})
			continue
		}
		parts := []contentPart{{Type: "text", Text: x.Content}}
		for _, img := range x.Images {
			parts = append(parts, contentPart{Type: "image_url", ImageURL: map[string]string{"url": "data:image/png;base64," + img}})
		}
		msgs = append(msgs, chatMessage{Role: x.Role, Content: parts})
	}
	body := map[string]any{"messages": msgs, "stream": true, "timings_per_token": false, "cache_prompt": true}
	if len(req.Tools) > 0 {
		body["tools"] = req.Tools
		body["parallel_tool_calls"] = true
	}
	opts := m.options(req.Options)
	for k, v := range opts {
		if f, ok := sampling[k]; ok {
			body[f] = v
		}
	}
	if n, ok := num(opts["num_predict"]); ok && n > 0 {
		body["max_tokens"] = int(n)
	}
	switch t := req.Think.(type) {
	case bool:
		body["chat_template_kwargs"] = map[string]any{"enable_thinking": t}
		if !t {
			body["reasoning_budget"] = 0
		}
	case string:
		body["chat_template_kwargs"] = map[string]any{"enable_thinking": true}
		body["reasoning_effort"] = t
	}
	for k, v := range replyFields(opts) {
		body[k] = v
	}
	return body
}

// replyFields maps the reply format settings to llama-server's request
// fields: nothing for text, response_format for JSON (any object, or a
// schema) and grammar for a GBNF grammar. Options are validated when saved,
// so a schema that doesn't parse is just left out here.
func replyFields(opts map[string]any) map[string]any {
	switch mode, _ := opts["reply_format"].(string); mode {
	case "json":
		return map[string]any{"response_format": map[string]any{"type": "json_object"}}
	case "schema":
		s, _ := opts["reply_schema"].(string)
		if !json.Valid([]byte(s)) {
			return nil
		}
		return map[string]any{"response_format": map[string]any{"type": "json_schema", "json_schema": map[string]any{"schema": json.RawMessage(s)}}}
	case "grammar":
		if g, _ := opts["reply_grammar"].(string); g != "" {
			return map[string]any{"grammar": g}
		}
	}
	return nil
}

type chunk struct {
	Choices []struct {
		Delta struct {
			Content          string     `json:"content"`
			ReasoningContent string     `json:"reasoning_content"`
			ToolCalls        []wireCall `json:"tool_calls"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Timings *struct {
		CacheN      int     `json:"cache_n"` // prompt tokens reused from the cache
		PromptN     int     `json:"prompt_n"`
		PromptMS    float64 `json:"prompt_ms"`
		PredictedN  int     `json:"predicted_n"`
		PredictedMS float64 `json:"predicted_ms"`
	} `json:"timings"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

// Chat streams a reply, loading the model first if needed. It calls fn
// for each piece and once more with Done set and the timings filled in.
func (e *Engine) Chat(ctx context.Context, req ollama.ChatRequest, fn func(ollama.ChatResponse) error) error {
	start := time.Now()
	p, loaded, err := e.ensure(ctx, req.Model, req.Options)
	if err != nil {
		return err
	}
	loadDur := time.Duration(0)
	if loaded {
		loadDur = time.Since(start)
	}
	ka, _ := parseKeepAlive(req.KeepAlive)
	if req.KeepAlive == nil || req.KeepAlive == "" {
		ka = modelKeepAlive(p.model, req.Options)
	}
	p.touch(ka, e.opt.KeepAlive)
	p.mu.Lock()
	p.busy++
	p.mu.Unlock()
	defer func() {
		p.mu.Lock()
		p.busy--
		p.mu.Unlock()
		p.touch(ka, e.opt.KeepAlive)
	}()

	// The slot serves this chat now. A server that was just started gets the
	// chat's saved cache first (kvcache.go); the slot forgets the chat again
	// if the reply fails.
	chat := chatKey(req.ChatID)
	restored := 0
	if chat != "" && p.takeFresh() {
		restored = e.restoreSlot(p, chat)
	}
	p.beginChat(chat, restored)
	replied := false
	defer func() {
		if !replied {
			p.beginChat("", 0)
		}
	}()

	b, _ := json.Marshal(chatBody(p.model, req))
	hreq, _ := http.NewRequestWithContext(ctx, http.MethodPost, p.base+"/v1/chat/completions", bytes.NewReader(b))
	hreq.Header.Set("Content-Type", "application/json")
	res, err := e.http.Do(hreq)
	if err != nil {
		if !p.running() {
			return fmt.Errorf("llama-server stopped: %s", p.lastLines())
		}
		return err
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		return serverError(res)
	}

	final := ollama.ChatResponse{Model: req.Model, Done: true, DoneReason: "stop", LoadDuration: int64(loadDur)}
	sc := bufio.NewScanner(res.Body)
	sc.Buffer(make([]byte, 64<<10), 8<<20)
	var calls []wireCall // streamed in pieces, joined by index
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		data, ok := strings.CutPrefix(line, "data:")
		if !ok {
			continue
		}
		data = strings.TrimSpace(data)
		if data == "[DONE]" {
			break
		}
		var c chunk
		if err := json.Unmarshal([]byte(data), &c); err != nil {
			continue
		}
		if c.Error != nil {
			return &ollama.APIError{StatusCode: http.StatusInternalServerError, Message: c.Error.Message}
		}
		if c.Timings != nil {
			final.PromptEvalCount, final.EvalCount = c.Timings.PromptN, c.Timings.PredictedN
			final.PromptEvalDuration = int64(c.Timings.PromptMS * float64(time.Millisecond))
			final.EvalDuration = int64(c.Timings.PredictedMS * float64(time.Millisecond))
			if chat != "" {
				// The slot holds the whole history and the reply now.
				p.beginChat(chat, c.Timings.CacheN+c.Timings.PromptN+c.Timings.PredictedN)
			}
		}
		for _, ch := range c.Choices {
			if ch.FinishReason != nil && *ch.FinishReason != "" {
				final.DoneReason = *ch.FinishReason
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
			if ch.Delta.Content == "" && ch.Delta.ReasoningContent == "" {
				continue
			}
			out := ollama.ChatResponse{Model: req.Model, CreatedAt: time.Now()}
			out.Message = ollama.Message{Role: "assistant", Content: ch.Delta.Content, Thinking: ch.Delta.ReasoningContent}
			if err := fn(out); err != nil {
				return err
			}
		}
	}
	if err := sc.Err(); err != nil {
		return err
	}
	replied = true
	final.CreatedAt = time.Now()
	final.TotalDuration = int64(time.Since(start))
	final.Message = ollama.Message{Role: "assistant", ToolCalls: fromWire(calls)}
	return fn(final)
}

// Embed turns texts into vectors with an embedding model.
func (e *Engine) Embed(ctx context.Context, model string, input []string) ([][]float32, error) {
	p, _, err := e.ensure(ctx, model, nil)
	if err != nil {
		return nil, err
	}
	if !p.opts.Embedding {
		return nil, &ollama.APIError{StatusCode: http.StatusBadRequest, Message: model + " isn't an embedding model"}
	}
	p.touch(0, e.opt.KeepAlive)
	b, _ := json.Marshal(map[string]any{"input": input, "encoding_format": "float"})
	hreq, _ := http.NewRequestWithContext(ctx, http.MethodPost, p.base+"/v1/embeddings", bytes.NewReader(b))
	hreq.Header.Set("Content-Type", "application/json")
	res, err := e.http.Do(hreq)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		return nil, serverError(res)
	}
	var out struct {
		Data []struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.NewDecoder(res.Body).Decode(&out); err != nil {
		return nil, err
	}
	if len(out.Data) != len(input) {
		return nil, fmt.Errorf("embed: got %d vectors for %d inputs", len(out.Data), len(input))
	}
	vecs := make([][]float32, len(input))
	for _, d := range out.Data {
		if d.Index >= 0 && d.Index < len(vecs) {
			vecs[d.Index] = d.Embedding
		}
	}
	return vecs, nil
}

func serverError(res *http.Response) error {
	b, _ := io.ReadAll(io.LimitReader(res.Body, 8192))
	var e struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	msg := strings.TrimSpace(string(b))
	if json.Unmarshal(b, &e) == nil && e.Error.Message != "" {
		msg = e.Error.Message
	}
	return &ollama.APIError{StatusCode: res.StatusCode, Message: msg}
}

// Tokenize counts the tokens text is for a model, with llama-server's own
// tokenizer (no chat template). The model loads with options first if it
// isn't loaded, as a chat request would. It also returns the context length
// the server runs with.
func (e *Engine) Tokenize(ctx context.Context, model string, options map[string]any, text string) (int, int, error) {
	p, _, err := e.ensure(ctx, model, options)
	if err != nil {
		return 0, 0, err
	}
	p.touch(0, e.opt.KeepAlive)
	b, _ := json.Marshal(map[string]string{"content": text})
	hreq, _ := http.NewRequestWithContext(ctx, http.MethodPost, p.base+"/tokenize", bytes.NewReader(b))
	hreq.Header.Set("Content-Type", "application/json")
	res, err := e.http.Do(hreq)
	if err != nil {
		return 0, 0, err
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		return 0, 0, serverError(res)
	}
	var out struct {
		Tokens []int `json:"tokens"`
	}
	if err := json.NewDecoder(res.Body).Decode(&out); err != nil {
		return 0, 0, err
	}
	p.mu.Lock()
	n := p.ctx
	p.mu.Unlock()
	return len(out.Tokens), n, nil
}

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

	"ollamui/internal/ollama"
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
	Role    string `json:"role"`
	Content any    `json:"content"`
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
			msgs = append(msgs, chatMessage{Role: x.Role, Content: x.Content})
			continue
		}
		parts := []contentPart{{Type: "text", Text: x.Content}}
		for _, img := range x.Images {
			parts = append(parts, contentPart{Type: "image_url", ImageURL: map[string]string{"url": "data:image/png;base64," + img}})
		}
		msgs = append(msgs, chatMessage{Role: x.Role, Content: parts})
	}
	body := map[string]any{"messages": msgs, "stream": true, "timings_per_token": false, "cache_prompt": true}
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
	return body
}

type chunk struct {
	Choices []struct {
		Delta struct {
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content"`
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
	final.Message = ollama.Message{Role: "assistant"}
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

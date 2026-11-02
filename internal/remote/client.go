package remote

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"
)

// HTTP is the client used for API calls; tests swap it.
var HTTP = &http.Client{Timeout: 20 * time.Minute}

// Message is one chat message.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Request is one chat completion.
type Request struct {
	Messages  []Message
	MaxTokens int
	// Sampling holds temperature, top_p, seed and the like; an endpoint
	// that refuses them gets the request again without.
	Sampling map[string]any
}

// Usage is what the API reports about a reply.
type Usage struct {
	Tokens int
	// Cost is in US dollars, when the API says (OpenRouter does).
	Cost float64
	// Length says the reply hit the token limit.
	Length bool
}

// Chat streams a reply, calling onDelta with each piece of the answer and
// of the reasoning; returning an error from onDelta stops it.
func Chat(ctx context.Context, ep Endpoint, m Model, req Request, onDelta func(content, reasoning string) error) (Usage, error) {
	body := map[string]any{
		"model":          m.Model,
		"messages":       req.Messages,
		"stream":         true,
		"max_tokens":     req.MaxTokens,
		"stream_options": map[string]any{"include_usage": true},
	}
	for k, v := range req.Sampling {
		body[k] = v
	}
	if ep.OpenRouter() {
		body["usage"] = map[string]any{"include": true}
		switch m.Reasoning {
		case "off":
			body["reasoning"] = map[string]any{"enabled": false}
		case "low", "medium", "high":
			body["reasoning"] = map[string]any{"effort": m.Reasoning}
		}
	} else if m.Reasoning != "" && m.Reasoning != "off" {
		body["reasoning_effort"] = m.Reasoning
	}
	u, err := chat(ctx, ep, body, onDelta)
	var se *statusError
	if errors.As(err, &se) && se.code == http.StatusBadRequest {
		// Some APIs refuse sampling options or want max_completion_tokens
		// (OpenAI's reasoning models); try once with the plainest request.
		for k := range req.Sampling {
			delete(body, k)
		}
		delete(body, "max_tokens")
		body["max_completion_tokens"] = req.MaxTokens
		u, err = chat(ctx, ep, body, onDelta)
	}
	return u, err
}

type statusError struct {
	code int
	msg  string
}

func (e *statusError) Error() string { return e.msg }

func chat(ctx context.Context, ep Endpoint, body map[string]any, onDelta func(content, reasoning string) error) (Usage, error) {
	var u Usage
	b, _ := json.Marshal(body)
	hr, err := http.NewRequestWithContext(ctx, http.MethodPost, ep.URL+"/chat/completions", bytes.NewReader(b))
	if err != nil {
		return u, err
	}
	header(hr, ep)
	hr.Header.Set("Content-Type", "application/json")
	resp, err := HTTP.Do(hr)
	if err != nil {
		return u, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return u, &statusError{resp.StatusCode, apiError(resp)}
	}
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 64<<10), 8<<20)
	for sc.Scan() {
		line, ok := strings.CutPrefix(sc.Text(), "data:")
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
					Content          string `json:"content"`
					Reasoning        string `json:"reasoning"`
					ReasoningContent string `json:"reasoning_content"`
				} `json:"delta"`
				FinishReason string `json:"finish_reason"`
			} `json:"choices"`
			Usage *struct {
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
			return u, errors.New(c.Error.Message)
		}
		if c.Usage != nil {
			u.Tokens, u.Cost = c.Usage.CompletionTokens, c.Usage.Cost
		}
		for _, ch := range c.Choices {
			if ch.FinishReason == "length" {
				u.Length = true
			}
			r := ch.Delta.Reasoning + ch.Delta.ReasoningContent
			if ch.Delta.Content != "" || r != "" {
				if err := onDelta(ch.Delta.Content, r); err != nil {
					return u, err
				}
			}
		}
	}
	return u, sc.Err()
}

func header(hr *http.Request, ep Endpoint) {
	if ep.Key != "" {
		hr.Header.Set("Authorization", "Bearer "+ep.Key)
	}
	if ep.OpenRouter() {
		hr.Header.Set("X-Title", "Assay")
		hr.Header.Set("HTTP-Referer", "https://github.com/autumnnippert/assay")
	}
}

// apiError reads the message out of an error response.
func apiError(resp *http.Response) string {
	b, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
	var e struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	msg := strings.TrimSpace(string(b))
	if json.Unmarshal(b, &e) == nil && e.Error.Message != "" {
		msg = e.Error.Message
	}
	if len(msg) > 400 {
		msg = msg[:400] + "…"
	}
	switch resp.StatusCode {
	case http.StatusUnauthorized, http.StatusForbidden:
		return "the API refused the key: " + msg
	case http.StatusPaymentRequired:
		return "out of credit: " + msg
	}
	return fmt.Sprintf("the API said %d: %s", resp.StatusCode, msg)
}

// Listed is a model an endpoint offers.
type Listed struct {
	ID      string `json:"id"`
	Name    string `json:"name,omitempty"`
	Context int    `json:"context,omitempty"`
	// In and Out are US dollars per million tokens, when the API says.
	In  float64 `json:"in,omitempty"`
	Out float64 `json:"out,omitempty"`
}

// List asks an endpoint for its models.
func List(ctx context.Context, ep Endpoint) ([]Listed, error) {
	hr, err := http.NewRequestWithContext(ctx, http.MethodGet, ep.URL+"/models", nil)
	if err != nil {
		return nil, err
	}
	header(hr, ep)
	resp, err := HTTP.Do(hr)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, errors.New(apiError(resp))
	}
	var r struct {
		Data []struct {
			ID            string `json:"id"`
			Name          string `json:"name"`
			ContextLength int    `json:"context_length"`
			Pricing       struct {
				Prompt     string `json:"prompt"`
				Completion string `json:"completion"`
			} `json:"pricing"`
		} `json:"data"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 32<<20)).Decode(&r); err != nil {
		return nil, err
	}
	out := make([]Listed, 0, len(r.Data))
	for _, d := range r.Data {
		l := Listed{ID: d.ID, Name: d.Name, Context: d.ContextLength}
		var p, c float64
		if _, err := fmt.Sscan(d.Pricing.Prompt, &p); err == nil {
			l.In = p * 1e6
		}
		if _, err := fmt.Sscan(d.Pricing.Completion, &c); err == nil {
			l.Out = c * 1e6
		}
		out = append(out, l)
	}
	sort.Slice(out, func(a, b int) bool { return out[a].ID < out[b].ID })
	return out, nil
}

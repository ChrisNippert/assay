package server

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"time"

	"assay/internal/ollama"
	"assay/internal/remote"
)

// /v1 for API models and teams: Assay runs them itself (s.chat, as the chat
// page does) and answers in OpenAI's shape, so an editor or any OpenAI
// client can use them like a local model. A team's planning and helpers'
// results come as reasoning_content, its answer as content.

// v1Message is a chat message in OpenAI's shape.
type v1Message struct {
	Role       string            `json:"role"`
	Content    json.RawMessage   `json:"content"`
	ToolCalls  []remote.WireCall `json:"tool_calls,omitempty"`
	ToolCallID string            `json:"tool_call_id,omitempty"`
}

type v1ChatRequest struct {
	Model         string        `json:"model"`
	Messages      []v1Message   `json:"messages"`
	Tools         []ollama.Tool `json:"tools"`
	Stream        bool          `json:"stream"`
	StreamOptions struct {
		IncludeUsage bool `json:"include_usage"`
	} `json:"stream_options"`
	Temperature         *float64 `json:"temperature"`
	TopP                *float64 `json:"top_p"`
	Seed                *int     `json:"seed"`
	Stop                any      `json:"stop"`
	MaxTokens           int      `json:"max_tokens"`
	MaxCompletionTokens int      `json:"max_completion_tokens"`
	ReasoningEffort     string   `json:"reasoning_effort"`
}

// assayModel says the name is one Assay runs itself rather than the
// engine: an API model or a team.
func (s *Server) assayModel(name string) bool {
	if _, ok := s.remote.Team(name); ok {
		return true
	}
	_, _, ok := s.remote.Lookup(name)
	return ok
}

// v1Content reads a message's content: a string, or parts of text and
// images (data URLs; others are left out).
func v1Content(raw json.RawMessage) (text string, images []string) {
	var str string
	if json.Unmarshal(raw, &str) == nil {
		return str, nil
	}
	var parts []struct {
		Type     string `json:"type"`
		Text     string `json:"text"`
		ImageURL struct {
			URL string `json:"url"`
		} `json:"image_url"`
	}
	json.Unmarshal(raw, &parts)
	var texts []string
	for _, p := range parts {
		switch p.Type {
		case "text":
			texts = append(texts, p.Text)
		case "image_url":
			if _, data, ok := strings.Cut(p.ImageURL.URL, ";base64,"); ok && strings.HasPrefix(p.ImageURL.URL, "data:") {
				images = append(images, data)
			}
		}
	}
	return strings.Join(texts, "\n"), images
}

// toChat turns an OpenAI request into Assay's own.
func (in v1ChatRequest) toChat() ollama.ChatRequest {
	req := ollama.ChatRequest{Model: in.Model, Tools: in.Tools, Options: map[string]any{}}
	for _, m := range in.Messages {
		text, images := v1Content(m.Content)
		role := m.Role
		if role == "developer" {
			role = "system"
		}
		req.Messages = append(req.Messages, ollama.Message{Role: role, Content: text, Images: images, ToolCalls: remote.FromWire(m.ToolCalls), ToolCallID: m.ToolCallID})
	}
	if in.Temperature != nil {
		req.Options["temperature"] = *in.Temperature
	}
	if in.TopP != nil {
		req.Options["top_p"] = *in.TopP
	}
	if in.Seed != nil {
		req.Options["seed"] = *in.Seed
	}
	if in.Stop != nil {
		req.Options["stop"] = in.Stop
	}
	if n := max(in.MaxTokens, in.MaxCompletionTokens); n > 0 {
		req.Options["num_predict"] = n
	}
	switch in.ReasoningEffort {
	case "none":
		req.Think = false
	case "minimal":
		req.Think = "low"
	case "low", "medium", "high":
		req.Think = in.ReasoningEffort
	}
	return req
}

func v1ID() string {
	b := make([]byte, 12)
	rand.Read(b)
	return "chatcmpl-" + hex.EncodeToString(b)
}

// v1Usage is OpenAI's usage, plus what an API charged (as OpenRouter
// reports it).
func v1Usage(r ollama.ChatResponse) map[string]any {
	u := map[string]any{"prompt_tokens": r.PromptEvalCount, "completion_tokens": r.EvalCount, "total_tokens": r.PromptEvalCount + r.EvalCount}
	if r.Cost > 0 {
		u["cost"] = r.Cost
	}
	return u
}

func v1Finish(r ollama.ChatResponse) string {
	switch {
	case len(r.Message.ToolCalls) > 0:
		return "tool_calls"
	case r.DoneReason == "length":
		return "length"
	}
	return "stop"
}

// v1Assay answers a chat completion with an API model or a team.
func (s *Server) v1Assay(w http.ResponseWriter, r *http.Request, raw []byte) {
	var in v1ChatRequest
	if err := json.Unmarshal(raw, &in); err != nil {
		writeV1Error(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}
	req := in.toChat()
	id, created := v1ID(), time.Now().Unix()
	if !in.Stream {
		s.v1AssayWhole(w, r.Context(), req, id, created)
		return
	}

	started := false
	fl, _ := w.(http.Flusher)
	send := func(v any) error {
		if !started {
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			w.Header().Set("X-Accel-Buffering", "no")
			w.WriteHeader(http.StatusOK)
			started = true
		}
		b, _ := json.Marshal(v)
		if _, err := w.Write([]byte("data: " + string(b) + "\n\n")); err != nil {
			return err
		}
		if fl != nil {
			fl.Flush()
		}
		return nil
	}
	chunk := func(delta map[string]any, finish any) map[string]any {
		return map[string]any{"id": id, "object": "chat.completion.chunk", "created": created, "model": in.Model,
			"choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": finish}}}
	}
	first := true
	err := s.chat(r.Context(), req, func(c ollama.ChatResponse) error {
		delta := map[string]any{}
		if first {
			delta["role"] = "assistant"
		}
		if !c.Done {
			if c.Message.Content == "" && c.Message.Thinking == "" {
				return nil // a team's progress
			}
			if c.Message.Content != "" {
				delta["content"] = c.Message.Content
			}
			if c.Message.Thinking != "" {
				delta["reasoning_content"] = c.Message.Thinking
			}
			first = false
			return send(chunk(delta, nil))
		}
		if calls := remote.ToWire(c.Message.ToolCalls); len(calls) > 0 {
			delta["tool_calls"] = calls
		}
		out := chunk(delta, v1Finish(c))
		if in.StreamOptions.IncludeUsage {
			if err := send(out); err != nil {
				return err
			}
			out = map[string]any{"id": id, "object": "chat.completion.chunk", "created": created, "model": in.Model, "choices": []any{}, "usage": v1Usage(c)}
		}
		return send(out)
	})
	if err != nil {
		if !started {
			s.v1Fail(w, err)
			return
		}
		send(map[string]any{"error": map[string]any{"message": err.Error(), "type": "server_error"}})
	}
	if started {
		w.Write([]byte("data: [DONE]\n\n"))
	}
}

// v1AssayWhole answers without streaming: one chat.completion.
func (s *Server) v1AssayWhole(w http.ResponseWriter, ctx context.Context, req ollama.ChatRequest, id string, created int64) {
	var content, thinking strings.Builder
	var final ollama.ChatResponse
	err := s.chat(ctx, req, func(c ollama.ChatResponse) error {
		if c.Done {
			final = c
		} else {
			content.WriteString(c.Message.Content)
			thinking.WriteString(c.Message.Thinking)
		}
		return nil
	})
	if err != nil {
		s.v1Fail(w, err)
		return
	}
	msg := map[string]any{"role": "assistant", "content": content.String()}
	if thinking.Len() > 0 {
		msg["reasoning_content"] = thinking.String()
	}
	if calls := remote.ToWire(final.Message.ToolCalls); len(calls) > 0 {
		msg["tool_calls"] = calls
	}
	writeJSON(w, http.StatusOK, map[string]any{"id": id, "object": "chat.completion", "created": created, "model": req.Model,
		"choices": []any{map[string]any{"index": 0, "message": msg, "finish_reason": v1Finish(final)}}, "usage": v1Usage(final)})
}

func (s *Server) v1Fail(w http.ResponseWriter, err error) {
	status := http.StatusBadGateway
	var apiErr *ollama.APIError
	if errors.As(err, &apiErr) && apiErr.StatusCode >= 400 {
		status = apiErr.StatusCode
	}
	writeV1Error(w, status, err.Error())
}

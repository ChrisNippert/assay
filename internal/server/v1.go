package server

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"

	"ollamui/internal/ollama"
	"ollamui/internal/presets"
)

// apiEngine is a backend that can pass OpenAI-style requests to its
// models, which is what serves /v1 (the llama.cpp engine).
type apiEngine interface {
	Proxy(ctx context.Context, model string, options map[string]any, path string, body []byte) (*http.Response, error)
	IsEmbedding(name string) bool
}

// writeV1Error answers in OpenAI's error shape.
func writeV1Error(w http.ResponseWriter, status int, msg string) {
	typ := "invalid_request_error"
	switch {
	case status == http.StatusNotFound:
		typ = "not_found_error"
	case status >= 500:
		typ = "server_error"
	}
	writeJSON(w, status, map[string]any{"error": map[string]any{"message": msg, "type": typ, "param": nil, "code": nil}})
}

func (s *Server) v1Engine(w http.ResponseWriter) (apiEngine, bool) {
	e, ok := s.ollama.(apiEngine)
	if !ok {
		writeV1Error(w, http.StatusNotImplemented, "the /v1 API needs the llama.cpp engine, not Ollama")
	}
	return e, ok
}

// handleV1Models lists the installed chat models, then the presets.
func (s *Server) handleV1Models(w http.ResponseWriter, r *http.Request) {
	e, ok := s.v1Engine(w)
	if !ok {
		return
	}
	models, err := s.ollama.List(r.Context())
	if err != nil {
		writeV1Error(w, http.StatusBadGateway, err.Error())
		return
	}
	type entry struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		OwnedBy string `json:"owned_by"`
	}
	data := []entry{}
	seen := map[string]bool{}
	for _, m := range models {
		if e.IsEmbedding(m.Name) {
			continue
		}
		seen[strings.ToLower(m.Name)] = true
		data = append(data, entry{m.Name, "model", m.ModifiedAt.Unix(), "ollamui"})
	}
	if list, err := s.presets.List(); err == nil {
		for _, p := range list {
			if !seen[p.Name] {
				data = append(data, entry{p.Name, "model", 0, "ollamui"})
			}
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": data})
}

// v1Target resolves the name in a request to the model that runs it and
// the preset's settings, if the name is a preset. Installed models win
// over a preset of the same name.
func (s *Server) v1Target(ctx context.Context, name string) (model string, preset *presets.Preset) {
	if models, err := s.ollama.List(ctx); err == nil {
		for _, m := range models {
			if strings.EqualFold(m.Name, name) {
				return name, nil
			}
		}
	}
	if p, err := s.presets.Get(name); err == nil {
		return p.Base, p
	}
	return name, nil
}

func (s *Server) handleV1Chat(w http.ResponseWriter, r *http.Request) {
	s.v1Proxy(w, r, "/v1/chat/completions")
}

func (s *Server) handleV1Embeddings(w http.ResponseWriter, r *http.Request) {
	s.v1Proxy(w, r, "/v1/embeddings")
}

// v1Proxy passes the request to the model's server and streams the answer
// back as it comes.
func (s *Server) v1Proxy(w http.ResponseWriter, r *http.Request, path string) {
	e, ok := s.v1Engine(w)
	if !ok {
		return
	}
	raw, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 64<<20))
	if err != nil {
		writeV1Error(w, http.StatusBadRequest, "could not read the request: "+err.Error())
		return
	}
	var req map[string]any
	if err := json.Unmarshal(raw, &req); err != nil {
		writeV1Error(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}
	name, _ := req["model"].(string)
	if name == "" {
		writeV1Error(w, http.StatusBadRequest, "model is required")
		return
	}
	model, preset := s.v1Target(r.Context(), name)
	var options map[string]any
	if preset != nil {
		options = preset.Options
		if path == "/v1/chat/completions" {
			applyPreset(req, preset)
		}
	}
	req["model"] = model
	body, _ := json.Marshal(req)

	res, err := e.Proxy(r.Context(), model, options, path, body)
	if err != nil {
		status, msg := http.StatusBadGateway, err.Error()
		var apiErr *ollama.APIError
		if errors.As(err, &apiErr) {
			msg = apiErr.Message
			if apiErr.StatusCode >= 400 {
				status = apiErr.StatusCode
			}
		}
		writeV1Error(w, status, msg)
		return
	}
	defer res.Body.Close()
	ct := res.Header.Get("Content-Type")
	if ct != "" {
		w.Header().Set("Content-Type", ct)
	}
	if strings.HasPrefix(ct, "text/event-stream") {
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("X-Accel-Buffering", "no")
	}
	w.WriteHeader(res.StatusCode)
	fl, _ := w.(http.Flusher)
	buf := make([]byte, 16<<10)
	for {
		n, err := res.Body.Read(buf)
		if n > 0 {
			if _, werr := w.Write(buf[:n]); werr != nil {
				return
			}
			if fl != nil {
				fl.Flush() // per chunk, so tokens appear as they're made
			}
		}
		if err != nil {
			return
		}
	}
}

// applyPreset adds a preset's system prompt (unless the request brings its
// own) and thinking setting (unless the request sets one) to a chat body.
func applyPreset(req map[string]any, p *presets.Preset) {
	if p.System != "" {
		msgs, _ := req["messages"].([]any)
		has := false
		for _, m := range msgs {
			if mm, ok := m.(map[string]any); ok && (mm["role"] == "system" || mm["role"] == "developer") {
				has = true
			}
		}
		if !has {
			req["messages"] = append([]any{map[string]any{"role": "system", "content": p.System}}, msgs...)
		}
	}
	if p.Think != nil {
		if _, set := req["chat_template_kwargs"]; !set {
			req["chat_template_kwargs"] = map[string]any{"enable_thinking": *p.Think}
			if _, set := req["reasoning_budget"]; !set && !*p.Think {
				req["reasoning_budget"] = 0
			}
		}
	}
}

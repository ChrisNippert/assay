package server

import (
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"time"
	"unicode/utf8"

	"ollamui/internal/chats"
	"ollamui/internal/ollama"
	"ollamui/internal/presets"
)

// --- pages ---

type chatPageData struct {
	Chat   *chats.Chat
	Model  string // model selected in the picker
	Preset string // preset selected in the picker (takes precedence over Model)
	Tune   tuneState
}

// tuneState is this chat's own tweaks, handed to the Tune panel.
type tuneState struct {
	Options map[string]any `json:"options"`
	System  string         `json:"system"`
	Think   *bool          `json:"think"`
}

func (s *Server) handleChatPage(w http.ResponseWriter, r *http.Request) {
	p := s.basePage(r.Context(), "chat", "Chat")
	data := &chatPageData{Preset: r.URL.Query().Get("preset")}
	if id := r.PathValue("id"); id != "" {
		c, err := s.chats.Get(id)
		if errors.Is(err, chats.ErrNotFound) {
			http.Redirect(w, r, "/chat", http.StatusFound)
			return
		}
		if err != nil {
			s.log.Error("load chat", "id", id, "err", err)
			http.Error(w, "could not load chat", http.StatusInternalServerError)
			return
		}
		data.Chat = c
		data.Model = c.Model
		data.Preset = c.Preset
		data.Tune = tuneState{Options: c.Options, System: c.System, Think: c.Think}
		p.ChatID = c.ID
		p.Title = c.Title
	}
	// Pick a model for a new chat: ?model=, then whatever is loaded, then the
	// first installed one.
	if data.Model == "" {
		data.Model = r.URL.Query().Get("model")
	}
	if data.Model == "" && len(p.Loaded) > 0 {
		data.Model = p.Loaded[0].Name
	}
	if data.Model == "" && len(p.Models) > 0 {
		data.Model = p.Models[0].Name
	}
	p.Data = data
	s.render(w, "chat", p)
}

func (s *Server) handleLibrary(w http.ResponseWriter, r *http.Request) {
	p := s.basePage(r.Context(), "library", "Library")
	s.render(w, "library", p)
}

// --- models ---

type nameRequest struct {
	Name string `json:"name"`
}

func (s *Server) handleLoaded(w http.ResponseWriter, r *http.Request) {
	models, err := s.loaded(r.Context())
	if err != nil {
		writeError(w, ollamaStatus(err), err.Error())
		return
	}
	writeJSON(w, http.StatusOK, models)
}

func (s *Server) handleLoad(w http.ResponseWriter, r *http.Request) {
	var req nameRequest
	if !readJSON(w, r, &req) {
		return
	}
	if req.Name == "" {
		writeError(w, http.StatusBadRequest, "name is required")
		return
	}
	// Loaded from the UI means loaded until ejected, like Unsloth/LM Studio.
	if err := s.ollama.Load(r.Context(), req.Name, "-1"); err != nil {
		writeError(w, ollamaStatus(err), err.Error())
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (s *Server) handleEject(w http.ResponseWriter, r *http.Request) {
	var req nameRequest
	if !readJSON(w, r, &req) {
		return
	}
	if req.Name == "" {
		writeError(w, http.StatusBadRequest, "name is required")
		return
	}
	if err := s.ollama.Unload(r.Context(), req.Name); err != nil {
		writeError(w, ollamaStatus(err), err.Error())
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (s *Server) handleDeleteModel(w http.ResponseWriter, r *http.Request) {
	var req nameRequest
	if !readJSON(w, r, &req) {
		return
	}
	if req.Name == "" {
		writeError(w, http.StatusBadRequest, "name is required")
		return
	}
	if err := s.ollama.Delete(r.Context(), req.Name); err != nil {
		writeError(w, ollamaStatus(err), err.Error())
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// --- chat ---

// sendRequest is one message from the browser, plus the chat's current
// Tune settings. Model is ignored when Preset is set.
type sendRequest struct {
	ChatID  string         `json:"chatId"`
	Model   string         `json:"model"`
	Preset  string         `json:"preset"`
	Content string         `json:"content"`
	Options map[string]any `json:"options"`
	System  string         `json:"system"`
	Think   *bool          `json:"think"`
}

// event is one line of the NDJSON stream sent back to the browser.
type event struct {
	Type    string       `json:"type"` // chat, thinking, content, done, error
	ID      string       `json:"id,omitempty"`
	Text    string       `json:"text,omitempty"`
	Stats   *chats.Stats `json:"stats,omitempty"`
	Message string       `json:"message,omitempty"`
}

func (s *Server) handleChatSend(w http.ResponseWriter, r *http.Request) {
	var req sendRequest
	if !readJSON(w, r, &req) {
		return
	}
	req.Content = strings.TrimSpace(req.Content)
	overrides, err := presets.CleanOptions(req.Options)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	// Resolve what actually gets sent: preset settings, then this chat's tweaks.
	var preset *presets.Preset
	if req.Preset != "" {
		if preset, err = s.presets.Get(req.Preset); err != nil {
			writeError(w, http.StatusBadRequest, "preset "+req.Preset+" not found")
			return
		}
		req.Model = preset.Base
	}
	if req.Content == "" || req.Model == "" {
		writeError(w, http.StatusBadRequest, "model and content are required")
		return
	}
	options, system, think := overrides, req.System, req.Think
	if preset != nil {
		options = presets.Merge(preset.Options, overrides)
		if system == "" {
			system = preset.System
		}
		if think == nil {
			think = preset.Think
		}
	}

	var c *chats.Chat
	if req.ChatID != "" {
		if c, err = s.chats.Get(req.ChatID); err != nil {
			writeError(w, http.StatusNotFound, "chat not found")
			return
		}
	} else {
		c = chats.New(req.Model)
		c.Title = titleFrom(req.Content)
	}
	c.Model = req.Model // switching models mid-chat is allowed
	c.Preset, c.Options, c.System, c.Think = req.Preset, overrides, req.System, req.Think
	c.Messages = append(c.Messages, chats.Message{Role: "user", Content: req.Content, Created: time.Now()})
	if err := s.chats.Save(c); err != nil {
		s.log.Error("save chat", "err", err)
		writeError(w, http.StatusInternalServerError, "could not save chat")
		return
	}

	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("X-Accel-Buffering", "no")
	rc := http.NewResponseController(w)
	enc := json.NewEncoder(w)
	emit := func(e event) error {
		if err := enc.Encode(e); err != nil {
			return err
		}
		return rc.Flush()
	}
	if emit(event{Type: "chat", ID: c.ID}) != nil {
		return
	}

	msgs := toOllama(c.Messages)
	if system != "" {
		msgs = append([]ollama.Message{{Role: "system", Content: system}}, msgs...)
	}
	chatReq := ollama.ChatRequest{Model: c.Model, Messages: msgs, Options: options}
	if think != nil {
		chatReq.Think = *think
	}
	// Chat requests reset Ollama's keep-alive timer. If the user loaded this
	// model "until ejected", keep it that way.
	if s.isPinned(r, c.Model) {
		chatReq.KeepAlive = -1
	}

	var content, thinking strings.Builder
	var stats *chats.Stats
	err = s.ollama.Chat(r.Context(), chatReq, func(resp ollama.ChatResponse) error {
		if t := resp.Message.Thinking; t != "" {
			thinking.WriteString(t)
			if err := emit(event{Type: "thinking", Text: t}); err != nil {
				return err
			}
		}
		if t := resp.Message.Content; t != "" {
			content.WriteString(t)
			if err := emit(event{Type: "content", Text: t}); err != nil {
				return err
			}
		}
		if resp.Done {
			stats = statsFrom(resp)
		}
		return nil
	})

	// Keep whatever was generated, even if the user stopped it early.
	if content.Len() > 0 || thinking.Len() > 0 {
		c.Messages = append(c.Messages, chats.Message{
			Role:     "assistant",
			Content:  content.String(),
			Thinking: thinking.String(),
			Model:    c.Model,
			Stats:    stats,
			Created:  time.Now(),
		})
		if serr := s.chats.Save(c); serr != nil {
			s.log.Error("save chat", "err", serr)
		}
	}
	if err != nil {
		if r.Context().Err() == nil {
			emit(event{Type: "error", Message: err.Error()})
		}
		return
	}
	emit(event{Type: "done", Stats: stats})
}

func (s *Server) isPinned(r *http.Request, model string) bool {
	loaded, err := s.loaded(r.Context())
	if err != nil {
		return false
	}
	for _, m := range loaded {
		if m.Name == model {
			return m.Pinned
		}
	}
	return false
}

func (s *Server) handleDeleteChat(w http.ResponseWriter, r *http.Request) {
	if err := s.chats.Delete(r.PathValue("id")); err != nil {
		writeError(w, http.StatusNotFound, "chat not found")
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// toOllama converts stored messages to the API shape. Earlier thinking is
// left out: models are trained to see only final answers in the history.
func toOllama(msgs []chats.Message) []ollama.Message {
	out := make([]ollama.Message, 0, len(msgs))
	for _, m := range msgs {
		out = append(out, ollama.Message{Role: m.Role, Content: m.Content})
	}
	return out
}

func statsFrom(r ollama.ChatResponse) *chats.Stats {
	st := &chats.Stats{
		PromptTokens: r.PromptEvalCount,
		OutputTokens: r.EvalCount,
		TotalMS:      r.TotalDuration / 1e6,
		LoadMS:       r.LoadDuration / 1e6,
		DoneReason:   r.DoneReason,
	}
	if r.EvalDuration > 0 {
		st.TokensPerSec = float64(r.EvalCount) / (float64(r.EvalDuration) / 1e9)
	}
	return st
}

func titleFrom(s string) string {
	s, _, _ = strings.Cut(strings.TrimSpace(s), "\n")
	const max = 60
	if utf8.RuneCountInString(s) <= max {
		return s
	}
	r := []rune(s)
	return strings.TrimSpace(string(r[:max])) + "…"
}

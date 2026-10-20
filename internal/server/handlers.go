package server

import (
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"sync"
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
	// Pick a model for a new chat: ?model=, then whatever is loaded on the
	// GPU (a model only on the CPU is voice mode's talker), then the first
	// installed one.
	if data.Model == "" {
		data.Model = r.URL.Query().Get("model")
	}
	for _, m := range p.Loaded {
		if data.Model == "" && m.SizeVRAM > 0 {
			data.Model = m.Name
		}
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
	// Load with the context chats will use, so the first message doesn't
	// have to reload the model.
	var opts map[string]any
	if auto := s.pickContext(r.Context(), req.Name, true); auto.Ctx > 0 {
		opts = map[string]any{"num_ctx": auto.Ctx}
	}
	// Loaded from the UI means loaded until ejected, like Unsloth/LM Studio.
	if err := s.ollama.Load(r.Context(), req.Name, "-1", opts); err != nil {
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
	var req struct {
		Name  string   `json:"name"`
		Names []string `json:"names"`
	}
	if !readJSON(w, r, &req) {
		return
	}
	names := req.Names
	if req.Name != "" {
		names = append([]string{req.Name}, names...)
	}
	if len(names) == 0 {
		writeError(w, http.StatusBadRequest, "name is required")
		return
	}
	// Every name shares the blobs; the disk is only freed once all are gone.
	for _, name := range names {
		if err := s.ollama.Delete(r.Context(), name); err != nil {
			writeError(w, ollamaStatus(err), err.Error())
			return
		}
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
	// FromUser, when set, drops that user message (counting user messages
	// from 0) and everything after it before sending: edit and regenerate.
	FromUser *int `json:"fromUser"`
	// Voice asks for a spoken reply: see voiceTurn. Talker names the model
	// that talks while the chat model thinks: "" picks one, "off" none.
	Voice  bool   `json:"voice"`
	Talker string `json:"talker"`
}

// event is one line of the NDJSON stream sent back to the browser.
type event struct {
	Type    string       `json:"type"` // chat, thinking, content, say, done, error
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
	t, ok := s.prepareTurn(w, r, req)
	if !ok {
		return
	}
	if req.Voice {
		s.voiceTurn(w, r, t, req.Talker)
		return
	}
	emit := streamEvents(w)
	if emit(event{Type: "chat", ID: t.chat.ID}) != nil {
		return
	}
	var content, thinking strings.Builder
	var stats *chats.Stats
	err := s.ollama.Chat(r.Context(), t.req, func(resp ollama.ChatResponse) error {
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
			stats = t.stats(resp)
		}
		return nil
	})
	s.saveReply(t, content.String(), thinking.String(), "", stats)
	if err == nil {
		s.noteThinking(t.req, thinking.Len() > 0)
	}
	if err != nil {
		if r.Context().Err() == nil {
			emit(event{Type: "error", Message: err.Error()})
		}
		return
	}
	emit(event{Type: "done", Stats: stats})
}

// turn is one send, resolved: the chat with the new user message saved,
// and the request that goes to Ollama.
type turn struct {
	chat *chats.Chat
	req  ollama.ChatRequest
	auto autoContext
	// spoken is what voice mode said before the reply (lead-in only).
	spoken string
	// replyModel is the model that wrote the reply, when it isn't the
	// chat's (voice mode's talker answering small talk).
	replyModel string
}

// stats turns Ollama's final chunk into the stats saved with the reply.
func (t *turn) stats(resp ollama.ChatResponse) *chats.Stats {
	st := statsFrom(resp)
	switch n := t.req.Options["num_ctx"].(type) {
	case int:
		st.Context, st.AutoContext = n, t.auto.Ctx > 0
	case int64:
		st.Context = int(n)
	}
	return st
}

// prepareTurn resolves preset and chat settings, saves the user message and
// builds the Ollama request. On failure it has already written the error.
func (s *Server) prepareTurn(w http.ResponseWriter, r *http.Request, req sendRequest) (*turn, bool) {
	req.Content = strings.TrimSpace(req.Content)
	overrides, err := presets.CleanOptions(req.Options)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return nil, false
	}
	// Resolve what actually gets sent: preset settings, then this chat's tweaks.
	var preset *presets.Preset
	if req.Preset != "" {
		if preset, err = s.presets.Get(req.Preset); err != nil {
			writeError(w, http.StatusBadRequest, "preset "+req.Preset+" not found")
			return nil, false
		}
		req.Model = preset.Base
	}
	if req.Content == "" || req.Model == "" {
		writeError(w, http.StatusBadRequest, "model and content are required")
		return nil, false
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
			return nil, false
		}
	} else {
		c = chats.New(req.Model)
		c.Title = titleFrom(req.Content)
	}
	if req.FromUser != nil {
		if !c.TruncateAtUser(*req.FromUser) {
			writeError(w, http.StatusBadRequest, "no such message")
			return nil, false
		}
	}
	c.Model = req.Model // switching models mid-chat is allowed
	c.Preset, c.Options, c.System, c.Think = req.Preset, overrides, req.System, req.Think
	c.Messages = append(c.Messages, chats.Message{Role: "user", Content: req.Content, Created: time.Now()})
	if err := s.chats.Save(c); err != nil {
		s.log.Error("save chat", "err", err)
		writeError(w, http.StatusInternalServerError, "could not save chat")
		return nil, false
	}

	if req.Voice {
		system = strings.TrimSpace(system + "\n\n" + voiceSystem)
	}
	msgs := toOllama(c.Messages)
	if system != "" {
		msgs = append([]ollama.Message{{Role: "system", Content: system}}, msgs...)
	}
	// Without a context length from the chat or preset, Ollama would use its
	// small default (4k on most GPUs), which thinking models fill quickly.
	// Pick the largest that fits on the GPU instead.
	t := &turn{chat: c}
	if _, set := options["num_ctx"]; !set {
		if t.auto = s.pickContext(r.Context(), c.Model, true); t.auto.Ctx > 0 {
			options = presets.Merge(options, map[string]any{"num_ctx": t.auto.Ctx})
		}
	}
	t.req = ollama.ChatRequest{Model: c.Model, Messages: msgs, Options: options}
	if think != nil {
		t.req.Think = *think
	}
	// Chat requests reset Ollama's keep-alive timer. If the user loaded this
	// model "until ejected", keep it that way.
	if s.isPinned(r, c.Model) {
		t.req.KeepAlive = -1
	}
	return t, true
}

// streamEvents starts an NDJSON response and returns a function that sends
// one event and flushes it.
func streamEvents(w http.ResponseWriter) func(event) error {
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("X-Accel-Buffering", "no")
	rc := http.NewResponseController(w)
	enc := json.NewEncoder(w)
	var mu sync.Mutex
	return func(e event) error {
		mu.Lock()
		defer mu.Unlock()
		if err := enc.Encode(e); err != nil {
			return err
		}
		return rc.Flush()
	}
}

// saveReply keeps whatever was generated, even if the user stopped it early.
func (s *Server) saveReply(t *turn, content, thinking, spoken string, stats *chats.Stats) {
	if content == "" && thinking == "" && spoken == "" {
		return
	}
	model := t.chat.Model
	if t.replyModel != "" {
		model = t.replyModel
	}
	t.chat.Messages = append(t.chat.Messages, chats.Message{
		Role:     "assistant",
		Content:  content,
		Thinking: thinking,
		Spoken:   spoken,
		Model:    model,
		Stats:    stats,
		Created:  time.Now(),
	})
	if err := s.chats.Save(t.chat); err != nil {
		s.log.Error("save chat", "err", err)
	}
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

package server

import (
	"encoding/json"
	"errors"
	"net/http"
	"slices"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"assay/internal/chats"
	"assay/internal/hw"
	"assay/internal/ollama"
	"assay/internal/presets"
	"assay/internal/projects"
	"assay/internal/tools"
)

// --- pages ---

type chatPageData struct {
	Chat   *chats.Chat
	Model  string // model selected in the picker
	Preset string // preset selected in the picker (takes precedence over Model)
	Tune   tuneState
	// Project is the chat's project (or the one a new chat will join).
	Project *projects.Project
	// Recent are the presets to offer on an empty chat: the last used
	// first, then the rest, three at most.
	Recent []*presets.Preset
	// Running: a reply is being written in the background.
	Running bool
}

// tuneState is this chat's own tweaks, handed to the Tune panel.
type tuneState struct {
	Options  map[string]any `json:"options"`
	System   string         `json:"system"`
	Think    *bool          `json:"think"`
	Compress *bool          `json:"compress"`
	Tools    *bool          `json:"tools"`
	// Compression is the chat's summary, if it has one; Messages counts the
	// chat's messages.
	Compression *chats.Compression `json:"compression"`
	Messages    int                `json:"messages"`
}

func (s *Server) handleChatPage(w http.ResponseWriter, r *http.Request) {
	if r.PathValue("id") == "" && s.firstRun(r.Context()) {
		http.Redirect(w, r, "/setup", http.StatusFound)
		return
	}
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
		data.Running = s.getRun(c.ID) != nil
		data.Model = c.Model
		data.Preset = c.Preset
		data.Tune = tuneState{Options: c.Options, System: c.System, Think: c.Think, Compress: c.Compress, Tools: c.Tools, Compression: c.Compression, Messages: len(c.Messages)}
		p.ChatID = c.ID
		p.Title = c.Title
	}
	// A chat in a project, or a new one started from it, shows the project;
	// a new one starts with the project's model or preset.
	pid := r.URL.Query().Get("project")
	if data.Chat != nil {
		pid = data.Chat.Project
	}
	if pid != "" {
		if pr, err := s.projects.Get(pid); err == nil {
			data.Project = pr
			p.ProjectID = pr.ID
			if data.Chat == nil && r.URL.Query().Get("model") == "" {
				data.Model, data.Preset = pr.Model, pr.Preset
			}
		}
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
	if data.Model == "" && p.Starred != "" && slices.ContainsFunc(p.Models, func(m ollama.Model) bool { return m.Name == p.Starred }) {
		data.Model = p.Starred
	}
	if data.Model == "" && len(p.Models) > 0 {
		data.Model = p.Models[0].Name
	}
	if data.Chat == nil {
		data.Recent = recentPresets(p.Chats, p.Presets, 3)
	}
	p.Data = data
	s.render(w, "chat", p)
}

// recentPresets orders presets by when a chat last used them (chats come
// newest first), then the unused ones by name, and keeps n.
func recentPresets(list []chats.Summary, all []*presets.Preset, n int) []*presets.Preset {
	var out []*presets.Preset
	seen := map[string]bool{}
	byName := map[string]*presets.Preset{}
	for _, p := range all {
		byName[p.Name] = p
	}
	for _, c := range list {
		if p := byName[c.Preset]; p != nil && !seen[p.Name] {
			seen[p.Name] = true
			out = append(out, p)
		}
	}
	for _, p := range all {
		if !seen[p.Name] {
			out = append(out, p)
		}
	}
	return out[:min(n, len(out))]
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

// readLiveGPU is swapped out in tests.
var readLiveGPU = hw.ReadLive

func (s *Server) handleGPU(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	writeJSON(w, http.StatusOK, readLiveGPU())
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
	// have to reload the model. Loaded from the UI means loaded until
	// ejected, like Unsloth/LM Studio.
	if _, err := s.loadAuto(r.Context(), req.Name, "-1"); err != nil {
		writeError(w, ollamaStatus(err), err.Error())
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// handleStar stars a model as the default for new chats; an empty name
// clears it. One model is starred at a time.
func (s *Server) handleStar(w http.ResponseWriter, r *http.Request) {
	var req nameRequest
	if !readJSON(w, r, &req) {
		return
	}
	st := s.chatSt.Get()
	st.DefaultModel = req.Name
	if err := s.chatSt.Set(st); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
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
	// Compress is the chat's "Compress automatically" choice; nil keeps the
	// chat's own, else its preset's, else the default in Settings.
	Compress *bool `json:"compress"`
	// Tools lets the model call tools: nil leaves it to the preset, then
	// to on (when the model supports them).
	Tools *bool `json:"tools"`
	// FromUser, when set, drops that user message (counting user messages
	// from 0) and everything after it before sending: edit and regenerate.
	FromUser *int `json:"fromUser"`
	// Voice asks for a spoken reply: see voiceTurn. Talker names the model
	// that talks while the chat model thinks: "" picks one, "off" none.
	Voice  bool   `json:"voice"`
	Talker string `json:"talker"`
	// Spoken asks for a reply written to be read aloud (plain prose), as
	// dictation's "Spoken style" toggle does.
	Spoken bool `json:"spoken"`
	// Project puts a new chat in a project.
	Project string `json:"project"`
	// Attachments are tokens from the upload endpoint for the new message.
	// An edit or regenerate keeps the attachments the message already had.
	Attachments []string `json:"attachments"`
}

// event is one line of the NDJSON stream sent back to the browser.
type event struct {
	Type    string         `json:"type"` // chat, sources, thinking, content, tool, toolresult, say, status, compressed, uncompressed, done, stopped, error
	ID      string         `json:"id,omitempty"`
	Text    string         `json:"text,omitempty"`
	Stats   *chats.Stats   `json:"stats,omitempty"`
	Message string         `json:"message,omitempty"`
	Sources []chats.Source `json:"sources,omitempty"`
	// Compression is the summary a "compressed" event made.
	Compression *chats.Compression `json:"compression,omitempty"`
	// Call is a tool call: "tool" when it starts, "toolresult" when done.
	Call *chats.ToolCall `json:"call,omitempty"`
	// Attachments: the new message's files, with Cut set on those shortened.
	Attachments []chats.Attachment `json:"attachments,omitempty"`
}

func (s *Server) handleChatSend(w http.ResponseWriter, r *http.Request) {
	var req sendRequest
	if !readJSON(w, r, &req) {
		return
	}
	const busy = "This chat is still writing a reply. Stop it or wait for it to finish."
	// One reply per chat at a time. An existing chat is claimed before the
	// user message is saved; a new chat can't have a run yet.
	var ru *run
	if req.ChatID != "" && !req.Voice {
		if ru = s.beginRun(r, req.ChatID); ru == nil {
			writeError(w, http.StatusConflict, busy)
			return
		}
	} else if req.Voice && req.ChatID != "" && s.getRun(req.ChatID) != nil {
		writeError(w, http.StatusConflict, busy)
		return
	}
	t, ok := s.prepareTurn(w, r, req)
	if !ok {
		if ru != nil {
			s.endRun(req.ChatID, ru, nil)
		}
		return
	}
	// Voice turns stay tied to the request (see voiceTurn).
	if req.Voice {
		s.voiceTurn(w, r, t, req.Talker)
		return
	}
	if ru == nil {
		if ru = s.beginRun(r, t.chat.ID); ru == nil {
			writeError(w, http.StatusConflict, busy)
			return
		}
	}
	// The reply is written in the background; this request just follows it,
	// so the browser leaving doesn't stop it.
	go s.runTurn(ru, t)
	ru.follow(r.Context(), streamEvents(w))
}

// runTurn generates the reply for t, publishing events on ru, saves it
// exactly once and then removes the run.
func (s *Server) runTurn(ru *run, t *turn) {
	var last *event
	defer func() {
		if p := recover(); p != nil {
			s.log.Error("reply panicked", "chat", t.chat.ID, "panic", p)
			last = &event{Type: "error", Message: "the reply failed unexpectedly"}
		}
		s.endRun(t.chat.ID, ru, last)
	}()
	emit := ru.emit
	emit(event{Type: "chat", ID: t.chat.ID})
	if len(t.cutFiles) > 0 {
		emit(event{Type: "attachments", Attachments: t.cutFiles})
	}
	if len(t.sources) > 0 {
		emit(event{Type: "sources", Sources: t.sources})
	}
	s.compressTurn(ru.ctx, t, emit)
	content, thinking, stats, err := s.chatLoop(ru.ctx, t, emit)
	cancelled := ru.ctx.Err() != nil
	t.stopped = cancelled || errors.Is(err, errLooping)
	s.saveReply(t, content, thinking, "", stats)
	if err == nil {
		s.noteThinking(t.req, thinking != "")
	}
	switch {
	case cancelled:
		last = &event{Type: "stopped"}
	case err != nil:
		last = &event{Type: "error", Message: err.Error()}
	default:
		last = &event{Type: "done", Stats: stats}
	}
}

// turn is one send, resolved: the chat with the new user message saved,
// and the request that goes to Ollama.
type turn struct {
	chat *chats.Chat
	req  ollama.ChatRequest
	auto autoContext
	// stopped is set when the user stopped the reply.
	stopped bool
	// spoken is what voice mode said before the reply (lead-in only).
	spoken string
	// replyModel is the model that wrote the reply, when it isn't the
	// chat's (voice mode's talker answering small talk).
	replyModel string
	// sources are the project-file excerpts put in the prompt.
	sources []chats.Source
	// tools are what the model may call this reply; calls are the ones it made.
	tools []tools.Tool
	calls []chats.ToolCall
	// fitted is the context llama.cpp chose, once the reply is done.
	fitted int
	// ctxSaved: the model's own settings set its context, so it isn't auto.
	ctxSaved bool
	// system is the system prompt without a compression summary (see buildMessages).
	system string
	// compress: the chat compresses automatically (see compressOn).
	compress bool
	// dropped: an edit cut the summary away before this send.
	dropped bool
	// vision: the model can see images; docChars is the room documents have.
	vision   bool
	docChars int
	// cutFiles are the new message's attachments when some were shortened.
	cutFiles []chats.Attachment
}

// hist is the chat's messages with their attachments put in (see withFiles).
func (t *turn) hist(s *Server) []chats.Message {
	msgs, _ := s.withFiles(t.chat.ID, t.chat.Messages, t.vision, t.docChars)
	return msgs
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
	if st.Context == 0 && t.auto.Fit {
		st.Context, st.AutoContext = t.fitted, !t.ctxSaved
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
	if (req.Content == "" && len(req.Attachments) == 0) || req.Model == "" {
		writeError(w, http.StatusBadRequest, "model and content are required")
		return nil, false
	}
	options, system, think, wantTools := overrides, req.System, req.Think, req.Tools
	if preset != nil {
		options = presets.Merge(preset.Options, overrides)
		if system == "" {
			system = preset.System
		}
		if think == nil {
			think = preset.Think
		}
		if wantTools == nil {
			wantTools = preset.Tools
		}
	}
	if err := presets.CheckReplyFormat(options); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return nil, false
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
		if c.Title == "" {
			c.Title = "Attached files"
		}
		if req.Project != "" {
			if _, err := s.projects.Get(req.Project); err == nil {
				c.Project = req.Project
			}
		}
	}
	var dropped bool
	var files []chats.Attachment
	if req.FromUser != nil {
		at := c.UserIndex(*req.FromUser)
		if at >= 0 {
			files = slices.Clone(c.Messages[at].Attachments)
		}
		if !c.TruncateAtUser(*req.FromUser) {
			writeError(w, http.StatusBadRequest, "no such message")
			return nil, false
		}
		// A summary of messages the edit cut (or of any before them) no
		// longer stands for this chat.
		dropped = c.DropCompressionFrom(at)
		s.forgetChat(c.ID) // a saved cache may hold the messages just cut
	}
	c.Model = req.Model // switching models mid-chat is allowed
	c.Preset, c.Options, c.System, c.Think, c.Compress, c.Tools = req.Preset, overrides, req.System, req.Think, req.Compress, req.Tools
	if len(req.Attachments) > 0 {
		added, err := s.chats.Adopt(c.ID, req.Attachments)
		if err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return nil, false
		}
		files = append(files, added...)
	}
	vision := s.hasVision(r.Context(), c.Model)
	if !vision {
		for _, a := range files {
			if a.Kind == "image" {
				writeError(w, http.StatusBadRequest, c.Model+" can't see images. Pick a model with vision, or remove "+a.Name+".")
				return nil, false
			}
		}
	}
	c.Messages = append(c.Messages, chats.Message{Role: "user", Content: req.Content, Attachments: files, Created: time.Now()})
	if err := s.chats.Save(c); err != nil {
		s.log.Error("save chat", "err", err)
		writeError(w, http.StatusInternalServerError, "could not save chat")
		return nil, false
	}

	// A project's instructions come before the chat's own system prompt,
	// and excerpts from its files after.
	t := &turn{chat: c}
	if c.Project != "" {
		if pr, err := s.projects.Get(c.Project); err == nil {
			if pr.Instructions != "" {
				system = strings.TrimSpace(pr.Instructions + "\n\n" + system)
			}
			if pr.Ready() > 0 {
				t.sources = s.retrieve(r.Context(), pr.ID, ragQuery(c.Messages))
				if sp := sourcesPrompt(t.sources); sp != "" {
					system = strings.TrimSpace(system + "\n\n" + sp)
				}
			}
		}
	}
	if req.Spoken || req.Voice && s.voiceSt.Get().Prose {
		system = strings.TrimSpace(system + "\n\n" + voiceSystem)
	}
	if !req.Voice {
		t.tools = s.turnTools(r.Context(), c, wantTools, c.Model)
	}
	if len(t.tools) > 0 {
		system = strings.TrimSpace(system + "\n\n" + toolsSystem)
	}
	t.system, t.dropped, t.compress = system, dropped, s.compressOn(c, preset)
	// Without a context length from the chat or preset, Ollama would use its
	// small default (4k on most GPUs), which thinking models fill quickly.
	// Pick the largest that fits on the GPU instead.
	// GPU layers set by hand put part of the model on the CPU on purpose:
	// auto context (which keeps it all on the GPU) stays out of it, and the
	// spill isn't taken as a sign the context was too big.
	var saved map[string]any
	if ms, ok := s.ollama.(modelSettings); ok {
		saved, _ = ms.Settings(c.Model)
	}
	// Experts placed by hand (n_cpu_moe, often saved for the model) count too.
	_, manualGPU := options["num_gpu"]
	_, cpuMoE := options["n_cpu_moe"]
	_, savedMoE := saved["n_cpu_moe"]
	manualGPU = manualGPU || cpuMoE || savedMoE
	if manualGPU {
		s.manualGPU.Store(strings.ToLower(c.Model), true)
	} else {
		s.manualGPU.Delete(strings.ToLower(c.Model))
	}
	if _, set := options["num_ctx"]; !set && !manualGPU {
		if t.auto = s.pickContext(r.Context(), c.Model, true); t.auto.Ctx > 0 {
			options = presets.Merge(options, map[string]any{"num_ctx": t.auto.Ctx})
		}
	}
	_, t.ctxSaved = saved["num_ctx"]
	// Attached documents may fill part of the context; what doesn't fit is
	// cut, and the chat remembers which of the new files were.
	t.vision = vision
	ctxLen := intOf(options["num_ctx"])
	if ctxLen == 0 {
		ctxLen = intOf(saved["num_ctx"])
	}
	t.docChars = docBudget(ctxLen)
	hist, cut := s.withFiles(c.ID, c.Messages, vision, t.docChars)
	if last := &c.Messages[len(c.Messages)-1]; len(last.Attachments) > 0 {
		changed := false
		for i := range last.Attachments {
			if cut[last.Attachments[i].File] && !last.Attachments[i].Cut {
				last.Attachments[i].Cut, changed = true, true
			}
		}
		if changed {
			t.cutFiles = last.Attachments
			s.chats.Save(c)
		}
	}
	msgs := buildMessages(system, c.Compression, hist)
	t.req = ollama.ChatRequest{Model: c.Model, Messages: msgs, Options: options, ChatID: c.ID, Tools: toolDefs(t.tools)}
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
	if content == "" && thinking == "" && spoken == "" && len(t.calls) == 0 {
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
		Stopped:  t.stopped,
		Calls:    t.calls,
		Model:    model,
		Stats:    stats,
		Sources:  t.sources,
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
	s.forgetChat(r.PathValue("id"))
	w.WriteHeader(http.StatusNoContent)
}

// toOllama converts stored messages to the API shape. Earlier thinking is
// left out: models are trained to see only final answers in the history.
func toOllama(msgs []chats.Message) []ollama.Message {
	out := make([]ollama.Message, 0, len(msgs))
	for _, m := range msgs {
		// Tool calls the reply made come first, as the model saw them.
		out = append(out, toolMessages(m)...)
		content := m.Content
		if m.Stopped {
			// The model sees where it was cut off, and what it was thinking
			// if it hadn't started the answer yet.
			if content == "" && m.Thinking != "" {
				content = "(Thinking so far: " + m.Thinking + ")"
			}
			content = strings.TrimSpace(content + "\n\n[The user stopped this reply here.]")
		}
		out = append(out, ollama.Message{Role: m.Role, Content: content, Images: m.Images})
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

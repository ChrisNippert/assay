package server

import (
	"context"
	"encoding/json"
	"errors"
	"maps"
	"net/http"
	"regexp"
	"strings"
	"time"
	"unicode/utf8"

	"assay/internal/chats"
	"assay/internal/ollama"
	"assay/internal/presets"
)

// Automatic context compression. When a chat's next prompt would use most of
// the context, its oldest messages are summarized by the same model and the
// summary stands in for them (chats.Compression). The messages stay on disk
// and on the page; only what is sent changes.

const (
	// compressAt is the share of the context a prompt may fill.
	compressAt = 0.8
	// compressKeep is how many of the latest messages are always sent as they are.
	compressKeep = 4
	// compressShare is the share of the context the kept messages may take,
	// when they can be chosen to fit.
	compressShare = 0.5
	// compressReply is the room kept for the reply when num_predict doesn't say.
	compressReply = 1024
	// summaryTokens caps a summary's length.
	summaryTokens = 512
)

// tokenizer is a backend that counts tokens with the model's own tokenizer
// (the engine, through llama-server).
type tokenizer interface {
	Tokenize(ctx context.Context, model string, options map[string]any, text string) (tokens, ctxLen int, err error)
}

// approxTokens estimates a text's tokens without the model, for deciding what
// to keep. Three characters to a token is a little on the high side, which is
// the safe way round for keeping a prompt inside the context.
func approxTokens(s string) int { return utf8.RuneCountInString(s)/3 + 1 }

// intOf reads a number from an option value, however it was decoded.
func intOf(v any) int {
	switch n := v.(type) {
	case int:
		return n
	case int64:
		return int(n)
	case float64:
		return int(n)
	}
	return 0
}

// needsCompression reports whether a prompt of prompt tokens leaves too little
// of a ctxLen context for the reply: more than compressAt of it, or less room
// than the reply needs (num_predict, or compressReply). ctxLen 0 means unknown.
func needsCompression(prompt, ctxLen, numPredict int) bool {
	if ctxLen <= 0 {
		return false
	}
	reply := min(max(numPredict, compressReply), ctxLen/2)
	limit := min(int(float64(ctxLen)*compressAt), ctxLen-reply)
	return prompt > limit
}

// cutPoint picks how many of a chat's oldest messages to summarize: the fewest
// messages that are left out so the rest fit in compressShare of a ctxLen
// context, keeping at least compressKeep and starting at a user message. If
// the latest messages don't fit that share, it cuts before the latest user
// message it can. 0 means nothing can be summarized.
func cutPoint(msgs []chats.Message, ctxLen int) int {
	budget := int(float64(ctxLen) * compressShare)
	// sum is the tokens of msgs[i:], counted from the end.
	sum := 0
	for _, m := range msgs[max(len(msgs)-compressKeep, 0):] {
		sum += approxTokens(m.Content)
	}
	cut, fallback := 0, 0
	for i := len(msgs) - compressKeep; i >= 1; i-- {
		sum += approxTokens(msgs[i].Content)
		if msgs[i].Role != "user" {
			continue
		}
		if fallback == 0 {
			fallback = i
		}
		if sum > budget {
			break
		}
		cut = i
	}
	if cut == 0 {
		return fallback
	}
	return cut
}

// buildMessages is what a chat sends: its system prompt, with the summary of
// its oldest messages added to it (one system message, since some templates
// take only one), then the messages after the summary. Thinking is never
// sent: toOllama keeps only the answers.
func buildMessages(system string, comp *chats.Compression, hist []chats.Message) []ollama.Message {
	if comp != nil && comp.UpTo > 0 && comp.UpTo <= len(hist) {
		note := "Summary of the earlier conversation (those messages were condensed to save context):\n" + comp.Text
		system = strings.TrimSpace(system + "\n\n" + note)
		hist = hist[comp.UpTo:]
	}
	msgs := toOllama(hist)
	if system != "" {
		msgs = append([]ollama.Message{{Role: "system", Content: system}}, msgs...)
	}
	return msgs
}

// promptSize counts the tokens msgs take as a prompt and returns the context
// length the model runs with (0 if unknown). Without the engine's tokenizer
// it estimates. Each message adds a few tokens for its role markers.
func (s *Server) promptSize(ctx context.Context, model string, options map[string]any, msgs []ollama.Message) (int, int, error) {
	var b strings.Builder
	for _, m := range msgs {
		b.WriteString(m.Content)
		b.WriteString("\n")
	}
	overhead := 12 * len(msgs)
	for _, m := range msgs {
		overhead += 768 * len(m.Images) // an image is a few hundred tokens
	}
	ctxLen := intOf(options["num_ctx"])
	t, ok := s.ollama.(tokenizer)
	if !ok {
		return approxTokens(b.String()) + overhead, ctxLen, nil
	}
	n, engineCtx, err := t.Tokenize(ctx, model, options, b.String())
	if err != nil {
		return 0, 0, err
	}
	if ctxLen == 0 {
		ctxLen = engineCtx
	}
	return n + overhead, ctxLen, nil
}

// compressOn reports whether a chat compresses automatically: its own choice,
// else its preset's, else the default in Settings.
func (s *Server) compressOn(c *chats.Chat, p *presets.Preset) bool {
	if c.Compress != nil {
		return *c.Compress
	}
	if p != nil && p.Compress != nil {
		return *p.Compress
	}
	return s.chatSt.Get().CompressAuto
}

// compressTurn summarizes the turn's oldest messages when its prompt would
// fill the context, and sets the messages the reply is sent with. A summary
// the chat already has is always used. It reports progress on emit. A failure
// is logged, and the reply goes out with the history as it is.
func (s *Server) compressTurn(ctx context.Context, t *turn, emit func(event) error) {
	if t.dropped {
		emit(event{Type: "uncompressed"})
	}
	if !t.compress {
		return
	}
	est, ctxLen, err := s.promptSize(ctx, t.req.Model, t.req.Options, t.req.Messages)
	if err != nil {
		s.log.Warn("could not count the prompt", "chat", t.chat.ID, "err", err)
		return
	}
	// The tool schemas go with every request too (often 1k+ tokens); the
	// template's rendering isn't countable from here, so estimate them.
	if len(t.req.Tools) > 0 {
		if b, err := json.Marshal(t.req.Tools); err == nil {
			est += len(b) / 3
		}
	}
	if !needsCompression(est, ctxLen, intOf(t.req.Options["num_predict"])) {
		return
	}
	upTo := cutPoint(t.hist(s), ctxLen)
	if upTo <= compressedUpTo(t.chat) {
		return
	}
	emit(event{Type: "status", Text: "Summarizing earlier messages…"})
	if err := s.compressTo(ctx, t.chat, t.req.Model, t.req.Options, upTo, ctxLen); err != nil {
		s.log.Warn("could not summarize the chat", "chat", t.chat.ID, "err", err)
		emit(event{Type: "status"})
		return
	}
	t.req.Messages = buildMessages(t.system, t.chat.Compression, t.hist(s))
	emit(event{Type: "compressed", Compression: t.chat.Compression})
}

// compressedUpTo is how many messages of c a summary covers (0 if none).
func compressedUpTo(c *chats.Chat) int {
	if c.Compression == nil {
		return 0
	}
	return c.Compression.UpTo
}

// compressTo summarizes c's messages before upTo, folding in the summary it
// has, and saves the chat. The saved cache of the chat is dropped: it holds
// the history as it was sent before.
func (s *Server) compressTo(ctx context.Context, c *chats.Chat, model string, options map[string]any, upTo, ctxLen int) error {
	from, old := 0, ""
	if c.Compression != nil {
		from, old = c.Compression.UpTo, c.Compression.Text
	}
	budget := 2048
	if ctxLen > 0 {
		budget = ctxLen / 2
	}
	full, _ := s.withFiles(c.ID, c.Messages, false, 0)
	text, err := s.summarize(ctx, model, options, old, full[from:upTo], budget)
	if err != nil {
		return err
	}
	before := 0
	for _, m := range full[:upTo] {
		before += approxTokens(m.Content)
	}
	c.Compression = &chats.Compression{
		Text:    text,
		UpTo:    upTo,
		Model:   model,
		Saved:   max(before-approxTokens(text), 0),
		Created: time.Now(),
	}
	if err := s.chats.Save(c); err != nil {
		return err
	}
	s.forgetChat(c.ID)
	return nil
}

// summaryInstructions asks for a summary the chat can carry on from.
const summaryInstructions = `Condense the conversation below so the assistant can carry on without the full text. Write only the summary itself: a compact, factual account of decisions made, facts, the user's preferences and goals, open questions, names, numbers, dates and code identifiers, as they were. Use short bullets or short paragraphs. Don't add a title, don't mention this request, don't address the user, and don't add anything the conversation doesn't say. Keep it under 300 words.`

// summarize folds msgs into old (a summary, or "") and returns the new summary.
// Long input goes in pieces of about budget tokens, each folded into the last.
// The request is a single user message without thinking, at low temperature,
// with the chat's reply format left out so the summary is plain text.
func (s *Server) summarize(ctx context.Context, model string, options map[string]any, old string, msgs []chats.Message, budget int) (string, error) {
	opts := map[string]any{}
	maps.Copy(opts, options)
	for _, k := range []string{"reply_format", "reply_schema", "reply_grammar"} {
		delete(opts, k)
	}
	opts["temperature"] = 0.2
	opts["num_predict"] = summaryTokens

	size := max(budget-approxTokens(old)-400, 256)
	for _, chunk := range chunkMessages(msgs, size) {
		var b strings.Builder
		b.WriteString(summaryInstructions)
		b.WriteString("\n\n")
		if old != "" {
			b.WriteString("Summary of what came before (keep what still matters):\n")
			b.WriteString(old)
			b.WriteString("\n\n")
		}
		b.WriteString("The conversation:\n\n")
		b.WriteString(transcript(chunk, size*3))
		b.WriteString("\nWrite the updated summary now.")
		text, err := s.complete(ctx, model, opts, b.String())
		if err != nil {
			return "", err
		}
		old = text
	}
	return old, nil
}

// chunkMessages splits msgs into runs of about budget tokens each. A message
// longer than that is a run of its own.
func chunkMessages(msgs []chats.Message, budget int) [][]chats.Message {
	var out [][]chats.Message
	start, sum := 0, 0
	for i, m := range msgs {
		n := approxTokens(m.Content)
		if i > start && sum+n > budget {
			out = append(out, msgs[start:i])
			start, sum = i, 0
		}
		sum += n
	}
	if start < len(msgs) {
		out = append(out, msgs[start:])
	}
	return out
}

// transcript writes msgs as the dialogue the summarizer reads. Each message is
// cut to maxChars runes so one long paste can't fill the request.
func transcript(msgs []chats.Message, maxChars int) string {
	var b strings.Builder
	for _, m := range msgs {
		name := "User"
		switch m.Role {
		case "user":
		case "assistant":
			name = "Assistant"
		default:
			continue
		}
		text := m.Content
		if r := []rune(text); len(r) > maxChars {
			text = string(r[:maxChars]) + " …"
		}
		b.WriteString(name + ": " + text + "\n\n")
	}
	return b.String()
}

// thinkBlock matches a thinking block a model wrote into its answer anyway.
var thinkBlock = regexp.MustCompile(`(?s)<think>.*?</think>`)

// complete runs one non-streaming, non-thinking request and returns its text.
func (s *Server) complete(ctx context.Context, model string, opts map[string]any, prompt string) (string, error) {
	var b strings.Builder
	req := ollama.ChatRequest{
		Model:    model,
		Messages: []ollama.Message{{Role: "user", Content: prompt}},
		Options:  opts,
		Think:    false,
	}
	err := s.ollama.Chat(ctx, req, func(r ollama.ChatResponse) error {
		b.WriteString(r.Message.Content)
		return nil
	})
	if err != nil {
		return "", err
	}
	text := thinkBlock.ReplaceAllString(b.String(), "")
	if i := strings.LastIndex(text, "</think>"); i >= 0 {
		text = text[i+len("</think>"):]
	}
	text = strings.TrimSpace(text)
	if text == "" {
		return "", errors.New("the model wrote no summary")
	}
	return text, nil
}

// chatOptions is the model and the options a chat's requests run with, as
// prepareTurn resolves them: its preset's, then its own tweaks, then the
// context Assay picks when none is set.
func (s *Server) chatOptions(ctx context.Context, c *chats.Chat) (string, map[string]any) {
	model := c.Model
	options := map[string]any{}
	if c.Preset != "" {
		if p, err := s.presets.Get(c.Preset); err == nil {
			model = p.Base
			maps.Copy(options, p.Options)
		}
	}
	maps.Copy(options, c.Options)
	saved := map[string]any{}
	if ms, ok := s.ollama.(modelSettings); ok {
		saved, _ = ms.Settings(model)
	}
	_, manual := options["num_gpu"]
	_, cpuMoE := options["n_cpu_moe"]
	_, savedMoE := saved["n_cpu_moe"]
	if _, set := options["num_ctx"]; !set && !(manual || cpuMoE || savedMoE) {
		if a := s.pickContext(ctx, model, true); a.Ctx > 0 {
			options["num_ctx"] = a.Ctx
		}
	}
	return model, options
}

// handleCompress summarizes a chat's messages except the latest few, now:
// POST /api/chats/{id}/compress. It answers with the chat's compression.
func (s *Server) handleCompress(w http.ResponseWriter, r *http.Request) {
	c, err := s.chats.Get(r.PathValue("id"))
	if err != nil {
		writeError(w, http.StatusNotFound, "chat not found")
		return
	}
	upTo := len(c.Messages) - compressKeep
	if upTo <= 0 {
		writeError(w, http.StatusBadRequest, "nothing to summarize yet")
		return
	}
	if upTo <= compressedUpTo(c) {
		writeError(w, http.StatusBadRequest, "the earlier messages are already summarized")
		return
	}
	model, options := s.chatOptions(r.Context(), c)
	_, ctxLen, err := s.promptSize(r.Context(), model, options, nil)
	if err != nil {
		writeError(w, ollamaStatus(err), err.Error())
		return
	}
	if err := s.compressTo(r.Context(), c, model, options, upTo, ctxLen); err != nil {
		writeError(w, ollamaStatus(err), "could not summarize: "+err.Error())
		return
	}
	writeJSON(w, http.StatusOK, c.Compression)
}

// handleUncompress removes a chat's summary, so its full history is sent again:
// DELETE /api/chats/{id}/compression.
func (s *Server) handleUncompress(w http.ResponseWriter, r *http.Request) {
	c, err := s.chats.Get(r.PathValue("id"))
	if err != nil {
		writeError(w, http.StatusNotFound, "chat not found")
		return
	}
	if c.Compression != nil {
		c.Compression = nil
		if err := s.chats.Save(c); err != nil {
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
		s.forgetChat(c.ID)
	}
	w.WriteHeader(http.StatusNoContent)
}

// handleChatSettings reads and saves the chat preferences: GET and PUT
// /api/chats/settings.
func (s *Server) handleChatSettings(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodGet {
		writeJSON(w, http.StatusOK, s.chatSt.Get())
		return
	}
	var st chats.Settings
	if !readJSON(w, r, &st) {
		return
	}
	st.DefaultModel = s.chatSt.Get().DefaultModel // starred from the library or picker
	if err := s.chatSt.Set(st); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, st)
}

package server

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode"
	"unicode/utf8"

	"ollamui/internal/chats"
	"ollamui/internal/ollama"
	"ollamui/internal/presets"
)

// Voice mode. The browser records speech, ollamui has it transcribed and
// sends it as a chat message with voice set, then has each sentence of the
// reply spoken. The interesting part is the gap while a thinking model
// thinks: rather than sit silent, a second, non-thinking request on the same
// model (the "talker") says a short lead-in, and then, as the thinking
// grows, shares where it's heading. When the real answer starts the talker
// stops and the answer is spoken.

// The reply itself gets no extra instructions about being spoken: thinking
// models spend their thinking weighing any such rules (Qwen3.5-9B took 95 s
// instead of 8 s over "what is 2 plus 2" with a two-line style prompt). The
// browser strips markdown and skips code blocks before speaking instead.

// The talker's prompts are written for a small model: one short sentence,
// in a fixed shape, about what's being worked on rather than facts, so it
// can't contradict the real answer or give it away.
const talkerOpener = `You speak for a voice assistant while it thinks about the user's question. Reply with exactly one short spoken sentence, under 15 words, that acknowledges the question and says what you'll think about. Never answer the question and never state facts. No markdown. Examples: "Let me think about how light travels through the air." "Good question, let me work through the numbers." "Let me think about what changes between those two cases."`

const talkerInterim = `You speak for a voice assistant while it thinks about the user's question. From its private notes below, say in one short spoken sentence, under 15 words, what it is working on right now, in the first person. Never give the final answer, and don't repeat what was already said. No markdown. Examples: "I'm checking how the angle of the sun changes things." "Now I'm comparing the two options on cost." "I'm double-checking that count."`

const (
	// charsPerSecond is roughly how fast text-to-speech talks.
	charsPerSecond = 15
	// talkerLead is how long before the current speech runs out the next
	// talker line is started, to cover its generation and synthesis.
	talkerLead = 2500 * time.Millisecond
	// newNotes is how much new thinking is worth a new interim line.
	newNotes   = 400
	maxInterim = 12
	// lineGap is extra quiet added before each interim line after the
	// second, so a long think gets sparser updates instead of chatter.
	lineGap = 2 * time.Second
)

type voiceStatus struct {
	Available bool   `json:"available"`
	URL       string `json:"url,omitempty"`
	Error     string `json:"error,omitempty"`
	// Talker is the small model that talks while the chat model thinks
	// ("" if none is installed).
	Talker string `json:"talker"`
}

func (s *Server) handleVoiceStatus(w http.ResponseWriter, r *http.Request) {
	var st voiceStatus
	if m := r.URL.Query().Get("model"); m != "" {
		st.Talker = s.warmTalker(m)
	}
	if s.voice == nil {
		st.Error = "no speech service configured (start ollamui with -voice URL)"
	} else if err := s.voice.Available(r.Context()); err != nil {
		st.URL = s.voice.URL()
		st.Error = "speech service at " + s.voice.URL() + " isn't answering. Start it with: uv run tools/voice/voice_server.py"
	} else {
		st.Available, st.URL = true, s.voice.URL()
	}
	writeJSON(w, http.StatusOK, st)
}

// handleTranscribe turns a recording (the request body) into text.
func (s *Server) handleTranscribe(w http.ResponseWriter, r *http.Request) {
	if s.voice == nil {
		writeError(w, http.StatusServiceUnavailable, "no speech service configured")
		return
	}
	audio, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 25<<20))
	if err != nil || len(audio) == 0 {
		writeError(w, http.StatusBadRequest, "expected audio in the request body")
		return
	}
	name := "speech.wav"
	if ct := r.Header.Get("Content-Type"); strings.Contains(ct, "webm") {
		name = "speech.webm"
	} else if strings.Contains(ct, "ogg") {
		name = "speech.ogg"
	} else if strings.Contains(ct, "mp4") {
		name = "speech.m4a"
	}
	text, err := s.voice.Transcribe(r.Context(), audio, name, "")
	if err != nil {
		writeError(w, http.StatusBadGateway, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"text": text})
}

// handleSpeak returns text as audio.
func (s *Server) handleSpeak(w http.ResponseWriter, r *http.Request) {
	if s.voice == nil {
		writeError(w, http.StatusServiceUnavailable, "no speech service configured")
		return
	}
	var req struct {
		Text string `json:"text"`
	}
	if !readJSON(w, r, &req) {
		return
	}
	text := strings.TrimSpace(req.Text)
	if text == "" || len(text) > 4000 {
		writeError(w, http.StatusBadRequest, "text must be 1–4000 characters")
		return
	}
	body, ctype, err := s.voice.Speak(r.Context(), text)
	if err != nil {
		writeError(w, http.StatusBadGateway, err.Error())
		return
	}
	defer body.Close()
	w.Header().Set("Content-Type", ctype)
	w.Header().Set("Cache-Control", "no-store")
	io.Copy(w, body)
}

// ollamaParallel is how many requests Ollama serves at once per model.
// Unknown (a remote Ollama, no systemd) counts as 1.
func (s *Server) ollamaParallel(ctx context.Context) int {
	if n, err := strconv.Atoi(strings.TrimSpace(s.ollamaEnv(ctx).Vars["OLLAMA_NUM_PARALLEL"])); err == nil && n > 0 {
		return n
	}
	return 1
}

// voiceTurn streams a spoken reply. Besides the usual events it sends "say"
// events: talker lines to speak (and show) while the model is thinking.
//
// The talker is a small model run on the CPU (see pickTalker), so it works
// beside the main model without VRAM or GPU time, and without needing
// Ollama to serve two requests at once on one model, which hybrid models
// like Qwen3.5 can't. Without a small model, the main model itself says a
// lead-in before it starts thinking, and the rest of the thinking is quiet.
func (s *Server) voiceTurn(w http.ResponseWriter, r *http.Request, t *turn, talkerName string) {
	emit := streamEvents(w)
	if emit(event{Type: "chat", ID: t.chat.ID}) != nil {
		return
	}
	ctx := r.Context()
	var tk *talker
	if s.thinks(ctx, t.req) && talkerName != "off" {
		if name := s.pickTalker(ctx, t.req.Model, talkerName); name != "" {
			tk = s.newTalker(ctx, t, name, emit)
		} else {
			// One model, one request at a time: lead-in first. That delays
			// the reply by about as long as the lead-in takes to generate,
			// which its own speech covers.
			lead := &talker{s: s, t: t, req: ollama.ChatRequest{Model: t.req.Model, Options: t.req.Options, KeepAlive: t.req.KeepAlive, Think: false}, emit: emit}
			lead.line(ctx, "")
			t.spoken = strings.Join(lead.said, " ")
		}
	}

	var mu sync.Mutex
	var content, thinking strings.Builder
	notes := make(chan struct{}, 1)
	var stats *chats.Stats

	talkCtx, stopTalking := context.WithCancel(ctx)
	defer stopTalking()
	talkerDone := make(chan struct{})
	go func() {
		defer close(talkerDone)
		if tk != nil {
			tk.run(talkCtx, notes, func() string { mu.Lock(); defer mu.Unlock(); return thinking.String() })
		}
	}()

	err := s.ollama.Chat(ctx, t.req, func(resp ollama.ChatResponse) error {
		if txt := resp.Message.Thinking; txt != "" {
			mu.Lock()
			thinking.WriteString(txt)
			mu.Unlock()
			select {
			case notes <- struct{}{}:
			default:
			}
			if err := emit(event{Type: "thinking", Text: txt}); err != nil {
				return err
			}
		}
		if txt := resp.Message.Content; txt != "" {
			mu.Lock()
			first := content.Len() == 0
			content.WriteString(txt)
			mu.Unlock()
			if first {
				// The answer is here: no more talker lines, and wait for the
				// one in flight so its "say" can't land after the answer.
				stopTalking()
				<-talkerDone
			}
			if err := emit(event{Type: "content", Text: txt}); err != nil {
				return err
			}
		}
		if resp.Done {
			stats = t.stats(resp)
		}
		return nil
	})
	stopTalking()
	<-talkerDone
	spoken := t.spoken
	if tk != nil {
		spoken = strings.Join(tk.said, " ")
	}
	s.saveReply(t, content.String(), thinking.String(), spoken, stats)
	if err != nil {
		if ctx.Err() == nil {
			emit(event{Type: "error", Message: err.Error()})
		}
		return
	}
	emit(event{Type: "done", Stats: stats})
}

// thinks reports whether req will think.
func (s *Server) thinks(ctx context.Context, req ollama.ChatRequest) bool {
	show, err := s.ollama.Show(ctx, req.Model)
	if err != nil || !slices.Contains(show.Capabilities, "thinking") {
		return false
	}
	switch v := req.Think.(type) {
	case nil:
		return true // thinking models think unless told not to
	case bool:
		return v
	case string:
		return v != ""
	}
	return false
}

// maxTalkerSize is the biggest model pickTalker considers: small enough to
// write a sentence quickly on the CPU.
const maxTalkerSize = 3e9

// talkerOptions keep the talker on the CPU with a small context.
var talkerOptions = map[string]any{"num_gpu": 0, "num_ctx": 4096}

// pickTalker returns the model to talk while main thinks: want if it's
// installed, else the largest installed chat model under maxTalkerSize
// (other than main). "" means none.
func (s *Server) pickTalker(ctx context.Context, main, want string) string {
	models, err := s.ollama.List(ctx)
	if err != nil {
		return ""
	}
	best, bestSize := "", int64(0)
	for _, m := range models {
		if want != "" {
			if sameModel(m.Name, want) {
				return m.Name
			}
			continue
		}
		if m.Size > maxTalkerSize || m.Size <= bestSize || sameModel(m.Name, main) || strings.Contains(strings.ToLower(m.Details.Family), "bert") {
			continue
		}
		if show, err := s.ollama.Show(ctx, m.Name); err != nil || !slices.Contains(show.Capabilities, "completion") {
			continue
		}
		best, bestSize = m.Name, m.Size
	}
	return best
}

func (s *Server) newTalker(ctx context.Context, t *turn, name string, emit func(event) error) *talker {
	req := ollama.ChatRequest{Model: name, Options: talkerOptions, KeepAlive: "30m"}
	if show, err := s.ollama.Show(ctx, name); err == nil && slices.Contains(show.Capabilities, "thinking") {
		req.Think = false
	}
	return &talker{s: s, t: t, req: req, emit: emit}
}

// warmTalker loads the talker model ahead of the first voice turn, so its
// first line doesn't wait for a load.
func (s *Server) warmTalker(main string) string {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	name := s.pickTalker(ctx, main, "")
	if name == "" {
		cancel()
		return ""
	}
	go func() {
		defer cancel()
		if err := s.ollama.Load(ctx, name, "30m", talkerOptions); err != nil {
			s.log.Warn("load talker", "model", name, "err", err)
		}
	}()
	return name
}

// talker says short lines while the model thinks.
type talker struct {
	s    *Server
	t    *turn
	req  ollama.ChatRequest // model, options and think for talker requests
	emit func(event) error
	said []string
	// until is when the speech sent so far is expected to finish.
	until time.Time
}

// run speaks a lead-in right away, then an interim line whenever the speech
// is about to run out and there's enough new thinking to share.
func (tk *talker) run(ctx context.Context, notes <-chan struct{}, thinking func() string) {
	tk.line(ctx, "")
	tick := time.NewTicker(250 * time.Millisecond)
	defer tick.Stop()
	shared := 0 // how much of the thinking the last line could see
	drops := 0  // lines dropped in a row; each doubles the notes wanted next
	for {
		select {
		case <-ctx.Done():
			return
		case <-notes:
		case <-tick.C:
		}
		th := thinking()
		gap := time.Duration(max(0, len(tk.said)-2)) * lineGap
		if time.Until(tk.until) > talkerLead-gap || len(tk.said) > maxInterim || len(th)-shared < newNotes<<min(drops, 4) {
			continue
		}
		if tk.line(ctx, th) {
			drops = 0
		} else {
			drops++
		}
		shared = len(th)
	}
}

// line generates one talker line and sends it as a "say" event. notes empty
// means the lead-in. It reports whether a line was said.
func (tk *talker) line(ctx context.Context, notes string) bool {
	question := ""
	if n := len(tk.t.chat.Messages); n > 0 {
		question = tail(tk.t.chat.Messages[n-1].Content, 600)
	}
	// Everything goes in one user message: small models follow that far
	// better than a system prompt followed by the user's own question,
	// which they just answer.
	var ask strings.Builder
	prompt := talkerOpener
	fmt.Fprintf(&ask, "The user's question: %q", question)
	if notes != "" {
		prompt = talkerInterim
		if len(tk.said) > 0 {
			fmt.Fprintf(&ask, "\n\nAlready said: %s", strings.Join(tk.said, " "))
		}
		fmt.Fprintf(&ask, "\n\nNotes, latest at the end:\n%s", tail(notes, 1500))
	}
	ask.WriteString("\n\nYour one sentence:")
	req := tk.req
	req.Messages = []ollama.Message{{Role: "system", Content: prompt}, {Role: "user", Content: ask.String()}}
	req.Options = presets.Merge(req.Options, map[string]any{"num_predict": 48})
	var b strings.Builder
	err := tk.s.ollama.Chat(ctx, req, func(resp ollama.ChatResponse) error {
		b.WriteString(resp.Message.Content)
		return nil
	})
	text := firstSentence(cleanSpoken(b.String()))
	if !goodLine(text, question, tk.said) {
		tk.s.log.Debug("talker line dropped", "line", b.String())
		return false
	}
	if err != nil || ctx.Err() != nil {
		if err != nil && ctx.Err() == nil {
			tk.s.log.Warn("talker", "model", req.Model, "err", err)
		}
		return false
	}
	if tk.emit(event{Type: "say", Text: text}) != nil {
		return false
	}
	tk.said = append(tk.said, text)
	start := time.Now()
	if tk.until.After(start) {
		start = tk.until
	}
	tk.until = start.Add(time.Duration(len(text)*int(time.Second)/charsPerSecond) + 500*time.Millisecond)
	return true
}

// goodLine rejects talker lines that would do more harm than silence: cut
// off mid-sentence, a near-repeat of an earlier line, or quoting numbers
// the question didn't, which from a small model reading half-finished notes
// are usually wrong intermediate results.
func goodLine(line, question string, said []string) bool {
	if line == "" || !strings.ContainsAny(line[len(line)-1:], ".!?") {
		return false
	}
	if conclusionRe.MatchString(line) {
		return false
	}
	for _, n := range numberRe.FindAllString(line, -1) {
		if !strings.Contains(question, n) {
			return false
		}
	}
	words := wordSet(line)
	asked := wordSet(question)
	for w := range words {
		if numberWords[w] && !asked[w] {
			return false
		}
	}
	for _, prev := range said {
		common := 0
		for w := range wordSet(prev) {
			if words[w] {
				common++
			}
		}
		if common*10 >= len(words)*6 {
			return false
		}
	}
	return true
}

var (
	numberRe     = regexp.MustCompile(`\d+(?:[.,]\d+)*`)
	conclusionRe = regexp.MustCompile(`(?i)\b(i('ve| have) (found|got|determined|confirmed|concluded)|i found|the answer is|turns out|ready to tell)\b`)
	numberWords  = map[string]bool{}
)

func init() {
	// "one" is left out: "one thing to check" is too common to reject.
	for _, w := range strings.Fields("zero two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion half dozen") {
		numberWords[w] = true
	}
}

func wordSet(s string) map[string]bool {
	set := map[string]bool{}
	for _, w := range strings.FieldsFunc(strings.ToLower(s), func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsDigit(r) }) {
		set[w] = true
	}
	return set
}

// firstSentence keeps the first sentence of s, which is all a talker line
// should be.
func firstSentence(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	for i, r := range s {
		if (r == '.' || r == '!' || r == '?') && (i+1 == len(s) || s[i+1] == ' ') && i > 10 {
			return s[:i+1]
		}
	}
	return s
}

// cleanSpoken strips what a model sometimes adds around a spoken line.
func cleanSpoken(s string) string {
	s = strings.TrimSpace(s)
	s = strings.NewReplacer("**", "", "*", "", "#", "", "`", "").Replace(s)
	s = strings.Trim(s, `"“” `)
	return s
}

// tail keeps the last n bytes of s, starting at a word.
func tail(s string, n int) string {
	if len(s) <= n {
		return s
	}
	s = s[len(s)-n:]
	for len(s) > 0 && !utf8.RuneStart(s[0]) {
		s = s[1:]
	}
	if i := strings.IndexAny(s, " \n"); i >= 0 && i < 40 {
		s = s[i+1:]
	}
	return "…" + s
}

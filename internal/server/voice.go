package server

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"slices"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode"
	"unicode/utf8"

	"assay/internal/chats"
	"assay/internal/ollama"
	"assay/internal/presets"
	"assay/internal/voice"
)

// Voice mode. The browser records speech, Assay has it transcribed and
// sends it as a chat message with voice set, then has each sentence of the
// reply spoken. The interesting part is the gap while a thinking model
// thinks: rather than sit silent, a second, non-thinking request on the same
// model (the "talker") says a short lead-in, and then, as the thinking
// grows, shares where it's heading. When the real answer starts the talker
// stops and the answer is spoken.

// voiceSystem is added to the system prompt of a spoken reply. Measured
// on Gemma 4 12B it removes all markup and costs no extra time (answers
// get shorter). Keep it to this one line: a longer list of style rules made
// Qwen3.5-9B think for 95 s instead of 8 s over "what is 2 plus 2", weighing
// each rule. The browser still strips markdown before speaking.
const voiceSystem = `Your reply will be read aloud, so write plain spoken prose: no markdown, LaTeX, code, tables or lists.`

// The talker's prompts are written for a small model: one short sentence,
// in a fixed shape, about what's being worked on rather than facts, so it
// can't contradict the real answer or give it away.
//
// Its first call also decides whether the big model is needed at all: a
// greeting or small talk gets a REPLY from the talker, which becomes the
// answer, and the big model's request is cancelled.
const talkerOpener = `You speak for a voice assistant. Read the user's last message and answer in one of two ways.
If it is only a greeting, thanks, goodbye or small talk that needs no facts or reasoning, write REPLY: and then a short, friendly spoken reply of one or two sentences.
Otherwise a slower, careful part of you will answer it. Write WAIT: and then one short spoken sentence, under 15 words, that acknowledges it and says what you'll think about. Never answer it and never state facts.
No markdown.
Examples:
"hey there" -> REPLY: Hey! What can I do for you?
"thanks, that helped" -> REPLY: Glad it helped. Anything else?
"why is the sky blue?" -> WAIT: Let me think about how light travels through the air.
"what's 17 times 23?" -> WAIT: Good one, let me work through the numbers.`

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
	// Settings carries what the browser needs: pause and sensitivity.
	Settings voice.Settings `json:"settings"`
	// Starting: Assay is starting the service. Stopped: it isn't running
	// and will be started on demand. Managed: Assay started it.
	Starting bool `json:"starting,omitempty"`
	Stopped  bool `json:"stopped,omitempty"`
	Managed  bool `json:"managed,omitempty"`
	// NeedsDownload: the bundled service hasn't been downloaded yet, and
	// starting it fetches DownloadSize, so the browser asks first.
	NeedsDownload bool   `json:"needsDownload,omitempty"`
	DownloadSize  string `json:"downloadSize,omitempty"`
}

func (s *Server) handleVoiceStatus(w http.ResponseWriter, r *http.Request) {
	st := voiceStatus{Settings: s.voiceSt.Get()}
	if m := r.URL.Query().Get("model"); m != "" {
		st.Talker = s.warmTalker(m)
	}
	// install=1 is the go-ahead to download the bundled service.
	if r.URL.Query().Get("install") == "1" && s.speech != nil {
		s.speech.Ready(r.Context(), true)
	}
	s.fillSpeechStatus(r.Context(), &st, r.URL.Query().Get("start") == "1")
	writeJSON(w, http.StatusOK, st)
}

// errNotDownloaded keeps the bundled service from downloading itself
// until someone agrees to it.
var errNotDownloaded = errors.New("voice mode hasn't been downloaded yet")

// fillSpeechStatus says whether the speech service is up, starting it
// first if start is set and Assay manages it.
func (s *Server) fillSpeechStatus(ctx context.Context, st *voiceStatus, start bool) {
	if s.voice == nil {
		st.Error = "Voice mode is off (Assay was started with -voice \"\")."
		return
	}
	st.URL = s.voice.URL()
	err := s.speechReady(ctx, start, false)
	switch {
	case err == nil:
		st.Available = true
	case errors.Is(err, voice.ErrStarting):
		st.Starting = true
	case errors.Is(err, errNotDownloaded):
		st.Error = "Voice mode is optional. Pressing the mic the first time downloads its speech models (" + voice.DownloadSize + "), after asking."
		st.Stopped, st.NeedsDownload, st.DownloadSize = true, true, voice.DownloadSize
	case s.speech != nil && !start:
		st.Error = "Starts when you press the mic in chat, and stops after 10 minutes without voice use."
		st.Stopped = true
	case s.speech != nil:
		st.Error = err.Error()
	default:
		st.Error = "The speech service at " + s.voice.URL() + " isn't answering. Start it with: uv run tools/voice/voice_server.py"
	}
	st.Managed = s.speech != nil && s.speech.Managed()
}

// speechReady checks the speech service. With start, a service assay
// manages is started; with wait, it waits (up to two minutes, the first
// start installs packages) until it answers.
func (s *Server) speechReady(ctx context.Context, start, wait bool) error {
	if s.voice == nil {
		return errors.New("no speech service configured")
	}
	if s.speech == nil {
		return s.voice.Available(ctx)
	}
	if !s.speech.Installed() && !s.speech.Managed() {
		if s.voice.Available(ctx) == nil {
			return nil // a service someone started themselves
		}
		if err := s.speech.CanStart(); err != nil {
			return err
		}
		return errNotDownloaded
	}
	if wait {
		ctx, cancel := context.WithTimeout(ctx, 2*time.Minute)
		defer cancel()
		return s.speech.Wait(ctx)
	}
	return s.speech.Ready(ctx, start)
}

// handleTranscribe turns a recording (the request body) into text.
func (s *Server) handleTranscribe(w http.ResponseWriter, r *http.Request) {
	if s.voice == nil {
		writeError(w, http.StatusServiceUnavailable, "no speech service configured")
		return
	}
	if err := s.speechReady(r.Context(), true, true); err != nil {
		writeError(w, http.StatusServiceUnavailable, err.Error())
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
	text, err := s.voice.Transcribe(r.Context(), audio, name, "", s.voiceSt.Get().STTModel)
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
	if err := s.speechReady(r.Context(), true, true); err != nil {
		writeError(w, http.StatusServiceUnavailable, err.Error())
		return
	}
	// Voice and Speed override the saved settings, for previews.
	var req struct {
		Text  string  `json:"text"`
		Voice string  `json:"voice"`
		Speed float64 `json:"speed"`
	}
	if !readJSON(w, r, &req) {
		return
	}
	text := strings.TrimSpace(req.Text)
	if text == "" || len(text) > 4000 {
		writeError(w, http.StatusBadRequest, "text must be 1–4000 characters")
		return
	}
	st := s.voiceSt.Get()
	if req.Voice != "" {
		st.Voice = req.Voice
	}
	if req.Speed > 0 && req.Speed <= 2 {
		st.Speed = req.Speed
	}
	body, ctype, err := s.voice.Speak(r.Context(), voice.SpeechText(text), st.Voice, st.Speed)
	if err != nil {
		writeError(w, http.StatusBadGateway, err.Error())
		return
	}
	defer body.Close()
	w.Header().Set("Content-Type", ctype)
	w.Header().Set("Cache-Control", "no-store")
	io.Copy(w, body)
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
	if len(t.sources) > 0 && emit(event{Type: "sources", Sources: t.sources}) != nil {
		return
	}
	s.compressTurn(r.Context(), t, emit)
	ctx := r.Context()
	if talkerName == "" {
		talkerName = s.voiceSt.Get().Talker
	}
	// mainCtx is the big model's request; small talk cancels it.
	mainCtx, cancelMain := context.WithCancel(ctx)
	defer cancelMain()
	smallTalk := make(chan string, 1)
	var tk *talker
	if s.thinks(ctx, t.req) && talkerName != "off" {
		if name := s.pickTalker(ctx, t.req.Model, talkerName); name != "" {
			tk = s.newTalker(ctx, t, name, emit)
			tk.onReply = func(text string) {
				smallTalk <- text
				cancelMain()
			}
		} else {
			// One model, one request at a time: lead-in first. That delays
			// the reply by about as long as the lead-in takes to generate,
			// which its own speech covers.
			lead := &talker{s: s, t: t, req: ollama.ChatRequest{Model: t.req.Model, Options: t.req.Options, KeepAlive: t.req.KeepAlive, Think: false}, emit: emit}
			lead.onReply = func(text string) { smallTalk <- text }
			lead.line(ctx, "")
			t.spoken = strings.Join(lead.said, " ")
			if len(smallTalk) > 0 {
				cancelMain()
			}
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

	var err error
	if mainCtx.Err() == nil {
		err = s.chat(mainCtx, t.req, s.voiceChunk(emit, &mu, &content, &thinking, notes, &stats, t, stopTalking, talkerDone))
	}
	stopTalking()
	<-talkerDone
	select {
	case text := <-smallTalk:
		// The talker answered on its own; whatever the big model started
		// (thinking about "hi") is dropped.
		if emit(event{Type: "content", Text: text}) != nil {
			return
		}
		model := t.req.Model
		if tk != nil {
			model = tk.req.Model
		}
		t.replyModel = model
		s.saveReply(t, text, "", "", nil)
		emit(event{Type: "done"})
		return
	default:
	}
	if err == nil {
		s.noteThinking(t.req, thinking.Len() > 0)
	}
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

// voiceChunk handles one streamed chunk of the big model's spoken reply.
func (s *Server) voiceChunk(emit func(event) error, mu *sync.Mutex, content, thinking *strings.Builder, notes chan struct{}, stats **chats.Stats, t *turn, stopTalking func(), talkerDone chan struct{}) func(ollama.ChatResponse) error {
	return func(resp ollama.ChatResponse) error {
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
			*stats = t.stats(resp)
		}
		return nil
	}
}

// thinks reports whether req will probably think. Ollama's capability list
// can't be trusted for this: Gemma 4 thinks without listing "thinking". So
// unless thinking is switched off, a model counts as thinking until it's
// been seen replying without (see noteThinking). Guessing wrong costs a
// lead-in line before a quick answer; the other way it's a silent wait.
func (s *Server) thinks(ctx context.Context, req ollama.ChatRequest) bool {
	switch v := req.Think.(type) {
	case bool:
		return v
	case string:
		return v != ""
	}
	if show, err := s.show(ctx, req.Model); err == nil && slices.Contains(show.Capabilities, "thinking") {
		return true
	}
	if v, ok := s.thinkers.Load(strings.ToLower(req.Model)); ok {
		return v.(bool)
	}
	return true
}

// noteThinking remembers whether model thought in a reply where thinking
// was left to the model.
func (s *Server) noteThinking(req ollama.ChatRequest, thought bool) {
	if req.Think == nil {
		s.thinkers.Store(strings.ToLower(req.Model), thought)
	}
}

// maxTalkerSize is the biggest model pickTalker considers: small enough to
// write a sentence quickly on the CPU.
const maxTalkerSize = 3e9

// talkerOptions keep the talker on the CPU with a small context and a few
// threads, leaving cores for speech recognition and synthesis, which run
// at the same time.
var talkerOptions = map[string]any{"num_gpu": 0, "num_ctx": 4096, "num_thread": 4}

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
	want := s.voiceSt.Get().Talker
	name := ""
	if want != "off" {
		name = s.pickTalker(ctx, main, want)
	}
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
	// onReply, if set, takes over when the lead-in call decides the message
	// is small talk and answers it itself.
	onReply func(string)
	said    []string
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
	err := tk.s.chat(ctx, req, func(resp ollama.ChatResponse) error {
		b.WriteString(resp.Message.Content)
		return nil
	})
	raw := cleanSpoken(b.String())
	if notes == "" {
		if reply, ok := cutPrefixFold(raw, "REPLY:"); ok && tk.onReply != nil && err == nil && ctx.Err() == nil {
			if reply = strings.TrimSpace(reply); reply != "" {
				tk.onReply(reply)
				return true
			}
		}
		raw, _ = cutPrefixFold(raw, "WAIT:")
		raw = strings.TrimSpace(raw)
	}
	text := firstSentence(raw)
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

// cutPrefixFold is strings.CutPrefix ignoring case.
func cutPrefixFold(s, prefix string) (string, bool) {
	if len(s) >= len(prefix) && strings.EqualFold(s[:len(prefix)], prefix) {
		return s[len(prefix):], true
	}
	return s, false
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

func (s *Server) handleVoiceSettings(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, s.voiceSt.Get())
}

// handleVoiceRemove deletes the bundled speech service's downloads.
func (s *Server) handleVoiceRemove(w http.ResponseWriter, r *http.Request) {
	if s.speech == nil {
		writeError(w, http.StatusBadRequest, "Assay doesn't manage a speech service")
		return
	}
	if err := s.speech.Remove(); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (s *Server) handleVoiceSettingsSave(w http.ResponseWriter, r *http.Request) {
	var in voice.Settings
	if !readJSON(w, r, &in) {
		return
	}
	if err := s.voiceSt.Set(in); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, in)
}

// voicePage is the Voice section of the settings page.
type voicePage struct {
	Settings   voice.Settings
	Status     voiceStatus
	AutoTalker string // what "Auto" picks right now
	Talkers    []talkerChoice
	Voices     []voiceChoice
	STTModels  []sttChoice
	Speeds     []float64
	Pauses     []int
	// Downloaded is the size of the bundled service's downloads, if any.
	Downloaded int64
}

type talkerChoice struct {
	Name  string
	Size  int64
	Small bool // small enough to run on the CPU comfortably
}

type voiceChoice struct{ ID, Label string }

type sttChoice struct{ Name, Note string }

func (s *Server) voicePage(ctx context.Context) voicePage {
	v := voicePage{
		Settings: s.voiceSt.Get(),
		Speeds:   []float64{0.8, 0.9, 1, 1.1, 1.2, 1.35, 1.5},
		Pauses:   []int{200, 300, 450, 600, 800, 1000},
	}
	for _, m := range voice.STTModels {
		v.STTModels = append(v.STTModels, sttChoice{m.Name, m.Note})
	}
	s.fillSpeechStatus(ctx, &v.Status, false)
	if s.speech != nil {
		v.Downloaded = s.speech.Downloads()
	}
	ids := kokoroVoices
	if v.Status.Available {
		if list, err := s.voice.Voices(ctx); err == nil && len(list) > 0 {
			ids = list
		}
	}
	for _, id := range ids {
		if label, ok := voiceLabel(id); ok {
			v.Voices = append(v.Voices, voiceChoice{id, label})
		}
	}
	if models, err := s.ollama.List(ctx); err == nil {
		sort.Slice(models, func(i, j int) bool { return models[i].Size < models[j].Size })
		for _, m := range models {
			if strings.Contains(strings.ToLower(m.Details.Family), "bert") {
				continue // embedding models can't talk
			}
			v.Talkers = append(v.Talkers, talkerChoice{m.Name, m.Size, m.Size <= maxTalkerSize})
		}
	}
	v.AutoTalker = s.pickTalker(ctx, "", "")
	return v
}

// kokoroVoices are Kokoro v1.0's English voices, for when the speech
// service can't list its own.
var kokoroVoices = strings.Fields("af_heart af_alloy af_aoede af_bella af_jessica af_kore af_nicole af_nova af_river af_sarah af_sky am_adam am_echo am_eric am_fenrir am_liam am_michael am_onyx am_puck am_santa bf_alice bf_emma bf_isabella bf_lily bm_daniel bm_fable bm_george bm_lewis")

// voiceLabel turns a Kokoro voice id like "bf_emma" into "Emma · British,
// female". Only English voices are offered: the service speaks English.
func voiceLabel(id string) (string, bool) {
	accent, name, ok := strings.Cut(id, "_")
	if !ok || len(accent) != 2 || name == "" {
		return "", false
	}
	var where string
	switch accent[0] {
	case 'a':
		where = "American"
	case 'b':
		where = "British"
	default:
		return "", false
	}
	who := "female"
	if accent[1] == 'm' {
		who = "male"
	}
	return strings.ToUpper(name[:1]) + name[1:] + " · " + where + ", " + who, true
}

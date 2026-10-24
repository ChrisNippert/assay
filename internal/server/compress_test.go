package server

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http/httptest"
	"strings"
	"testing"

	"ollamui/internal/chats"
	"ollamui/internal/ollama"
)

// fakeLlama stands in for the engine: its tokenizer always counts tokens, and
// a chat answers with summary (or fails), recording each request.
type fakeLlama struct {
	backend // only the methods below are used
	tokens  int
	ctxLen  int
	summary string
	fail    error
	reqs    []ollama.ChatRequest
}

func (f *fakeLlama) Tokenize(ctx context.Context, model string, options map[string]any, text string) (int, int, error) {
	return f.tokens, f.ctxLen, nil
}

func (f *fakeLlama) Chat(ctx context.Context, req ollama.ChatRequest, fn func(ollama.ChatResponse) error) error {
	f.reqs = append(f.reqs, req)
	if f.fail != nil {
		return f.fail
	}
	return fn(ollama.ChatResponse{Model: req.Model, Message: ollama.Message{Role: "assistant", Content: f.summary}, Done: true})
}

// longChat is n alternating messages, starting with the user, each about 240
// tokens by approxTokens. n should be odd so the last one is the user's.
func longChat(n int) *chats.Chat {
	c := chats.New("m")
	for i := range n {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		c.Messages = append(c.Messages, chats.Message{Role: role, Content: strings.Repeat(fmt.Sprintf("message %d ", i), 60)})
	}
	return c
}

func TestNeedsCompression(t *testing.T) {
	for _, c := range []struct {
		prompt, ctx, predict int
		want                 bool
	}{
		{3000, 4096, -1, false}, // under 80%, and 1k is left for the reply
		{3100, 4096, -1, true},  // past 80% of 4096 (3276) and past 4096-1024
		{2100, 4096, 8192, true},
		{2000, 4096, 8192, false},
		{3100, 0, -1, false}, // no context length known
	} {
		if got := needsCompression(c.prompt, c.ctx, c.predict); got != c.want {
			t.Errorf("needsCompression(%d, %d, %d) = %v, want %v", c.prompt, c.ctx, c.predict, got, c.want)
		}
	}
}

func TestCutPoint(t *testing.T) {
	c := longChat(11)
	// Each message is about 200 tokens, so the nine latest (about 1800) are the
	// most that fit in half of 4096.
	cut := cutPoint(c.Messages, 4096)
	if cut != 2 || c.Messages[cut].Role != "user" || len(c.Messages)-cut < compressKeep {
		t.Errorf("cut = %d (%s), want 2 at a user message keeping at least %d", cut, c.Messages[cut].Role, compressKeep)
	}
	// Half of 2048 is about four messages: cut at 6 (user) keeps 5, and 4 is too few.
	if got := cutPoint(c.Messages, 2048); got != 6 {
		t.Errorf("cut at 2048 = %d, want 6", got)
	}
	// Too short to keep 4 and still summarize anything.
	if got := cutPoint(longChat(5).Messages[:4], 4096); got != 0 {
		t.Errorf("short chat cut at %d", got)
	}
	// The latest messages alone are over the share: cut before the last user message it can.
	if got := cutPoint(c.Messages, 512); got != 6 {
		t.Errorf("tiny context cut at %d, want 6", got)
	}
}

func TestBuildMessages(t *testing.T) {
	c := longChat(11)
	c.Messages[3].Thinking = "private reasoning"

	plain := buildMessages("Be brief.", nil, c.Messages)
	if len(plain) != 12 || plain[0].Role != "system" || plain[0].Content != "Be brief." {
		t.Fatalf("no summary: %d messages, first %+v", len(plain), plain[0])
	}
	for _, m := range plain {
		if strings.Contains(m.Content, "private reasoning") {
			t.Error("thinking was sent")
		}
	}

	comp := &chats.Compression{Text: "The user's dog is Biscuit.", UpTo: 4}
	msgs := buildMessages("Be brief.", comp, c.Messages)
	if len(msgs) != 1+len(c.Messages)-4 {
		t.Fatalf("with summary: %d messages", len(msgs))
	}
	if msgs[0].Role != "system" || !strings.HasPrefix(msgs[0].Content, "Be brief.") || !strings.Contains(msgs[0].Content, "Biscuit") {
		t.Errorf("system = %q", msgs[0].Content)
	}
	if msgs[1].Content != c.Messages[4].Content {
		t.Error("the messages after the summary don't start at UpTo")
	}

	// No system prompt: the summary is the system message.
	if got := buildMessages("", comp, c.Messages)[0]; got.Role != "system" || !strings.Contains(got.Content, "Biscuit") {
		t.Errorf("summary without a system prompt: %+v", got)
	}
}

func TestDropCompressionFrom(t *testing.T) {
	c := longChat(11)
	c.Compression = &chats.Compression{Text: "s", UpTo: 4}
	if c.DropCompressionFrom(4) || c.Compression == nil {
		t.Error("an edit from the first unsummarized message dropped the summary")
	}
	if c.DropCompressionFrom(9) || c.Compression == nil {
		t.Error("an edit later on dropped the summary")
	}
	if !c.DropCompressionFrom(2) || c.Compression != nil {
		t.Error("an edit inside the summarized messages kept the summary")
	}
	if i := c.UserIndex(2); i != 4 {
		t.Errorf("UserIndex(2) = %d, want 4", i)
	}
	if i := c.UserIndex(99); i != -1 {
		t.Errorf("UserIndex(99) = %d, want -1", i)
	}
}

func TestChunkMessages(t *testing.T) {
	c := longChat(7)
	chunks := chunkMessages(c.Messages, 500) // two messages of about 240 each per run
	total := 0
	for _, ch := range chunks {
		total += len(ch)
		if len(ch) > 2 {
			t.Errorf("a run of %d messages is over the budget", len(ch))
		}
	}
	if total != len(c.Messages) || len(chunks) != 4 {
		t.Errorf("chunks = %d runs over %d messages", len(chunks), total)
	}
	if got := transcript([]chats.Message{{Role: "user", Content: strings.Repeat("x", 100)}, {Role: "system", Content: "no"}}, 10); got != "User: xxxxxxxxxx …\n\n" {
		t.Errorf("transcript = %q", got)
	}
}

func newCompressServer(t *testing.T, fb *fakeLlama) *Server {
	t.Helper()
	st, err := chats.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	cs, _ := chats.OpenSettings("")
	return &Server{ollama: fb, chats: st, chatSt: cs, log: slog.New(slog.NewTextHandler(io.Discard, nil))}
}

func testTurn(c *chats.Chat, system string) *turn {
	return &turn{
		chat:     c,
		system:   system,
		compress: true,
		req: ollama.ChatRequest{
			Model:    "m",
			Messages: buildMessages(system, c.Compression, c.Messages),
			Options:  map[string]any{"num_ctx": 4096},
		},
	}
}

func collect(events *[]event) func(event) error {
	return func(e event) error {
		*events = append(*events, e)
		return nil
	}
}

func TestCompressTurnSummarizesOnce(t *testing.T) {
	fb := &fakeLlama{tokens: 3500, ctxLen: 4096, summary: "The user's dog is called Biscuit."}
	s := newCompressServer(t, fb)
	c := longChat(11)
	if err := s.chats.Save(c); err != nil {
		t.Fatal(err)
	}
	tn := testTurn(c, "Be brief.")
	var events []event
	s.compressTurn(context.Background(), tn, collect(&events))

	if len(fb.reqs) != 1 {
		t.Fatalf("%d summary requests, want 1", len(fb.reqs))
	}
	req := fb.reqs[0]
	if req.Think != false || req.Options["temperature"] != 0.2 || req.Options["num_predict"] != summaryTokens {
		t.Errorf("summary request: think %v, options %v", req.Think, req.Options)
	}
	if len(req.Messages) != 1 || req.Messages[0].Role != "user" {
		t.Errorf("summary request should be one user message: %+v", req.Messages)
	}
	if c.Compression == nil || c.Compression.UpTo != 2 || c.Compression.Text != fb.summary || c.Compression.Model != "m" {
		t.Fatalf("compression = %+v", c.Compression)
	}
	if c.Compression.Saved <= 0 {
		t.Errorf("saved = %d, want a positive estimate", c.Compression.Saved)
	}
	if msgs := tn.req.Messages; msgs[0].Role != "system" || !strings.Contains(msgs[0].Content, "Biscuit") || !strings.HasPrefix(msgs[0].Content, "Be brief.") {
		t.Errorf("sent system = %q", msgs[0].Content)
	}
	if len(tn.req.Messages) != 1+len(c.Messages)-2 {
		t.Errorf("sent %d messages", len(tn.req.Messages))
	}
	if len(events) != 2 || events[0].Type != "status" || events[1].Type != "compressed" || events[1].Compression.UpTo != 2 {
		t.Errorf("events = %+v", events)
	}
	saved, err := s.chats.Get(c.ID)
	if err != nil || saved.Compression == nil || saved.Compression.UpTo != 2 {
		t.Errorf("summary not saved with the chat: %+v, %v", saved, err)
	}

	// A later turn under the limit sends the same summary without asking again.
	fb.tokens = 100
	tn2 := testTurn(c, "Be brief.")
	events = nil
	s.compressTurn(context.Background(), tn2, collect(&events))
	if len(fb.reqs) != 1 || len(events) != 0 {
		t.Errorf("second turn: %d requests, events %+v", len(fb.reqs), events)
	}
	if !strings.Contains(tn2.req.Messages[0].Content, "Biscuit") {
		t.Error("the summary wasn't sent again")
	}
}

func TestCompressTurnExtendsSummary(t *testing.T) {
	fb := &fakeLlama{tokens: 3500, ctxLen: 4096, summary: "Old summary: Biscuit."}
	s := newCompressServer(t, fb)
	c := longChat(11)
	c.Compression = &chats.Compression{Text: "Old summary: Biscuit.", UpTo: 4}
	for i := range 8 {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		c.Messages = append(c.Messages, chats.Message{Role: role, Content: strings.Repeat("more ", 150)})
	}
	fb.summary = "Updated summary: Biscuit and a new question."
	var events []event
	s.compressTurn(context.Background(), testTurn(c, ""), collect(&events))
	// The new messages may take more than one run, so the summary is folded in
	// run by run; the first run carries the old summary.
	if len(fb.reqs) < 1 {
		t.Fatal("no summary request")
	}
	prompt := fb.reqs[0].Messages[0].Content
	if !strings.Contains(prompt, "Summary of what came before") || !strings.Contains(prompt, "Old summary: Biscuit.") {
		t.Error("the old summary wasn't folded in")
	}
	if c.Compression.UpTo <= 4 || c.Compression.Text != fb.summary {
		t.Errorf("summary not extended: %+v", c.Compression)
	}
}

func TestCompressTurnOffOrFailing(t *testing.T) {
	fb := &fakeLlama{tokens: 3500, ctxLen: 4096, summary: "unused"}
	s := newCompressServer(t, fb)
	c := longChat(11)
	tn := testTurn(c, "")
	tn.compress = false
	s.compressTurn(context.Background(), tn, collect(new([]event)))
	if len(fb.reqs) != 0 || c.Compression != nil {
		t.Fatal("compressed with compression off")
	}

	fb.fail = errors.New("boom")
	var events []event
	tn = testTurn(c, "")
	s.compressTurn(context.Background(), tn, collect(&events))
	if c.Compression != nil || len(tn.req.Messages) != len(c.Messages) {
		// no system prompt, so the messages are the whole history as sent
		t.Errorf("after a failed summary: compression %+v, %d messages", c.Compression, len(tn.req.Messages))
	}
	if len(events) != 2 || events[len(events)-1].Type != "status" || events[len(events)-1].Text != "" {
		t.Errorf("a failed summary should clear the status: %+v", events)
	}
}

func TestCompressTurnUncompressed(t *testing.T) {
	fb := &fakeLlama{tokens: 100, ctxLen: 4096}
	s := newCompressServer(t, fb)
	c := longChat(3)
	tn := testTurn(c, "")
	tn.dropped = true
	var events []event
	s.compressTurn(context.Background(), tn, collect(&events))
	if len(events) != 1 || events[0].Type != "uncompressed" {
		t.Errorf("events = %+v", events)
	}
}

func TestSummarizeChunks(t *testing.T) {
	fb := &fakeLlama{summary: "a summary"}
	s := newCompressServer(t, fb)
	c := longChat(7)
	text, err := s.summarize(context.Background(), "m", map[string]any{"reply_format": "json", "temperature": 0.9}, "", c.Messages[:6], 1000)
	if err != nil || text != "a summary" {
		t.Fatalf("summarize = %q, %v", text, err)
	}
	if len(fb.reqs) != 3 {
		t.Fatalf("%d requests for 6 messages in runs of two, want 3", len(fb.reqs))
	}
	if !strings.Contains(fb.reqs[1].Messages[0].Content, "a summary") {
		t.Error("a later run doesn't carry the summary so far")
	}
	if _, set := fb.reqs[0].Options["reply_format"]; set || fb.reqs[0].Options["temperature"] != 0.2 {
		t.Errorf("options = %v", fb.reqs[0].Options)
	}
}

func TestPromptSizeEstimates(t *testing.T) {
	s := newCompressServer(t, &fakeLlama{})
	s.ollama = nil // a backend that can't tokenize
	n, ctxLen, err := s.promptSize(context.Background(), "m", map[string]any{"num_ctx": 2048}, []ollama.Message{{Role: "user", Content: strings.Repeat("a", 300)}})
	if err != nil || ctxLen != 2048 || n < 100 || n > 200 {
		t.Errorf("estimate = %d tokens, context %d, %v", n, ctxLen, err)
	}
}

func TestManualCompressAndUndo(t *testing.T) {
	fb := &fakeLlama{tokens: 3500, ctxLen: 4096, summary: "Summary by hand."}
	s := newCompressServer(t, fb)
	c := longChat(11)
	if err := s.chats.Save(c); err != nil {
		t.Fatal(err)
	}

	w := httptest.NewRecorder()
	r := httptest.NewRequest("POST", "/api/chats/"+c.ID+"/compress", nil)
	r.SetPathValue("id", c.ID)
	s.handleCompress(w, r)
	if w.Code != 200 {
		t.Fatalf("compress: %d %s", w.Code, w.Body)
	}
	saved, _ := s.chats.Get(c.ID)
	if saved.Compression == nil || saved.Compression.UpTo != len(c.Messages)-compressKeep || saved.Compression.Text != "Summary by hand." {
		t.Fatalf("saved compression = %+v", saved.Compression)
	}

	// Nothing new to summarize the second time.
	w = httptest.NewRecorder()
	s.handleCompress(w, r)
	if w.Code != 400 {
		t.Errorf("second compress: %d, want 400", w.Code)
	}

	w = httptest.NewRecorder()
	r = httptest.NewRequest("DELETE", "/api/chats/"+c.ID+"/compression", nil)
	r.SetPathValue("id", c.ID)
	s.handleUncompress(w, r)
	if w.Code != 204 {
		t.Fatalf("undo: %d", w.Code)
	}
	saved, _ = s.chats.Get(c.ID)
	if saved.Compression != nil {
		t.Error("undo kept the summary")
	}
}

func TestManualCompressTooShort(t *testing.T) {
	s := newCompressServer(t, &fakeLlama{})
	c := longChat(3)
	if err := s.chats.Save(c); err != nil {
		t.Fatal(err)
	}
	w := httptest.NewRecorder()
	r := httptest.NewRequest("POST", "/api/chats/"+c.ID+"/compress", nil)
	r.SetPathValue("id", c.ID)
	s.handleCompress(w, r)
	if w.Code != 400 {
		t.Errorf("compress of 3 messages: %d, want 400", w.Code)
	}
}

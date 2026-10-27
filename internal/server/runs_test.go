package server

import (
	"bufio"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

const gatedReply = `{"message":{"role":"assistant","content":"one "}}
{"message":{"role":"assistant","content":"two"}}
{"message":{"role":"assistant","content":""},"done":true}
`

// startGated starts a reply that waits halfway, over a real HTTP server, and
// returns the chat id once the first event has arrived.
func startGated(t *testing.T, s *Server, fake *fakeOllama) (ts *httptest.Server, chatID string, cancel context.CancelFunc, body *bufio.Scanner) {
	t.Helper()
	fake.gate = make(chan struct{})
	fake.reply = gatedReply
	ts = httptest.NewServer(s)
	t.Cleanup(ts.Close)
	ctx, cancel := context.WithCancel(context.Background())
	req, _ := http.NewRequestWithContext(ctx, "POST", ts.URL+"/api/chat", strings.NewReader(`{"model":"qwen3:8b","content":"hi"}`))
	req.Header.Set("Content-Type", "application/json")
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	if res.StatusCode != 200 {
		t.Fatalf("status %d", res.StatusCode)
	}
	body = bufio.NewScanner(res.Body)
	if !body.Scan() {
		t.Fatal("no first event")
	}
	var e event
	json.Unmarshal(body.Bytes(), &e)
	if e.Type != "chat" {
		t.Fatalf("first event %+v", e)
	}
	return ts, e.ID, cancel, body
}

func waitFor(t *testing.T, what string, ok func() bool) {
	t.Helper()
	for i := 0; i < 300; i++ {
		if ok() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

func readEvents(t *testing.T, sc *bufio.Scanner) []event {
	t.Helper()
	var out []event
	for sc.Scan() {
		var e event
		if err := json.Unmarshal(sc.Bytes(), &e); err != nil {
			t.Fatal(err)
		}
		out = append(out, e)
	}
	return out
}

func TestRunSurvivesDisconnectAndReplays(t *testing.T) {
	s, fake, store := newTestServer(t)
	ts, id, cancel, _ := startGated(t, s, fake)
	cancel() // the browser leaves
	time.Sleep(50 * time.Millisecond)
	if s.getRun(id) == nil {
		t.Fatal("run ended when the client left")
	}
	if rec := do(s, "GET", "/api/chats/running", ""); !strings.Contains(rec.Body.String(), id) {
		t.Errorf("running list = %s", rec.Body)
	}
	// A second send while it runs is refused.
	rec := do(s, "POST", "/api/chat", `{"chatId":"`+id+`","model":"qwen3:8b","content":"again"}`)
	if rec.Code != http.StatusConflict {
		t.Errorf("second send: status %d, want 409", rec.Code)
	}
	// Coming back replays what was sent, then follows to the end.
	res, err := http.Get(ts.URL + "/api/chats/" + id + "/stream")
	if err != nil {
		t.Fatal(err)
	}
	sc := bufio.NewScanner(res.Body)
	if !sc.Scan() {
		t.Fatal("no replayed event")
	}
	close(fake.gate)
	evs := readEvents(t, sc)
	text := ""
	for _, e := range evs {
		text += e.Text
	}
	last := evs[len(evs)-1]
	if !strings.Contains(text, "two") || last.Type != "done" {
		t.Errorf("rest of stream = %+v", evs)
	}
	waitFor(t, "run to end", func() bool { return s.getRun(id) == nil })
	c, _ := store.Get(id)
	if len(c.Messages) != 2 || c.Messages[1].Content != "one two" || c.Messages[1].Stopped {
		t.Fatalf("saved = %+v", c.Messages)
	}
	if res, _ := http.Get(ts.URL + "/api/chats/" + id + "/stream"); res.StatusCode != 204 {
		t.Errorf("stream after done: %d, want 204", res.StatusCode)
	}
}

func TestStopEndpointSavesStopped(t *testing.T) {
	s, fake, store := newTestServer(t)
	ts, id, _, body := startGated(t, s, fake)
	res, err := http.Post(ts.URL+"/api/chats/"+id+"/stop", "", nil)
	if err != nil || res.StatusCode != 204 {
		t.Fatalf("stop: %v %v", err, res)
	}
	evs := readEvents(t, body)
	if last := evs[len(evs)-1]; last.Type != "stopped" {
		t.Errorf("last event = %+v", last)
	}
	waitFor(t, "run to end", func() bool { return s.getRun(id) == nil })
	c, _ := store.Get(id)
	if len(c.Messages) != 2 || !c.Messages[1].Stopped || c.Messages[1].Content != "one " {
		t.Fatalf("saved = %+v", c.Messages)
	}
	if rec := do(s, "POST", "/api/chats/"+id+"/stop", ""); rec.Code != 404 {
		t.Errorf("stop with nothing running: %d, want 404", rec.Code)
	}
	if rec := do(s, "GET", "/api/chats/"+id+"/stream", ""); rec.Code != 204 {
		t.Errorf("stream: %d", rec.Code)
	}
}

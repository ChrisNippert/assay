package server

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"assay/internal/evals"
	"assay/internal/remote"
)

// fakeAPI is an OpenAI-style endpoint that answers 391 to everything,
// thinking first, and records what it was sent.
type fakeAPI struct {
	mu   sync.Mutex
	reqs []map[string]any
	auth []string
	// refuse makes it answer 400 to requests with a temperature.
	refuse bool
}

func (f *fakeAPI) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/v1/models" {
		fmt.Fprint(w, `{"data":[{"id":"z/b","name":"B","context_length":8192,"pricing":{"prompt":"0.000001","completion":"0.000004"}},{"id":"a/a"}]}`)
		return
	}
	var body map[string]any
	json.NewDecoder(r.Body).Decode(&body)
	f.mu.Lock()
	f.reqs = append(f.reqs, body)
	f.auth = append(f.auth, r.Header.Get("Authorization"))
	refuse := f.refuse
	f.mu.Unlock()
	if refuse && body["temperature"] != nil {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, `{"error":{"message":"temperature is not supported"}}`)
		return
	}
	for _, line := range []string{
		`{"choices":[{"delta":{"reasoning":"17*23 = 391"}}]}`,
		`{"choices":[{"delta":{"content":"Answer: "}}]}`,
		`{"choices":[{"delta":{"content":"391"},"finish_reason":"stop"}]}`,
		`{"choices":[],"usage":{"completion_tokens":9,"cost":0.0025}}`,
		`[DONE]`,
	} {
		fmt.Fprintf(w, "data: %s\n\n", line)
	}
}

func TestAPIModelEval(t *testing.T) {
	s, fake, _ := newTestServer(t)
	fake.tags = threeModels
	lib, _ := evals.OpenLibrary(t.TempDir())
	s.evalLib, s.evalRes = lib, evals.OpenResults(t.TempDir())
	if _, err := lib.Save("two", []byte(twoQuestions)); err != nil {
		t.Fatal(err)
	}
	api := &fakeAPI{}
	ts := httptest.NewServer(api)
	defer ts.Close()

	// Set up an endpoint and two ways of running one model.
	if rec := do(s, "PUT", "/api/remote/endpoints/new", `{"name":"Fake","url":"`+ts.URL+`/v1/","key":"sk-secret-123456"}`); rec.Code != 200 {
		t.Fatalf("endpoint: %d %s", rec.Code, rec.Body)
	}
	for _, body := range []string{
		`{"endpoint":"fake","model":"z/b","title":"B"}`,
		`{"endpoint":"fake","model":"z/b","title":"B terse","system":"Be terse.","reasoning":"low"}`,
	} {
		if rec := do(s, "PUT", "/api/remote/models/new", body); rec.Code != 200 {
			t.Fatalf("model: %d %s", rec.Code, rec.Body)
		}
	}
	rec := do(s, "GET", "/api/remote", "")
	if strings.Contains(rec.Body.String(), "sk-secret") || !strings.Contains(rec.Body.String(), `"keyHint":"…3456"`) {
		t.Errorf("the key leaked or has no hint: %s", rec.Body)
	}
	if rec := do(s, "GET", "/api/remote/endpoints/fake/models", ""); !strings.Contains(rec.Body.String(), `"id":"a/a"`) || !strings.Contains(rec.Body.String(), `"out":4`) {
		t.Errorf("model list: %s", rec.Body)
	}

	if _, err := s.startEvals(context.Background(), evalRunRequest{Models: []string{"api:b", "api:b-terse"}, Sets: []string{"two"}}); err != nil {
		t.Fatal(err)
	}
	st := waitEvals(t, s)
	for _, it := range st.Items {
		if it.State != itemDone || it.Answered != 2 || it.Correct != 1 || it.Title == "" {
			t.Fatalf("item %+v", it)
		}
	}
	api.mu.Lock()
	if len(api.reqs) != 4 || api.auth[0] != "Bearer sk-secret-123456" {
		t.Fatalf("requests %d, auth %q", len(api.reqs), api.auth)
	}
	var terse map[string]any
	for _, r := range api.reqs {
		msgs := r["messages"].([]any)
		if first := msgs[0].(map[string]any); first["role"] == "system" && strings.HasPrefix(first["content"].(string), "Be terse.") {
			terse = r
		}
		if r["temperature"] != float64(1) || r["model"] != "z/b" || r["stream"] != true {
			t.Errorf("request %v", r)
		}
	}
	api.mu.Unlock()
	if terse == nil || terse["reasoning_effort"] != "low" {
		t.Errorf("own system prompt or reasoning not sent: %v", terse)
	}

	// Each way of running it has its own row, with tokens and cost.
	res, err := s.evalResultsFor(context.Background(), "", []string{"two"})
	if err != nil {
		t.Fatal(err)
	}
	rows := map[string]evalRow{}
	for _, row := range res.Rows {
		rows[row.Name] = row
	}
	b, bt := rows["api:b"], rows["api:b-terse"]
	if !b.API || b.Digest == bt.Digest || bt.System != "Be terse." || !strings.Contains(bt.Quant, "own prompt") {
		t.Fatalf("rows %+v / %+v", b, bt)
	}
	if c := b.Cells["two"]; !c.Done || c.Score != 0.5 || c.Tokens != 18 || c.Cost != 0.005 {
		t.Errorf("cell %+v", c)
	}
	if sp := s.spend.Summary(time.Now()).Models["api:b"]; sp.Today.Calls != 2 || math.Abs(sp.Today.Cost-0.005) > 1e-9 || sp.Today.Tokens != 18 {
		t.Errorf("spending not recorded: %+v", sp)
	}
	r, _ := s.evalRes.Get(b.Digest, "two", "")
	if r.Answers[0].Thinking != "17*23 = 391" {
		t.Errorf("thinking %q", r.Answers[0].Thinking)
	}

	// Editing the prompt is a new model as far as results go.
	do(s, "PUT", "/api/remote/models/b-terse", `{"endpoint":"fake","model":"z/b","title":"B terse","system":"Be very terse."}`)
	res, _ = s.evalResultsFor(context.Background(), "", []string{"two"})
	for _, row := range res.Rows {
		if row.Name == "api:b-terse" && len(row.Cells) != 0 {
			t.Errorf("edited prompt kept the old results: %+v", row)
		}
	}
}

func TestAPIRetriesWithoutSampling(t *testing.T) {
	api := &fakeAPI{refuse: true}
	ts := httptest.NewServer(api)
	defer ts.Close()
	ep := remote.Endpoint{ID: "x", Name: "X", URL: ts.URL + "/v1"}
	a, err := (&Server{}).askAPI(context.Background(), ep, remote.Model{ID: "m", Model: "o"}, "", "17×23?", 100)
	if err != nil || a.Reply != "Answer: 391" || a.Tokens != 9 {
		t.Fatalf("%+v %v", a, err)
	}
	api.mu.Lock()
	defer api.mu.Unlock()
	if last := api.reqs[len(api.reqs)-1]; last["temperature"] != nil || last["max_completion_tokens"] != float64(100) {
		t.Errorf("retry %v", last)
	}
}

func TestRemoteKeyStaysWithURL(t *testing.T) {
	st, _ := remote.Open("")
	st.PutEndpoint(remote.Endpoint{ID: "o", Name: "O", URL: "https://openrouter.ai/api/v1", Key: "sk-1"})
	st.PutEndpoint(remote.Endpoint{ID: "o", Name: "Renamed", URL: "https://openrouter.ai/api/v1"})
	if e, _ := st.Endpoint("o"); e.Key != "sk-1" {
		t.Errorf("rename lost the key: %+v", e)
	}
	st.PutEndpoint(remote.Endpoint{ID: "o", Name: "O", URL: "https://evil.example/v1"})
	if e, _ := st.Endpoint("o"); e.Key != "" {
		t.Errorf("the key followed a new URL: %+v", e)
	}
}

// teamAPI plays every team role: it plans, helps and answers depending on
// the prompt it gets, and records each call's last message.
type teamAPI struct {
	mu    sync.Mutex
	calls []string
}

func (f *teamAPI) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Messages []remote.Message `json:"messages"`
	}
	json.NewDecoder(r.Body).Decode(&body)
	sys, last := "", body.Messages[len(body.Messages)-1].Content
	if body.Messages[0].Role == "system" {
		sys = body.Messages[0].Content
	}
	f.mu.Lock()
	f.calls = append(f.calls, last)
	f.mu.Unlock()
	var reply string
	switch {
	case strings.Contains(sys, "JSON array"):
		reply = `Here: ["multiply 17 by 20", "multiply 17 by 3", "unused third part"]`
	case strings.Contains(sys, "one of several helpers"):
		reply = "part done: " + last[strings.LastIndex(last, "\n")+1:]
	case strings.Contains(sys, "<task>") && !strings.Contains(last, "Results of your tasks"):
		reply = "Let me delegate.\n<task>compute 17*20</task>\n<task>compute 17*3</task>"
	case strings.Contains(sys, "helper doing one task"):
		reply = "result for " + last
	default:
		reply = "Adding the parts: Answer: 391"
	}
	b, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{"delta": map[string]string{"content": reply}, "finish_reason": "stop"}}})
	fmt.Fprintf(w, "data: %s\n\ndata: {\"choices\":[],\"usage\":{\"completion_tokens\":5,\"cost\":0.001}}\n\ndata: [DONE]\n\n", b)
}

func TestTeamEval(t *testing.T) {
	s, fake, _ := newTestServer(t)
	fake.tags = threeModels
	lib, _ := evals.OpenLibrary(t.TempDir())
	s.evalLib, s.evalRes = lib, evals.OpenResults(t.TempDir())
	if _, err := lib.Save("two", []byte(twoQuestions)); err != nil {
		t.Fatal(err)
	}
	api := &teamAPI{}
	ts := httptest.NewServer(api)
	defer ts.Close()
	do(s, "PUT", "/api/remote/endpoints/new", `{"name":"Fake","url":"`+ts.URL+`"}`)
	do(s, "PUT", "/api/remote/models/new", `{"endpoint":"fake","model":"m","title":"M"}`)
	for _, body := range []string{
		`{"title":"Pipe","strategy":"pipeline","lead":"api:m","worker":"api:m","size":2}`,
		`{"title":"Loop","strategy":"loop","lead":"api:m","worker":"api:m","size":4,"rounds":2}`,
	} {
		if rec := do(s, "PUT", "/api/teams/new", body); rec.Code != 200 {
			t.Fatalf("team: %d %s", rec.Code, rec.Body)
		}
	}
	if rec := do(s, "PUT", "/api/teams/new", `{"title":"Bad","strategy":"loop","lead":"api:nope","worker":"api:m","size":2,"rounds":1}`); rec.Code != 400 {
		t.Errorf("a team with a missing model saved: %d", rec.Code)
	}
	if _, err := s.startEvals(context.Background(), evalRunRequest{Models: []string{"team:pipe", "team:loop"}, Sets: []string{"two"}}); err != nil {
		t.Fatal(err)
	}
	for _, it := range waitEvals(t, s).Items {
		if it.State != itemDone || it.Answered != 2 || it.Correct != 1 {
			t.Fatalf("item %+v", it)
		}
	}
	res, _ := s.evalResultsFor(context.Background(), "", []string{"two"})
	rows := map[string]evalRow{}
	for _, row := range res.Rows {
		rows[row.Name] = row
	}
	pipe, loop := rows["team:pipe"], rows["team:loop"]
	if !pipe.Team || !strings.Contains(pipe.Quant, "M → M ×2") || !strings.Contains(loop.Quant, "Free loop, 2 rounds") {
		t.Fatalf("rows %+v / %+v", pipe, loop)
	}
	// Pipeline: plan, 2 parts (the third is dropped), combine = 4 calls of 5 tokens.
	if c := pipe.Cells["two"]; c.Tokens != 2*4*5 || math.Abs(c.Cost-2*4*0.001) > 1e-9 {
		t.Errorf("pipeline cell %+v", c)
	}
	// Loop: hand out 2 tasks, then answer = 4 calls.
	if c := loop.Cells["two"]; c.Tokens != 2*4*5 {
		t.Errorf("loop cell %+v", c)
	}
	r, _ := s.evalRes.Get(pipe.Digest, "two", "")
	tr := r.Answers[0].Thinking
	if !strings.Contains(tr, "## Plan (M)") || !strings.Contains(tr, "## Part 2 (M): multiply 17 by 3") || strings.Contains(tr, "## Part 3") {
		t.Errorf("pipeline transcript:\n%s", tr)
	}
	r, _ = s.evalRes.Get(loop.Digest, "two", "")
	if tr := r.Answers[0].Thinking; !strings.Contains(tr, "## Round 1, task 2 (M)") || !strings.Contains(tr, "result for compute 17*3") {
		t.Errorf("loop transcript:\n%s", tr)
	}
	// The lead saw the helpers' results before answering.
	api.mu.Lock()
	defer api.mu.Unlock()
	var sawParts, sawResults bool
	for _, c := range api.calls {
		sawParts = sawParts || strings.Contains(c, "### Part 1: multiply 17 by 20\n\npart done")
		sawResults = sawResults || strings.Contains(c, "<result task=\"2\">\nresult for compute 17*3")
	}
	if !sawParts || !sawResults {
		t.Errorf("lead never saw the results: %v %v", sawParts, sawResults)
	}
}

func TestParts(t *testing.T) {
	for _, c := range []struct {
		in   string
		want []string
	}{
		{`["a", "b", "c"]`, []string{"a", "b"}},
		{"Plan:\n1. first thing\n2) second thing\n", []string{"first thing", "second thing"}},
		{"- one\n- two", []string{"one", "two"}},
		{"just do it", nil},
	} {
		if got := parts(c.in, 2); strings.Join(got, "|") != strings.Join(c.want, "|") {
			t.Errorf("parts(%q) = %q", c.in, got)
		}
	}
}

func TestTeamChat(t *testing.T) {
	s, fake, _ := newTestServer(t)
	fake.tags = threeModels
	api := &teamAPI{}
	ts := httptest.NewServer(api)
	defer ts.Close()
	do(s, "PUT", "/api/remote/endpoints/new", `{"name":"Fake","url":"`+ts.URL+`"}`)
	do(s, "PUT", "/api/remote/models/new", `{"endpoint":"fake","model":"m","title":"M"}`)
	do(s, "PUT", "/api/teams/new", `{"title":"Pipe","strategy":"pipeline","lead":"api:m","worker":"api:m","size":2}`)
	do(s, "PUT", "/api/teams/new", `{"title":"Loop","strategy":"loop","lead":"api:m","worker":"api:m","size":2,"rounds":1}`)
	for _, team := range []string{"team:pipe", "team:loop"} {
		rec := do(s, "POST", "/api/chat", `{"model":"`+team+`","content":"What is 17 × 23?"}`)
		var thinking, content string
		var done map[string]any
		for _, line := range strings.Split(strings.TrimSpace(rec.Body.String()), "\n") {
			var ev map[string]any
			json.Unmarshal([]byte(line), &ev)
			switch ev["type"] {
			case "thinking":
				thinking += ev["text"].(string)
			case "content":
				content += ev["text"].(string)
			case "done":
				done = ev["stats"].(map[string]any)
			case "error":
				t.Fatalf("%s: %v", team, ev["message"])
			}
		}
		if content != "Adding the parts: Answer: 391" || done == nil || done["outputTokens"] != float64(20) || math.Abs(done["cost"].(float64)-0.004) > 1e-9 {
			t.Errorf("%s: content %q done %v", team, content, done)
		}
		want := "## Plan (M)"
		if team == "team:loop" {
			want = "## Round 1, task 2 (M)"
		}
		if !strings.Contains(thinking, want) {
			t.Errorf("%s thinking:\n%s", team, thinking)
		}
	}
	// The chat was saved with the team as its model, the work as thinking.
	list, _ := s.chats.List()
	c, _ := s.chats.Get(list[0].ID)
	if last := c.Messages[len(c.Messages)-1]; last.Role != "assistant" || !strings.Contains(last.Thinking, "## ") || last.Stats.Cost == 0 {
		t.Errorf("saved %+v", last)
	}
}

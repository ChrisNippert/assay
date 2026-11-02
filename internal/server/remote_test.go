package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

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
	a, err := askAPI(context.Background(), ep, remote.Model{ID: "m", Model: "o"}, "", "17×23?", 100)
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

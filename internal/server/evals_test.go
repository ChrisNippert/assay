package server

import (
	"context"
	"encoding/json"
	"math"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"assay/internal/evals"
	"assay/internal/lean"
)

func waitEvals(t *testing.T, s *Server) evalState {
	t.Helper()
	for range 500 {
		if st := s.evalJob.state(); !st.Running {
			return st
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("benchmark run never finished")
	return evalState{}
}

const twoQuestions = `name = "Two"
check = "number"
[[questions]]
id = "a"
prompt = "What is 17 × 23?"
answer = "391"
[[questions]]
id = "b"
prompt = "Eggs for 10?"
answer = "7.5"
`

func TestEvalRun(t *testing.T) {
	s, fake, _ := newTestServer(t)
	fake.tags = threeModels
	lib, _ := evals.OpenLibrary(t.TempDir())
	s.evalLib, s.evalRes = lib, evals.OpenResults(t.TempDir())
	if _, err := lib.Save("two", []byte(twoQuestions)); err != nil {
		t.Fatal(err)
	}
	fake.reply = `{"message":{"role":"assistant","thinking":"17*23 = 391"}}
{"message":{"role":"assistant","content":"So:\nAnswer: 391"}}
{"done":true,"done_reason":"stop","eval_count":12}
`
	st, err := s.startEvalLevels(context.Background(), evalRunRequest{Models: []string{"small:q4", "embed:0.6b"}, Sets: []string{"two"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(st.Items) != 2 || st.Items[1].State != itemSkipped {
		t.Fatalf("items %+v", st.Items)
	}
	st = waitEvals(t, s)
	if it := st.Items[0]; it.State != itemDone || it.Answered != 2 || it.Correct != 1 {
		t.Fatalf("item %+v", it)
	}
	r, ok := s.evalRes.Get("d1", "two", "")
	if !ok || !r.Done() || r.Correct != 1 || r.Answers[0].Got != "391" || r.Answers[0].Thinking == "" || r.Answers[1].Correct || r.Answers[0].Tokens != 12 {
		t.Fatalf("result %+v", r)
	}
	// The eval sampling, the limit for thinking runs, and no think field.
	fake.mu.Lock()
	req := fake.chatReqs[len(fake.chatReqs)-1]
	fake.mu.Unlock()
	if req.Options["temperature"] != float64(1) || req.Options["seed"] != float64(1) || req.Options["num_predict"] != float64(evalTokensThink) || req.Think != nil {
		t.Errorf("request %+v think %v", req.Options, req.Think)
	}
	// Done already: nothing to run unless asked to redo.
	if _, err := s.startEvalLevels(context.Background(), evalRunRequest{Models: []string{"small:q4"}, Sets: []string{"two"}}); err == nil {
		t.Error("ran a finished set again")
	}
	// Thinking off is a separate result.
	if _, err := s.startEvalLevels(context.Background(), evalRunRequest{Models: []string{"small:q4"}, Sets: []string{"two"}, Think: "off"}); err != nil {
		t.Fatal(err)
	}
	waitEvals(t, s)
	fake.mu.Lock()
	req = fake.chatReqs[len(fake.chatReqs)-1]
	fake.mu.Unlock()
	if req.Think != false || req.Options["num_predict"] != float64(evalTokensNoThink) {
		t.Errorf("think off request: %v %v", req.Think, req.Options)
	}
	if _, ok := s.evalRes.Get("d1", "two", "off"); !ok {
		t.Error("no think-off result")
	}

	// The table shows the cell; with the user's set as the suite, an overall.
	res, err := s.evalResultsFor(context.Background(), "", []string{"two"})
	if err != nil {
		t.Fatal(err)
	}
	var found bool
	for _, row := range res.Rows {
		if row.Name == "small:q4" {
			found = true
			if c := row.Cells["two"]; !c.Done || c.Score != 0.5 || row.Overall == nil || *row.Overall != 0.5 {
				t.Errorf("row %+v", row)
			}
		}
		if row.Name == "embed:0.6b" {
			t.Error("embedding model listed")
		}
	}
	if !found || res.Rows[0].Name != "small:q4" {
		t.Errorf("rows %+v", res.Rows)
	}

	// Editing the set marks the result stale and drops the overall.
	lib.Save("two", []byte(strings.Replace(twoQuestions, `"7.5"`, `"8"`, 1)))
	res, _ = s.evalResultsFor(context.Background(), "", []string{"two"})
	for _, row := range res.Rows {
		if row.Name == "small:q4" && (!row.Cells["two"].Stale || row.Overall != nil) {
			t.Errorf("not stale: %+v", row)
		}
	}
}

func TestEvalLevels(t *testing.T) {
	s, fake, _ := newTestServer(t)
	fake.tags = threeModels
	lib, _ := evals.OpenLibrary(t.TempDir())
	s.evalLib, s.evalRes = lib, evals.OpenResults(t.TempDir())
	if _, err := lib.Save("two", []byte(twoQuestions)); err != nil {
		t.Fatal(err)
	}
	fake.reply = `{"message":{"role":"assistant","content":"Answer: 391"}}
{"done":true,"done_reason":"stop","eval_count":5}
`
	req := evalRunRequest{Models: []string{"small:q4"}, Sets: []string{"two"}}
	req.Thinks = []string{"max"}
	if _, err := s.startEvalLevels(context.Background(), req); err == nil {
		t.Fatal("accepted an unknown level")
	}
	req.Thinks = []string{"low", "high"}
	if _, err := s.startEvalLevels(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(10 * time.Second)
	for {
		_, lo := s.evalRes.Get("d1", "two", "low")
		_, hi := s.evalRes.Get("d1", "two", "high")
		if lo && hi {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("low %v high %v", lo, hi)
		}
		time.Sleep(20 * time.Millisecond)
	}
	fake.mu.Lock()
	last := fake.chatReqs[len(fake.chatReqs)-1]
	fake.mu.Unlock()
	if last.Think != "high" {
		t.Errorf("think %v, want high", last.Think)
	}
	// The table of every mode has a row per model and level with results.
	out, err := s.evalResultsFor(context.Background(), thinkAll, nil)
	if err != nil {
		t.Fatal(err)
	}
	var titles []string
	for _, r := range out.Rows {
		if r.Name == "small:q4" {
			titles = append(titles, r.Think)
		}
	}
	slices.Sort(titles)
	if !slices.Equal(titles, []string{"", "high", "low"}) {
		t.Errorf("rows by mode: %q", titles)
	}
}

func TestEvalResumes(t *testing.T) {
	s, fake, _ := newTestServer(t)
	fake.tags = threeModels
	lib, _ := evals.OpenLibrary(t.TempDir())
	s.evalLib, s.evalRes = lib, evals.OpenResults(t.TempDir())
	set, _ := lib.Save("two", []byte(twoQuestions))
	s.evalRes.Put(&evals.Result{Model: "small:q4", Digest: "d1", Set: "two", SetHash: set.Hash, Sampling: evals.Sampling, Total: 2, MaxTokens: 100, Answers: []evals.Answer{{ID: "a", Correct: true}}, Correct: 1})
	fake.reply = `{"message":{"role":"assistant","content":"Answer: 7.5"}}
{"done":true,"done_reason":"length","eval_count":100}
`
	if _, err := s.startEvalLevels(context.Background(), evalRunRequest{Models: []string{"small:q4"}, Sets: []string{"two"}}); err != nil {
		t.Fatal(err)
	}
	waitEvals(t, s)
	r, _ := s.evalRes.Get("d1", "two", "")
	if len(r.Answers) != 2 || r.Correct != 2 || !r.Answers[1].Cut || r.MaxTokens != 100 {
		t.Errorf("resumed %+v", r)
	}
	fake.mu.Lock()
	n := len(fake.chatReqs)
	fake.mu.Unlock()
	if n != 1 {
		t.Errorf("asked %d questions, want 1", n)
	}
}

func TestFitRows(t *testing.T) {
	f := func(v float64) *float64 { return &v }
	rows := []evalRow{
		{Name: "a", Overall: f(0.2), Index: f(10), Of: "x/a"},
		{Name: "a2", Overall: f(0.4), Index: f(10), Of: "x/a"}, // same base: counted once at 0.3
		{Name: "b", Overall: f(0.5), Index: f(20), Of: "x/b"},
		{Name: "c", Overall: f(0.7), Index: f(30), Of: "x/c"},
		{Name: "d", Overall: f(0.9)},
		{Name: "e"},
	}
	fit := fitRows(rows)
	if !fit.OK || fit.N != 3 || math.Abs(fit.B-50) > 1e-9 || fit.R2 < 0.999 {
		t.Fatalf("fit %+v", fit)
	}
	if rows[4].Estimate == nil || math.Abs(*rows[4].Estimate-40) > 1e-9 || !rows[4].Outside || rows[4].Anchor {
		t.Errorf("d: %+v", rows[4])
	}
	if !rows[1].Anchor || rows[5].Estimate != nil {
		t.Errorf("anchor/none: %+v %+v", rows[1], rows[5])
	}
	if fit := fitRows(rows[:2]); fit.OK || fit.Comment == "" {
		t.Errorf("too few: %+v", fit)
	}
}

func TestBenchmarksPage(t *testing.T) {
	s, _, _ := newTestServer(t)
	rec := httptest.NewRecorder()
	s.ServeHTTP(rec, httptest.NewRequest("GET", "http://localhost/benchmarks", nil))
	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	for _, want := range []string{`id="bm-table"`, `window.assayBench = {`, `"Math word problems"`, `/static/benchmarks.js`, `class="nav-item active"`} {
		if !strings.Contains(rec.Body.String(), want) {
			t.Errorf("page lacks %s", want)
		}
	}
	rec = httptest.NewRecorder()
	s.ServeHTTP(rec, httptest.NewRequest("PUT", "http://localhost/api/evals/sets/math", strings.NewReader(twoQuestions)))
	if rec.Code != 400 {
		t.Errorf("saved over a built-in set: %d", rec.Code)
	}
}

const codeAndProof = `name = "Mixed"
[[questions]]
id = "add"
check = "code"
prompt = "Write add(a, b)."
tests = "assert add(2, 3) == 5\n"
[[questions]]
id = "proof"
check = "judge"
prompt = "Prove 1 + 1 = 2."
answer = "By definition."
`

func line(content string) string {
	b, _ := json.Marshal(map[string]any{"message": map[string]string{"role": "assistant", "content": content}})
	return string(b) + "\n" + `{"done":true,"done_reason":"stop","eval_count":5}` + "\n"
}

func TestEvalCodeAndJudge(t *testing.T) {
	if !evals.HavePython() {
		t.Skip("no python3")
	}
	s, fake, _ := newTestServer(t)
	fake.tags = threeModels
	lib, _ := evals.OpenLibrary(t.TempDir())
	s.evalLib, s.evalRes = lib, evals.OpenResults(t.TempDir())
	if _, err := lib.Save("mixed", []byte(codeAndProof)); err != nil {
		t.Fatal(err)
	}
	// No judge: the proof waits.
	fake.replies = []string{line("```python\ndef add(a, b):\n    return a + b\n```"), line("It's obvious.")}
	if _, err := s.startEvalLevels(context.Background(), evalRunRequest{Models: []string{"small:q4"}, Sets: []string{"mixed"}}); err != nil {
		t.Fatal(err)
	}
	st := waitEvals(t, s)
	if it := st.Items[0]; it.State != itemDone || it.Waiting != 1 || it.Correct != 1 {
		t.Fatalf("item %+v", it)
	}
	r, _ := s.evalRes.Get("d1", "mixed", "")
	if r.Done() || !r.Answers[0].Correct || !r.Answers[1].Pending {
		t.Fatalf("result %+v", r)
	}
	// With a judge: only the judge is asked, and its verdict counts.
	fake.mu.Lock()
	asked := len(fake.chatReqs)
	fake.replies = []string{line("A one-line claim, no proof.\nVerdict: FAIL")}
	fake.mu.Unlock()
	if _, err := s.startEvalLevels(context.Background(), evalRunRequest{Models: []string{"small:q4"}, Sets: []string{"mixed"}, Judge: "mid:q4"}); err != nil {
		t.Fatal(err)
	}
	waitEvals(t, s)
	fake.mu.Lock()
	reqs := fake.chatReqs[asked:]
	fake.mu.Unlock()
	if len(reqs) != 1 || reqs[0].Model != "mid:q4" || !strings.Contains(reqs[0].Messages[1].Content, "By definition.") {
		t.Fatalf("judge requests %+v", reqs)
	}
	r, _ = s.evalRes.Get("d1", "mixed", "")
	if !r.Done() || r.Correct != 1 || r.Answers[1].Pending || r.Answers[1].JudgeModel != "mid:q4" || !strings.Contains(r.Answers[1].Judge, "no proof") {
		t.Fatalf("judged %+v", r.Answers[1])
	}
}

const leanSet = `name = "L"
check = "lean"
[[questions]]
id = "double"
prompt = "Prove double n = 2n."
preamble = "def double (n : Nat) : Nat := n + n"
statement = "∀ n : Nat, double n = 2 * n"
[[questions]]
id = "sorry"
prompt = "Prove it again."
preamble = "def double (n : Nat) : Nat := n + n"
statement = "∀ n : Nat, double n = 2 * n"
`

func TestEvalLean(t *testing.T) {
	home, _ := os.UserHomeDir()
	inst := &lean.Installer{Dir: filepath.Join(home, ".local/share/assay/lean")}
	if inst.Binary() == "" {
		t.Skip("Lean isn't installed")
	}
	s, fake, _ := newTestServer(t)
	fake.tags = threeModels
	lib, _ := evals.OpenLibrary(t.TempDir())
	s.evalLib, s.evalRes = lib, evals.OpenResults(t.TempDir())
	lib.Save("l", []byte(leanSet))
	// Without Lean the run is refused up front.
	s.leanInst = &lean.Installer{Dir: t.TempDir()}
	if _, err := s.startEvalLevels(context.Background(), evalRunRequest{Models: []string{"small:q4"}, Sets: []string{"l"}}); err == nil || !strings.Contains(err.Error(), "Lean") {
		t.Fatalf("ran without Lean: %v", err)
	}
	if onPath, _ := exec.LookPath("lean"); onPath != "" {
		t.Skip("a lean on PATH makes the refusal untestable")
	}
	s.leanInst = inst
	fake.replies = []string{
		line("Unfold and use omega.\n```lean\ntheorem goal : ∀ n : Nat, double n = 2 * n := by\n  intro n; unfold double; omega\n```"),
		line("```lean\ntheorem goal : ∀ n : Nat, double n = 2 * n := by sorry\n```"),
	}
	if _, err := s.startEvalLevels(context.Background(), evalRunRequest{Models: []string{"small:q4"}, Sets: []string{"l"}}); err != nil {
		t.Fatal(err)
	}
	waitEvals(t, s)
	r, _ := s.evalRes.Get("d1", "l", "")
	if r == nil || !r.Done() || r.Correct != 1 || !r.Answers[0].Correct || r.Answers[1].Correct || !strings.Contains(r.Answers[1].Output, "sorry") {
		t.Fatalf("result %+v", r)
	}
	// The prompt gives the definitions and the exact theorem to prove.
	fake.mu.Lock()
	msg := fake.chatReqs[len(fake.chatReqs)-2].Messages[0].Content
	fake.mu.Unlock()
	if !strings.Contains(msg, "def double") || !strings.Contains(msg, "theorem goal : ∀ n : Nat, double n = 2 * n") {
		t.Errorf("prompt: %s", msg)
	}
}

// A model that repeats itself is stopped before the token limit.
func TestEvalStopsLoops(t *testing.T) {
	s, fake, _ := newTestServer(t)
	fake.tags = threeModels
	var b strings.Builder
	for range 400 {
		b.WriteString(`{"message":{"role":"assistant","content":"","thinking":"Let me recheck: 8 = 20.5 + 10.348t - 4.905t^2\n"}}` + "\n")
	}
	b.WriteString(`{"message":{"role":"assistant","content":"Answer: 7.5"}}` + "\n" + `{"done":true,"done_reason":"stop","eval_count":401}` + "\n")
	fake.reply = b.String()
	a, err := s.askModel(context.Background(), "small:q4", "", "q", "", 8192, 0)
	if err != nil || !a.Looped || a.Reply != "" || a.Tokens == 0 || a.Tokens >= 400 {
		t.Fatalf("err %v looped %v reply %q tokens %d", err, a.Looped, a.Reply, a.Tokens)
	}
}

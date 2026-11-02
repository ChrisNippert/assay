package evals

import (
	"context"
	"encoding/json"
	"math"
	"math/rand/v2"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"assay/internal/lean"
)

func TestBuiltinSetsLoad(t *testing.T) {
	l, err := OpenLibrary("")
	if err != nil {
		t.Fatal(err)
	}
	sets, errs := l.List()
	if len(errs) > 0 || len(sets) < 5 {
		t.Fatalf("sets %d errs %v", len(sets), errs)
	}
	for _, s := range sets {
		if !s.BuiltIn || s.Hash == "" || len(s.Questions) < 5 {
			t.Errorf("%s: builtIn %v hash %q questions %d", s.ID, s.BuiltIn, s.Hash, len(s.Questions))
		}
	}
}

// The right answer, written the way a model would, passes every built-in
// question; this catches answers the grader can't read.
func TestBuiltinAnswersGrade(t *testing.T) {
	l, _ := OpenLibrary("")
	sets, _ := l.List()
	samples := map[string]string{ // regex questions: a reply that should pass
		"three-words":    "Vast, deep, restless.",
		"yes-caps":       "YES",
		"bullets":        "- apple\n- pear\n- plum",
		"no-e":           "My cats nap all day.",
		"json":           `{"name": "Ana", "age": 31}`,
		"commas":         "1,2,3,4,5,6,7,8,9,10",
		"lowercase":      "the capital of france is paris.",
		"haiku-end":      "cold breath on the glass\nsnow hides the sleeping garden\nthe kettle whistles\nTHE END",
		"reverse-words":  "jumps fox brown quick the",
		"rhyme-brackets": "[HAT]",
		"exclaim":        "The moon is bright tonight! It lights the whole valley!",
		"spanish":        "buenos días",
		"countdown":      "5\n4\n3\n2\n1",
		"q-k":            "quick",
		"s-words":        "Sam sells seven silver spoons.",
		"no-comma":       "Sunlight hits air molecules and blue light scatters the most. So we see blue all around us.",
	}
	for _, s := range sets {
		for _, q := range s.Questions {
			if c := s.CheckOf(q); c == CheckCode || c == CheckJudge || c == CheckLean {
				continue // see TestCodeRuns and TestParseVerdict
			}
			reply := "Let me think.\n\nAnswer: " + q.Answer
			if s.CheckOf(q) == CheckRegex {
				var ok bool
				reply, ok = samples[q.ID]
				if !ok {
					t.Errorf("%s/%s: no sample reply", s.ID, q.ID)
					continue
				}
			}
			if got, ok := s.Grade(q, reply); !ok {
				t.Errorf("%s/%s: right answer graded wrong (read %q)", s.ID, q.ID, got)
			}
		}
	}
}

func TestGrade(t *testing.T) {
	s := &Set{ID: "t", Name: "t"}
	num := Question{Prompt: "p", Answer: "70.20", Check: CheckNumber}
	frac := Question{Prompt: "p", Answer: "5/36", Check: CheckNumber, Tolerance: 0.0005}
	choice := Question{Prompt: "p", Answer: "C", Choices: []string{"a", "b", "c", "d"}}
	exact := Question{Prompt: "p", Answer: "('c', 2)", Accept: []string{"(c, 2)"}, Check: CheckExact}
	sci := Question{Prompt: "p", Answer: "32000", Check: CheckNumber, Tolerance: 100}
	small := Question{Prompt: "p", Answer: "0.0045", Check: CheckNumber, Tolerance: 0.00002}
	notRe := Question{Prompt: "p", Answer: "(?i)e", Check: CheckRegex, Not: true}
	set := Question{Prompt: "p", Answer: "3, 5, 7", Check: CheckSet}
	setStr := Question{Prompt: "p", Answer: "AB, BA", Check: CheckSet}
	setPair := Question{Prompt: "p", Answer: "1:2, 3:4", Check: CheckSet}
	cases := []struct {
		q     Question
		reply string
		ok    bool
	}{
		{num, "So she pays $70.20 in total.\n\n**Answer: $70.20**", true},
		{num, "Answer: 70.2 dollars", true},
		{num, "The total is 60 + 10.20 = 70.20", true},
		{num, "Answer: 75", false},
		{num, "<think>maybe 75?</think>\nAnswer: 70.20", true},
		{num, "Answer: 1,070.20", false},
		{frac, "Answer: 5/36", true},
		{frac, "Answer: ≈ 0.1389", true},
		{frac, "\\boxed{\\frac{5}{36}}", true},
		{frac, "Answer: 0.15", false},
		{choice, "The labels are wrong, so...\nAnswer: C", true},
		{choice, "Answer: (C) The one labelled Mixed", true},
		{choice, "Answer: C.", true},
		{choice, "**Final answer:** B", false},
		{choice, "I pick C", true},
		{exact, "Answer: ('c', 2)", true},
		{exact, "Answer: `(c, 2)`", true},
		{exact, "Answer: ('b', 1)", false},
		{sci, "Answer: 3.2 × 10^4 N", true},
		{sci, "Answer: $3.2 \\times 10^{4}$", true},
		{sci, "Answer: 3.2·10⁴", true},
		{sci, "Answer: 3.2e4", true},
		{sci, "Answer: 32,000 newtons", true},
		{sci, "Answer: 3.2 × 10^3", false},
		{small, "Answer: 4.5 \\cdot 10^{-3} s", true},
		{small, "Answer: 4.5×10⁻³", true},
		{small, "Answer: -4.5×10⁻³", false},
		{notRe, "A cat naps", true},
		{notRe, "The cat naps", false},
		{set, "Answer: 3, 5, 7", true},
		{set, "Answer: 7 5 3", true},
		{set, "**Answer:** {3, 5, 7}", true},
		{set, "Answer: [3,5,7]", true},
		{set, "Answer: 3, 5", false},
		{set, "Answer: 3, 5, 7, 8", false},
		{set, "Answer: 3, 5, 5, 7", true},
		{setStr, "Answer: {\"BA\", \"ab\"}", true},
		{setStr, "Answer: AB", false},
		{setStr, "Answer: AB, BA, BB", false},
		{setPair, "Answer: 1:2, 3:4", true},
		{setPair, "Answer: 1:2", false},
	}
	for _, c := range cases {
		if got, ok := s.Grade(c.q, c.reply); ok != c.ok {
			t.Errorf("%q vs %q: ok=%v (read %q), want %v", c.reply, c.q.Answer, ok, got, c.ok)
		}
	}
}

func TestParseRejects(t *testing.T) {
	bad := map[string]string{
		"no questions": `name = "x"`,
		"bad number":   "name = \"x\"\ncheck = \"number\"\n[[questions]]\nprompt = \"p\"\nanswer = \"many\"",
		"bad letter":   "name = \"x\"\n[[questions]]\nprompt = \"p\"\nchoices = [\"a\", \"b\"]\nanswer = \"C\"",
		"bad regex":    "name = \"x\"\ncheck = \"regex\"\n[[questions]]\nprompt = \"p\"\nanswer = \"(\"",
		"unknown key":  "name = \"x\"\nfoo = 1\n[[questions]]\nprompt = \"p\"\nanswer = \"a\"",
		"dup id":       "name = \"x\"\n[[questions]]\nid = \"a\"\nprompt = \"p\"\nanswer = \"a\"\n[[questions]]\nid = \"a\"\nprompt = \"p\"\nanswer = \"a\"",
	}
	for name, src := range bad {
		if _, err := Parse("x", []byte(src)); err == nil {
			t.Errorf("%s: parsed", name)
		}
	}
	s, err := Parse("x", []byte("name = \"x\"\n[[questions]]\nprompt = \"p\"\nanswer = \"a\""))
	if err != nil || s.Questions[0].ID != "1" {
		t.Fatalf("good set: %v %+v", err, s)
	}
}

func TestLibrarySaveDelete(t *testing.T) {
	l, _ := OpenLibrary(t.TempDir())
	if _, err := l.Save("math", []byte("name = \"x\"\n[[questions]]\nprompt = \"p\"\nanswer = \"a\"")); err == nil {
		t.Error("saved over a built-in set")
	}
	src := "name = \"Mine\"\n[[questions]]\nprompt = \"p\"\nanswer = \"a\"\n"
	s, err := l.Save("mine", []byte(src))
	if err != nil || s.Name != "Mine" {
		t.Fatal(err)
	}
	if raw, _ := l.Raw("mine"); string(raw) != src {
		t.Errorf("raw %q", raw)
	}
	if _, err := l.Get("mine"); err != nil {
		t.Fatal(err)
	}
	sets, _ := l.List()
	if sets[len(sets)-1].ID != "mine" || sets[len(sets)-1].BuiltIn {
		t.Errorf("list: %+v", sets[len(sets)-1])
	}
	if err := l.Delete("mine"); err != nil {
		t.Fatal(err)
	}
	if _, err := l.Get("mine"); err == nil {
		t.Error("still there")
	}
}

func TestHashChangesWithAnswers(t *testing.T) {
	a, _ := Parse("x", []byte("name = \"x\"\n[[questions]]\nprompt = \"p\"\nanswer = \"a\""))
	b, _ := Parse("x", []byte("name = \"renamed\"\n[[questions]]\nprompt = \"p\"\nanswer = \"a\""))
	c, _ := Parse("x", []byte("name = \"x\"\n[[questions]]\nprompt = \"p\"\nanswer = \"b\""))
	if a.Hash != b.Hash || a.Hash == c.Hash {
		t.Errorf("hashes %s %s %s", a.Hash, b.Hash, c.Hash)
	}
}

func TestResults(t *testing.T) {
	rs := OpenResults(t.TempDir())
	r := &Result{Model: "m", Digest: "sha256:abc", Set: "math", Total: 2, Answers: []Answer{{ID: "1", Correct: true, Reply: "x"}}}
	r.Recount()
	if r.Done() || r.Score() != 1 {
		t.Errorf("done %v score %v", r.Done(), r.Score())
	}
	if err := rs.Put(r); err != nil {
		t.Fatal(err)
	}
	r2 := *r
	r2.Think = ThinkOff
	rs.Put(&r2)
	if got, ok := rs.Get("sha256:abc", "math", ""); !ok || got.Answers[0].Reply != "x" {
		t.Fatalf("get %+v", got)
	}
	all := rs.All()
	if len(all) != 2 || all[0].Answers[0].Reply != "" {
		t.Errorf("all: %d", len(all))
	}
	rs.Delete("sha256:abc", "math", ThinkOff)
	if _, ok := rs.Get("sha256:abc", "math", ThinkOff); ok {
		t.Error("not deleted")
	}
	a := Answer{Thinking: strings.Repeat("x", maxThinking*2)}
	a.Clip()
	if len(a.Thinking) > maxThinking+100 {
		t.Errorf("clip: %d", len(a.Thinking))
	}
}

func TestFitLine(t *testing.T) {
	if _, ok := FitLine([]Point{{0.1, 5}, {0.5, 20}}); ok {
		t.Error("fitted two points")
	}
	f, ok := FitLine([]Point{{0.2, 10}, {0.4, 20}, {0.6, 30}})
	if !ok || math.Abs(f.B-50) > 1e-9 || math.Abs(f.A) > 1e-9 || f.R2 < 0.999 || f.SE > 1e-9 {
		t.Fatalf("fit %+v", f)
	}
	if p := f.Predict(0.5); math.Abs(p-25) > 1e-9 {
		t.Errorf("predict %v", p)
	}
	if !f.Outside(0.8) || f.Outside(0.3) {
		t.Error("outside")
	}
	f, _ = FitLine([]Point{{0.2, 12}, {0.4, 18}, {0.6, 31}, {0.5, 22}})
	if f.R2 <= 0.8 || f.R2 >= 1 || f.SE <= 0 {
		t.Errorf("noisy fit %+v", f)
	}
}

func TestExtractCode(t *testing.T) {
	reply := "<think>```python\nwrong()\n```</think>Here:\n```python\ndef f(): return 1\n```\nand a fix:\n```py\ndef f(): return 2\n```\nDone."
	if code, ok := ExtractCode(reply); !ok || code != "def f(): return 2\n" {
		t.Errorf("got %q %v", code, ok)
	}
	if code, ok := ExtractCode("```\nx = 1\n```"); !ok || code != "x = 1\n" {
		t.Errorf("unlabelled: %q", code)
	}
	if _, ok := ExtractCode("no code here"); ok {
		t.Error("found code in plain text")
	}
}

func TestCodeRuns(t *testing.T) {
	if !HavePython() {
		t.Skip("no python3")
	}
	ctx := context.Background()
	old := CodeTimeout
	CodeTimeout = 2 * time.Second
	defer func() { CodeTimeout = old }()
	t.Logf("network blocked: %v", noNetwork())
	if noNetwork() {
		net := "import socket\ns = socket.socket()\ns.settimeout(2)\ns.connect(('1.1.1.1', 53))\n"
		if ok, out, _ := RunCode(ctx, net, "pass\n"); ok {
			t.Errorf("reached the network: %s", out)
		}
	}
	tests := "assert add(2, 3) == 5\nassert add(-1, 1) == 0\n"
	if ok, out, err := RunCode(ctx, "def add(a, b):\n    return a + b\n", tests); !ok || err != nil {
		t.Errorf("right code failed: %v %s", err, out)
	}
	if ok, out, _ := RunCode(ctx, "def add(a, b):\n    return a - b\n", tests); ok || !strings.Contains(out, "AssertionError") {
		t.Errorf("wrong code passed: %s", out)
	}
	// A solution can't fake success by printing the marker and exiting.
	if ok, _, _ := RunCode(ctx, "print('ALL TESTS PASSED')\nimport sys\nsys.exit(0)\n", tests); ok {
		t.Error("exit(0) before the tests counted as a pass")
	}
	if ok, out, _ := RunCode(ctx, "while True: pass\n", tests); ok || !strings.Contains(out, "Stopped") && !strings.Contains(out, "CPU") {
		t.Logf("endless loop output: %q", out)
		if ok {
			t.Error("endless loop passed")
		}
	}
}

// Every built-in code question's tests are valid Python.
func TestBuiltinCodeTestsCompile(t *testing.T) {
	if !HavePython() {
		t.Skip("no python3")
	}
	py, _ := pythonPath()
	l, _ := OpenLibrary("")
	sets, _ := l.List()
	for _, s := range sets {
		for _, q := range s.Questions {
			if s.CheckOf(q) != CheckCode {
				continue
			}
			cmd := exec.Command(py, "-c", "import sys; compile(sys.stdin.read(), 'tests', 'exec')")
			cmd.Stdin = strings.NewReader(q.Tests)
			if out, err := cmd.CombinedOutput(); err != nil {
				t.Errorf("%s/%s: %v %s", s.ID, q.ID, err, out)
			}
		}
	}
}

// Each "Simulation projects" question has a reference solution under
// testdata/projects-sims that passes its hidden tests; an empty program does
// not. The tests' expected values were computed independently (mpmath
// quadrature, sympy-derived equations, analytic formulas), see the set's
// doc in ARCHITECTURE.md.
func TestProjectsSimsReferences(t *testing.T) {
	if !HavePython() {
		t.Skip("no python3")
	}
	l, _ := OpenLibrary("")
	sets, _ := l.List()
	var set *Set
	for _, s := range sets {
		if s.ID == "projects-sims" {
			set = s
		}
	}
	if set == nil {
		t.Fatal("projects-sims set missing")
	}
	if set.MaxTokens != 16384 || len(set.Questions) < 5 {
		t.Errorf("max_tokens %d, %d questions", set.MaxTokens, len(set.Questions))
	}
	for _, q := range set.Questions {
		t.Run(q.ID, func(t *testing.T) {
			ref, err := os.ReadFile(filepath.Join("testdata", "projects-sims", q.ID+".py"))
			if err != nil {
				t.Fatal(err)
			}
			start := time.Now()
			ok, out, err := RunCode(context.Background(), string(ref), q.Tests)
			if err != nil || !ok {
				t.Fatalf("reference failed: %v\n%s", err, out)
			}
			if d := time.Since(start); d > 10*time.Second {
				t.Errorf("reference took %v", d)
			}
			if ok, _, _ := RunCode(context.Background(), "pass\n", q.Tests); ok {
				t.Error("an empty solution passed")
			}
		})
	}
}

func TestParseVerdict(t *testing.T) {
	cases := map[string][2]bool{
		"The proof is fine.\nVerdict: PASS":                      {true, true},
		"Gap in step 2.\n\n**Verdict: FAIL**":                    {false, true},
		"verdict: pass ... wait, no.\nVerdict: FAIL":             {false, true},
		"<think>Verdict: PASS</think>It is wrong. Verdict: fail": {false, true},
		"I can't decide.": {false, false},
	}
	for in, want := range cases {
		if pass, ok := ParseVerdict(in); pass != want[0] || ok != want[1] {
			t.Errorf("%q: %v %v", in, pass, ok)
		}
	}
	sys, user := JudgeMessages(Question{Prompt: "Prove X.", Answer: "Because Y."}, "<think>hmm</think>Proof: Z.")
	if !strings.Contains(sys, "Verdict: PASS") || !strings.Contains(user, "Because Y.") || !strings.Contains(user, "Proof: Z.") || strings.Contains(user, "hmm") {
		t.Errorf("judge prompt: %s", user)
	}
}

// Every built-in Lean question's reference proof passes the checker.
func TestBuiltinLeanReferences(t *testing.T) {
	home, _ := os.UserHomeDir()
	bin := os.Getenv("ASSAY_LEAN")
	if bin == "" {
		bin = (&lean.Installer{Dir: filepath.Join(home, ".local/share/assay/lean")}).Binary()
	}
	if bin == "" {
		t.Skip("Lean isn't installed")
	}
	l, _ := OpenLibrary("")
	sets, _ := l.List()
	n := 0
	for _, s := range sets {
		for _, q := range s.Questions {
			if s.CheckOf(q) != CheckLean {
				continue
			}
			n++
			r, err := lean.Check(context.Background(), bin, q.Preamble, q.Statement, q.Answer)
			if err != nil || !r.OK {
				t.Errorf("%s/%s: %v\n%s", s.ID, q.ID, err, r.Output)
			}
		}
	}
	if n == 0 {
		t.Error("no Lean questions")
	}
}

// Generated sets build the same questions every time, with unique ids and
// finite answers.
func TestGeneratedSets(t *testing.T) {
	for _, g := range genSpecs {
		a, err := g.Build()
		if err != nil {
			t.Fatal(err)
		}
		b, _ := g.Build()
		if a.Hash != b.Hash {
			t.Errorf("%s: not deterministic", g.ID)
		}
		seen := map[string]bool{}
		for _, q := range a.Questions {
			if seen[q.Prompt] {
				t.Errorf("%s/%s: same prompt twice", g.ID, q.ID)
			}
			seen[q.Prompt] = true
			if x, _ := parseNumber(q.Answer); math.IsNaN(x) || math.IsInf(x, 0) {
				t.Errorf("%s/%s: answer %s", g.ID, q.ID, q.Answer)
			}
		}
		verifyGenerated(t, a)
	}
}

// verifyGenerated runs verify/<id>.py, an independent checker written in
// Python with sympy/mpmath, on the set's questions (as JSON: a list of
// {id, prompt, answer, tolerance}). It reads each prompt's numbers from the
// text and recomputes the answer another way, so a slip in the Go code or
// a prompt that says something else than the code computed fails here.
// Skipped when python3 with sympy and mpmath isn't installed. -E -P
// (not -I) so packages installed with pip --user are found.
func verifyGenerated(t *testing.T, s *Set) {
	script, _ := filepath.Abs(filepath.Join("verify", s.ID+".py"))
	if _, err := os.Stat(script); err != nil {
		t.Errorf("%s: no checker at verify/%s.py", s.ID, s.ID)
		return
	}
	if s.ID != "concurrency" && exec.Command("python3", "-E", "-P", "-c", "import sympy, mpmath").Run() != nil { // concurrency needs only the standard library
		t.Logf("%s: python3 with sympy and mpmath not found; not verified", s.ID)
		return
	}
	b, _ := json.Marshal(s.Questions)
	f := filepath.Join(t.TempDir(), s.ID+".json")
	os.WriteFile(f, b, 0o644)
	out, err := exec.Command("python3", "-E", "-P", script, f).CombinedOutput()
	if err != nil {
		t.Errorf("%s: checker failed: %v\n%s", s.ID, err, out)
	}
}

func TestGenBuild(t *testing.T) {
	g := &GenSpec{ID: "toy", Name: "Toy", Seed: 7, PerGen: 5, RelTol: 0.005, Gens: []Generator{{
		ID: "add",
		Make: func(r *rand.Rand) (Question, bool) {
			x, y := decimal(r, 0.1, 9.9, 0.1), between(r, 1, 9)
			return Question{Prompt: "What is " + show(x) + " + " + strconv.Itoa(y) + "?", Answer: num(x + float64(y))}, true
		},
	}}}
	s, err := g.Build()
	if err != nil {
		t.Fatal(err)
	}
	if len(s.Questions) != 5 || s.Questions[4].ID != "add-5" || !s.Generated {
		t.Fatalf("%+v", s.Questions)
	}
	for _, q := range s.Questions {
		if strings.Contains(q.Prompt, "0000") || strings.Contains(q.Prompt, "9999") {
			t.Errorf("float noise in %q", q.Prompt)
		}
		if got, ok := s.Grade(q, "Answer: "+q.Answer); !ok {
			t.Errorf("%s: %q graded wrong", q.Prompt, got)
		}
	}
}

func TestLooping(t *testing.T) {
	step := "*   $v_{0x} = 19 \\cos(33^\\circ) = 15.93473$\n*   $t = 2.96826$\n*   Rounding to 3 significant figures: 47.3\n\n"
	normal := ""
	for i := range 300 {
		normal += "Step " + strconv.Itoa(i) + ": now we compute the next quantity carefully.\n"
	}
	if Looping(normal) || Looping("short") {
		t.Error("normal text flagged as a loop")
	}
	if !Looping(normal + strings.Repeat(step, 12)) {
		t.Error("repeated block not flagged")
	}
	varied := normal
	for i := range 12 { // same line again and again among changing ones
		varied += "Let me double-check the quadratic formula once more.\nvalue " + strconv.Itoa(i) + "\n"
	}
	if !Looping(varied) {
		t.Error("repeated line not flagged")
	}
}

// Every Game projects question has a reference solution in
// testdata/projects-games/<id>.py (written and run when the set was made);
// the hidden tests must pass against it, through the real harness.
func TestProjectsGamesReferences(t *testing.T) {
	if !HavePython() {
		t.Skip("no python3")
	}
	l, _ := OpenLibrary("")
	s, err := l.Get("projects-games")
	if err != nil || s == nil {
		t.Fatalf("projects-games set: %v", err)
	}
	if s.MaxTokens < 16384 || len(s.Questions) < 5 {
		t.Errorf("max_tokens %d, questions %d", s.MaxTokens, len(s.Questions))
	}
	for _, q := range s.Questions {
		t.Run(q.ID, func(t *testing.T) {
			ref, err := os.ReadFile(filepath.Join("testdata", "projects-games", q.ID+".py"))
			if err != nil {
				t.Fatal(err)
			}
			ok, out, err := RunCode(context.Background(), string(ref), q.Tests)
			if err != nil || !ok {
				t.Errorf("reference fails its tests: %v\n%s", err, out)
			}
			// as a model would send it: inside a reply
			code, found := ExtractCode("Here you go.\n```python\n" + string(ref) + "```\n")
			if !found || strings.TrimSpace(code) != strings.TrimSpace(string(ref)) {
				t.Errorf("reference not extracted from a reply")
			}
		})
	}
}

// Every Concurrent code question has a reference solution in
// testdata/concurrency-code/<id>.py; the hidden tests (which hammer it from
// many threads with a 1 µs switch interval) must pass against it, through
// the real harness. The broken variants they were checked against (no lock,
// check-then-act, lock-ordering deadlock, notify instead of notify_all,
// ignored timeouts…) are not kept: each failed its tests every time.
func TestConcurrencyCodeReferences(t *testing.T) {
	if !HavePython() {
		t.Skip("no python3")
	}
	l, _ := OpenLibrary("")
	s, err := l.Get("concurrency-code")
	if err != nil || s == nil {
		t.Fatalf("concurrency-code set: %v", err)
	}
	if s.MaxTokens < 16384 || len(s.Questions) < 6 || len(s.Questions) > 8 {
		t.Errorf("max_tokens %d, questions %d", s.MaxTokens, len(s.Questions))
	}
	for _, q := range s.Questions {
		t.Run(q.ID, func(t *testing.T) {
			ref, err := os.ReadFile(filepath.Join("testdata", "concurrency-code", q.ID+".py"))
			if err != nil {
				t.Fatal(err)
			}
			ok, out, err := RunCode(context.Background(), string(ref), q.Tests)
			if err != nil || !ok {
				t.Errorf("reference fails its tests: %v\n%s", err, out)
			}
			code, found := ExtractCode("Here you go.\n```python\n" + string(ref) + "```\n")
			if !found || strings.TrimSpace(code) != strings.TrimSpace(string(ref)) {
				t.Errorf("reference not extracted from a reply")
			}
			// a missing solution must fail the tests, not pass or hang
			if ok, _, _ := RunCode(context.Background(), "x = 1\n", q.Tests); ok {
				t.Errorf("tests pass without a solution")
			}
		})
	}
}

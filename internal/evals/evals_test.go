package evals

import (
	"context"
	"math"
	"os/exec"
	"strings"
	"testing"
	"time"
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
		if !s.BuiltIn || s.Hash == "" || len(s.Questions) < 10 {
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
			if c := s.CheckOf(q); c == CheckCode || c == CheckJudge {
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
	notRe := Question{Prompt: "p", Answer: "(?i)e", Check: CheckRegex, Not: true}
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
		{notRe, "A cat naps", true},
		{notRe, "The cat naps", false},
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

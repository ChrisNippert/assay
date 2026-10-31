package evals

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"
)

// Think modes a run can use.
const (
	ThinkDefault = ""    // whatever the model does when left alone
	ThinkOff     = "off" // asked not to think
)

// Answer is one question's outcome.
type Answer struct {
	ID       string `json:"id"`
	Reply    string `json:"reply"`
	Thinking string `json:"thinking,omitempty"`
	// Got is the answer read from the reply; empty for regex and contains.
	Got     string `json:"got,omitempty"`
	Correct bool   `json:"correct"`
	// Tokens is how many the model wrote, thinking included; Cut says it
	// hit the token limit.
	Tokens int    `json:"tokens"`
	Cut    bool   `json:"cut,omitempty"`
	Ms     int64  `json:"ms"`
	Error  string `json:"error,omitempty"`
}

// Result is one model's run of one set.
type Result struct {
	Model   string    `json:"model"`
	Digest  string    `json:"digest"`
	Set     string    `json:"set"`
	SetHash string    `json:"setHash"`
	Think   string    `json:"think"`
	At      time.Time `json:"at"`
	Ctx     int       `json:"ctx,omitempty"`
	// MaxTokens is the reply limit per question.
	MaxTokens int `json:"maxTokens"`
	Correct   int `json:"correct"`
	// Total is the set's size; len(Answers) < Total while a run is unfinished.
	Total   int      `json:"total"`
	Answers []Answer `json:"answers"`
}

// Done says every question has an answer.
func (r *Result) Done() bool { return len(r.Answers) >= r.Total && r.Total > 0 }

// Score is the share right, 0–1, of the questions answered so far.
func (r *Result) Score() float64 {
	if len(r.Answers) == 0 {
		return 0
	}
	return float64(r.Correct) / float64(len(r.Answers))
}

// Recount sets Correct from the answers.
func (r *Result) Recount() {
	r.Correct = 0
	for _, a := range r.Answers {
		if a.Correct {
			r.Correct++
		}
	}
}

// maxThinking caps the thinking kept per answer.
const maxThinking = 40_000

// Clip shortens very long thinking before saving.
func (a *Answer) Clip() {
	if len(a.Thinking) > maxThinking {
		a.Thinking = a.Thinking[:maxThinking/2] + "\n\n[… cut …]\n\n" + a.Thinking[len(a.Thinking)-maxThinking/2:]
	}
}

// Results keeps runs as <dir>/<digest>/<set>[@nothink].json.
type Results struct {
	dir string
	mu  sync.Mutex
}

// OpenResults keeps results in dir; "" keeps nothing.
func OpenResults(dir string) *Results { return &Results{dir: dir} }

var unsafeRe = regexp.MustCompile(`[^a-zA-Z0-9]+`)

func (rs *Results) path(digest, set, think string) string {
	d := unsafeRe.ReplaceAllString(strings.TrimPrefix(digest, "sha256:"), "_")
	name := set
	if think == ThinkOff {
		name += "@nothink"
	}
	return filepath.Join(rs.dir, d, name+".json")
}

// Get returns a saved run.
func (rs *Results) Get(digest, set, think string) (*Result, bool) {
	if rs.dir == "" || digest == "" {
		return nil, false
	}
	rs.mu.Lock()
	defer rs.mu.Unlock()
	b, err := os.ReadFile(rs.path(digest, set, think))
	if err != nil {
		return nil, false
	}
	var r Result
	if json.Unmarshal(b, &r) != nil {
		return nil, false
	}
	return &r, true
}

// Put saves a run, replacing the last one for the same model, set and mode.
func (rs *Results) Put(r *Result) error {
	if r.Digest == "" {
		return errors.New("evals: result has no digest")
	}
	if rs.dir == "" {
		return nil
	}
	b, err := json.MarshalIndent(r, "", " ")
	if err != nil {
		return err
	}
	rs.mu.Lock()
	defer rs.mu.Unlock()
	p := rs.path(r.Digest, r.Set, r.Think)
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(p+".tmp", b, 0o644); err != nil {
		return err
	}
	return os.Rename(p+".tmp", p)
}

// Delete removes a run.
func (rs *Results) Delete(digest, set, think string) error {
	if rs.dir == "" {
		return nil
	}
	rs.mu.Lock()
	defer rs.mu.Unlock()
	err := os.Remove(rs.path(digest, set, think))
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	return err
}

// All returns every saved run without the answers' texts (for tables).
func (rs *Results) All() []*Result {
	if rs.dir == "" {
		return nil
	}
	rs.mu.Lock()
	defer rs.mu.Unlock()
	files, _ := filepath.Glob(filepath.Join(rs.dir, "*", "*.json"))
	var out []*Result
	for _, f := range files {
		b, err := os.ReadFile(f)
		if err != nil {
			continue
		}
		var r Result
		if json.Unmarshal(b, &r) != nil || r.Digest == "" {
			continue
		}
		for i := range r.Answers {
			r.Answers[i].Reply, r.Answers[i].Thinking = "", ""
		}
		out = append(out, &r)
	}
	return out
}

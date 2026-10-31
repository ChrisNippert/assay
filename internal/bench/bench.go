// Package bench keeps speed measurements of installed models: one JSON file
// per model, under ~/.local/share/assay/bench/. Results are keyed by the
// model's digest, so every name for the same weights shares one result and
// a re-pulled model starts over.
package bench

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

// Result is one speed test.
type Result struct {
	Model  string    `json:"model"`
	Digest string    `json:"digest"`
	At     time.Time `json:"at"`
	// Ctx is the context the model ran with; GPUPct how much of it was on
	// the GPU.
	Ctx    int `json:"ctx"`
	GPUPct int `json:"gpuPct"`
	// LoadMs is loading the model from disk; TTFTMs the time from sending
	// the prompt to the first word of the reply.
	LoadMs       int64   `json:"loadMs"`
	TTFTMs       int64   `json:"ttftMs"`
	PromptTokens int     `json:"promptTokens"`
	PromptTPS    float64 `json:"promptTps"`
	GenTokens    int     `json:"genTokens"`
	GenTPS       float64 `json:"genTps"`
}

// Store keeps results in a folder.
type Store struct {
	dir string
	mu  sync.Mutex
}

// Open uses dir, creating it when the first result is saved. With dir ""
// nothing is kept.
func Open(dir string) *Store { return &Store{dir: dir} }

var unsafe = regexp.MustCompile(`[^a-zA-Z0-9]+`)

func (s *Store) path(digest string) string {
	d := strings.TrimPrefix(digest, "sha256:")
	return filepath.Join(s.dir, unsafe.ReplaceAllString(d, "_")+".json")
}

// Get returns the result for a digest, if there is one.
func (s *Store) Get(digest string) (Result, bool) {
	if digest == "" || s.dir == "" {
		return Result{}, false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	b, err := os.ReadFile(s.path(digest))
	if err != nil {
		return Result{}, false
	}
	var r Result
	if json.Unmarshal(b, &r) != nil {
		return Result{}, false
	}
	return r, true
}

// Put saves r, replacing an older result for the same digest.
func (s *Store) Put(r Result) error {
	if r.Digest == "" {
		return errors.New("bench: result has no digest")
	}
	if s.dir == "" {
		return nil
	}
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := os.MkdirAll(s.dir, 0o755); err != nil {
		return err
	}
	p := s.path(r.Digest)
	if err := os.WriteFile(p+".tmp", append(b, '\n'), 0o644); err != nil {
		return err
	}
	return os.Rename(p+".tmp", p)
}

// Prompt is the fixed text every model reads: about a thousand tokens, so
// prompt speed is measured on a realistic amount of input.
var Prompt = func() string {
	notes := []string{
		"The town council met on Tuesday to discuss the river path. Residents asked for better lighting near the old mill, more benches along the east bank and a safer crossing at the bridge. The council agreed to collect quotes for lighting before the spring meeting.",
		"The library extended its weekend hours after a survey showed that most visitors come on Saturday afternoons. It also plans a repair café on the first Sunday of each month, where volunteers fix lamps, bicycles, clothes and small appliances for free.",
		"The bakery on Market Street changed hands in March. The new owners kept the sourdough recipe, added a rye loaf and began opening an hour earlier for commuters. They hope to offer a small breakfast menu by the summer.",
		"Volunteers planted forty fruit trees in the community orchard: apples, pears, plums and two quince trees. A watering rota runs through the dry months, and the first harvest festival is planned for late September, with a cider press borrowed from the next village.",
		"The school's robotics club placed third in the regional contest. Their robot sorted coloured blocks using a cheap camera and a small neural network trained on photos the students took themselves. The club meets on Thursdays and is looking for a mentor who knows electronics.",
		"Road works on the north road will close one lane for six weeks while the drains are replaced. Buses keep their routes but may run up to ten minutes late at peak times. Cyclists are asked to use the river path, which is now lit as far as the mill.",
	}
	var b strings.Builder
	b.WriteString("Here are this month's notes from a small town newsletter.\n\n")
	for round := 0; round < 3; round++ {
		for i, n := range notes {
			b.WriteString(n)
			if i%2 == round%2 {
				b.WriteString(" More details are on the notice board outside the town hall.")
			}
			b.WriteString("\n\n")
		}
	}
	b.WriteString("Write a friendly summary of these notes for the newsletter's front page, in about two hundred words.")
	return b.String()
}()

// GenTokens caps the reply, so every test generates the same amount.
const GenTokens = 256

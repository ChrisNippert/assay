// Package evals runs your own benchmarks on local models: sets of questions
// in plain TOML files, each with a way to check the answer. Results are kept
// per model digest with every answer, and a straight-line fit through the
// models that also have a public Intelligence Index score places the others
// on that index as an estimate.
package evals

import (
	"crypto/sha256"
	"embed"
	"encoding/hex"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"sync"

	"github.com/BurntSushi/toml"
)

// Checks say how an answer is graded.
const (
	CheckNumber   = "number"   // the final answer is this number (within Tolerance)
	CheckExact    = "exact"    // the final answer is this text (case, spaces and end punctuation ignored)
	CheckChoice   = "choice"   // the final answer is this letter of Choices
	CheckContains = "contains" // the whole reply contains this text (case ignored)
	CheckRegex    = "regex"    // the whole reply matches this regular expression
)

// Checks lists every check, for the UI.
var Checks = []string{CheckNumber, CheckExact, CheckChoice, CheckContains, CheckRegex}

// Question is one item of a set.
type Question struct {
	ID     string `toml:"id" json:"id"`
	Prompt string `toml:"prompt" json:"prompt"`
	// Answer is what a right reply gives; Accept lists other right answers.
	Answer string   `toml:"answer" json:"answer"`
	Accept []string `toml:"accept,omitempty" json:"accept,omitempty"`
	// Check overrides the set's check for this question.
	Check string `toml:"check,omitempty" json:"check,omitempty"`
	// Choices are shown as A, B, C… for the choice check.
	Choices []string `toml:"choices,omitempty" json:"choices,omitempty"`
	// Tolerance is the allowed difference for the number check (default:
	// one part in ten thousand).
	Tolerance float64 `toml:"tolerance,omitempty" json:"tolerance,omitempty"`
	// Not, for regex and contains, fails a reply that matches it instead.
	Not bool `toml:"not,omitempty" json:"not,omitempty"`
}

// Set is one benchmark.
type Set struct {
	ID   string `toml:"-" json:"id"`
	Name string `toml:"name" json:"name"`
	// Short labels the set's column in tables; default: the name.
	Short       string `toml:"short,omitempty" json:"short,omitempty"`
	Description string `toml:"description,omitempty" json:"description,omitempty"`
	// Check is the default check for its questions.
	Check string `toml:"check,omitempty" json:"check,omitempty"`
	// System is an optional system message for every question.
	System    string     `toml:"system,omitempty" json:"system,omitempty"`
	Questions []Question `toml:"questions" json:"questions"`
	// BuiltIn sets come with Assay and can't be edited (copy one instead).
	BuiltIn bool `toml:"-" json:"builtIn"`
	// Hash changes whenever a question, answer or check changes, so old
	// results can be told apart from current ones.
	Hash string `toml:"-" json:"hash"`
}

// ShortName is the column label.
func (s *Set) ShortName() string {
	if s.Short != "" {
		return s.Short
	}
	return s.Name
}

// CheckOf is the check a question uses.
func (s *Set) CheckOf(q Question) string {
	if q.Check != "" {
		return q.Check
	}
	if s.Check != "" {
		return s.Check
	}
	if len(q.Choices) > 0 {
		return CheckChoice
	}
	return CheckExact
}

var idRe = regexp.MustCompile(`^[a-z0-9][a-z0-9-]{0,63}$`)

// ValidID says whether id can name a set (it becomes a file name).
func ValidID(id string) bool { return idRe.MatchString(id) }

// Validate checks a set is usable and fills in missing question ids.
func (s *Set) Validate() error {
	if !ValidID(s.ID) {
		return fmt.Errorf("the id %q should be lowercase letters, digits and dashes", s.ID)
	}
	if strings.TrimSpace(s.Name) == "" {
		return errors.New("the set needs a name")
	}
	if s.Check != "" && !slices.Contains(Checks, s.Check) {
		return fmt.Errorf("unknown check %q (use %s)", s.Check, strings.Join(Checks, ", "))
	}
	if len(s.Questions) == 0 {
		return errors.New("the set has no questions")
	}
	seen := map[string]bool{}
	for i := range s.Questions {
		q := &s.Questions[i]
		if q.ID == "" {
			q.ID = fmt.Sprint(i + 1)
		}
		n := fmt.Sprintf("question %s", q.ID)
		if seen[q.ID] {
			return fmt.Errorf("%s: the id is used twice", n)
		}
		seen[q.ID] = true
		if strings.TrimSpace(q.Prompt) == "" {
			return fmt.Errorf("%s has no prompt", n)
		}
		if q.Answer == "" {
			return fmt.Errorf("%s has no answer", n)
		}
		c := s.CheckOf(*q)
		switch c {
		case CheckNumber:
			for _, a := range append([]string{q.Answer}, q.Accept...) {
				if _, ok := parseNumber(a); !ok {
					return fmt.Errorf("%s: %q isn't a number", n, a)
				}
			}
		case CheckChoice:
			if len(q.Choices) < 2 || len(q.Choices) > 26 {
				return fmt.Errorf("%s needs 2 to 26 choices", n)
			}
			if i := letterIndex(q.Answer); i < 0 || i >= len(q.Choices) {
				return fmt.Errorf("%s: the answer should be a letter from A to %c", n, 'A'+len(q.Choices)-1)
			}
		case CheckRegex:
			if _, err := regexp.Compile(q.Answer); err != nil {
				return fmt.Errorf("%s: %v", n, err)
			}
		case CheckExact, CheckContains:
		default:
			return fmt.Errorf("%s: unknown check %q", n, c)
		}
	}
	s.Hash = s.hash()
	return nil
}

func (s *Set) hash() string {
	h := sha256.New()
	fmt.Fprintf(h, "%s\x00%s\x00", s.Check, s.System)
	for _, q := range s.Questions {
		fmt.Fprintf(h, "%s\x00%s\x00%s\x00%q\x00%s\x00%q\x00%g\x00%v\x01", q.ID, q.Prompt, q.Answer, q.Accept, q.Check, q.Choices, q.Tolerance, q.Not)
	}
	return hex.EncodeToString(h.Sum(nil))[:12]
}

// Parse reads a set from TOML.
func Parse(id string, b []byte) (*Set, error) {
	var s Set
	md, err := toml.Decode(string(b), &s)
	if err != nil {
		return nil, err
	}
	if un := md.Undecoded(); len(un) > 0 {
		return nil, fmt.Errorf("unknown field %q", un[0].String())
	}
	s.ID = id
	if err := s.Validate(); err != nil {
		return nil, err
	}
	return &s, nil
}

// Encode writes a set as TOML.
func Encode(s *Set) ([]byte, error) {
	var b strings.Builder
	if err := toml.NewEncoder(&b).Encode(s); err != nil {
		return nil, err
	}
	return []byte(b.String()), nil
}

//go:embed builtin/*.toml
var builtinFS embed.FS

// Library holds the built-in sets and the user's own, which live as
// <dir>/<id>.toml.
type Library struct {
	dir     string
	builtin []*Set
	mu      sync.Mutex
}

// OpenLibrary loads the built-in sets; the user's are read from dir on
// every List. With dir "" only the built-in ones exist.
func OpenLibrary(dir string) (*Library, error) {
	l := &Library{dir: dir}
	files, err := fs.Glob(builtinFS, "builtin/*.toml")
	if err != nil {
		return nil, err
	}
	for _, f := range files {
		b, _ := builtinFS.ReadFile(f)
		id := strings.TrimSuffix(filepath.Base(f), ".toml")
		s, err := Parse(id, b)
		if err != nil {
			return nil, fmt.Errorf("built-in set %s: %w", id, err)
		}
		s.BuiltIn = true
		l.builtin = append(l.builtin, s)
	}
	return l, nil
}

// List returns every set, built-in first. Damaged user files are reported
// in errs and left out.
func (l *Library) List() (sets []*Set, errs map[string]string) {
	sets = append(sets, l.builtin...)
	if l.dir == "" {
		return sets, nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	entries, _ := os.ReadDir(l.dir)
	for _, e := range entries {
		id, ok := strings.CutSuffix(e.Name(), ".toml")
		if !ok || e.IsDir() || !ValidID(id) || l.isBuiltin(id) {
			continue
		}
		b, err := os.ReadFile(filepath.Join(l.dir, e.Name()))
		var s *Set
		if err == nil {
			s, err = Parse(id, b)
		}
		if err != nil {
			if errs == nil {
				errs = map[string]string{}
			}
			errs[id] = err.Error()
			continue
		}
		sets = append(sets, s)
	}
	return sets, errs
}

func (l *Library) isBuiltin(id string) bool {
	return slices.ContainsFunc(l.builtin, func(s *Set) bool { return s.ID == id })
}

// Get finds one set.
func (l *Library) Get(id string) (*Set, error) {
	for _, s := range l.builtin {
		if s.ID == id {
			return s, nil
		}
	}
	if l.dir == "" || !ValidID(id) {
		return nil, fs.ErrNotExist
	}
	l.mu.Lock()
	b, err := os.ReadFile(filepath.Join(l.dir, id+".toml"))
	l.mu.Unlock()
	if err != nil {
		return nil, err
	}
	return Parse(id, b)
}

// Raw returns a set's TOML as written (built-in or the user's file).
func (l *Library) Raw(id string) ([]byte, error) {
	if l.isBuiltin(id) {
		return builtinFS.ReadFile("builtin/" + id + ".toml")
	}
	if l.dir == "" || !ValidID(id) {
		return nil, fs.ErrNotExist
	}
	return os.ReadFile(filepath.Join(l.dir, id+".toml"))
}

// Save checks the TOML and writes it as the user's set id.
func (l *Library) Save(id string, raw []byte) (*Set, error) {
	if l.isBuiltin(id) {
		return nil, errors.New("built-in sets can't be changed; save it under another id")
	}
	if l.dir == "" {
		return nil, errors.New("no folder to save sets in")
	}
	s, err := Parse(id, raw)
	if err != nil {
		return nil, err
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if err := os.MkdirAll(l.dir, 0o755); err != nil {
		return nil, err
	}
	p := filepath.Join(l.dir, id+".toml")
	if err := os.WriteFile(p+".tmp", raw, 0o644); err != nil {
		return nil, err
	}
	return s, os.Rename(p+".tmp", p)
}

// Delete removes one of the user's sets.
func (l *Library) Delete(id string) error {
	if l.isBuiltin(id) {
		return errors.New("built-in sets can't be deleted")
	}
	if l.dir == "" || !ValidID(id) {
		return fs.ErrNotExist
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	return os.Remove(filepath.Join(l.dir, id+".toml"))
}

package evals

import (
	"cmp"
	"fmt"
	"hash/fnv"
	"math"
	"math/rand/v2"
	"strconv"
)

// Generated sets are built-in sets whose questions are written by code:
// each Generator picks random numbers, words the question and computes the
// answer, so the exact problem can't be in any model's training data. The
// seed is fixed, so a set is the same on every run and machine and its
// results compare; changing the seed (or a generator) changes the set's
// hash, which marks old results as being from another version.

// Generator writes one kind of question.
type Generator struct {
	// ID names the kind; questions are <ID>-1, <ID>-2, …
	ID string
	// Make words one question and its answer from r. It sets Prompt and
	// Answer (and Tolerance if the default relative one doesn't fit); it
	// may return ok false to ask for another draw (say, the numbers made a
	// degenerate problem).
	Make func(r *rand.Rand) (q Question, ok bool)
}

// GenSpec describes one generated set.
type GenSpec struct {
	ID, Name, Short, Description, System string
	// Check is how answers are graded (default number).
	Check string
	// Seed picks the questions; bump it to write a fresh set.
	Seed uint64
	// PerGen is how many questions each generator writes.
	PerGen int
	// RelTol is the default allowed relative error for number answers
	// (e.g. 0.005 for half a percent).
	RelTol float64
	// MaxTokens raises the reply limit for this set when the run uses the
	// default limit (hard questions need long reasoning).
	MaxTokens int
	Gens      []Generator
}

var genSpecs []*GenSpec

// registerGen adds a generated set; call it from an init function.
func registerGen(g *GenSpec) { genSpecs = append(genSpecs, g) }

// Build writes the set's questions.
func (g *GenSpec) Build() (*Set, error) {
	s := &Set{ID: g.ID, Name: g.Name, Short: g.Short, Description: g.Description, System: g.System, Check: cmp.Or(g.Check, CheckNumber), MaxTokens: g.MaxTokens, BuiltIn: true, Generated: true}
	for _, gen := range g.Gens {
		h := fnv.New64a()
		h.Write([]byte(gen.ID))
		r := rand.New(rand.NewPCG(g.Seed, h.Sum64()))
		for k := 1; k <= g.PerGen; k++ {
			var q Question
			ok := false
			for try := 0; try < 1000 && !ok; try++ {
				q, ok = gen.Make(r)
			}
			if !ok {
				return nil, fmt.Errorf("generator %s: no usable draw", gen.ID)
			}
			q.ID = fmt.Sprintf("%s-%d", gen.ID, k)
			if q.Tolerance == 0 && g.RelTol > 0 {
				if want, okN := parseNumber(q.Answer); okN {
					q.Tolerance = math.Max(math.Abs(want)*g.RelTol, 1e-12)
				}
			}
			s.Questions = append(s.Questions, q)
		}
	}
	if err := s.Validate(); err != nil {
		return nil, fmt.Errorf("generated set %s: %w", g.ID, err)
	}
	return s, nil
}

// Helpers for generators.

// num writes a computed answer with enough digits for any sane tolerance.
func num(x float64) string { return strconv.FormatFloat(x, 'g', 10, 64) }

// pick returns a random element.
func pick[T any](r *rand.Rand, xs ...T) T { return xs[r.IntN(len(xs))] }

// between returns a random integer in [lo, hi].
func between(r *rand.Rand, lo, hi int) int { return lo + r.IntN(hi-lo+1) }

// decimal returns a random number in [lo, hi] rounded to step (e.g. 0.1),
// so prompts show tidy values.
func decimal(r *rand.Rand, lo, hi, step float64) float64 {
	n := int(math.Round((hi - lo) / step))
	v := lo + float64(r.IntN(n+1))*step
	d := max(0, int(math.Ceil(-math.Log10(step)-1e-9)))
	v, _ = strconv.ParseFloat(strconv.FormatFloat(v, 'f', d, 64), 64)
	return v
}

// show writes a value for a prompt without float noise (2.3, not 2.2999…).
func show(x float64) string { return strconv.FormatFloat(x, 'f', -1, 64) }

package remote

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

// Spend keeps what API models cost, per day and model, in a JSON file
// ("" keeps it in memory). Every API call adds to it: chats, benchmarks,
// teams and speed tests.
type Spend struct {
	path string
	mu   sync.Mutex
	// days is "2026-11-01" → model name → tally.
	days map[string]map[string]*Tally
}

// Tally is what one model cost over some time.
type Tally struct {
	Cost   float64 `json:"cost"`
	Calls  int     `json:"calls"`
	Tokens int     `json:"tokens"`
}

func (t *Tally) add(o Tally) {
	t.Cost += o.Cost
	t.Calls += o.Calls
	t.Tokens += o.Tokens
}

// OpenSpend reads path; a missing file is no spending yet.
func OpenSpend(path string) (*Spend, error) {
	sp := &Spend{path: path, days: map[string]map[string]*Tally{}}
	if path == "" {
		return sp, nil
	}
	b, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return sp, nil
	}
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(b, &sp.days); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return sp, nil
}

// Add records one call of model (its benchmark name, "api:…").
func (sp *Spend) Add(model string, cost float64, tokens int) {
	if sp == nil {
		return
	}
	day := time.Now().Format(time.DateOnly)
	sp.mu.Lock()
	defer sp.mu.Unlock()
	if sp.days[day] == nil {
		sp.days[day] = map[string]*Tally{}
	}
	t := sp.days[day][model]
	if t == nil {
		t = &Tally{}
		sp.days[day][model] = t
	}
	t.add(Tally{Cost: cost, Calls: 1, Tokens: tokens})
	if sp.path == "" {
		return
	}
	b, _ := json.Marshal(sp.days)
	os.MkdirAll(filepath.Dir(sp.path), 0o755)
	tmp := sp.path + ".tmp"
	if os.WriteFile(tmp, b, 0o600) == nil {
		os.Rename(tmp, sp.path)
	}
}

// Summary is spending today, over the last 7 and 30 days (today
// included) and in all, per model and in total.
type Summary struct {
	Total  Periods            `json:"total"`
	Models map[string]Periods `json:"models"`
	// Since is the first day with any spending.
	Since string `json:"since,omitempty"`
}

// Periods is one model's (or everything's) spending over the periods.
type Periods struct {
	Today Tally `json:"today"`
	Week  Tally `json:"week"`
	Month Tally `json:"month"`
	All   Tally `json:"all"`
}

func (p *Periods) add(age int, t Tally) {
	if age == 0 {
		p.Today.add(t)
	}
	if age < 7 {
		p.Week.add(t)
	}
	if age < 30 {
		p.Month.add(t)
	}
	p.All.add(t)
}

// Summary adds the days up as of now.
func (sp *Spend) Summary(now time.Time) Summary {
	out := Summary{Models: map[string]Periods{}}
	if sp == nil {
		return out
	}
	today, _ := time.Parse(time.DateOnly, now.Format(time.DateOnly))
	sp.mu.Lock()
	defer sp.mu.Unlock()
	days := make([]string, 0, len(sp.days))
	for d := range sp.days {
		days = append(days, d)
	}
	sort.Strings(days)
	if len(days) > 0 {
		out.Since = days[0]
	}
	for _, d := range days {
		at, err := time.Parse(time.DateOnly, d)
		if err != nil {
			continue
		}
		age := int(today.Sub(at).Hours() / 24)
		for model, t := range sp.days[d] {
			p := out.Models[model]
			p.add(age, *t)
			out.Models[model] = p
			out.Total.add(age, *t)
		}
	}
	return out
}

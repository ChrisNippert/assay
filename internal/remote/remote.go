// Package remote keeps API models (OpenRouter or any OpenAI-compatible
// endpoint) that the benchmarks can run alongside local ones, and talks to
// them.
//
// An Endpoint is a base URL and key. A Model is one way of running a model
// there: the model id plus an optional system prompt and reasoning effort,
// so the same API model with two prompts is two Models with separate
// results.
package remote

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"sync"
)

// OpenRouterURL is the default endpoint.
const OpenRouterURL = "https://openrouter.ai/api/v1"

// Endpoint is an OpenAI-compatible API.
type Endpoint struct {
	ID   string `json:"id"`
	Name string `json:"name"`
	URL  string `json:"url"`
	Key  string `json:"key,omitempty"`
}

// OpenRouter says the endpoint is OpenRouter, which takes its own
// reasoning and usage options.
func (e Endpoint) OpenRouter() bool { return strings.Contains(e.URL, "openrouter.ai") }

// Model is one way of running an API model.
type Model struct {
	ID       string `json:"id"`
	Endpoint string `json:"endpoint"`
	// Model is the id the API knows it by, e.g. "openai/gpt-5-mini".
	Model string `json:"model"`
	Title string `json:"title"`
	// System goes before each set's own system prompt.
	System string `json:"system,omitempty"`
	// Reasoning is "", "off", "low", "medium" or "high"; "" leaves it to
	// the API.
	Reasoning string `json:"reasoning,omitempty"`
}

// Name is how the benchmarks name the model ("api:" + ID).
func (m Model) Name() string { return Prefix + m.ID }

// Prefix marks API model names.
const Prefix = "api:"

// Digest keys the model's results: it changes when anything that changes
// the answers does, so an edited prompt starts fresh results.
func (m Model) Digest(ep Endpoint) string {
	h := sha256.Sum256([]byte(strings.Join([]string{strings.TrimRight(ep.URL, "/"), m.Model, m.System, m.Reasoning}, "\x00")))
	return "api-" + hex.EncodeToString(h[:12])
}

var idRe = regexp.MustCompile(`^[a-z0-9][a-z0-9-]{0,63}$`)

var reasonings = []string{"", "off", "low", "medium", "high"}

func (e Endpoint) validate() error {
	if !idRe.MatchString(e.ID) {
		return errors.New("endpoint id: lowercase letters, digits and dashes")
	}
	if strings.TrimSpace(e.Name) == "" {
		return errors.New("give the endpoint a name")
	}
	u, err := url.Parse(e.URL)
	if err != nil || (u.Scheme != "https" && u.Scheme != "http") || u.Host == "" {
		return errors.New("the URL should look like https://openrouter.ai/api/v1")
	}
	return nil
}

func (m Model) validate() error {
	if !idRe.MatchString(m.ID) {
		return errors.New("model id: lowercase letters, digits and dashes")
	}
	if strings.TrimSpace(m.Model) == "" {
		return errors.New("which model? e.g. openai/gpt-5-mini")
	}
	if !slices.Contains(reasonings, m.Reasoning) {
		return errors.New(`reasoning should be "", off, low, medium or high`)
	}
	return nil
}

// Team is an orchestrator–worker setup the benchmarks run as one entry:
// the Lead plans and answers, Workers (up to Size at a time) do the parts.
// Lead and Worker are benchmark model names, local or "api:…".
type Team struct {
	ID       string `json:"id"`
	Title    string `json:"title"`
	Strategy string `json:"strategy"` // Pipeline or Loop
	Lead     string `json:"lead"`
	Worker   string `json:"worker"`
	// Size is the most workers per round (1–8); Rounds the most rounds of
	// delegation in a Loop (1–5). An Agent has neither.
	Size   int `json:"size,omitempty"`
	Rounds int `json:"rounds,omitempty"`
	// An Agent's rails: tokens for one answer across every call, the most
	// it may cost in US dollars (0: no cap) and minutes.
	Budget  int     `json:"budget,omitempty"`
	MaxCost float64 `json:"maxCost,omitempty"`
	Minutes int     `json:"minutes,omitempty"`
	// Notes are extra instructions for the lead.
	Notes string `json:"notes,omitempty"`
}

// Team strategies.
const (
	// Pipeline: the lead splits the task, the workers each do a part, the
	// lead writes the answer from their results. Always three steps.
	Pipeline = "pipeline"
	// Loop: the lead hands out tasks when it wants, reads the results and
	// may hand out more, for up to Rounds rounds, then answers.
	Loop = "loop"
	// Agent: the lead spawns as many helpers as often as it likes (a
	// spawn_agents tool), within a budget of tokens, cost and time.
	Agent = "agent"
)

// TeamPrefix marks team names.
const TeamPrefix = "team:"

// Name is how the benchmarks name the team.
func (t Team) Name() string { return TeamPrefix + t.ID }

func (t Team) validate() error {
	if !idRe.MatchString(t.ID) {
		return errors.New("team id: lowercase letters, digits and dashes")
	}
	if t.Strategy != Pipeline && t.Strategy != Loop && t.Strategy != Agent {
		return errors.New("strategy should be pipeline, loop or agent")
	}
	if t.Lead == "" || t.Worker == "" {
		return errors.New("pick a lead and a worker model")
	}
	if strings.HasPrefix(t.Lead, TeamPrefix) || strings.HasPrefix(t.Worker, TeamPrefix) {
		return errors.New("a team can't be a member of a team")
	}
	if t.Strategy == Agent {
		if t.Budget < 1000 || t.Budget > 5_000_000 {
			return errors.New("token budget: 1,000 to 5,000,000")
		}
		if !(t.MaxCost >= 0 && t.MaxCost <= 1000) {
			return errors.New("cost cap: $0 to $1,000 (0 for none)")
		}
		if t.Minutes < 1 || t.Minutes > 240 {
			return errors.New("time limit: 1 to 240 minutes")
		}
		return nil
	}
	if t.Size < 1 || t.Size > 8 {
		return errors.New("workers per round: 1 to 8")
	}
	if t.Strategy == Loop && (t.Rounds < 1 || t.Rounds > 5) {
		return errors.New("rounds: 1 to 5")
	}
	return nil
}

type file struct {
	Endpoints []Endpoint `json:"endpoints"`
	Models    []Model    `json:"models"`
	Teams     []Team     `json:"teams,omitempty"`
}

// Store keeps endpoints and models in a JSON file readable only by the
// user, since it holds API keys ("" keeps them in memory).
type Store struct {
	path string
	mu   sync.Mutex
	f    file
}

// Open reads path; a missing file means nothing set up yet.
func Open(path string) (*Store, error) {
	st := &Store{path: path}
	if path == "" {
		return st, nil
	}
	b, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return st, nil
	}
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(b, &st.f); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return st, nil
}

func (st *Store) save() error {
	if st.path == "" {
		return nil
	}
	b, _ := json.MarshalIndent(st.f, "", "  ")
	if err := os.MkdirAll(filepath.Dir(st.path), 0o755); err != nil {
		return err
	}
	tmp := st.path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, st.path)
}

// Endpoints returns the endpoints, keys included.
func (st *Store) Endpoints() []Endpoint {
	if st == nil {
		return nil
	}
	st.mu.Lock()
	defer st.mu.Unlock()
	return slices.Clone(st.f.Endpoints)
}

// Models returns the models.
func (st *Store) Models() []Model {
	if st == nil {
		return nil
	}
	st.mu.Lock()
	defer st.mu.Unlock()
	return slices.Clone(st.f.Models)
}

// Endpoint finds an endpoint by id.
func (st *Store) Endpoint(id string) (Endpoint, bool) {
	for _, e := range st.Endpoints() {
		if e.ID == id {
			return e, true
		}
	}
	return Endpoint{}, false
}

// Lookup finds a model by its benchmark name ("api:id") with its endpoint.
func (st *Store) Lookup(name string) (Model, Endpoint, bool) {
	id, ok := strings.CutPrefix(name, Prefix)
	if !ok {
		return Model{}, Endpoint{}, false
	}
	for _, m := range st.Models() {
		if m.ID == id {
			ep, ok := st.Endpoint(m.Endpoint)
			return m, ep, ok
		}
	}
	return Model{}, Endpoint{}, false
}

// PutEndpoint adds or replaces an endpoint; an empty Key keeps the saved
// one if the URL is unchanged.
func (st *Store) PutEndpoint(e Endpoint) error {
	e.URL = strings.TrimRight(strings.TrimSpace(e.URL), "/")
	e.Key = strings.TrimSpace(e.Key)
	if err := e.validate(); err != nil {
		return err
	}
	st.mu.Lock()
	defer st.mu.Unlock()
	if i := slices.IndexFunc(st.f.Endpoints, func(x Endpoint) bool { return x.ID == e.ID }); i >= 0 {
		// The saved key stays only with the URL it was given for, so it
		// can't be sent somewhere else without typing it again.
		if e.Key == "" && e.URL == st.f.Endpoints[i].URL {
			e.Key = st.f.Endpoints[i].Key
		}
		st.f.Endpoints[i] = e
	} else {
		st.f.Endpoints = append(st.f.Endpoints, e)
	}
	return st.save()
}

// DeleteEndpoint removes an endpoint and its models.
func (st *Store) DeleteEndpoint(id string) error {
	st.mu.Lock()
	defer st.mu.Unlock()
	st.f.Endpoints = slices.DeleteFunc(st.f.Endpoints, func(x Endpoint) bool { return x.ID == id })
	st.f.Models = slices.DeleteFunc(st.f.Models, func(x Model) bool { return x.Endpoint == id })
	return st.save()
}

// PutModel adds or replaces a model.
func (st *Store) PutModel(m Model) error {
	m.Model, m.Title = strings.TrimSpace(m.Model), strings.TrimSpace(m.Title)
	m.System = strings.TrimSpace(m.System)
	if m.Title == "" {
		m.Title = m.Model
	}
	if err := m.validate(); err != nil {
		return err
	}
	if _, ok := st.Endpoint(m.Endpoint); !ok {
		return errors.New("no such endpoint: " + m.Endpoint)
	}
	st.mu.Lock()
	defer st.mu.Unlock()
	if i := slices.IndexFunc(st.f.Models, func(x Model) bool { return x.ID == m.ID }); i >= 0 {
		st.f.Models[i] = m
	} else {
		st.f.Models = append(st.f.Models, m)
	}
	return st.save()
}

// DeleteModel removes a model (its results stay on disk).
func (st *Store) DeleteModel(id string) error {
	st.mu.Lock()
	defer st.mu.Unlock()
	st.f.Models = slices.DeleteFunc(st.f.Models, func(x Model) bool { return x.ID == id })
	return st.save()
}

// Slug makes an id from a name, unique among taken.
func Slug(name string, taken func(string) bool) string {
	s := strings.Trim(regexp.MustCompile(`[^a-z0-9]+`).ReplaceAllString(strings.ToLower(name), "-"), "-")
	if len(s) > 48 {
		s = strings.Trim(s[:48], "-")
	}
	if s == "" {
		s = "model"
	}
	id := s
	for n := 2; taken(id); n++ {
		id = fmt.Sprintf("%s-%d", s, n)
	}
	return id
}

// Teams returns the teams.
func (st *Store) Teams() []Team {
	if st == nil {
		return nil
	}
	st.mu.Lock()
	defer st.mu.Unlock()
	return slices.Clone(st.f.Teams)
}

// Team finds a team by its benchmark name ("team:id").
func (st *Store) Team(name string) (Team, bool) {
	id, ok := strings.CutPrefix(name, TeamPrefix)
	if !ok {
		return Team{}, false
	}
	for _, t := range st.Teams() {
		if t.ID == id {
			return t, true
		}
	}
	return Team{}, false
}

// PutTeam adds or replaces a team.
func (st *Store) PutTeam(t Team) error {
	t.Title, t.Notes = strings.TrimSpace(t.Title), strings.TrimSpace(t.Notes)
	switch t.Strategy {
	case Pipeline:
		t.Rounds = 0
	case Agent:
		t.Size, t.Rounds = 0, 0
	}
	if t.Strategy != Agent {
		t.Budget, t.MaxCost, t.Minutes = 0, 0, 0
	}
	if err := t.validate(); err != nil {
		return err
	}
	if t.Title == "" {
		return errors.New("give the team a name")
	}
	st.mu.Lock()
	defer st.mu.Unlock()
	if i := slices.IndexFunc(st.f.Teams, func(x Team) bool { return x.ID == t.ID }); i >= 0 {
		st.f.Teams[i] = t
	} else {
		st.f.Teams = append(st.f.Teams, t)
	}
	return st.save()
}

// DeleteTeam removes a team (its results stay on disk).
func (st *Store) DeleteTeam(id string) error {
	st.mu.Lock()
	defer st.mu.Unlock()
	st.f.Teams = slices.DeleteFunc(st.f.Teams, func(x Team) bool { return x.ID == id })
	return st.save()
}

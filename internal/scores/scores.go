// Package scores looks up Artificial Analysis Intelligence Index scores for
// models. The data comes from OpenRouter's keyless model list, which embeds
// the index and (for open-weight models) the Hugging Face repo id. It is
// cached on disk, refreshed at most daily, and everything works without it.
package scores

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// DefaultURL is OpenRouter's public model list.
const DefaultURL = "https://openrouter.ai/api/v1/models"

// AttributionURL is where the scores come from; show a link to it wherever
// a score is shown.
const AttributionURL = "https://artificialanalysis.ai"

// MaxAge is how old the cache may get before a refresh is attempted.
const MaxAge = 24 * time.Hour

// Entry is one rated model.
type Entry struct {
	Score float64 `json:"score"`
	Model string  `json:"model"` // lowercase "org/name" the score belongs to
}

type snapshot struct {
	FetchedAt time.Time        `json:"fetchedAt"`
	ByID      map[string]Entry `json:"byId"`   // lowercase HF "org/name"
	ByName    map[string]Entry `json:"byName"` // lowercase name alone, only when unambiguous
}

// Store holds the scores and keeps them fresh.
type Store struct {
	url  string
	path string
	http *http.Client
	log  *slog.Logger

	mu   sync.RWMutex
	snap snapshot
}

// Open loads the cache at path (missing or damaged is fine) and returns a
// store that fetches from url when asked to refresh.
func Open(url, path string, log *slog.Logger) *Store {
	if log == nil {
		log = slog.Default()
	}
	s := &Store{url: url, path: path, http: &http.Client{Timeout: 30 * time.Second}, log: log}
	if b, err := os.ReadFile(path); err == nil {
		var snap snapshot
		if json.Unmarshal(b, &snap) == nil {
			s.snap = snap
		}
	}
	return s
}

// Lookup finds the score for a Hugging Face repo id such as "Qwen/Qwen3-8B".
// It tries the full id first, then the name alone if only one model has it.
func (s *Store) Lookup(repoID string) (Entry, bool) {
	if s == nil {
		return Entry{}, false
	}
	id := strings.ToLower(repoID)
	s.mu.RLock()
	defer s.mu.RUnlock()
	if e, ok := s.snap.ByID[id]; ok {
		return e, true
	}
	if _, name, ok := strings.Cut(id, "/"); ok {
		if e, ok := s.snap.ByName[name]; ok {
			return e, true
		}
	}
	return Entry{}, false
}

// Count is how many models have a score.
func (s *Store) Count() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.snap.ByID) + len(s.snap.ByName)
}

// Max is the highest score known, for scaling bars; 0 with no data.
func (s *Store) Max() float64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	var m float64
	for _, e := range s.snap.ByID {
		m = max(m, e.Score)
	}
	for _, e := range s.snap.ByName {
		m = max(m, e.Score)
	}
	return m
}

// Stale reports whether the data is missing or older than MaxAge.
func (s *Store) Stale() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return time.Since(s.snap.FetchedAt) > MaxAge
}

// Run refreshes now if the data is stale, then re-checks every hour until
// ctx ends. Failures are logged and the old data keeps being served.
func (s *Store) Run(ctx context.Context) {
	for {
		if s.Stale() {
			if err := s.Refresh(ctx); err != nil {
				s.log.Warn("couldn't refresh intelligence scores", "err", err)
			}
		}
		select {
		case <-ctx.Done():
			return
		case <-time.After(time.Hour):
		}
	}
}

type feed struct {
	Data []struct {
		ID         string `json:"id"`
		HFID       string `json:"hugging_face_id"`
		Benchmarks *struct {
			AA *struct {
				Intelligence *float64 `json:"intelligence_index"`
			} `json:"artificial_analysis"`
		} `json:"benchmarks"`
	} `json:"data"`
}

// Refresh downloads the feed and replaces the data and cache file. On error
// nothing changes.
func (s *Store) Refresh(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, s.url, nil)
	if err != nil {
		return err
	}
	res, err := s.http.Do(req)
	if err != nil {
		return err
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		return fmt.Errorf("score feed: HTTP %d", res.StatusCode)
	}
	body, err := io.ReadAll(io.LimitReader(res.Body, 64<<20))
	if err != nil {
		return err
	}
	snap, err := parse(body)
	if err != nil {
		return err
	}
	snap.FetchedAt = time.Now()
	if err := s.save(snap); err != nil {
		s.log.Warn("couldn't save score cache", "err", err)
	}
	s.mu.Lock()
	s.snap = snap
	s.mu.Unlock()
	return nil
}

func (s *Store) save(snap snapshot) error {
	b, err := json.Marshal(snap)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(s.path), 0o755); err != nil {
		return err
	}
	tmp := s.path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, s.path)
}

// parse builds the lookup tables from a feed. Models are indexed by their
// Hugging Face id when they have one, and by bare name (from the HF id and
// from OpenRouter's own id) as a fallback; a bare name used by models with
// different scores is dropped rather than guessed.
func parse(body []byte) (snapshot, error) {
	var f feed
	if err := json.Unmarshal(body, &f); err != nil {
		return snapshot{}, fmt.Errorf("score feed: %w", err)
	}
	snap := snapshot{ByID: map[string]Entry{}, ByName: map[string]Entry{}}
	ambiguous := map[string]bool{}
	addName := func(name string, e Entry) {
		if name == "" || ambiguous[name] {
			return
		}
		if old, ok := snap.ByName[name]; ok && old.Score != e.Score {
			delete(snap.ByName, name)
			ambiguous[name] = true
			return
		}
		snap.ByName[name] = e
	}
	for _, m := range f.Data {
		if m.Benchmarks == nil || m.Benchmarks.AA == nil || m.Benchmarks.AA.Intelligence == nil {
			continue
		}
		score := *m.Benchmarks.AA.Intelligence
		if score <= 0 {
			continue
		}
		orID, _, _ := strings.Cut(strings.ToLower(m.ID), ":") // drop ":free" and friends
		if m.HFID != "" {
			hf := strings.ToLower(m.HFID)
			e := Entry{Score: score, Model: hf}
			snap.ByID[hf] = e
			addName(afterSlash(hf), e)
		}
		if orID != "" {
			addName(afterSlash(orID), Entry{Score: score, Model: orID})
		}
	}
	return snap, nil
}

func afterSlash(id string) string {
	_, name, ok := strings.Cut(id, "/")
	if !ok {
		return ""
	}
	return name
}

// BaseModel picks the repo a quantized repo was made from out of its Hugging
// Face tags ("base_model:quantized:org/name"). Finetunes, merges and
// adapters return "" because they don't inherit the base model's score.
func BaseModel(tags []string) string {
	for _, t := range tags {
		if rest, ok := strings.CutPrefix(t, "base_model:quantized:"); ok && strings.Contains(rest, "/") {
			return rest
		}
	}
	return ""
}

// All lists every model that has a Hugging Face id, best first (for showing
// where a score sits among similar models).
func (s *Store) All() []Entry {
	if s == nil {
		return nil
	}
	s.mu.RLock()
	out := make([]Entry, 0, len(s.snap.ByID))
	for _, e := range s.snap.ByID {
		out = append(out, e)
	}
	s.mu.RUnlock()
	sort.Slice(out, func(i, j int) bool {
		if out[i].Score != out[j].Score {
			return out[i].Score > out[j].Score
		}
		return out[i].Model < out[j].Model
	})
	return out
}

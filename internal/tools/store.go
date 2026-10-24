package tools

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sync"
)

// Store keeps Settings in a JSON file ("" keeps them in memory).
type Store struct {
	path string
	mu   sync.Mutex
	s    Settings
}

// OpenStore reads path, or starts from the defaults (web search off, the
// other tools on).
func OpenStore(path string) (*Store, error) {
	st := &Store{path: path, s: Settings{Search: SearchOff}}
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
	if err := json.Unmarshal(b, &st.s); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if st.s.Validate() != nil {
		st.s = Settings{Search: SearchOff}
	}
	return st, nil
}

// Get returns the current settings.
func (st *Store) Get() Settings {
	st.mu.Lock()
	defer st.mu.Unlock()
	s := st.s
	s.Off = slices.Clone(s.Off)
	return s
}

// Set validates and saves s.
func (st *Store) Set(s Settings) error {
	if err := s.Validate(); err != nil {
		return err
	}
	if s.Search == "" {
		s.Search = SearchOff
	}
	slices.Sort(s.Off)
	s.Off = slices.Compact(s.Off)
	st.mu.Lock()
	defer st.mu.Unlock()
	if st.path != "" {
		b, _ := json.MarshalIndent(s, "", "  ")
		if err := os.MkdirAll(filepath.Dir(st.path), 0o755); err != nil {
			return err
		}
		if err := os.WriteFile(st.path+".tmp", append(b, '\n'), 0o644); err != nil {
			return err
		}
		if err := os.Rename(st.path+".tmp", st.path); err != nil {
			return err
		}
	}
	st.s = s
	return nil
}

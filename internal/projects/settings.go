package projects

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// Settings are RAG's preferences, shared by every project.
type Settings struct {
	// EmbedModel is the Ollama embedding model files are indexed with; ""
	// until someone picks one. Changing it means re-indexing.
	EmbedModel string `json:"embedModel"`
	// TopK is how many excerpts go into the prompt.
	TopK int `json:"topK"`
}

// DefaultSettings are used until something is saved.
func DefaultSettings() Settings { return Settings{TopK: 6} }

// Validate checks values a browser could send wrong.
func (s Settings) Validate() error {
	switch {
	case s.TopK < 1 || s.TopK > 20:
		return errors.New("excerpts must be between 1 and 20")
	case len(s.EmbedModel) > 200:
		return errors.New("model name too long")
	}
	return nil
}

// SettingsStore keeps Settings in a JSON file ("" keeps them in memory).
type SettingsStore struct {
	path string
	mu   sync.Mutex
	s    Settings
}

// OpenSettings reads path, or starts from the defaults.
func OpenSettings(path string) (*SettingsStore, error) {
	st := &SettingsStore{path: path, s: DefaultSettings()}
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
		st.s = DefaultSettings()
	}
	return st, nil
}

// Get returns the current settings.
func (st *SettingsStore) Get() Settings {
	st.mu.Lock()
	defer st.mu.Unlock()
	return st.s
}

// Set validates and saves s.
func (st *SettingsStore) Set(s Settings) error {
	if err := s.Validate(); err != nil {
		return err
	}
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

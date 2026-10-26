package chats

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// Settings are the preferences for chats, shared by every chat.
type Settings struct {
	// CompressAuto is the default for "Compress automatically" where neither
	// the chat nor its preset says otherwise.
	CompressAuto bool `json:"compressAuto"`
	// DefaultModel is the starred model: a new chat starts on it when no
	// model is loaded.
	DefaultModel string `json:"defaultModel,omitempty"`
}

// DefaultSettings are used until something is saved.
func DefaultSettings() Settings { return Settings{CompressAuto: true} }

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
	if os.IsNotExist(err) {
		return st, nil
	}
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(b, &st.s); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return st, nil
}

// Get returns the current settings.
func (st *SettingsStore) Get() Settings {
	st.mu.Lock()
	defer st.mu.Unlock()
	return st.s
}

// Set saves s.
func (st *SettingsStore) Set(s Settings) error {
	st.mu.Lock()
	defer st.mu.Unlock()
	if st.path != "" {
		b, _ := json.MarshalIndent(s, "", "  ")
		if err := os.MkdirAll(filepath.Dir(st.path), 0o755); err != nil {
			return err
		}
		if err := os.WriteFile(st.path, b, 0o644); err != nil {
			return err
		}
	}
	st.s = s
	return nil
}

package voice

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sync"
)

// Settings are voice mode's preferences, edited on the Settings page and
// kept in one JSON file so every device sees the same.
type Settings struct {
	// Talker is the model that talks while the chat model thinks: "" picks
	// one automatically, "off" turns it off.
	Talker string `json:"talker"`
	// Voice and Speed go to the speech service; "" and 0 mean its default.
	Voice string  `json:"voice"`
	Speed float64 `json:"speed"`
	// STTModel is the speech-recognition model, e.g. "small.en"; "" is the
	// service's default.
	STTModel string `json:"sttModel"`
	// PauseMs is the silence after a spoken sentence.
	PauseMs int `json:"pauseMs"`
	// Sensitivity is how readily speech is detected: "low", "normal", "high".
	Sensitivity string `json:"sensitivity"`
	// Prose asks spoken replies for plain prose (no markdown, LaTeX, code).
	Prose bool `json:"prose"`
}

// DefaultSettings are used until something is saved.
func DefaultSettings() Settings {
	return Settings{PauseMs: 450, Sensitivity: "normal", Prose: true}
}

// Sensitivities are the allowed Sensitivity values, least first.
var Sensitivities = []string{"low", "normal", "high"}

// Validate checks values a browser could send wrong.
func (s Settings) Validate() error {
	switch {
	case s.Speed != 0 && (s.Speed < 0.5 || s.Speed > 2):
		return errors.New("speed must be between 0.5 and 2")
	case s.PauseMs < 0 || s.PauseMs > 3000:
		return errors.New("pause must be between 0 and 3000 ms")
	case !slices.Contains(Sensitivities, s.Sensitivity):
		return fmt.Errorf("sensitivity must be one of %v", Sensitivities)
	case len(s.Talker) > 200 || len(s.Voice) > 100 || len(s.STTModel) > 100:
		return errors.New("value too long")
	}
	return nil
}

// SettingsStore keeps Settings in a file.
type SettingsStore struct {
	path string
	mu   sync.Mutex
	s    Settings
}

// OpenSettings reads path, or starts from the defaults if it doesn't exist.
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

// Set validates and saves s. A store opened with no path keeps it in memory.
func (st *SettingsStore) Set(s Settings) error {
	if err := s.Validate(); err != nil {
		return err
	}
	if st.path == "" {
		st.mu.Lock()
		st.s = s
		st.mu.Unlock()
		return nil
	}
	b, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
	st.mu.Lock()
	defer st.mu.Unlock()
	if err := os.MkdirAll(filepath.Dir(st.path), 0o755); err != nil {
		return err
	}
	tmp := st.path + ".tmp"
	if err := os.WriteFile(tmp, append(b, '\n'), 0o644); err != nil {
		return err
	}
	if err := os.Rename(tmp, st.path); err != nil {
		return err
	}
	st.s = s
	return nil
}

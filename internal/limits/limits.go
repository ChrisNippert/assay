// Package limits keeps the app-wide resource limits for models.
package limits

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// Limits apply from the next model load.
type Limits struct {
	// MaxVRAMGB caps the GPU memory Assay's models use together; 0 is no limit.
	MaxVRAMGB float64 `json:"maxVramGb"`
	// Gentle makes llama-server yield the GPU and CPU to other apps.
	Gentle bool `json:"gentle"`
}

// Validate rejects limits that make no sense.
func (l Limits) Validate() error {
	if !(l.MaxVRAMGB >= 0 && l.MaxVRAMGB <= 4096) {
		return errors.New("max VRAM must be between 0 and 4096 GB")
	}
	if l.MaxVRAMGB > 0 && l.MaxVRAMGB < 0.5 {
		return errors.New("max VRAM must be at least 0.5 GB")
	}
	return nil
}

// Store keeps Limits in a JSON file ("" keeps them in memory).
type Store struct {
	path string
	mu   sync.Mutex
	l    Limits
}

// Open reads path; a missing file means no limits.
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
	if err := json.Unmarshal(b, &st.l); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if st.l.Validate() != nil {
		st.l = Limits{}
	}
	return st, nil
}

// Get returns the current limits; a nil Store has none.
func (st *Store) Get() Limits {
	if st == nil {
		return Limits{}
	}
	st.mu.Lock()
	defer st.mu.Unlock()
	return st.l
}

// Set validates and saves l.
func (st *Store) Set(l Limits) error {
	if err := l.Validate(); err != nil {
		return err
	}
	st.mu.Lock()
	defer st.mu.Unlock()
	if st.path != "" {
		b, _ := json.MarshalIndent(l, "", "  ")
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
	st.l = l
	return nil
}

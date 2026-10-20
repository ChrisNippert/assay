// Package chats stores conversations as one JSON file each.
package chats

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
)

// ErrNotFound is returned when a chat doesn't exist.
var ErrNotFound = errors.New("chat not found")

// Stats are the speed numbers Ollama reports for one reply.
type Stats struct {
	PromptTokens int     `json:"promptTokens"`
	OutputTokens int     `json:"outputTokens"`
	TokensPerSec float64 `json:"tokensPerSec"`
	TotalMS      int64   `json:"totalMs"`
	LoadMS       int64   `json:"loadMs"`
	// DoneReason is why generation stopped: "stop" normally, "length" when
	// the reply hit the context or max-tokens limit.
	DoneReason string `json:"doneReason,omitempty"`
	// Context is the context length the reply ran with, when ollamui set it;
	// AutoContext means ollamui picked it to fit the GPU.
	Context     int  `json:"context,omitempty"`
	AutoContext bool `json:"autoContext,omitempty"`
}

// Message is one turn in a chat.
type Message struct {
	Role     string    `json:"role"`
	Content  string    `json:"content"`
	Thinking string    `json:"thinking,omitempty"`
	Model    string    `json:"model,omitempty"`
	Stats    *Stats    `json:"stats,omitempty"`
	Created  time.Time `json:"created"`
}

// Chat is a whole conversation. Options, System and Think are this chat's
// own tweaks, applied on top of the preset (if any).
type Chat struct {
	ID       string         `json:"id"`
	Title    string         `json:"title"`
	Model    string         `json:"model"`
	Preset   string         `json:"preset,omitempty"`
	Options  map[string]any `json:"options,omitempty"`
	System   string         `json:"system,omitempty"`
	Think    *bool          `json:"think,omitempty"`
	Created  time.Time      `json:"created"`
	Updated  time.Time      `json:"updated"`
	Messages []Message      `json:"messages"`
}

// Summary is what the sidebar needs to list a chat.
type Summary struct {
	ID      string
	Title   string
	Model   string
	Updated time.Time
}

// Store keeps chats in a directory.
type Store struct {
	dir string
	mu  sync.Mutex
}

// Open returns a store rooted at dir, creating it if needed.
func Open(dir string) (*Store, error) {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, err
	}
	return &Store{dir: dir}, nil
}

var idPattern = regexp.MustCompile(`^[0-9]{8}-[0-9a-f]{8}$`)

// New returns an unsaved chat with a fresh ID.
func New(model string) *Chat {
	var b [4]byte
	rand.Read(b[:])
	now := time.Now()
	return &Chat{
		ID:      now.Format("20060102") + "-" + hex.EncodeToString(b[:]),
		Model:   model,
		Created: now,
		Updated: now,
	}
}

// ValidID reports whether id has the shape New produces. Anything else is
// rejected before it gets near the filesystem.
func ValidID(id string) bool { return idPattern.MatchString(id) }

func (s *Store) path(id string) (string, error) {
	if !ValidID(id) {
		return "", ErrNotFound
	}
	return filepath.Join(s.dir, id+".json"), nil
}

// Get loads a chat.
func (s *Store) Get(id string) (*Chat, error) {
	p, err := s.path(id)
	if err != nil {
		return nil, err
	}
	b, err := os.ReadFile(p)
	if errors.Is(err, os.ErrNotExist) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, err
	}
	var c Chat
	if err := json.Unmarshal(b, &c); err != nil {
		return nil, fmt.Errorf("chat %s: %w", id, err)
	}
	return &c, nil
}

// Save writes a chat. It writes a temp file and renames it into place, so a
// crash never leaves a half-written chat.
func (s *Store) Save(c *Chat) error {
	p, err := s.path(c.ID)
	if err != nil {
		return fmt.Errorf("invalid chat id %q", c.ID)
	}
	c.Updated = time.Now()
	b, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	f, err := os.CreateTemp(s.dir, ".tmp-*")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name()) // no-op after a successful rename
	if _, err := f.Write(b); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(f.Name(), p)
}

// Delete removes a chat. Deleting a chat that doesn't exist is not an error.
func (s *Store) Delete(id string) error {
	p, err := s.path(id)
	if err != nil {
		return err
	}
	if err := os.Remove(p); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return nil
}

// List returns all chats, most recently updated first. Files that fail to
// parse are skipped rather than hiding every other chat.
func (s *Store) List() ([]Summary, error) {
	entries, err := os.ReadDir(s.dir)
	if err != nil {
		return nil, err
	}
	var out []Summary
	for _, e := range entries {
		id, ok := strings.CutSuffix(e.Name(), ".json")
		if !ok || !ValidID(id) {
			continue
		}
		c, err := s.Get(id)
		if err != nil {
			continue
		}
		out = append(out, Summary{ID: c.ID, Title: c.Title, Model: c.Model, Updated: c.Updated})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Updated.After(out[j].Updated) })
	return out, nil
}

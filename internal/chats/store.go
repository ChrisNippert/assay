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
	"slices"
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
	// PromptPerSec is how fast the prompt was read (0 when it was cached).
	PromptPerSec float64 `json:"promptPerSec,omitempty"`
	TotalMS      int64   `json:"totalMs"`
	LoadMS       int64   `json:"loadMs"`
	// DoneReason is why generation stopped: "stop" normally, "length" when
	// the reply hit the context or max-tokens limit.
	DoneReason string `json:"doneReason,omitempty"`
	// Context is the context length the reply ran with, when Assay set it;
	// AutoContext means Assay picked it to fit the GPU.
	Context     int  `json:"context,omitempty"`
	AutoContext bool `json:"autoContext,omitempty"`
	// Cost is what an API model charged for the reply, in US dollars.
	Cost float64 `json:"cost,omitempty"`
}

// Message is one turn in a chat.
type Message struct {
	Role     string `json:"role"`
	Content  string `json:"content"`
	Thinking string `json:"thinking,omitempty"`
	// Spoken is what voice mode said aloud while the model was still
	// thinking, before Content.
	Spoken string `json:"spoken,omitempty"`
	// Stopped is set on a reply the user stopped before it finished.
	Stopped bool `json:"stopped,omitempty"`
	// Calls are the tool calls made on the way to Content, in order.
	Calls []ToolCall `json:"calls,omitempty"`
	Model string     `json:"model,omitempty"`
	Stats *Stats     `json:"stats,omitempty"`
	// Sources are the project-file excerpts the reply was given (RAG).
	Sources []Source `json:"sources,omitempty"`
	// Attachments are the files sent with a user message (see attach.go).
	Attachments []Attachment `json:"attachments,omitempty"`
	// Images are the attached images, base64, loaded for one request only.
	Images  []string  `json:"-"`
	Created time.Time `json:"created"`
}

// ToolCall is one tool call a reply made. Calls the model asked for in the
// same step share Step; Said and Think (what the model wrote and thought
// before asking) sit on the first call of a step.
type ToolCall struct {
	Step   int    `json:"step"`
	ID     string `json:"id"`
	Name   string `json:"name"`
	Args   string `json:"args"`   // the JSON arguments as the model wrote them
	Result string `json:"result"` // what went back to the model (truncated)
	// Block is edit_code's whole updated block, shown to the user and used
	// by later edits (the model may have got only the changed lines).
	Block string `json:"block,omitempty"`
	Label string `json:"label"` // "Searched the web for “…” · 5 results"
	Error bool   `json:"error,omitempty"`
	Trim  string `json:"trim,omitempty"` // the trim a forget call made
	Said  string `json:"said,omitempty"`
	Think string `json:"think,omitempty"`
}

// Source is one excerpt from a project file that went into a reply's
// prompt, numbered as the model was asked to cite it ([1], [2], ...).
type Source struct {
	N       int     `json:"n"`
	File    string  `json:"file"`
	Page    int     `json:"page,omitempty"`
	Heading string  `json:"heading,omitempty"`
	Text    string  `json:"text"`
	Score   float32 `json:"score"`
}

// Chat is a whole conversation. Options, System and Think are this chat's
// own tweaks, applied on top of the preset (if any).
type Chat struct {
	ID      string         `json:"id"`
	Title   string         `json:"title"`
	Model   string         `json:"model"`
	Preset  string         `json:"preset,omitempty"`
	Project string         `json:"project,omitempty"` // project ID, "" for none
	Options map[string]any `json:"options,omitempty"`
	System  string         `json:"system,omitempty"`
	Think   *bool          `json:"think,omitempty"`
	// Compress is this chat's "Compress automatically" choice; nil follows
	// the preset, then the default in Settings.
	Compress *bool `json:"compress,omitempty"`
	// Compression, if set, stands for the messages before its UpTo index
	// when the chat is sent. The messages themselves stay in Messages.
	Compression *Compression `json:"compression,omitempty"`
	// Tools is whether the model may call tools: nil leaves it to the
	// preset, then to the default (on when the model supports tools).
	Tools *bool `json:"tools,omitempty"`
	// EditCode lets the model change an earlier code block with edit_code
	// (see edit.go). Off unless the chat turns it on.
	EditCode bool `json:"editCode,omitempty"`
	// Forget lets the model leave earlier messages out of later prompts with
	// the forget tool (see trim.go). Off unless the chat turns it on.
	Forget bool `json:"forget,omitempty"`
	// Trims are the messages left out of later prompts. The saved messages
	// are not changed; a trim can be undone.
	Trims    []Trim    `json:"trims,omitempty"`
	Created  time.Time `json:"created"`
	Updated  time.Time `json:"updated"`
	Messages []Message `json:"messages"`
}

// Trim leaves messages out of later prompts: each one is sent as
// "[trimmed: Note]". Messages are indexes into the chat's Messages.
type Trim struct {
	ID       string    `json:"id"`
	Messages []int     `json:"messages"`
	Note     string    `json:"note"`
	Created  time.Time `json:"created"`
}

// DropTrimsFrom removes the trims of message i (counting from 0) and later
// messages: an edit or regenerate from there replaces them.
func (c *Chat) DropTrimsFrom(i int) {
	var kept []Trim
	for _, t := range c.Trims {
		var idx []int
		for _, m := range t.Messages {
			if m < i {
				idx = append(idx, m)
			}
		}
		if len(idx) > 0 {
			t.Messages = idx
			kept = append(kept, t)
		}
	}
	c.Trims = kept
}

// UndoTrim removes the trim with the given ID and reports whether it was there.
func (c *Chat) UndoTrim(id string) bool {
	i := slices.IndexFunc(c.Trims, func(t Trim) bool { return t.ID == id })
	if i < 0 {
		return false
	}
	c.Trims = slices.Delete(c.Trims, i, i+1)
	return true
}

// Compression is a summary of a chat's oldest messages, sent in their
// place to keep the prompt within the context length.
type Compression struct {
	Text string `json:"text"`
	// UpTo is the index of the first message not covered: Messages[:UpTo]
	// are summarized.
	UpTo int `json:"upTo"`
	// Model wrote the summary.
	Model string `json:"model,omitempty"`
	// Saved is about how many tokens the summarized messages took, less the
	// summary's own, as the page reports it.
	Saved   int       `json:"saved,omitempty"`
	Created time.Time `json:"created"`
}

// DropCompressionFrom removes the summary if message i (counting from 0)
// is summarized or cut: an edit or regenerate from there changes what the
// summary stands for. It reports whether it removed one.
func (c *Chat) DropCompressionFrom(i int) bool {
	if c.Compression == nil || i >= c.Compression.UpTo {
		return false
	}
	c.Compression = nil
	return true
}

// UserIndex returns the index in Messages of the n-th user message (from
// 0), or -1.
func (c *Chat) UserIndex(n int) int {
	for i, m := range c.Messages {
		if m.Role != "user" {
			continue
		}
		if n == 0 {
			return i
		}
		n--
	}
	return -1
}

// TruncateAtUser drops the n-th user message (from 0) and everything after
// it. It reports false if the chat has no such message.
func (c *Chat) TruncateAtUser(n int) bool {
	for i, m := range c.Messages {
		if m.Role != "user" {
			continue
		}
		if n == 0 {
			c.Messages = c.Messages[:i]
			return true
		}
		n--
	}
	return false
}

// Summary is what the sidebar needs to list a chat.
type Summary struct {
	ID      string
	Title   string
	Model   string
	Preset  string
	Project string
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
	if _, err := s.path(c.ID); err != nil {
		return fmt.Errorf("invalid chat id %q", c.ID)
	}
	c.Updated = time.Now()
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.write(c)
}

// Rename sets a chat's title. Unlike Save it leaves Updated alone, so a
// rename doesn't move the chat to the top of the sidebar.
func (s *Store) Rename(id, title string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	c, err := s.Get(id)
	if err != nil {
		return err
	}
	c.Title = title
	return s.write(c)
}

// SetProject moves a chat into a project ("" takes it out). Like Rename,
// it leaves Updated alone so the chat keeps its place.
func (s *Store) SetProject(id, project string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	c, err := s.Get(id)
	if err != nil {
		return err
	}
	c.Project = project
	return s.write(c)
}

// LeaveProject takes every chat out of a project, e.g. when it's deleted.
func (s *Store) LeaveProject(project string) error {
	list, err := s.List()
	if err != nil {
		return err
	}
	for _, c := range list {
		if c.Project == project {
			if err := s.SetProject(c.ID, ""); err != nil {
				return err
			}
		}
	}
	return nil
}

// write saves c atomically. The caller holds s.mu.
func (s *Store) write(c *Chat) error {
	p, err := s.path(c.ID)
	if err != nil {
		return err
	}
	b, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
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
	return os.RemoveAll(filepath.Join(s.dir, id+filesSuffix))
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
		out = append(out, Summary{ID: c.ID, Title: c.Title, Model: c.Model, Preset: c.Preset, Project: c.Project, Updated: c.Updated})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Updated.After(out[j].Updated) })
	return out, nil
}

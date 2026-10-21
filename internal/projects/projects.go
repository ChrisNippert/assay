// Package projects keeps projects: named groups of chats with optional
// instructions, a default model or preset, and files for RAG. Each project
// is a folder: project.json, files/ (the originals) and index/ (their
// embedded chunks, see package index).
package projects

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"ollamui/internal/index"
)

// ErrNotFound is returned when a project or file doesn't exist.
var ErrNotFound = errors.New("project not found")

// File states.
const (
	FileQueued   = "queued"
	FileIndexing = "indexing"
	FileReady    = "ready"
	FileError    = "error"
)

// File is a document added to a project.
type File struct {
	Name   string    `json:"name"`
	Size   int64     `json:"size"`
	Added  time.Time `json:"added"`
	Status string    `json:"status"`
	Chunks int       `json:"chunks,omitempty"`
	Error  string    `json:"error,omitempty"`
	// Done and Total count chunks embedded so far while indexing.
	Done  int `json:"done,omitempty"`
	Total int `json:"total,omitempty"`
}

// Project is a group of chats with shared settings and files.
type Project struct {
	ID   string `json:"id"`
	Name string `json:"name"`
	// Instructions are added to the system prompt of every chat in it.
	Instructions string `json:"instructions,omitempty"`
	// Model or Preset is what new chats in the project start with.
	Model   string    `json:"model,omitempty"`
	Preset  string    `json:"preset,omitempty"`
	Files   []File    `json:"files,omitempty"`
	Created time.Time `json:"created"`
	Updated time.Time `json:"updated"`
}

// File returns the named file, or nil.
func (p *Project) File(name string) *File {
	for i := range p.Files {
		if p.Files[i].Name == name {
			return &p.Files[i]
		}
	}
	return nil
}

// Ready counts files that are indexed.
func (p *Project) Ready() int {
	n := 0
	for _, f := range p.Files {
		if f.Status == FileReady {
			n++
		}
	}
	return n
}

// Store keeps projects in a directory.
type Store struct {
	dir string
	mu  sync.Mutex
	ix  map[string]*index.Index // open indexes by project ID
}

// Open returns a store rooted at dir.
func Open(dir string) (*Store, error) {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, err
	}
	return &Store{dir: dir, ix: map[string]*index.Index{}}, nil
}

var idPattern = regexp.MustCompile(`^p[0-9a-f]{10}$`)

// ValidID reports whether id has the shape Create gives.
func ValidID(id string) bool { return idPattern.MatchString(id) }

func (s *Store) path(id string, parts ...string) (string, error) {
	if !ValidID(id) {
		return "", ErrNotFound
	}
	return filepath.Join(append([]string{s.dir, id}, parts...)...), nil
}

// CleanName tidies a project name: one line, trimmed, at most 80 characters.
func CleanName(name string) string {
	name = strings.Join(strings.Fields(name), " ")
	for utf8.RuneCountInString(name) > 80 {
		_, size := utf8.DecodeLastRuneInString(name)
		name = name[:len(name)-size]
	}
	return name
}

// Create makes a new, empty project.
func (s *Store) Create(name string) (*Project, error) {
	name = CleanName(name)
	if name == "" {
		return nil, errors.New("a project needs a name")
	}
	var b [5]byte
	rand.Read(b[:])
	now := time.Now()
	p := &Project{ID: "p" + hex.EncodeToString(b[:]), Name: name, Created: now, Updated: now}
	return p, s.Save(p)
}

// Get loads a project.
func (s *Store) Get(id string) (*Project, error) {
	path, err := s.path(id, "project.json")
	if err != nil {
		return nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return read(path)
}

func read(path string) (*Project, error) {
	b, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, err
	}
	var p Project
	if err := json.Unmarshal(b, &p); err != nil {
		return nil, err
	}
	return &p, nil
}

// Save writes a project.
func (s *Store) Save(p *Project) error {
	path, err := s.path(p.ID, "project.json")
	if err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return write(path, p)
}

func write(path string, p *Project) error {
	b, err := json.MarshalIndent(p, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	if err := os.WriteFile(path+".tmp", append(b, '\n'), 0o600); err != nil {
		return err
	}
	return os.Rename(path+".tmp", path)
}

// Update loads a project, lets fn change it and saves it, under one lock,
// so background indexing and edits from the page don't overwrite each other.
func (s *Store) Update(id string, fn func(*Project) error) (*Project, error) {
	path, err := s.path(id, "project.json")
	if err != nil {
		return nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	p, err := read(path)
	if err != nil {
		return nil, err
	}
	if err := fn(p); err != nil {
		return nil, err
	}
	p.Updated = time.Now()
	return p, write(path, p)
}

// List returns all projects, by name.
func (s *Store) List() ([]*Project, error) {
	entries, err := os.ReadDir(s.dir)
	if err != nil {
		return nil, err
	}
	var out []*Project
	for _, e := range entries {
		if !e.IsDir() || !ValidID(e.Name()) {
			continue
		}
		if p, err := s.Get(e.Name()); err == nil {
			out = append(out, p)
		}
	}
	sort.Slice(out, func(i, j int) bool { return strings.ToLower(out[i].Name) < strings.ToLower(out[j].Name) })
	return out, nil
}

// Delete removes a project and its files. Its chats are the caller's to move.
func (s *Store) Delete(id string) error {
	path, err := s.path(id)
	if err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, err := os.Stat(path); err != nil {
		return ErrNotFound
	}
	delete(s.ix, id)
	return os.RemoveAll(path)
}

var unsafeChars = regexp.MustCompile(`[^\p{L}\p{N} ._()+,=@-]+`)

// CleanFileName makes an uploaded file's name safe to store: no folders,
// no leading dots, no odd characters, at most 120 bytes.
func CleanFileName(name string) string {
	name = filepath.Base(strings.ReplaceAll(name, `\`, "/"))
	name = strings.TrimLeft(unsafeChars.ReplaceAllString(name, "_"), ". ")
	if len(name) > 120 {
		ext := filepath.Ext(name)
		if len(ext) > 20 {
			ext = ""
		}
		name = name[:120-len(ext)] + ext
	}
	if name == "" || name == "_" {
		return ""
	}
	return name
}

// FilePath is where a project's file is stored.
func (s *Store) FilePath(id, name string) (string, error) {
	if name != CleanFileName(name) || name == "" {
		return "", ErrNotFound
	}
	return s.path(id, "files", name)
}

// Index opens (once) and returns a project's vector index.
func (s *Store) Index(id string) (*index.Index, error) {
	dir, err := s.path(id, "index")
	if err != nil {
		return nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if ix := s.ix[id]; ix != nil {
		return ix, nil
	}
	ix, err := index.Open(dir)
	if err != nil {
		return nil, err
	}
	s.ix[id] = ix
	return ix, nil
}

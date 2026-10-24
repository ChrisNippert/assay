package engine

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// ErrNotFound is returned for a model name the store doesn't have.
var ErrNotFound = errors.New("model not found")

// Model is an installed model: one or more GGUF files (several when the
// model is split), an optional projector for images and audio, and names.
type Model struct {
	// Name is the full name, e.g. "hf.co/unsloth/Qwen3.5-9B-GGUF:Q8_0";
	// Aliases are short ones, e.g. "qwen3.5-9b:q8_0".
	Name    string   `json:"name"`
	Aliases []string `json:"aliases,omitempty"`
	// Files are the model's GGUF files in order, relative to the store's
	// blobs folder; Projector likewise ("" if none).
	Files     []string `json:"files"`
	Projector string   `json:"projector,omitempty"`
	Size      int64    `json:"size"` // all files, projector included
	// Digest identifies the weights: the first file's sha256 when known.
	Digest   string    `json:"digest"`
	Modified time.Time `json:"modified"`
	// Repo and Quant say where it came from on Hugging Face, if it did.
	Repo  string `json:"repo,omitempty"`
	Quant string `json:"quant,omitempty"`
	// Params are default sampling settings (temperature, top_k, stop...).
	Params map[string]any `json:"params,omitempty"`
	// Settings are the user's own for this model (context, cache type,
	// sampling...), applied over Params and under each request's options.
	Settings map[string]any `json:"settings,omitempty"`
	// Link is "hard" or "symlink" for a model linked from another app's
	// files, and Origin the first file's path there. Source is "hf"
	// (downloaded here) or where it is linked from: "ollama", "hf-cache",
	// "llama.cpp", "lmstudio" or "folder".
	Link   string `json:"link,omitempty"`
	Origin string `json:"origin,omitempty"`
	Source string `json:"source"`
}

// options layers the model's Params, its Settings and a request's options.
func (m *Model) options(req map[string]any) map[string]any {
	out := map[string]any{}
	for _, o := range []map[string]any{m.Params, m.Settings, req} {
		for k, v := range o {
			out[k] = v
		}
	}
	return out
}

// Names lists the full name and the aliases.
func (m *Model) Names() []string { return append([]string{m.Name}, m.Aliases...) }

// Store keeps models in a folder: blobs/ for the files and one JSON
// manifest per model in manifests/.
type Store struct {
	dir string

	mu     sync.Mutex
	models []*Model
	// moving is set while MoveTo copies the files; changes wait.
	moving bool
}

func (s *Store) isMoving() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.moving
}

// OpenStore loads the store in dir, creating it if needed.
func OpenStore(dir string) (*Store, error) {
	for _, d := range []string{"blobs", "manifests"} {
		if err := os.MkdirAll(filepath.Join(dir, d), 0o755); err != nil {
			return nil, err
		}
	}
	s := &Store{dir: dir}
	return s, s.reload()
}

// Blob is a file's full path.
// Blob is where a model file is: in blobs/, or, for a file used in place
// (an absolute name), right where it is.
func (s *Store) Blob(name string) string {
	if filepath.IsAbs(name) {
		return name
	}
	return filepath.Join(s.Dir(), "blobs", name)
}

// Dir is the store's folder. It changes when the store is moved.
func (s *Store) Dir() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.dir
}

func (s *Store) reload() error {
	dir := s.Dir()
	entries, err := os.ReadDir(filepath.Join(dir, "manifests"))
	if err != nil {
		return err
	}
	var models []*Model
	for _, e := range entries {
		if !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		b, err := os.ReadFile(filepath.Join(dir, "manifests", e.Name()))
		if err != nil {
			continue
		}
		var m Model
		if json.Unmarshal(b, &m) == nil && m.Name != "" && len(m.Files) > 0 {
			models = append(models, &m)
		}
	}
	sort.Slice(models, func(i, j int) bool { return models[i].Name < models[j].Name })
	s.mu.Lock()
	s.models = models
	s.mu.Unlock()
	return nil
}

// List returns the installed models, by name.
func (s *Store) List() []*Model {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]*Model(nil), s.models...)
}

// Get finds a model by its name or an alias. "x" and "x:latest" match.
func (s *Store) Get(name string) (*Model, error) {
	want := normName(name)
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, m := range s.models {
		for _, n := range m.Names() {
			if normName(n) == want {
				return m, nil
			}
		}
	}
	return nil, fmt.Errorf("%w: %s", ErrNotFound, name)
}

func normName(n string) string {
	n = strings.ToLower(strings.TrimSpace(n))
	if !strings.Contains(n[strings.LastIndex(n, "/")+1:], ":") {
		n += ":latest"
	}
	return n
}

func manifestFile(name string) string {
	h := sha256.Sum256([]byte(strings.ToLower(name)))
	return hex.EncodeToString(h[:8]) + ".json"
}

// Put saves a model's manifest (adding or replacing it).
func (s *Store) Put(m *Model) error {
	if s.isMoving() {
		return ErrMoving
	}
	b, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	path := filepath.Join(s.Dir(), "manifests", manifestFile(m.Name))
	if err := os.WriteFile(path+".tmp", append(b, '\n'), 0o644); err != nil {
		return err
	}
	if err := os.Rename(path+".tmp", path); err != nil {
		return err
	}
	return s.reload()
}

// Delete removes a model and any files no other model uses. With an
// alias it only drops that alias.
func (s *Store) Delete(name string) error {
	if s.isMoving() {
		return ErrMoving
	}
	m, err := s.Get(name)
	if err != nil {
		return err
	}
	if normName(name) != normName(m.Name) {
		cp := *m
		cp.Aliases = nil
		for _, a := range m.Aliases {
			if normName(a) != normName(name) {
				cp.Aliases = append(cp.Aliases, a)
			}
		}
		return s.Put(&cp)
	}
	if err := os.Remove(filepath.Join(s.Dir(), "manifests", manifestFile(m.Name))); err != nil && !errors.Is(err, fs.ErrNotExist) {
		return err
	}
	if err := s.reload(); err != nil {
		return err
	}
	used := map[string]bool{}
	for _, o := range s.List() {
		for _, f := range append(o.Files, o.Projector) {
			used[f] = true
		}
	}
	for _, f := range append(m.Files, m.Projector) {
		// A file used in place belongs to another app: only forget it.
		if f != "" && !used[f] && !filepath.IsAbs(f) {
			os.Remove(s.Blob(f))
			os.Remove(s.Blob(f) + ".partial")
			if dir := filepath.Dir(f); dir != "." {
				os.Remove(s.Blob(dir)) // a split model's folder, once empty
			}
		}
	}
	return nil
}

// ---- importing Ollama's models ----

type ollamaManifest struct {
	Layers []struct {
		MediaType string `json:"mediaType"`
		Digest    string `json:"digest"`
		Size      int64  `json:"size"`
	} `json:"layers"`
}

// ImportOllama adds the models of an Ollama install (its models folder,
// usually ~/.ollama/models) that the store doesn't have yet. Files are
// hard-linked, so nothing is copied and Ollama keeps working; a symlink
// is used when the folders are on different disks. Names that point at
// the same weights become one model with aliases. It returns how many
// models it added.
func (s *Store) ImportOllama(ollamaDir string) (int, error) {
	type found struct {
		names     []string
		model     string // blob digest of the weights
		projector string
		size      int64
		params    map[string]any
		modified  time.Time
	}
	byDigest := map[string]*found{}
	var order []string
	for _, root := range []string{"manifests", "manifests-v2"} {
		base := filepath.Join(ollamaDir, root)
		filepath.WalkDir(base, func(path string, d fs.DirEntry, err error) error {
			if err != nil || d.IsDir() {
				return nil
			}
			rel, _ := filepath.Rel(base, path)
			name := ollamaName(filepath.ToSlash(rel))
			if name == "" {
				return nil
			}
			b, err := os.ReadFile(path) // follows manifests-v2's symlinks
			if err != nil {
				return nil
			}
			var man ollamaManifest
			if json.Unmarshal(b, &man) != nil {
				return nil
			}
			f := &found{}
			for _, l := range man.Layers {
				switch l.MediaType {
				case "application/vnd.ollama.image.model":
					f.model = l.Digest
					f.size += l.Size
				case "application/vnd.ollama.image.projector":
					f.projector = l.Digest
					f.size += l.Size
				case "application/vnd.ollama.image.params":
					if pb, err := os.ReadFile(ollamaBlob(ollamaDir, l.Digest)); err == nil {
						json.Unmarshal(pb, &f.params)
					}
				}
			}
			if f.model == "" {
				return nil
			}
			if info, err := os.Stat(path); err == nil {
				f.modified = info.ModTime()
			}
			if have := byDigest[f.model]; have != nil {
				if !contains(have.names, name) {
					have.names = append(have.names, name)
				}
				if have.params == nil {
					have.params = f.params
				}
				return nil
			}
			f.names = []string{name}
			byDigest[f.model] = f
			order = append(order, f.model)
			return nil
		})
	}

	added := 0
	for _, digest := range order {
		f := byDigest[digest]
		dup := false
		for _, n := range f.names {
			if _, err := s.Get(n); err == nil {
				dup = true
			}
		}
		if dup {
			continue
		}
		file, err := s.linkBlob(ollamaDir, f.model)
		if err != nil {
			return added, err
		}
		m := &Model{Files: []string{file}, Size: f.size, Digest: strings.TrimPrefix(f.model, "sha256:"), Modified: f.modified, Params: f.params, Source: "ollama", Link: s.linkKind(file), Origin: ollamaBlob(ollamaDir, f.model)}
		if f.projector != "" {
			if m.Projector, err = s.linkBlob(ollamaDir, f.projector); err != nil {
				return added, err
			}
		}
		// The full hf.co name leads; short names are aliases.
		sort.SliceStable(f.names, func(i, j int) bool {
			return strings.HasPrefix(f.names[i], "hf.co/") && !strings.HasPrefix(f.names[j], "hf.co/")
		})
		m.Name, m.Aliases = f.names[0], f.names[1:]
		if repo, quant, ok := strings.Cut(strings.TrimPrefix(m.Name, "hf.co/"), ":"); ok && strings.HasPrefix(m.Name, "hf.co/") {
			m.Repo, m.Quant = repo, quant
		}
		if err := s.Put(m); err != nil {
			return added, err
		}
		added++
	}
	return added, nil
}

// ollamaName turns a manifest path into a model name:
// "hf.co/unsloth/Qwen3.5-9B-GGUF/Q8_0" → "hf.co/unsloth/Qwen3.5-9B-GGUF:Q8_0",
// "registry.ollama.ai/library/qwen3/8b" and "ollama.com/library/qwen3/8b" → "qwen3:8b".
func ollamaName(rel string) string {
	parts := strings.Split(rel, "/")
	if len(parts) < 3 {
		return ""
	}
	host, tag := parts[0], parts[len(parts)-1]
	repo := strings.Join(parts[1:len(parts)-1], "/")
	switch host {
	case "registry.ollama.ai", "ollama.com":
		repo = strings.TrimPrefix(repo, "library/")
		return repo + ":" + tag
	}
	return host + "/" + repo + ":" + tag
}

func ollamaBlob(ollamaDir, digest string) string {
	return filepath.Join(ollamaDir, "blobs", strings.Replace(digest, ":", "-", 1))
}

// linkBlob hard-links an Ollama blob into the store and returns its name
// there. On another disk the blob is used in place: its full path is the
// name (no symlink, so no admin rights on Windows).
func (s *Store) linkBlob(ollamaDir, digest string) (string, error) {
	src := ollamaBlob(ollamaDir, digest)
	name := strings.Replace(digest, ":", "-", 1) + ".gguf"
	dst := s.Blob(name)
	if _, err := os.Stat(dst); err == nil {
		return name, nil
	}
	if _, err := os.Stat(src); err != nil {
		return "", fmt.Errorf("import %s: %w", digest, err)
	}
	if err := hardLink(src, dst); err != nil {
		return src, nil
	}
	return name, nil
}

func contains(list []string, s string) bool {
	for _, x := range list {
		if x == s {
			return true
		}
	}
	return false
}

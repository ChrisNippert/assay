// Package index is a small vector index kept in plain files: one JSON file
// with the chunks and one file of float32 vectors beside it. Search is a
// linear scan, which is plenty for a project's worth of documents (tens of
// thousands of chunks) and needs no database.
package index

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"sync"
)

// Entry is one indexed chunk of a file.
type Entry struct {
	File    string `json:"file"`
	Page    int    `json:"page,omitempty"`
	Heading string `json:"heading,omitempty"`
	Text    string `json:"text"`
}

// Hit is a search result.
type Hit struct {
	Entry
	Score float32 `json:"score"` // cosine similarity, -1 to 1
}

// Index holds every chunk of one project, embedded with one model.
type Index struct {
	dir string

	mu      sync.RWMutex
	model   string
	dims    int
	entries []Entry
	vecs    []float32 // len(entries) * dims, each vector normalised
}

type meta struct {
	Model   string  `json:"model"`
	Dims    int     `json:"dims"`
	Entries []Entry `json:"entries"`
}

// Open loads the index in dir, or starts an empty one.
func Open(dir string) (*Index, error) {
	ix := &Index{dir: dir}
	b, err := os.ReadFile(filepath.Join(dir, "index.json"))
	if errors.Is(err, os.ErrNotExist) {
		return ix, nil
	}
	if err != nil {
		return nil, err
	}
	var m meta
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, fmt.Errorf("index: %w", err)
	}
	f, err := os.Open(filepath.Join(dir, "vectors.f32"))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	vecs := make([]float32, len(m.Entries)*m.Dims)
	if err := binary.Read(f, binary.LittleEndian, vecs); err != nil && !(len(vecs) == 0 && errors.Is(err, io.EOF)) {
		return nil, fmt.Errorf("index vectors: %w", err)
	}
	ix.model, ix.dims, ix.entries, ix.vecs = m.Model, m.Dims, m.Entries, vecs
	return ix, nil
}

// Model is the embedding model the index was built with ("" when empty).
func (ix *Index) Model() string {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	return ix.model
}

// Len is the number of chunks.
func (ix *Index) Len() int {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	return len(ix.entries)
}

// Put replaces file's chunks with entries and their vectors. All vectors
// in an index come from one model; putting with another model fails, so
// the caller re-indexes everything instead of mixing incomparable vectors.
func (ix *Index) Put(model, file string, entries []Entry, vecs [][]float32) error {
	if len(entries) != len(vecs) {
		return errors.New("index: entries and vectors differ in number")
	}
	ix.mu.Lock()
	defer ix.mu.Unlock()
	if len(ix.entries) > 0 && ix.model != model {
		return fmt.Errorf("index was built with %s, not %s", ix.model, model)
	}
	ix.removeLocked(file)
	if len(ix.entries) == 0 {
		ix.model, ix.dims = model, 0
	}
	for i, v := range vecs {
		if ix.dims == 0 {
			ix.dims = len(v)
		}
		if len(v) != ix.dims || ix.dims == 0 {
			return fmt.Errorf("index: vector has %d dimensions, want %d", len(v), ix.dims)
		}
		ix.entries = append(ix.entries, entries[i])
		ix.vecs = append(ix.vecs, normalize(v)...)
	}
	if len(ix.entries) == 0 {
		ix.model = ""
	}
	return ix.saveLocked()
}

// Remove drops a file's chunks.
func (ix *Index) Remove(file string) error {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	ix.removeLocked(file)
	if len(ix.entries) == 0 {
		ix.model, ix.dims = "", 0
	}
	return ix.saveLocked()
}

// Clear drops everything, e.g. before re-indexing with another model.
func (ix *Index) Clear() error {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	ix.entries, ix.vecs, ix.model, ix.dims = nil, nil, "", 0
	return ix.saveLocked()
}

func (ix *Index) removeLocked(file string) {
	keep := 0
	for i, e := range ix.entries {
		if e.File == file {
			continue
		}
		ix.entries[keep] = e
		copy(ix.vecs[keep*ix.dims:(keep+1)*ix.dims], ix.vecs[i*ix.dims:(i+1)*ix.dims])
		keep++
	}
	ix.entries = ix.entries[:keep]
	ix.vecs = ix.vecs[:keep*ix.dims]
}

// Search returns the k chunks closest to q.
func (ix *Index) Search(q []float32, k int) []Hit {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	if len(q) != ix.dims || ix.dims == 0 || k <= 0 {
		return nil
	}
	q = normalize(q)
	hits := make([]Hit, 0, k+1)
	for i := range ix.entries {
		v := ix.vecs[i*ix.dims : (i+1)*ix.dims]
		var dot float32
		for j, x := range q {
			dot += x * v[j]
		}
		if len(hits) == k && dot <= hits[k-1].Score {
			continue
		}
		hits = append(hits, Hit{Entry: ix.entries[i], Score: dot})
		sort.Slice(hits, func(a, b int) bool { return hits[a].Score > hits[b].Score })
		if len(hits) > k {
			hits = hits[:k]
		}
	}
	return hits
}

func normalize(v []float32) []float32 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	out := make([]float32, len(v))
	if sum == 0 {
		return out
	}
	n := float32(1 / math.Sqrt(sum))
	for i, x := range v {
		out[i] = x * n
	}
	return out
}

// saveLocked writes both files atomically (each via a temp file).
func (ix *Index) saveLocked() error {
	if err := os.MkdirAll(ix.dir, 0o755); err != nil {
		return err
	}
	b, err := json.Marshal(meta{Model: ix.model, Dims: ix.dims, Entries: ix.entries})
	if err != nil {
		return err
	}
	vecPath := filepath.Join(ix.dir, "vectors.f32")
	f, err := os.Create(vecPath + ".tmp")
	if err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, ix.vecs); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(ix.dir, "index.json.tmp"), b, 0o644); err != nil {
		return err
	}
	if err := os.Rename(vecPath+".tmp", vecPath); err != nil {
		return err
	}
	return os.Rename(filepath.Join(ix.dir, "index.json.tmp"), filepath.Join(ix.dir, "index.json"))
}

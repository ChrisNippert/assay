package engine

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"assay/internal/gguf"
	"assay/internal/presets"
)

// Adapter is a LoRA adapter in the library: a file llama.cpp applies on
// top of a model of its architecture.
type Adapter struct {
	Name string `json:"name"`
	Size int64  `json:"size"`
	// Arch is the base architecture it was trained for; only models of
	// that architecture can load it.
	Arch string `json:"arch"`
}

// Adapters lists the installed LoRA adapters, by name.
func (e *Engine) Adapters() []Adapter {
	var out []Adapter
	for _, m := range e.store.Adapters() {
		out = append(out, Adapter{Name: m.Name, Size: m.Size, Arch: m.Adapter.Arch})
	}
	return out
}

// AdaptersFor lists the installed adapters a model can load with: those
// for its architecture. An adapter has none.
func (e *Engine) AdaptersFor(name string) ([]Adapter, error) {
	m, err := e.store.Get(name)
	if err != nil {
		return nil, err
	}
	if m.Adapter != nil {
		return nil, nil
	}
	f, err := e.gguf(m)
	if err != nil {
		return nil, err
	}
	var out []Adapter
	for _, a := range e.Adapters() {
		if a.Arch == f.Arch() {
			out = append(out, a)
		}
	}
	return out, nil
}

// AddAdapter adds a LoRA adapter file to the library under name (the file's
// name without .gguf if empty). Like an import, it's hard-linked when the
// store is on the same disk and otherwise used where it is. The file is
// only read, never copied or changed.
func (e *Engine) AddAdapter(src, name string) (Adapter, error) {
	if e.moving.Load() {
		return Adapter{}, errMovingModels
	}
	src = expandHome(strings.TrimSpace(src))
	if !filepath.IsAbs(src) || !strings.HasSuffix(strings.ToLower(src), ".gguf") {
		return Adapter{}, errors.New("enter the full path of a .gguf file")
	}
	src = filepath.Clean(src)
	info, err := os.Stat(src)
	if err != nil || !info.Mode().IsRegular() {
		return Adapter{}, errors.New("that file isn't there")
	}
	f, err := gguf.Read(src)
	if err != nil {
		return Adapter{}, errors.New("that isn't a GGUF file")
	}
	if !f.IsLoRA() {
		return Adapter{}, errors.New("that file isn't a LoRA adapter")
	}
	if strings.Contains(src, ",") {
		return Adapter{}, errors.New("llama.cpp can't load an adapter whose path has a comma in it; rename the file")
	}
	name = strings.TrimSpace(name)
	if name == "" {
		name = strings.TrimSuffix(filepath.Base(src), filepath.Ext(src))
	}
	if _, err := e.store.Get(name); err == nil {
		return Adapter{}, fmt.Errorf("%s is already in the library", name)
	}
	m := &Model{Name: name, Size: info.Size(), Modified: info.ModTime(), Digest: pathHash(src), Source: "file", Origin: src, Link: "hard", Adapter: &AdapterInfo{Arch: f.Arch()}}
	if file, _, err := e.store.linkFile(src, "", "", ""); err == nil {
		m.Files, m.Link = []string{file}, e.store.linkKind(file)
	} else {
		m.Files, m.Link = []string{src}, "in place"
	}
	if err := e.store.Put(m); err != nil {
		return Adapter{}, err
	}
	e.opt.Log.Info("added LoRA adapter", "adapter", name, "arch", f.Arch())
	return Adapter{Name: name, Size: m.Size, Arch: f.Arch()}, nil
}

// checkLoRA says why a model can't load with these adapters, or nil: each
// must be an installed adapter for the model's architecture, with its file.
func (e *Engine) checkLoRA(m *Model, uses []presets.LoRA) error {
	if len(uses) == 0 {
		return nil
	}
	mf, err := e.gguf(m)
	if err != nil {
		return err
	}
	for _, u := range uses {
		a, err := e.store.Get(u.Name)
		if err != nil || a.Adapter == nil {
			return fmt.Errorf("adapter %s isn't installed", u.Name)
		}
		if a.Adapter.Arch != mf.Arch() {
			return fmt.Errorf("adapter %s is for %s models, and %s is %s", a.Name, a.Adapter.Arch, m.Name, mf.Arch())
		}
		if len(e.store.Missing(a)) > 0 {
			return fmt.Errorf("adapter %s's file is missing: it was linked from %s", a.Name, a.Origin)
		}
	}
	return nil
}

// setScales sets the scales of a running server's adapters, through its
// /lora-adapters. They're numbered in the order the server was given them.
func (e *Engine) setScales(ctx context.Context, p *proc, uses []presets.LoRA) error {
	type scale struct {
		ID    int     `json:"id"`
		Scale float64 `json:"scale"`
	}
	body := make([]scale, len(uses))
	for i, u := range uses {
		body[i] = scale{ID: i, Scale: u.Scale}
	}
	b, _ := json.Marshal(body)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.base+"/lora-adapters", bytes.NewReader(b))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	res, err := e.http.Do(req)
	if err != nil {
		return err
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		said, _ := io.ReadAll(io.LimitReader(res.Body, 512))
		return fmt.Errorf("llama-server said %s: %s", res.Status, strings.TrimSpace(string(said)))
	}
	return nil
}

// expandHome turns a leading ~ into the home folder.
func expandHome(dir string) string {
	if rest, ok := strings.CutPrefix(dir, "~"); ok && (rest == "" || rest[0] == '/' || rest[0] == '\\') {
		home, _ := os.UserHomeDir()
		return home + rest
	}
	return dir
}

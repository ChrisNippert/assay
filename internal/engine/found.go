package engine

import (
	"fmt"
)

func (e *Engine) finder() *Finder {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.Finder == nil {
		e.Finder = NewFinder("")
	}
	return e.Finder
}

// Found lists the models other apps have that this store doesn't.
func (e *Engine) Found(refresh bool) []Source { return e.finder().Scan(e.store, refresh) }

// Folders lists the folders the user asked to look in.
func (e *Engine) Folders() []string { return e.finder().Folders() }

// AddFolder asks to look for models in a folder.
func (e *Engine) AddFolder(dir string) error { return e.finder().AddFolder(e.store, dir) }

// RemoveFolder stops looking in a folder (it isn't touched).
func (e *Engine) RemoveFolder(dir string) error { return e.finder().RemoveFolder(dir) }

// ImportFound links models from a source (id and folder, as Found lists
// them) into the store. With no names, it takes all of them.
func (e *Engine) ImportFound(id, dir string, names []string) (int, error) {
	if e.moving.Load() {
		return 0, errMovingModels
	}
	picked, err := pickFound(e.Found(false), id, dir, names)
	if err != nil {
		return 0, err
	}
	n, err := e.store.ImportFound(id, picked)
	if n > 0 {
		e.opt.Log.Info("linked models", "source", id, "count", n)
	}
	return n, err
}

// pickFound returns the candidates an import asks for: the named ones, or
// with no names every model the source lists. Adapters are left out of
// that, so they're only imported by name.
func pickFound(sources []Source, id, dir string, names []string) ([]Candidate, error) {
	var picked []Candidate
	if len(names) == 0 {
		for _, s := range sources {
			if s.ID == id && s.Dir == dir {
				for _, c := range s.Models {
					if !c.Adapter {
						picked = append(picked, c)
					}
				}
			}
		}
		return picked, nil
	}
	for _, n := range names {
		c, ok := FindCandidate(sources, id, dir, n)
		if !ok {
			return nil, fmt.Errorf("%s isn't there any more; refresh the list", n)
		}
		picked = append(picked, c)
	}
	return picked, nil
}

// missingError explains a model whose files are gone.
func (e *Engine) missingError(m *Model) error {
	if m.Origin == "" || len(e.store.Missing(m)) == 0 {
		return nil
	}
	return fmt.Errorf("%s's file is missing: it was linked from %s, which has gone. Restore it there, or delete the model", m.Name, m.Origin)
}

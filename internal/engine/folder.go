package engine

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"

	"ollamui/internal/gguf"
)

// folderConfig is the models folder setting, kept in the config folder.
type folderConfig struct {
	Dir string `json:"dir"`
}

// LoadFolder reads the saved models folder from path ("" if none).
func LoadFolder(path string) string {
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	var c folderConfig
	if json.Unmarshal(b, &c) != nil {
		return ""
	}
	return c.Dir
}

// SaveFolder remembers the models folder in path.
func SaveFolder(path, dir string) error {
	b, err := json.MarshalIndent(folderConfig{Dir: dir}, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return writeAtomic(path, append(b, '\n'))
}

// FolderSetting says where the models folder is remembered.
type FolderSetting struct {
	// File is the config file; "" keeps nothing.
	File string
	// Pinned is set when the -models flag chose the folder: it wins over
	// the saved one on the next start.
	Pinned bool
}

// errMovingModels is returned for starting a model while the store moves.
var errMovingModels = errors.New("the models folder is being moved; try again when it's done")

// FolderPinned reports whether the -models flag chose the folder.
func (e *Engine) FolderPinned() bool { return e.Folder.Pinned }

// PlanFolder checks a new models folder.
func (e *Engine) PlanFolder(dst string) (*MovePlan, error) { return e.store.Plan(dst) }

// MoveFolder moves the models to dst (or switches to the store there),
// with every model stopped first, and remembers the choice. The engine
// uses the new folder from then on, no restart needed.
func (e *Engine) MoveFolder(ctx context.Context, dst string, opt MoveOptions) error {
	if _, err := e.store.Plan(dst); err != nil {
		return err
	}
	e.moving.Store(true)
	defer e.moving.Store(false)
	// Wait for a start in progress, then stop everything.
	e.loading.Lock()
	defer e.loading.Unlock()
	e.StopAll()
	if err := e.store.MoveTo(ctx, dst, opt); err != nil {
		return err
	}
	e.mu.Lock()
	e.meta = map[string]*gguf.File{}
	e.mu.Unlock()
	if e.Folder.File != "" {
		return SaveFolder(e.Folder.File, e.store.Dir())
	}
	return nil
}

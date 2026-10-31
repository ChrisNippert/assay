package engine

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
)

// by-name/ holds a readable symlink for every model file, e.g.
// "unsloth-Qwen3.5-2B-GGUF-Q4_K_M.gguf" -> ../blobs/sha256-….gguf, so
// the models folder can be browsed by hand. The blobs keep their hashed
// names (a file two models share is stored once). Only symlinks in
// by-name/ are ever created or removed. Windows needs admin rights for
// symlinks, so it has none.

var unsafeName = regexp.MustCompile(`[^A-Za-z0-9._+-]+`)

// readableName turns "hf.co/unsloth/Qwen3.5-2B-GGUF:Q4_K_M" into
// "unsloth-Qwen3.5-2B-GGUF-Q4_K_M".
func readableName(name string) string {
	name = strings.TrimPrefix(name, "hf.co/")
	return strings.Trim(unsafeName.ReplaceAllString(name, "-"), "-.")
}

// byNameLinks maps each link name to the file it points at.
func byNameLinks(models []*Model) map[string]string {
	want := map[string]string{}
	target := func(f string) string {
		if filepath.IsAbs(f) {
			return f
		}
		return filepath.Join("..", "blobs", f)
	}
	for _, m := range models {
		base := readableName(m.Name)
		if base == "" {
			continue
		}
		for i, f := range m.Files {
			n := base + ".gguf"
			if len(m.Files) > 1 {
				n = fmt.Sprintf("%s-%05d-of-%05d.gguf", base, i+1, len(m.Files))
			}
			want[n] = target(f)
		}
		if m.Projector != "" {
			want[base+".mmproj.gguf"] = target(m.Projector)
		}
		if m.MTP != "" {
			want[base+".mtp.gguf"] = target(m.MTP)
		}
	}
	return want
}

// syncByName brings by-name/ in line with models. Best effort: a failure
// only leaves a link missing or stale.
func syncByName(dir string, models []*Model) {
	if runtime.GOOS == "windows" {
		return
	}
	links := filepath.Join(dir, "by-name")
	if os.MkdirAll(links, 0o755) != nil {
		return
	}
	want := byNameLinks(models)
	entries, _ := os.ReadDir(links)
	for _, e := range entries {
		p := filepath.Join(links, e.Name())
		if e.Type()&os.ModeSymlink == 0 {
			continue // not ours
		}
		if t, err := os.Readlink(p); err == nil && t == want[e.Name()] {
			delete(want, e.Name())
			continue
		}
		os.Remove(p)
	}
	for n, t := range want {
		os.Symlink(t, filepath.Join(links, n))
	}
}

// clearByName removes dir's by-name/ links and then the folder, if that
// leaves it empty. The new folder makes its own.
func clearByName(dir string) {
	links := filepath.Join(dir, "by-name")
	entries, _ := os.ReadDir(links)
	for _, e := range entries {
		if e.Type()&os.ModeSymlink != 0 {
			os.Remove(filepath.Join(links, e.Name()))
		}
	}
	os.Remove(links)
}

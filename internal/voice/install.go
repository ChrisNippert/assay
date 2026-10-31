package voice

import (
	"context"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"time"
)

// Voice mode's downloads (Python packages, speech recognition and voice
// models, together most of a gigabyte) only happen after someone agrees to
// them, and Remove takes them all away again.

// DownloadSize is roughly what the first start downloads.
const DownloadSize = "about 800 MB"

// Installed reports whether the bundled service has been downloaded, so
// starting it won't fetch anything big.
func (l *Launcher) Installed() bool {
	for _, p := range []string{markerPath(), filepath.Join(modelDir(), "kokoro-v1.0.int8.onnx")} {
		if p != "" {
			if _, err := os.Stat(p); err == nil {
				return true
			}
		}
	}
	return false
}

func (l *Launcher) markInstalled() {
	if p := markerPath(); p != "" {
		if _, err := os.Stat(p); err != nil {
			os.WriteFile(p, []byte("the bundled speech service has been downloaded\n"), 0o644)
		}
	}
}

// Downloads is how much space the bundled service's downloads take.
func (l *Launcher) Downloads() (size int64) {
	seen := map[uint64]bool{}
	for _, p := range downloadPaths() {
		size += dirSize(p, seen)
	}
	return size
}

// Remove stops the bundled service and deletes everything it downloaded.
// uv hard-links the script's packages from its own cache, so those are
// cleaned from uv's cache too (other projects using them download them
// again when they need them).
func (l *Launcher) Remove() error {
	l.Stop()
	if uv, err := findUV(); err == nil {
		if pkgs := scriptPackages(); len(pkgs) > 0 {
			out, err := exec.Command(uv, append([]string{"cache", "clean"}, pkgs...)...).CombinedOutput()
			if err != nil {
				l.log.Warn("uv cache clean", "err", err, "out", string(out))
			}
		}
	}
	var first error
	for _, p := range downloadPaths() {
		if err := os.RemoveAll(p); err != nil && first == nil {
			first = err
		}
	}
	l.log.Info("removed the speech service's downloads")
	return first
}

func cacheDir() string {
	dir, err := os.UserCacheDir()
	if err != nil {
		return ""
	}
	return dir
}

func markerPath() string {
	if c := cacheDir(); c != "" {
		return filepath.Join(c, "assay", "voice-installed")
	}
	return ""
}

// modelDir is where voice_server.py keeps the Kokoro voice model.
func modelDir() string {
	if c := cacheDir(); c != "" {
		return filepath.Join(c, "assay-voice")
	}
	return ""
}

// downloadPaths are the files and folders the bundled service creates: the
// marker and script, the Kokoro model, the whisper models (in Hugging
// Face's cache) and uv's environment for the script.
func downloadPaths() []string {
	c := cacheDir()
	if c == "" {
		return nil
	}
	paths := []string{
		markerPath(),
		filepath.Join(c, "assay", "voice_server.py"),
		modelDir(),
	}
	hub := filepath.Join(c, "huggingface", "hub")
	if h := os.Getenv("HF_HUB_CACHE"); h != "" {
		hub = h
	} else if h := os.Getenv("HF_HOME"); h != "" {
		hub = filepath.Join(h, "hub")
	}
	whisper, _ := filepath.Glob(filepath.Join(hub, "models--Systran--faster-whisper-*"))
	paths = append(paths, whisper...)
	paths = append(paths, linkedBlobs(hub, whisper)...)
	envs, _ := filepath.Glob(filepath.Join(uvCacheDir(c), "environments-v2", "voice-server-*"))
	return append(paths, envs...)
}

// linkedBlobs finds the files that symlinks under dirs point to inside hub.
// Newer huggingface_hub versions keep weights in a blobs folder shared by
// every model, so removing a model's folder alone would leave them behind.
func linkedBlobs(hub string, dirs []string) []string {
	if h, err := filepath.EvalSymlinks(hub); err == nil {
		hub = h
	}
	var blobs []string
	for _, dir := range dirs {
		filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
			if err != nil || d.Type()&fs.ModeSymlink == 0 {
				return nil
			}
			if t, err := filepath.EvalSymlinks(p); err == nil {
				if rel, err := filepath.Rel(hub, t); err == nil && !strings.HasPrefix(rel, "..") && !strings.HasPrefix(rel, filepath.Join(filepath.Base(dir), "")) {
					blobs = append(blobs, t)
				}
			}
			return nil
		})
	}
	return blobs
}

func uvCacheDir(cache string) string {
	if d := os.Getenv("UV_CACHE_DIR"); d != "" {
		return d
	}
	if uv, err := findUV(); err == nil {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		if out, err := exec.CommandContext(ctx, uv, "cache", "dir").Output(); err == nil {
			if d := strings.TrimSpace(string(out)); d != "" {
				return d
			}
		}
	}
	return filepath.Join(cache, "uv")
}

// scriptPackages names the packages installed in uv's environments for the
// script, from their .dist-info folders.
func scriptPackages() []string {
	c := cacheDir()
	if c == "" {
		return nil
	}
	dirs, _ := filepath.Glob(filepath.Join(uvCacheDir(c), "environments-v2", "voice-server-*", "lib", "python*", "site-packages", "*.dist-info"))
	var pkgs []string
	for _, d := range dirs {
		name, _, _ := strings.Cut(filepath.Base(d), "-")
		if name != "" && !slices.Contains(pkgs, name) {
			pkgs = append(pkgs, name)
		}
	}
	return pkgs
}

// dirSize adds up the files under root, skipping any already in seen.
func dirSize(root string, seen map[uint64]bool) int64 {
	var n int64
	filepath.WalkDir(root, func(_ string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		info, err := d.Info()
		if err != nil || !info.Mode().IsRegular() {
			return nil
		}
		if id, ok := fileID(info); ok {
			if seen[id] {
				return nil
			}
			seen[id] = true
		}
		n += info.Size()
		return nil
	})
	return n
}

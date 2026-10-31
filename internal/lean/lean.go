// Package lean installs the Lean 4 theorem prover (core only, no Mathlib)
// from its GitHub releases and checks proofs with it, so proof benchmarks
// are graded by a machine instead of a judge.
package lean

import (
	"archive/zip"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
)

// DefaultAPI is the latest Lean 4 release on GitHub.
const DefaultAPI = "https://api.github.com/repos/leanprover/lean4/releases/latest"

// Installer keeps one Lean release in Dir.
type Installer struct {
	Dir  string
	API  string
	HTTP *http.Client

	mu       sync.Mutex
	busy     bool
	progress Progress
	lastErr  string
	cancel   context.CancelFunc
}

// Progress reports an install under way.
type Progress struct {
	Phase      string `json:"phase"` // Downloading, Unpacking
	Done       int64  `json:"done"`
	Total      int64  `json:"total"`
	Version    string `json:"version,omitempty"`
	DownloadMB int64  `json:"downloadMb,omitempty"`
}

// Status is what the UI shows.
type Status struct {
	Installed string    `json:"installed,omitempty"` // version, e.g. v4.35.0
	Path      string    `json:"path,omitempty"`
	Busy      bool      `json:"busy"`
	Progress  *Progress `json:"progress,omitempty"`
	Error     string    `json:"error,omitempty"`
	OnPath    string    `json:"onPath,omitempty"` // a lean found on PATH, used when nothing is installed
}

type record struct {
	Tag       string    `json:"tag"`
	Asset     string    `json:"asset"`
	Installed time.Time `json:"installed"`
}

func exeName() string {
	if runtime.GOOS == "windows" {
		return "lean.exe"
	}
	return "lean"
}

func (in *Installer) record() (r record) {
	b, err := os.ReadFile(filepath.Join(in.Dir, "release.json"))
	if err == nil {
		json.Unmarshal(b, &r)
	}
	return r
}

// Binary is the lean to run: the installed one, else one on PATH, else "".
func (in *Installer) Binary() string {
	if in != nil && in.Dir != "" {
		p := filepath.Join(in.Dir, "bin", exeName())
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	if p, err := lookPath(); err == nil {
		return p
	}
	return ""
}

func lookPath() (string, error) { return execLookPath("lean") }

// Status reports what is installed and any install in progress.
func (in *Installer) Status() Status {
	st := Status{}
	if r := in.record(); r.Tag != "" {
		st.Installed, st.Path = r.Tag, filepath.Join(in.Dir, "bin", exeName())
	} else if p, err := lookPath(); err == nil {
		st.OnPath = p
	}
	in.mu.Lock()
	defer in.mu.Unlock()
	st.Busy, st.Error = in.busy, in.lastErr
	if in.busy {
		p := in.progress
		st.Progress = &p
	}
	return st
}

type release struct {
	Tag    string `json:"tag_name"`
	Assets []struct {
		Name string `json:"name"`
		URL  string `json:"browser_download_url"`
		Size int64  `json:"size"`
	} `json:"assets"`
}

func (in *Installer) client() *http.Client {
	if in.HTTP != nil {
		return in.HTTP
	}
	return &http.Client{Timeout: 0}
}

// assetName is the zip for this system ("lean-4.35.0-linux.zip").
func assetName(tag, goos, goarch string) string {
	plat := map[string]string{"linux": "linux", "darwin": "darwin", "windows": "windows"}[goos]
	if goarch == "arm64" && goos != "windows" {
		plat += "_aarch64"
	}
	return "lean-" + strings.TrimPrefix(tag, "v") + "-" + plat + ".zip"
}

// Start installs the latest release in the background.
func (in *Installer) Start() error {
	in.mu.Lock()
	if in.busy {
		in.mu.Unlock()
		return errors.New("Lean is already being installed")
	}
	ctx, cancel := context.WithCancel(context.Background())
	in.busy, in.lastErr, in.cancel, in.progress = true, "", cancel, Progress{Phase: "Starting"}
	in.mu.Unlock()
	go func() {
		err := in.install(ctx)
		in.mu.Lock()
		in.busy, in.cancel = false, nil
		if err != nil {
			in.lastErr = err.Error()
			if ctx.Err() != nil {
				in.lastErr = "Cancelled."
			}
		}
		in.mu.Unlock()
		cancel()
	}()
	return nil
}

// Cancel stops an install under way.
func (in *Installer) Cancel() {
	in.mu.Lock()
	if in.cancel != nil {
		in.cancel()
	}
	in.mu.Unlock()
}

func (in *Installer) set(p Progress) {
	in.mu.Lock()
	in.progress = p
	in.mu.Unlock()
}

func (in *Installer) install(ctx context.Context) error {
	api := in.API
	if api == "" {
		api = DefaultAPI
	}
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, api, nil)
	req.Header.Set("User-Agent", "assay")
	resp, err := in.client().Do(req)
	if err != nil {
		return fmt.Errorf("couldn't reach GitHub: %w", err)
	}
	var rel release
	err = json.NewDecoder(resp.Body).Decode(&rel)
	resp.Body.Close()
	if err != nil || resp.StatusCode != http.StatusOK {
		return fmt.Errorf("couldn't read the Lean release list (%s)", resp.Status)
	}
	want := assetName(rel.Tag, runtime.GOOS, runtime.GOARCH)
	var url string
	var size int64
	for _, a := range rel.Assets {
		if a.Name == want {
			url, size = a.URL, a.Size
		}
	}
	if url == "" {
		return fmt.Errorf("release %s has no %s", rel.Tag, want)
	}
	if err := os.MkdirAll(filepath.Dir(in.Dir), 0o755); err != nil {
		return err
	}
	zipPath := in.Dir + ".download.zip"
	defer os.Remove(zipPath)
	if err := in.download(ctx, url, size, zipPath, rel.Tag); err != nil {
		return err
	}
	staging := in.Dir + ".new"
	os.RemoveAll(staging)
	defer os.RemoveAll(staging)
	in.set(Progress{Phase: "Unpacking", Version: rel.Tag})
	if err := unzip(ctx, zipPath, staging, func(done, total int64) {
		in.set(Progress{Phase: "Unpacking", Done: done, Total: total, Version: rel.Tag})
	}); err != nil {
		return err
	}
	// The zip holds one folder, lean-4.x.y-linux/.
	root := staging
	if entries, _ := os.ReadDir(staging); len(entries) == 1 && entries[0].IsDir() {
		root = filepath.Join(staging, entries[0].Name())
	}
	if _, err := os.Stat(filepath.Join(root, "bin", exeName())); err != nil {
		return errors.New("the download has no bin/" + exeName())
	}
	b, _ := json.Marshal(record{Tag: rel.Tag, Asset: want, Installed: time.Now()})
	if err := os.WriteFile(filepath.Join(root, "release.json"), b, 0o644); err != nil {
		return err
	}
	old := in.Dir + ".old"
	os.RemoveAll(old)
	if _, err := os.Stat(in.Dir); err == nil {
		if err := os.Rename(in.Dir, old); err != nil {
			return err
		}
	}
	if err := os.Rename(root, in.Dir); err != nil {
		os.Rename(old, in.Dir)
		return err
	}
	os.RemoveAll(old)
	return nil
}

func (in *Installer) download(ctx context.Context, url string, size int64, dest, tag string) error {
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	req.Header.Set("User-Agent", "assay")
	resp, err := in.client().Do(req)
	if err != nil {
		return fmt.Errorf("download failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download failed: %s", resp.Status)
	}
	if resp.ContentLength > 0 {
		size = resp.ContentLength
	}
	f, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer f.Close()
	buf := make([]byte, 512<<10)
	var done int64
	for {
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			if _, err := f.Write(buf[:n]); err != nil {
				return err
			}
			done += int64(n)
			in.set(Progress{Phase: "Downloading", Done: done, Total: size, Version: tag})
		}
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			return fmt.Errorf("download interrupted: %w", rerr)
		}
	}
	if size > 0 && done != size {
		return fmt.Errorf("download cut short: %d of %d bytes", done, size)
	}
	return f.Close()
}

func unzip(ctx context.Context, path, dest string, progress func(done, total int64)) error {
	zr, err := zip.OpenReader(path)
	if err != nil {
		return err
	}
	defer zr.Close()
	var total, done int64
	for _, f := range zr.File {
		total += int64(f.UncompressedSize64)
	}
	if total > 12<<30 {
		return errors.New("archive is too large")
	}
	for i, f := range zr.File {
		if i%200 == 0 {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			progress(done, total)
		}
		p := filepath.Join(dest, filepath.FromSlash(f.Name))
		if p != dest && !strings.HasPrefix(p, dest+string(filepath.Separator)) {
			return fmt.Errorf("archive entry %q is outside the folder", f.Name)
		}
		if f.FileInfo().IsDir() {
			if err := os.MkdirAll(p, 0o755); err != nil {
				return err
			}
			continue
		}
		if !f.Mode().IsRegular() && f.Mode()&os.ModeSymlink == 0 {
			continue
		}
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			return err
		}
		rc, err := f.Open()
		if err != nil {
			return err
		}
		if f.Mode()&os.ModeSymlink != 0 {
			target, err := io.ReadAll(io.LimitReader(rc, 4096))
			rc.Close()
			if err != nil {
				return err
			}
			t := string(target)
			if filepath.IsAbs(t) || strings.HasPrefix(filepath.Clean(filepath.Join(filepath.Dir(f.Name), t)), "..") {
				return fmt.Errorf("archive link %q points outside the folder", f.Name)
			}
			if err := os.Symlink(t, p); err != nil {
				return err
			}
			continue
		}
		out, err := os.OpenFile(p, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, f.Mode().Perm()|0o644)
		if err == nil {
			_, err = io.Copy(out, rc)
			if cerr := out.Close(); err == nil {
				err = cerr
			}
		}
		rc.Close()
		if err != nil {
			return err
		}
		done += int64(f.UncompressedSize64)
	}
	progress(total, total)
	return nil
}

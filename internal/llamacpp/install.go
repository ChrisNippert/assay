package llamacpp

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Installer keeps one release of llama.cpp in Dir. It only ever writes to
// Dir and to the siblings Dir+".new" and Dir+".old", never beside them, so
// a llama.cpp the user built or installed elsewhere is left alone.
type Installer struct {
	Dir string
	// API is the release list (default: GitHub's).
	API  string
	HTTP *http.Client
	// GOOS and GOARCH default to this system's.
	GOOS, GOARCH string
	// BeforeSwap, if set, runs just before the new files replace the old
	// (the caller stops the models using the old binary there).
	BeforeSwap func() error
}

// Progress reports an install under way.
type Progress struct {
	Phase       string // "Downloading", "Unpacking", "Installing"
	Done, Total int64
}

// record is what release.json in Dir says about the install.
type record struct {
	Tag       string    `json:"tag"`
	Asset     string    `json:"asset"`
	Installed time.Time `json:"installed"`
}

func (in *Installer) exe() string {
	if goos, _ := in.platform(); goos == "windows" {
		return "llama-server.exe"
	}
	return "llama-server"
}

// Binary is the installed llama-server ("" if there isn't one).
func (in *Installer) Binary() string {
	p := filepath.Join(in.Dir, in.exe())
	if info, err := os.Stat(p); err == nil && !info.IsDir() {
		return p
	}
	return ""
}

// Installed is the tag of the installed release ("" if none).
func (in *Installer) Installed() string {
	if in.Binary() == "" {
		return ""
	}
	b, err := os.ReadFile(filepath.Join(in.Dir, "release.json"))
	if err != nil {
		return ""
	}
	var r record
	if json.Unmarshal(b, &r) != nil {
		return ""
	}
	return r.Tag
}

// Install downloads rel and swaps it into Dir. The old install stays
// working until the new one is complete and unpacked; if the swap fails it
// is put back.
func (in *Installer) Install(ctx context.Context, rel *Release, progress func(Progress)) error {
	if progress == nil {
		progress = func(Progress) {}
	}
	if in.Dir == "" || filepath.Base(in.Dir) == "llama.cpp" {
		return errors.New("refusing to install into a folder that may hold the user's own build")
	}
	parent := filepath.Dir(in.Dir)
	if err := os.MkdirAll(parent, 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(parent, filepath.Base(in.Dir)+"-download-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	defer tmp.Close()
	if err := in.download(ctx, rel.Asset, tmp, progress); err != nil {
		return err
	}

	staging := in.Dir + ".new"
	os.RemoveAll(staging)
	if err := os.MkdirAll(staging, 0o755); err != nil {
		return err
	}
	defer os.RemoveAll(staging)
	progress(Progress{Phase: "Unpacking"})
	if strings.HasSuffix(rel.Asset.Name, ".zip") {
		err = unzip(tmp.Name(), staging)
	} else if _, err = tmp.Seek(0, io.SeekStart); err == nil {
		err = untar(tmp, staging)
	}
	if err != nil {
		return fmt.Errorf("unpacking %s: %w", rel.Asset.Name, err)
	}
	if err := flatten(staging, in.exe()); err != nil {
		return err
	}
	rec, _ := json.Marshal(record{Tag: rel.Tag, Asset: rel.Asset.Name, Installed: time.Now().UTC()})
	if err := os.WriteFile(filepath.Join(staging, "release.json"), rec, 0o644); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	progress(Progress{Phase: "Installing"})
	if in.BeforeSwap != nil {
		if err := in.BeforeSwap(); err != nil {
			return err
		}
	}
	return swap(staging, in.Dir)
}

// swap puts the new folder in place of dir, keeping the old one until the
// new one is in.
func swap(staging, dir string) error {
	old := dir + ".old"
	os.RemoveAll(old)
	hadOld := false
	if _, err := os.Lstat(dir); err == nil {
		if err := os.Rename(dir, old); err != nil {
			return fmt.Errorf("moving the old install aside: %w", err)
		}
		hadOld = true
	}
	if err := os.Rename(staging, dir); err != nil {
		if hadOld {
			os.Rename(old, dir)
		}
		return fmt.Errorf("putting the new install in place: %w", err)
	}
	os.RemoveAll(old)
	return nil
}

// download fetches a into f, checking its sha256 when GitHub gave one.
func (in *Installer) download(ctx context.Context, a Asset, f *os.File, progress func(Progress)) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, a.URL, nil)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", "ollamui")
	resp, err := in.client().Do(req)
	if err != nil {
		return fmt.Errorf("download failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download failed: %s", resp.Status)
	}
	total := resp.ContentLength
	if total <= 0 {
		total = a.Size
	}
	h := sha256.New()
	var done int64
	buf := make([]byte, 256<<10)
	var last time.Time
	for {
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			if _, err := f.Write(buf[:n]); err != nil {
				return err
			}
			h.Write(buf[:n])
			done += int64(n)
			if time.Since(last) > 200*time.Millisecond {
				last = time.Now()
				progress(Progress{Phase: "Downloading", Done: done, Total: total})
			}
		}
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			return fmt.Errorf("download interrupted: %w", rerr)
		}
	}
	if total > 0 && done != total {
		return fmt.Errorf("download cut short: got %d of %d bytes", done, total)
	}
	progress(Progress{Phase: "Downloading", Done: done, Total: total})
	if want, ok := strings.CutPrefix(a.Digest, "sha256:"); ok && !strings.EqualFold(want, hex.EncodeToString(h.Sum(nil))) {
		return errors.New("the download doesn't match its checksum")
	}
	return nil
}

// maxUnpacked bounds what an archive may expand to.
const maxUnpacked = 8 << 30

// inside resolves an archive entry name below dest, or fails if it would
// land elsewhere.
func inside(dest, name string) (string, error) {
	p := filepath.Join(dest, filepath.FromSlash(name))
	if p != dest && !strings.HasPrefix(p, dest+string(filepath.Separator)) {
		return "", fmt.Errorf("archive entry %q is outside the folder", name)
	}
	return p, nil
}

func untar(r io.Reader, dest string) error {
	gz, err := gzip.NewReader(r)
	if err != nil {
		return err
	}
	defer gz.Close()
	tr := tar.NewReader(gz)
	var total int64
	for {
		h, err := tr.Next()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		p, err := inside(dest, h.Name)
		if err != nil {
			return err
		}
		switch h.Typeflag {
		case tar.TypeDir:
			err = os.MkdirAll(p, 0o755)
		case tar.TypeReg:
			if total += h.Size; total > maxUnpacked {
				return errors.New("archive is too large")
			}
			err = writeFile(p, tr, os.FileMode(h.Mode)&0o777)
		case tar.TypeSymlink:
			// Only links that stay inside the folder (libfoo.so -> libfoo.so.0).
			target := filepath.ToSlash(filepath.Join(filepath.Dir(h.Name), h.Linkname))
			if _, lerr := inside(dest, target); lerr != nil || filepath.IsAbs(h.Linkname) {
				return fmt.Errorf("archive link %q points outside the folder", h.Name)
			}
			if err = os.MkdirAll(filepath.Dir(p), 0o755); err == nil {
				err = os.Symlink(h.Linkname, p)
			}
		}
		if err != nil {
			return err
		}
	}
}

func unzip(path, dest string) error {
	zr, err := zip.OpenReader(path)
	if err != nil {
		return err
	}
	defer zr.Close()
	var total int64
	for _, f := range zr.File {
		p, err := inside(dest, f.Name)
		if err != nil {
			return err
		}
		if f.FileInfo().IsDir() {
			if err := os.MkdirAll(p, 0o755); err != nil {
				return err
			}
			continue
		}
		if !f.Mode().IsRegular() {
			continue
		}
		if total += int64(f.UncompressedSize64); total > maxUnpacked {
			return errors.New("archive is too large")
		}
		rc, err := f.Open()
		if err != nil {
			return err
		}
		err = writeFile(p, rc, f.Mode().Perm())
		rc.Close()
		if err != nil {
			return err
		}
	}
	return nil
}

func writeFile(p string, r io.Reader, mode os.FileMode) error {
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(p, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, mode|0o644)
	if err != nil {
		return err
	}
	if _, err := io.Copy(f, r); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// flatten moves a lone top-level folder's contents up ("llama-b1234/…"),
// then checks the server binary is there.
func flatten(dir, exe string) error {
	if _, err := os.Stat(filepath.Join(dir, exe)); err != nil {
		entries, err := os.ReadDir(dir)
		if err != nil {
			return err
		}
		if len(entries) != 1 || !entries[0].IsDir() {
			return fmt.Errorf("the download has no %s in it", exe)
		}
		inner := filepath.Join(dir, entries[0].Name())
		items, err := os.ReadDir(inner)
		if err != nil {
			return err
		}
		for _, it := range items {
			if err := os.Rename(filepath.Join(inner, it.Name()), filepath.Join(dir, it.Name())); err != nil {
				return err
			}
		}
		os.Remove(inner)
	}
	if info, err := os.Stat(filepath.Join(dir, exe)); err != nil || info.IsDir() {
		return fmt.Errorf("the download has no %s in it", exe)
	}
	return nil
}

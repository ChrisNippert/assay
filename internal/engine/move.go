package engine

import (
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"syscall"
)

// Plan kinds: what choosing a folder would do.
const (
	PlanMove   = "move"   // move the store's files there
	PlanSwitch = "switch" // the folder already is a store: just use it
)

// MovePlan says what moving the store to a folder involves.
type MovePlan struct {
	From, To string
	Kind     string
	SameDisk bool // a rename: instant, nothing is copied
	// Bytes are copied when the disks differ; of those, Linked bytes
	// (LinkedCount files) are hard links, shared with an Ollama install,
	// that become real copies here.
	Bytes       int64
	Linked      int64
	LinkedCount int
	Free        int64 // free space where it's going
	// Models is how many models a switch finds in the folder.
	Models int
}

// MoveOptions tune MoveTo.
type MoveOptions struct {
	// ForceCopy copies even on one disk (for tests).
	ForceCopy bool
	// Progress is told how much of the total has been copied.
	Progress func(done, total int64, file string)
}

// ErrMoving is returned for changes while the store is being moved.
var ErrMoving = errors.New("the models folder is being moved")

// resolve makes path absolute with symlinks followed as far as it exists.
func resolve(path string) string {
	path = filepath.Clean(path)
	rest := ""
	for p := path; ; p = filepath.Dir(p) {
		if r, err := filepath.EvalSymlinks(p); err == nil {
			return filepath.Join(r, rest)
		}
		if filepath.Dir(p) == p {
			return path
		}
		rest = filepath.Join(filepath.Base(p), rest)
	}
}

// ExpandPath turns "~/x" and relative paths into absolute ones.
func ExpandPath(p string) (string, error) {
	p = strings.TrimSpace(p)
	if p == "" {
		return "", errors.New("enter a folder")
	}
	if p == "~" || strings.HasPrefix(p, "~/") {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		p = filepath.Join(home, strings.TrimPrefix(p, "~"))
	}
	if !filepath.IsAbs(p) {
		return "", errors.New("use a full path, starting with / or ~/")
	}
	return filepath.Clean(p), nil
}

func inside(parent, child string) bool {
	rel, err := filepath.Rel(parent, child)
	return err == nil && rel != "." && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}

// isStore reports whether dir looks like a model store.
func isStore(dir string) bool {
	for _, d := range []string{"blobs", "manifests"} {
		if info, err := os.Stat(filepath.Join(dir, d)); err != nil || !info.IsDir() {
			return false
		}
	}
	return true
}

// Plan checks that the store can move to dst and says what that takes.
// The errors are written to be shown.
func (s *Store) Plan(dst string) (*MovePlan, error) {
	dst, err := ExpandPath(dst)
	if err != nil {
		return nil, err
	}
	cur := s.Dir()
	rc, rd := resolve(cur), resolve(dst)
	switch {
	case rc == rd:
		return nil, errors.New("models are stored there already")
	case inside(rc, rd):
		return nil, errors.New("that folder is inside the current models folder")
	case inside(rd, rc):
		return nil, errors.New("that folder contains the current models folder; pick an empty one, or a new subfolder")
	}
	p := &MovePlan{From: cur, To: dst, Kind: PlanMove}
	info, err := os.Stat(dst)
	switch {
	case errors.Is(err, fs.ErrNotExist):
		if np := nearest(dst); !isDir(np) {
			return nil, fmt.Errorf("%s is a file, not a folder", np)
		}
	case errors.Is(err, syscall.ENOTDIR):
		return nil, errors.New("part of that path is a file, not a folder")
	case err != nil:
		return nil, err
	case !info.IsDir():
		return nil, errors.New("that is a file, not a folder")
	case isStore(dst):
		p.Kind = PlanSwitch
	default:
		entries, err := os.ReadDir(dst)
		if err != nil {
			return nil, err
		}
		for _, e := range entries {
			if e.Name() != "lost+found" {
				return nil, errors.New("that folder isn't empty and isn't a models folder; pick an empty one, or a new subfolder")
			}
		}
	}
	p.Free, _ = DiskFree(dst)
	if p.Kind == PlanSwitch {
		if ents, err := os.ReadDir(filepath.Join(dst, "manifests")); err == nil {
			for _, e := range ents {
				if strings.HasSuffix(e.Name(), ".json") {
					p.Models++
				}
			}
		}
		return p, nil
	}
	p.SameDisk = sameDisk(cur, dst)
	if !p.SameDisk {
		p.Bytes, p.Linked, p.LinkedCount = scan(cur)
	}
	return p, nil
}

func isDir(p string) bool {
	info, err := os.Stat(p)
	return err == nil && info.IsDir()
}

// scan adds up the regular files under root (symlinks are not followed),
// and how many of them are hard-linked elsewhere.
func scan(root string) (total, linked int64, linkedCount int) {
	filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil || !d.Type().IsRegular() {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return nil
		}
		total += info.Size()
		if linkCount(info) > 1 {
			linked += info.Size()
			linkedCount++
		}
		return nil
	})
	return
}

// Usage is the space the store's own files take (symlinked ones, which
// live elsewhere, are not counted), and how much is free on its disk.
func (s *Store) Usage() (used, free int64) {
	dir := s.Dir()
	used, _, _ = scan(dir)
	free, _ = DiskFree(dir)
	return
}

// MoveTo moves the store to dst and switches to it, or, when dst already
// is a store, only switches. Nothing may be using the files: stop the
// models first. On the same disk it renames; across disks it copies,
// checks every size, and deletes the originals only afterwards. A failure
// or cancel leaves the original store in place and removes the copies.
func (s *Store) MoveTo(ctx context.Context, dst string, opt MoveOptions) error {
	plan, err := s.Plan(dst)
	if err != nil {
		return err
	}
	dst = plan.To
	s.mu.Lock()
	if s.moving {
		s.mu.Unlock()
		return ErrMoving
	}
	s.moving = true
	s.mu.Unlock()
	defer func() { s.mu.Lock(); s.moving = false; s.mu.Unlock() }()

	src := plan.From
	if plan.Kind == PlanSwitch {
		return s.swap(dst)
	}
	if err := os.MkdirAll(dst, 0o755); err != nil {
		return fmt.Errorf("can't use that folder: %w", err)
	}
	if plan.SameDisk && !opt.ForceCopy {
		err := renameStore(src, dst)
		if err == nil {
			return s.finish(src, dst)
		}
		if !errors.Is(err, syscall.EXDEV) {
			return err
		}
		plan.Bytes, plan.Linked, plan.LinkedCount = scan(src)
	}
	if plan.Bytes == 0 {
		plan.Bytes, _, _ = scan(src)
	}
	if free, err := DiskFree(dst); err == nil && free < plan.Bytes {
		return fmt.Errorf("not enough space on that disk: the models need %s and %s is free", fmtBytes(plan.Bytes), fmtBytes(free))
	}
	copied, err := copyStore(ctx, src, dst, plan.Bytes, opt.Progress)
	if err != nil {
		copied.undo()
		return err
	}
	if err := s.swap(dst); err != nil {
		copied.undo()
		return err
	}
	copied.removeOriginals(src)
	return nil
}

// finish switches to dst after a rename and tidies the old folder.
func (s *Store) finish(src, dst string) error {
	if err := s.swap(dst); err != nil {
		return err
	}
	os.Remove(src) // only if empty
	return nil
}

// swap makes dst the store's folder and reads its manifests.
func (s *Store) swap(dst string) error {
	for _, d := range []string{"blobs", "manifests"} {
		if err := os.MkdirAll(filepath.Join(dst, d), 0o755); err != nil {
			return err
		}
	}
	s.mu.Lock()
	old := s.dir
	s.dir = dst
	s.mu.Unlock()
	if err := s.reload(); err != nil {
		s.mu.Lock()
		s.dir = old
		s.mu.Unlock()
		s.reload()
		return err
	}
	return nil
}

// renameStore renames blobs/ and manifests/ from src to dst, undoing the
// first if the second fails.
func renameStore(src, dst string) error {
	var done []string
	for _, d := range []string{"blobs", "manifests"} {
		os.Remove(filepath.Join(dst, d)) // an empty one in the way
		if err := os.Rename(filepath.Join(src, d), filepath.Join(dst, d)); err != nil {
			for _, x := range done {
				os.Rename(filepath.Join(dst, x), filepath.Join(src, x))
			}
			return err
		}
		done = append(done, d)
	}
	return nil
}

// copied tracks what a cross-disk move has written, to undo or finish it.
type copied struct {
	files []string // destination files, in order
	srcs  []string // their originals
	dirs  []string // destination folders made
}

func (c *copied) undo() {
	for _, f := range c.files {
		os.Remove(f)
		os.Remove(f + ".part")
	}
	for i := len(c.dirs) - 1; i >= 0; i-- {
		os.Remove(c.dirs[i]) // only if empty
	}
}

// removeOriginals deletes the source files that were copied (and any
// symlinks recreated), then the folders left empty.
func (c *copied) removeOriginals(src string) {
	for _, f := range c.srcs {
		os.Remove(f)
	}
	for _, d := range []string{"blobs", "manifests"} {
		removeEmptyDirs(filepath.Join(src, d))
	}
	os.Remove(src)
}

func removeEmptyDirs(root string) {
	var dirs []string
	filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err == nil && d.IsDir() {
			dirs = append(dirs, path)
		}
		return nil
	})
	for i := len(dirs) - 1; i >= 0; i-- {
		os.Remove(dirs[i])
	}
}

// copyStore copies blobs/ and manifests/ from src to dst. Symlinks stay
// symlinks (to an absolute target); hard links become real copies.
func copyStore(ctx context.Context, src, dst string, total int64, progress func(done, total int64, file string)) (*copied, error) {
	c := &copied{}
	var done int64
	for _, top := range []string{"blobs", "manifests"} {
		root := filepath.Join(src, top)
		err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			rel, _ := filepath.Rel(src, path)
			to := filepath.Join(dst, rel)
			switch {
			case d.IsDir():
				if _, err := os.Stat(to); errors.Is(err, fs.ErrNotExist) {
					c.dirs = append(c.dirs, to)
				}
				return os.MkdirAll(to, 0o755)
			case d.Type()&fs.ModeSymlink != 0:
				target, err := os.Readlink(path)
				if err != nil {
					return err
				}
				if !filepath.IsAbs(target) {
					target = filepath.Join(filepath.Dir(path), target)
				}
				os.Remove(to)
				if err := os.Symlink(target, to); err != nil {
					return err
				}
				c.files = append(c.files, to)
				c.srcs = append(c.srcs, path)
			case d.Type().IsRegular():
				n, err := copyFile(ctx, path, to, func(n int64) {
					if progress != nil {
						progress(done+n, total, rel)
					}
				})
				c.files = append(c.files, to)
				if err != nil {
					return err
				}
				c.srcs = append(c.srcs, path)
				done += n
			}
			return nil
		})
		if err != nil {
			return c, err
		}
	}
	// Check every copy against its original before anything is deleted.
	for i, f := range c.files {
		a, err := os.Lstat(c.srcs[i])
		if err != nil {
			return c, err
		}
		b, err := os.Lstat(f)
		if err != nil {
			return c, err
		}
		if a.Mode().IsRegular() && (!b.Mode().IsRegular() || a.Size() != b.Size()) {
			return c, fmt.Errorf("the copy of %s doesn't match the original (%d of %d bytes)", filepath.Base(f), b.Size(), a.Size())
		}
	}
	return c, nil
}

// copyFile copies src to dst through a ".part" file and returns its size.
func copyFile(ctx context.Context, src, dst string, progress func(n int64)) (int64, error) {
	in, err := os.Open(src)
	if err != nil {
		return 0, err
	}
	defer in.Close()
	info, err := in.Stat()
	if err != nil {
		return 0, err
	}
	out, err := os.OpenFile(dst+".part", os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o644)
	if err != nil {
		return 0, err
	}
	buf := make([]byte, 4<<20)
	var n int64
	for {
		if err := ctx.Err(); err != nil {
			out.Close()
			os.Remove(dst + ".part")
			return n, err
		}
		r, rerr := in.Read(buf)
		if r > 0 {
			if _, err := out.Write(buf[:r]); err != nil {
				out.Close()
				os.Remove(dst + ".part")
				return n, err
			}
			n += int64(r)
			progress(n)
		}
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			out.Close()
			os.Remove(dst + ".part")
			return n, rerr
		}
	}
	if err := out.Sync(); err != nil {
		out.Close()
		os.Remove(dst + ".part")
		return n, err
	}
	if err := out.Close(); err != nil {
		os.Remove(dst + ".part")
		return n, err
	}
	if err := os.Rename(dst+".part", dst); err != nil {
		os.Remove(dst + ".part")
		return n, err
	}
	os.Chtimes(dst, info.ModTime(), info.ModTime())
	return n, nil
}

func fmtBytes(n int64) string {
	const unit = 1000
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for m := n / unit; m >= unit; m /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(n)/float64(div), "kMGTPE"[exp])
}

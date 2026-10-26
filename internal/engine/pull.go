package engine

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"hash"
	"io"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"

	"ollamui/internal/hf"
	"ollamui/internal/ollama"
)

// HF is what downloading needs from Hugging Face (hf.Client).
type HF interface {
	Files(ctx context.Context, repo string) ([]hf.File, error)
	RecommendedParams(ctx context.Context, repo string) (map[string]any, error)
	Download(ctx context.Context, repo, file string, offset int64) (io.ReadCloser, int64, error)
}

// ParseHFName splits "hf.co/owner/repo:QUANT" (or huggingface.co/...) into
// repo and quant. The quant is "" when the name has none.
func ParseHFName(name string) (repo, quant string, ok bool) {
	rest, ok := strings.CutPrefix(name, "hf.co/")
	if !ok {
		rest, ok = strings.CutPrefix(name, "huggingface.co/")
	}
	if !ok {
		return "", "", false
	}
	repo, quant, _ = strings.Cut(rest, ":")
	if !hf.ValidRepo(repo) {
		return "", "", false
	}
	return repo, quant, true
}

// pickQuant finds the quant a tag asks for; no tag (or "latest") means
// Q4_K_M, as with Ollama, else the first that isn't a draft.
func pickQuant(quants []hf.Quant, tag string) (hf.Quant, bool) {
	if tag == "" || strings.EqualFold(tag, "latest") {
		for _, q := range quants {
			if !q.Draft && strings.EqualFold(q.Name, "Q4_K_M") {
				return q, true
			}
		}
		for _, q := range quants {
			if !q.Draft {
				return q, true
			}
		}
		return hf.Quant{}, false
	}
	for _, q := range quants {
		if strings.EqualFold(q.Name, tag) || strings.EqualFold(QuantTag(q.Name), tag) {
			return q, true
		}
	}
	return hf.Quant{}, false
}

// QuantTag is a quant's name as it goes in a model name: a draft's
// "DSpark Q8_0" becomes "DSpark-Q8_0".
func QuantTag(quant string) string { return strings.ReplaceAll(quant, " ", "-") }

var unsafeChars = regexp.MustCompile(`[^A-Za-z0-9._-]+`)

// Pull downloads a model from Hugging Face into the store, calling fn with
// progress the way Ollama's /api/pull reported it (one Digest per file).
// A stopped download resumes where it left off; files are checked against
// the sha256 Hugging Face lists for them.
func (e *Engine) Pull(ctx context.Context, name string, fn func(ollama.PullProgress) error) error {
	if e.HF == nil {
		return errors.New("downloads aren't set up")
	}
	repo, tag, ok := ParseHFName(name)
	if !ok {
		return fmt.Errorf("%s isn't a Hugging Face model (hf.co/owner/repo:QUANT)", name)
	}
	fn(ollama.PullProgress{Status: "pulling manifest"})
	files, err := e.HF.Files(ctx, repo)
	if err != nil {
		return err
	}
	quants, vision := hf.Quants(repo, files)
	q, ok := pickQuant(quants, tag)
	if !ok {
		return fmt.Errorf("%s has no %s quant", repo, tag)
	}
	byPath := map[string]hf.File{}
	for _, f := range files {
		byPath[f.Path] = f
	}

	// Where each file goes in blobs/. A single file is named by its hash,
	// so two names for the same weights share it; a split model keeps its
	// file names in a folder of its own, since llama.cpp finds the later
	// parts by name next to the first.
	dest := func(p string, split bool) string {
		if f := byPath[p]; f.SHA256() != "" && !split {
			return "sha256-" + f.SHA256() + ".gguf"
		}
		return "hf-" + unsafeChars.ReplaceAllString(repo+"-"+q.Name, "_") + "/" + path.Base(p)
	}
	m := &Model{Name: "hf.co/" + repo + ":" + QuantTag(q.Name), Repo: repo, Quant: q.Name, Source: "hf"}
	// A draft can't chat. It's a file beside the quants of the model it was
	// made for, or the whole repo is one (then the repo says what for).
	var info *hf.Repo
	if ir, ok := e.HF.(interface {
		Info(context.Context, string) (*hf.Repo, error)
	}); ok {
		info, _ = ir.Info(ctx, repo)
	}
	switch {
	case q.Draft:
		m.Draft, m.DraftFor = q.Kind, repo
		if info != nil && info.IsDraft() {
			m.DraftFor = info.DraftFor()
		}
	case info != nil && info.IsDraft():
		m.Draft, m.DraftFor = hf.DraftSlug(info.Arch()+" "+info.Name()), info.DraftFor()
	}
	isDraft := m.Draft != ""
	type job struct{ src, dst string }
	var jobs []job
	for _, p := range q.Files {
		d := dest(p, q.Split)
		m.Files = append(m.Files, d)
		jobs = append(jobs, job{p, d})
	}
	if pr := hf.Projector(files); vision && pr != "" && !isDraft {
		m.Projector = dest(pr, false)
		jobs = append(jobs, job{pr, m.Projector})
	}
	if mtp := hf.MTP(files); mtp != "" && !isDraft {
		m.MTP = dest(mtp, false)
		jobs = append(jobs, job{mtp, m.MTP})
	}
	for _, j := range jobs {
		f := byPath[j.src]
		m.Size += f.Size
		if err := e.download(ctx, repo, f, e.store.Blob(j.dst), fn); err != nil {
			return err
		}
	}
	m.Digest = byPath[q.Files[0]].SHA256()
	m.Modified = time.Now()
	if p, err := e.HF.RecommendedParams(ctx, repo); err == nil {
		m.Params = p
	}
	if old, err := e.store.Get(m.Name); err == nil {
		m.Aliases, m.Settings = old.Aliases, old.Settings
	}
	fn(ollama.PullProgress{Status: "writing manifest"})
	if err := e.store.Put(m); err != nil {
		return err
	}
	fn(ollama.PullProgress{Status: "success"})
	return nil
}

// download fetches one file to dst, resuming from dst.partial.
func (e *Engine) download(ctx context.Context, repo string, f hf.File, dst string, fn func(ollama.PullProgress) error) error {
	report := func(done int64) {
		fn(ollama.PullProgress{Status: "pulling " + path.Base(f.Path), Digest: f.Path, Total: f.Size, Completed: done})
	}
	// Two downloads can want the same file (quants sharing a projector or
	// MTP head, or a model fetched again for its head): one writes it, the
	// other waits and then finds it there.
	unlock := e.lockFile(dst)
	defer unlock()
	if info, err := os.Stat(dst); err == nil && (f.Size == 0 || info.Size() == f.Size) {
		report(info.Size())
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	part := dst + ".partial"
	var h hash.Hash
	if f.SHA256() != "" {
		h = sha256.New()
	}
	// A stall or a dropped connection resumes; a few in a row gives up.
	var lastErr error
	for attempt := 0; attempt < 5; attempt++ {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(time.Duration(attempt) * 2 * time.Second):
			}
		}
		done, err := e.downloadOnce(ctx, repo, f, part, h, report)
		if err == nil {
			break
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if errors.Is(err, hf.ErrNotFound) || strings.Contains(err.Error(), "access denied") {
			return err
		}
		lastErr = err
		e.opt.Log.Warn("download interrupted, resuming", "file", f.Path, "at", done, "err", err)
		if attempt == 4 {
			return fmt.Errorf("download %s: %w", path.Base(f.Path), lastErr)
		}
	}
	if h != nil {
		if got := hex.EncodeToString(h.Sum(nil)); got != f.SHA256() {
			os.Remove(part)
			return fmt.Errorf("%s arrived damaged (checksum mismatch); try again", path.Base(f.Path))
		}
	}
	return os.Rename(part, dst)
}

// downloadOnce appends to part from where it ends, feeding h from the
// start of the file. It returns how much part holds.
func (e *Engine) downloadOnce(ctx context.Context, repo string, f hf.File, part string, h hash.Hash, report func(int64)) (int64, error) {
	out, err := os.OpenFile(part, os.O_RDWR|os.O_CREATE, 0o644)
	if err != nil {
		return 0, err
	}
	defer out.Close()
	if h != nil {
		h.Reset()
	}
	// Re-read what's there for the hash (fast next to the network).
	var have int64
	if h != nil {
		have, err = io.Copy(h, out)
	} else {
		have, err = out.Seek(0, io.SeekEnd)
	}
	if err != nil {
		return 0, err
	}
	if f.Size > 0 && have > f.Size {
		have = 0
		if h != nil {
			h.Reset()
		}
	}
	body, _, err := e.HF.Download(ctx, repo, f.Path, have)
	if hf.IsNoRange(err) && have > 0 {
		have = 0
		if h != nil {
			h.Reset()
		}
		body, _, err = e.HF.Download(ctx, repo, f.Path, 0)
	}
	if err != nil {
		return have, err
	}
	defer body.Close()
	if err := out.Truncate(have); err != nil {
		return have, err
	}
	if _, err := out.Seek(have, io.SeekStart); err != nil {
		return have, err
	}
	if have > 0 {
		e.opt.Log.Info("resuming download", "file", f.Path, "at", have)
	}
	report(have)

	// A read that hangs for a minute counts as a dropped connection.
	stall := time.AfterFunc(time.Minute, func() { body.Close() })
	defer stall.Stop()
	buf := make([]byte, 1<<20)
	last := time.Now()
	for {
		n, rerr := body.Read(buf)
		if n > 0 {
			stall.Reset(time.Minute)
			if _, err := out.Write(buf[:n]); err != nil {
				return have, err
			}
			if h != nil {
				h.Write(buf[:n])
			}
			have += int64(n)
			if time.Since(last) > 200*time.Millisecond {
				report(have)
				last = time.Now()
			}
		}
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			if ctx.Err() != nil {
				return have, ctx.Err()
			}
			return have, rerr
		}
	}
	report(have)
	if f.Size > 0 && have != f.Size {
		return have, fmt.Errorf("got %d of %d bytes", have, f.Size)
	}
	return have, nil
}

// Copy gives a model another name, as Ollama's copy did.
func (e *Engine) Copy(ctx context.Context, src, dst string) error {
	m, err := e.store.Get(src)
	if err != nil {
		return err
	}
	if o, err := e.store.Get(dst); err == nil {
		if o.Name == m.Name {
			return nil
		}
		return fmt.Errorf("%s is already the name of another model", dst)
	}
	cp := *m
	cp.Aliases = append(append([]string(nil), m.Aliases...), dst)
	return e.store.Put(&cp)
}

// Has reports whether the store has a model by that name.
func (e *Engine) Has(name string) bool {
	_, err := e.store.Get(name)
	return err == nil
}

// Import picks up models pulled through Ollama now, rather than on the
// next list.
func (e *Engine) Import() {
	if e.OllamaDir == "" {
		return
	}
	if n, err := e.store.ImportOllama(e.OllamaDir); err == nil && n > 0 {
		e.opt.Log.Info("imported models from Ollama", "count", n)
	}
	e.mu.Lock()
	e.lastImport = time.Now()
	e.mu.Unlock()
}

// lockFile holds dst for one download at a time.
func (e *Engine) lockFile(dst string) func() {
	e.mu.Lock()
	if e.files == nil {
		e.files = map[string]*sync.Mutex{}
	}
	l := e.files[dst]
	if l == nil {
		l = &sync.Mutex{}
		e.files[dst] = l
	}
	e.mu.Unlock()
	l.Lock()
	return l.Unlock
}

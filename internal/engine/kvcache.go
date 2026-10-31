package engine

import (
	"bytes"
	"cmp"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"time"
)

// Saved slot caches. When a model unloads, the KV cache of the chat it
// last served is written to <KVDir>/<model key>/<chat id>.bin (llama-server
// does the writing, into the slot save path it was started with). When
// that chat sends again after a reload, the file is read back before the
// request, so the history isn't processed again. A file only ever holds a
// prefix of its chat, so a stale one costs a reprocess, not a wrong reply.

// kvMinTokens is the shortest history worth saving: below it, processing
// it again is about as quick as reading the file back.
const kvMinTokens = 1000

// kvLimitDefault caps the folder; the oldest files go first.
const kvLimitDefault = 2 << 30

// chatIDRe matches the chat ids Assay makes (date and hex), so an id is
// always a plain file name.
var chatIDRe = regexp.MustCompile(`^[A-Za-z0-9_-]{1,64}$`)

// chatKey returns the id if it names a chat that can have saved caches, or "".
func chatKey(id string) string {
	if chatIDRe.MatchString(id) {
		return id
	}
	return ""
}

// kvDir is the folder a model's saved caches go in for these load options,
// or "" when caches are off. The folder is keyed by everything that changes
// the cache's layout (the model, context, cache type, flash attention,
// draft), so a file is only ever read back by a server it fits.
func (e *Engine) kvDir(m *Model, o loadOpts) string {
	if e.opt.KVDir == "" || o.Embedding {
		return ""
	}
	ctx := o.Ctx
	if ctx == 0 {
		ctx = DefaultCtx
	}
	ct, fa := cmp.Or(o.CacheType, e.opt.CacheType), cmp.Or(o.FlashAttn, e.opt.FlashAttn)
	key := fmt.Sprintf("%s|%d|%s|%s|%s", m.Files[0], ctx, ct, fa, o.Draft)
	sum := sha256.Sum256([]byte(key))
	return filepath.Join(e.opt.KVDir, hex.EncodeToString(sum[:8]))
}

// slotRequest saves or restores the server's slot 0 to or from a file in
// its save path. It returns the number of tokens saved or restored.
func (e *Engine) slotRequest(p *proc, action, file string) (int, error) {
	b, _ := json.Marshal(map[string]string{"filename": file})
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, p.base+"/slots/0?action="+action, bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	res, err := e.http.Do(req)
	if err != nil {
		return 0, err
	}
	defer res.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(res.Body, 8192))
	if res.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("slot %s: %s: %s", action, res.Status, bytes.TrimSpace(body))
	}
	var out struct {
		Saved    int `json:"n_saved"`
		Restored int `json:"n_restored"`
	}
	if err := json.Unmarshal(body, &out); err != nil {
		return 0, err
	}
	return out.Saved + out.Restored, nil
}

// saveSlot writes the chat a process served last to its file, if that
// history is long enough to be worth it. It runs as the process unloads.
func (e *Engine) saveSlot(p *proc) {
	p.mu.Lock()
	chat, tokens, dir, busy := p.chat, p.chatTokens, p.kvDir, p.busy
	p.mu.Unlock()
	if chat == "" || dir == "" || tokens < kvMinTokens || busy > 0 || !p.running() {
		return
	}
	n, err := e.slotRequest(p, "save", chat+".bin")
	if err != nil {
		e.opt.Log.Warn("could not save the chat's cache", "model", p.model.Name, "chat", chat, "err", err)
		return
	}
	e.opt.Log.Info("saved the chat's cache", "model", p.model.Name, "chat", chat, "tokens", n)
	e.trimKV()
}

// restoreSlot loads a chat's saved cache into a freshly started server. The
// file is removed either way: a restored cache lives in the slot now, and
// the next unload writes it again. Returns the tokens restored, 0 if none.
func (e *Engine) restoreSlot(p *proc, chat string) int {
	if p.kvDir == "" {
		return 0
	}
	file := filepath.Join(p.kvDir, chat+".bin")
	if _, err := os.Stat(file); err != nil {
		return 0
	}
	n, err := e.slotRequest(p, "restore", chat+".bin")
	os.Remove(file)
	if err != nil {
		// Other settings, or a file cut short by a crash: process the history.
		e.opt.Log.Info("skipped the chat's saved cache", "model", p.model.Name, "chat", chat, "err", err)
		return 0
	}
	e.opt.Log.Info("restored the chat's cache", "model", p.model.Name, "chat", chat, "tokens", n)
	return n
}

// ForgetChat drops a chat's saved caches and stops a loaded server from
// saving one for it, after the chat was deleted or an earlier message
// was edited.
func (e *Engine) ForgetChat(chat string) {
	if e.opt.KVDir == "" || chatKey(chat) == "" {
		return
	}
	e.mu.Lock()
	for _, p := range e.procs {
		p.mu.Lock()
		if p.chat == chat {
			p.chat, p.chatTokens = "", 0
		}
		p.mu.Unlock()
	}
	e.mu.Unlock()
	files, _ := filepath.Glob(filepath.Join(e.opt.KVDir, "*", chat+".bin"))
	for _, f := range files {
		os.Remove(f)
	}
}

// kvFile is one saved cache, for the size cap.
type kvFile struct {
	path string
	size int64
	mod  time.Time
}

// overLimit returns the files to remove, oldest first, so that what is
// left comes to at most limit bytes.
func overLimit(files []kvFile, limit int64) []string {
	sort.SliceStable(files, func(i, j int) bool { return files[i].mod.Before(files[j].mod) })
	var total int64
	for _, f := range files {
		total += f.size
	}
	var out []string
	for _, f := range files {
		if total <= limit {
			break
		}
		out = append(out, f.path)
		total -= f.size
	}
	return out
}

// trimKV keeps the saved caches folder within the size cap.
func (e *Engine) trimKV() {
	if e.opt.KVDir == "" {
		return
	}
	var files []kvFile
	filepath.WalkDir(e.opt.KVDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || filepath.Ext(path) != ".bin" {
			return nil
		}
		if info, err := d.Info(); err == nil {
			files = append(files, kvFile{path: path, size: info.Size(), mod: info.ModTime()})
		}
		return nil
	})
	limit := e.opt.KVLimit
	if limit == 0 {
		limit = kvLimitDefault
	}
	for _, path := range overLimit(files, limit) {
		os.Remove(path)
	}
}

package engine

import (
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestArgsSlotCache(t *testing.T) {
	st, err := OpenStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	m := &Model{Name: "a:b", Files: []string{"sha256-abc.gguf"}}
	o := loadOpts{GPULayers: -1}

	e := New(st, Options{KVDir: t.TempDir()})
	a := strings.Join(e.args(m, o, 1), " ")
	if !strings.Contains(a, "--cache-reuse 256") {
		t.Errorf("no --cache-reuse: %s", a)
	}
	if want := "--slot-save-path " + e.kvDir(m, o); !strings.Contains(a, want) {
		t.Errorf("args %q lack %q", a, want)
	}

	a = strings.Join(e.args(m, loadOpts{GPULayers: -1, Embedding: true}, 1), " ")
	if strings.Contains(a, "--slot-save-path") || strings.Contains(a, "--cache-reuse") {
		t.Errorf("embedding model got cache flags: %s", a)
	}

	off := New(st, Options{})
	a = strings.Join(off.args(m, o, 1), " ")
	if strings.Contains(a, "--slot-save-path") {
		t.Errorf("caches off, but slot save path set: %s", a)
	}
	if !strings.Contains(a, "--cache-reuse 256") {
		t.Errorf("cache reuse should not depend on saved caches: %s", a)
	}
}

func TestKVDirKeyedBySettings(t *testing.T) {
	st, _ := OpenStore(t.TempDir())
	m := &Model{Name: "a:b", Files: []string{"sha256-abc.gguf"}}
	e := New(st, Options{KVDir: "/data/kvcache"})
	base := e.kvDir(m, loadOpts{GPULayers: -1})
	if filepath.Dir(base) != "/data/kvcache" {
		t.Fatalf("kvDir = %s, want a folder in /data/kvcache", base)
	}
	if again := e.kvDir(m, loadOpts{GPULayers: -1, Threads: 4}); again != base {
		t.Errorf("threads changed the folder: %s vs %s", again, base)
	}
	for name, o := range map[string]loadOpts{
		"context":    {GPULayers: -1, Ctx: 4096},
		"cache type": {GPULayers: -1, CacheType: "q8_0"},
		"flash attn": {GPULayers: -1, FlashAttn: "off"},
		"draft":      {GPULayers: -1, Draft: "other"},
	} {
		if got := e.kvDir(m, o); got == base {
			t.Errorf("%s should change the folder", name)
		}
	}
	other := &Model{Name: "c:d", Files: []string{"sha256-def.gguf"}}
	if e.kvDir(other, loadOpts{GPULayers: -1}) == base {
		t.Error("another model shares the folder")
	}
	if got := New(st, Options{}).kvDir(m, loadOpts{}); got != "" {
		t.Errorf("caches off: kvDir = %q", got)
	}
}

func TestChatKey(t *testing.T) {
	for id, ok := range map[string]bool{
		"20261024-a1b2c3d4":     true,
		"x":                     true,
		"":                      false,
		"../etc":                false,
		"a/b":                   false,
		"a.bin":                 false,
		strings.Repeat("a", 65): false,
	} {
		if (chatKey(id) != "") != ok {
			t.Errorf("chatKey(%q) accepted = %v, want %v", id, !ok, ok)
		}
	}
}

func TestOverLimit(t *testing.T) {
	now := time.Now()
	files := []kvFile{
		{path: "new", size: 40, mod: now},
		{path: "old", size: 40, mod: now.Add(-2 * time.Hour)},
		{path: "mid", size: 40, mod: now.Add(-time.Hour)},
	}
	if got := overLimit(files, 200); got != nil {
		t.Errorf("under the cap: removed %v", got)
	}
	got := overLimit(files, 80)
	if len(got) != 1 || got[0] != "old" {
		t.Errorf("cap 80: removed %v, want [old]", got)
	}
	got = overLimit(files, 10)
	if len(got) != 3 || got[0] != "old" || got[1] != "mid" || got[2] != "new" {
		t.Errorf("cap 10: removed %v, want oldest first", got)
	}
}

func TestTrimKVKeepsCap(t *testing.T) {
	dir := t.TempDir()
	st, _ := OpenStore(t.TempDir())
	e := New(st, Options{KVDir: dir, KVLimit: 1000})
	for i, n := range []int{600, 600} {
		sub := filepath.Join(dir, "m")
		os.MkdirAll(sub, 0o755)
		p := filepath.Join(sub, "c"+string(rune('a'+i))+".bin")
		os.WriteFile(p, make([]byte, n), 0o644)
		old := time.Now().Add(-time.Duration(2-i) * time.Hour)
		os.Chtimes(p, old, old)
	}
	e.trimKV()
	if _, err := os.Stat(filepath.Join(dir, "m", "ca.bin")); !os.IsNotExist(err) {
		t.Errorf("oldest file kept: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "m", "cb.bin")); err != nil {
		t.Errorf("newest file removed: %v", err)
	}
}

func TestForgetChat(t *testing.T) {
	dir := t.TempDir()
	st, _ := OpenStore(t.TempDir())
	e := New(st, Options{KVDir: dir})
	write := func(p string) string {
		os.MkdirAll(filepath.Dir(p), 0o755)
		os.WriteFile(p, []byte("kv"), 0o644)
		return p
	}
	gone := []string{write(filepath.Join(dir, "k1", "chat1.bin")), write(filepath.Join(dir, "k2", "chat1.bin"))}
	kept := write(filepath.Join(dir, "k1", "chat2.bin"))
	e.procs["m"] = &proc{model: &Model{Name: "m"}, chat: "chat1", chatTokens: 5000, done: make(chan struct{})}

	e.ForgetChat("chat1")
	for _, f := range gone {
		if _, err := os.Stat(f); !os.IsNotExist(err) {
			t.Errorf("%s not removed", f)
		}
	}
	if _, err := os.Stat(kept); err != nil {
		t.Errorf("another chat's cache removed: %v", err)
	}
	if p := e.procs["m"]; p.chat != "" || p.chatTokens != 0 {
		t.Errorf("loaded server still saves chat1: %q %d", p.chat, p.chatTokens)
	}
	e.ForgetChat("../k1") // not a chat id: nothing happens
	if _, err := os.Stat(kept); err != nil {
		t.Errorf("bad id removed files: %v", err)
	}
}

// slotServer answers llama-server's slot endpoints and records them.
func slotServer(t *testing.T, status int, reply string, calls *[]string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasPrefix(r.URL.Path, "/slots/0") {
			http.NotFound(w, r)
			return
		}
		*calls = append(*calls, r.URL.RawQuery)
		var b [256]byte
		n, _ := r.Body.Read(b[:])
		*calls = append(*calls, string(b[:n]))
		w.WriteHeader(status)
		w.Write([]byte(reply))
	}
}

func TestSaveSlotOnlyLongChats(t *testing.T) {
	var calls []string
	e, p := fakeProc(t, slotServer(t, http.StatusOK, `{"n_saved": 2000}`, &calls), nil)
	e.opt.KVDir = t.TempDir()
	p.kvDir = filepath.Join(e.opt.KVDir, "m")

	p.beginChat("chat1", 500)
	e.saveSlot(p)
	if len(calls) != 0 {
		t.Errorf("a short chat was saved: %v", calls)
	}

	p.beginChat("chat1", 2000)
	e.saveSlot(p)
	if len(calls) != 2 || calls[0] != "action=save" || !strings.Contains(calls[1], `"chat1.bin"`) {
		t.Errorf("save request = %v", calls)
	}

	calls = nil
	p.mu.Lock()
	p.busy = 1
	p.mu.Unlock()
	e.saveSlot(p)
	if len(calls) != 0 {
		t.Errorf("saved while a request was in flight: %v", calls)
	}
}

func TestRestoreSlot(t *testing.T) {
	dir := t.TempDir()
	var calls []string
	e, p := fakeProc(t, slotServer(t, http.StatusOK, `{"n_restored": 2000}`, &calls), nil)
	e.opt.KVDir = dir
	p.kvDir = dir
	file := filepath.Join(dir, "chat1.bin")

	if n := e.restoreSlot(p, "chat1"); n != 0 || len(calls) != 0 {
		t.Errorf("no file: restored %d, calls %v", n, calls)
	}

	os.WriteFile(file, []byte("kv"), 0o644)
	if n := e.restoreSlot(p, "chat1"); n != 2000 {
		t.Errorf("restored %d, want 2000", n)
	}
	if calls[0] != "action=restore" {
		t.Errorf("restore request = %v", calls)
	}
	if _, err := os.Stat(file); !os.IsNotExist(err) {
		t.Error("restored file kept")
	}

	// A file the server refuses (other settings) is dropped, not retried.
	calls = nil
	e2, p2 := fakeProc(t, slotServer(t, http.StatusBadRequest, `{"error":"mismatch"}`, &calls), nil)
	p2.kvDir = dir
	e2.opt.KVDir = dir
	os.WriteFile(file, []byte("kv"), 0o644)
	if n := e2.restoreSlot(p2, "chat1"); n != 0 {
		t.Errorf("refused restore reported %d tokens", n)
	}
	if _, err := os.Stat(file); !os.IsNotExist(err) {
		t.Error("refused file kept")
	}
}

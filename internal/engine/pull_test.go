package engine

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"ollamui/internal/hf"
	"ollamui/internal/ollama"
)

type fakeHF struct {
	files   map[string][]byte
	bad     map[string]bool // serve corrupted bytes
	offsets []int64
}

func (f *fakeHF) Files(ctx context.Context, repo string) ([]hf.File, error) {
	var out []hf.File
	for p, b := range f.files {
		sum := sha256.Sum256(b)
		fl := hf.File{Type: "file", Path: p, Size: int64(len(b))}
		fl.LFS = &struct {
			Oid string `json:"oid"`
		}{hex.EncodeToString(sum[:])}
		out = append(out, fl)
	}
	return out, nil
}

func (f *fakeHF) RecommendedParams(ctx context.Context, repo string) (map[string]any, error) {
	return map[string]any{"temperature": 0.7}, nil
}

func (f *fakeHF) Download(ctx context.Context, repo, file string, offset int64) (io.ReadCloser, int64, error) {
	b, ok := f.files[file]
	if !ok {
		return nil, 0, hf.ErrNotFound
	}
	f.offsets = append(f.offsets, offset)
	if f.bad[file] {
		b = bytes.Repeat([]byte("x"), len(b))
	}
	return io.NopCloser(bytes.NewReader(b[offset:])), int64(len(b)), nil
}

func testEngine(t *testing.T, h HF) *Engine {
	t.Helper()
	st, err := OpenStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	e := New(st, Options{})
	e.HF = h
	return e
}

func TestPullSplitWithProjectorAndResume(t *testing.T) {
	h := &fakeHF{files: map[string][]byte{
		"Q8_0/M-Q8_0-00001-of-00002.gguf": []byte("first part of the weights"),
		"Q8_0/M-Q8_0-00002-of-00002.gguf": []byte("second part"),
		"M-Q4_K_M.gguf":                   []byte("small one"),
		"mmproj-BF16.gguf":                []byte("bf16 projector"),
		"mmproj-F16.gguf":                 []byte("f16 projector"),
	}}
	e := testEngine(t, h)
	// Half of part one is already there from an earlier try.
	dir := e.store.Blob("hf-o_M-GGUF-Q8_0")
	os.MkdirAll(dir, 0o755)
	os.WriteFile(filepath.Join(dir, "M-Q8_0-00001-of-00002.gguf.partial"), []byte("first part"), 0o644)

	var last ollama.PullProgress
	if err := e.Pull(context.Background(), "hf.co/o/M-GGUF:q8_0", func(p ollama.PullProgress) error { last = p; return nil }); err != nil {
		t.Fatal(err)
	}
	if last.Status != "success" {
		t.Errorf("last status %q", last.Status)
	}
	m, err := e.store.Get("hf.co/o/M-GGUF:Q8_0")
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Files) != 2 || filepath.Base(m.Files[0]) != "M-Q8_0-00001-of-00002.gguf" || filepath.Dir(m.Files[0]) != filepath.Dir(m.Files[1]) {
		t.Errorf("files = %v", m.Files)
	}
	if b, _ := os.ReadFile(e.store.Blob(m.Files[0])); string(b) != "first part of the weights" {
		t.Errorf("resumed file = %q", b)
	}
	if h.offsets[0] != int64(len("first part")) {
		t.Errorf("first download started at %d, want resume", h.offsets[0])
	}
	if b, _ := os.ReadFile(e.store.Blob(m.Projector)); string(b) != "f16 projector" {
		t.Errorf("projector = %q", b)
	}
	if m.Params["temperature"] != 0.7 || m.Source != "hf" || m.Quant != "Q8_0" {
		t.Errorf("model = %+v", m)
	}

	// Again: nothing to fetch.
	n := len(h.offsets)
	if err := e.Pull(context.Background(), "hf.co/o/M-GGUF:Q8_0", func(ollama.PullProgress) error { return nil }); err != nil {
		t.Fatal(err)
	}
	if len(h.offsets) != n {
		t.Errorf("re-pull downloaded again")
	}

	if err := e.Copy(context.Background(), "hf.co/o/M-GGUF:Q8_0", "m:q8"); err != nil {
		t.Fatal(err)
	}
	if !e.Has("m:q8") {
		t.Error("alias missing")
	}
	if err := e.Delete(context.Background(), "hf.co/o/M-GGUF:Q8_0"); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Errorf("split folder left behind: %v", err)
	}
}

func TestPullDefaultQuantAndDamage(t *testing.T) {
	h := &fakeHF{files: map[string][]byte{
		"M-Q4_K_M.gguf": []byte("weights"),
		"M-Q8_0.gguf":   []byte("bigger weights"),
	}, bad: map[string]bool{"M-Q4_K_M.gguf": true}}
	e := testEngine(t, h)
	err := e.Pull(context.Background(), "hf.co/o/M-GGUF", func(ollama.PullProgress) error { return nil })
	if err == nil || !strings.Contains(err.Error(), "damaged") {
		t.Fatalf("err = %v, want checksum failure on the default Q4_K_M", err)
	}
	if _, err := e.store.Get("hf.co/o/M-GGUF:Q4_K_M"); err == nil {
		t.Error("damaged model was saved")
	}
	if err := e.Pull(context.Background(), "qwen3:8b", func(ollama.PullProgress) error { return nil }); err == nil {
		t.Error("non-HF name accepted")
	}
}

func TestParseHFName(t *testing.T) {
	for in, want := range map[string][2]string{
		"hf.co/unsloth/Qwen3-8B-GGUF:UD-Q4_K_XL": {"unsloth/Qwen3-8B-GGUF", "UD-Q4_K_XL"},
		"huggingface.co/a/b":                     {"a/b", ""},
	} {
		r, q, ok := ParseHFName(in)
		if !ok || r != want[0] || q != want[1] {
			t.Errorf("%s -> %s %s %v", in, r, q, ok)
		}
	}
	for _, in := range []string{"qwen3:8b", "hf.co/../x:Q4", "hf.co/onlyowner"} {
		if _, _, ok := ParseHFName(in); ok {
			t.Errorf("%s accepted", in)
		}
	}
}

func TestTwoPullsOfOneModelAtOnce(t *testing.T) {
	h := &fakeHF{files: map[string][]byte{
		"M-Q4_K_M.gguf":   bytes.Repeat([]byte("w"), 1<<20),
		"mmproj-F16.gguf": []byte("projector"),
		"mtp-M-Q8_0.gguf": []byte("mtp head"),
	}}
	e := testEngine(t, h)
	var wg sync.WaitGroup
	errs := make([]error, 4)
	for i := range errs {
		wg.Add(1)
		go func() {
			defer wg.Done()
			errs[i] = e.Pull(context.Background(), "hf.co/o/M-GGUF:Q4_K_M", func(ollama.PullProgress) error { return nil })
		}()
	}
	wg.Wait()
	for i, err := range errs {
		if err != nil {
			t.Fatalf("pull %d: %v", i, err)
		}
	}
	m, err := e.store.Get("hf.co/o/M-GGUF:Q4_K_M")
	if err != nil || m.MTP == "" {
		t.Fatalf("model %+v, %v", m, err)
	}
}

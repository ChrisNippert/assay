package presets

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRoundTrip(t *testing.T) {
	dir := t.TempDir()
	s, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	think := true
	opts, err := CleanOptions(map[string]any{"temperature": 0.7, "num_ctx": float64(32768), "stop": []any{"<|im_end|>"}})
	if err != nil {
		t.Fatal(err)
	}
	p := &Preset{Name: "coder", Base: "qwen3:8b", System: "Be terse.", Think: &think, Published: true, Options: opts}
	if err := s.Save(p); err != nil {
		t.Fatal(err)
	}

	raw, _ := os.ReadFile(filepath.Join(dir, "coder.toml"))
	if !strings.Contains(string(raw), `base = "qwen3:8b"`) || !strings.Contains(string(raw), "num_ctx = 32768") {
		t.Errorf("file is not the TOML we expect:\n%s", raw)
	}

	got, err := s.Get("coder")
	if err != nil {
		t.Fatal(err)
	}
	if got.Base != "qwen3:8b" || got.System != "Be terse." || got.Think == nil || !*got.Think || !got.Published {
		t.Fatalf("got %+v", got)
	}
	if got.Options["num_ctx"] != int64(32768) || got.Options["temperature"] != 0.7 {
		t.Fatalf("options = %#v", got.Options)
	}
	if stop := got.Options["stop"].([]string); len(stop) != 1 || stop[0] != "<|im_end|>" {
		t.Fatalf("stop = %#v", got.Options["stop"])
	}
	if got.OllamaName() != "ollamui/coder" {
		t.Error(got.OllamaName())
	}

	list, _ := s.List()
	if len(list) != 1 {
		t.Fatalf("list = %v", list)
	}
	if err := s.Delete("coder"); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Get("coder"); !errors.Is(err, ErrNotFound) {
		t.Fatal(err)
	}
}

func TestCleanOptions(t *testing.T) {
	if _, err := CleanOptions(map[string]any{"temprature": 0.5}); err == nil {
		t.Error("accepted a typo")
	}
	if _, err := CleanOptions(map[string]any{"top_k": 2.5}); err == nil {
		t.Error("accepted a fractional top_k")
	}
	if _, err := CleanOptions(map[string]any{"temperature": "hot"}); err == nil {
		t.Error("accepted a string temperature")
	}
	for _, bad := range []map[string]any{{"speedup": "fast"}, {"spec_ngram": "ngram-bogus"}, {"spec_ngram": 1.0}} {
		if _, err := CleanOptions(bad); err == nil {
			t.Errorf("accepted %v", bad)
		}
	}
	if _, err := CleanOptions(map[string]any{"speedup": "mtp", "spec_ngram": "ngram-mod"}); err != nil {
		t.Error(err)
	}
	got, err := CleanOptions(map[string]any{"top_k": float64(40), "temperature": int64(1)})
	if err != nil || got["top_k"] != int64(40) || got["temperature"] != 1.0 {
		t.Errorf("got %#v, %v", got, err)
	}
}

func TestNames(t *testing.T) {
	for _, n := range []string{"coder", "qwen3-8b.creative", "a_b"} {
		if !ValidName(n) {
			t.Errorf("%q rejected", n)
		}
	}
	for _, n := range []string{"", "../x", "Coder", "a b", "-x", strings.Repeat("a", 65)} {
		if ValidName(n) {
			t.Errorf("%q accepted", n)
		}
	}
}

func TestMerge(t *testing.T) {
	base := map[string]any{"temperature": 0.6, "top_k": int64(20)}
	got := Merge(base, map[string]any{"temperature": 0.9})
	if got["temperature"] != 0.9 || got["top_k"] != int64(20) || base["temperature"] != 0.6 {
		t.Errorf("merge = %v, base = %v", got, base)
	}
}

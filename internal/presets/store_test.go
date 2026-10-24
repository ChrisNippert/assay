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

func TestReplyFormatOptions(t *testing.T) {
	schema := `{"type":"object","properties":{"ok":{"type":"boolean"}}}`
	got, err := CleanOptions(map[string]any{"reply_format": "schema", "reply_schema": schema})
	if err != nil || got["reply_format"] != "schema" || got["reply_schema"] != schema {
		t.Errorf("got %#v, %v", got, err)
	}
	for _, bad := range []map[string]any{
		{"reply_format": "yaml"},
		{"reply_format": "json", "reply_schema": "{nope"},
		{"reply_format": "schema", "reply_schema": `[1,2]`},
		{"reply_schema": strings.Repeat(" ", maxReplyText+1) + "{}"},
		{"reply_grammar": strings.Repeat("a", maxReplyText+1)},
		{"reply_schema": 3.0},
	} {
		if _, err := CleanOptions(bad); err == nil {
			t.Errorf("accepted %v", bad)
		} else if strings.Contains(err.Error(), "nope") || len(err.Error()) > 200 {
			t.Errorf("error should not echo the value: %v", err)
		}
	}
	if err := CheckReplyFormat(map[string]any{"reply_format": "schema"}); err == nil {
		t.Error("schema mode without a schema passed")
	}
	if err := CheckReplyFormat(map[string]any{"reply_format": "grammar", "reply_grammar": `root ::= "a"`}); err != nil {
		t.Error(err)
	}
	if err := CheckReplyFormat(map[string]any{"reply_format": "json"}); err != nil {
		t.Error(err)
	}
}

func TestExpertOptions(t *testing.T) {
	got, err := CleanOptions(map[string]any{"n_cpu_moe": float64(-1), "load_mode": "mlock"})
	if err != nil || got["n_cpu_moe"] != int64(-1) || got["load_mode"] != "mlock" {
		t.Fatalf("clean = %v, %v", got, err)
	}
	if got, err := CleanOptions(map[string]any{"n_cpu_moe": 0, "load_mode": "none"}); err != nil || got["n_cpu_moe"] != int64(0) {
		t.Fatalf("zero layers: %v, %v", got, err)
	}
	for _, bad := range []map[string]any{
		{"n_cpu_moe": -2},
		{"n_cpu_moe": 1000},
		{"n_cpu_moe": 2.5},
		{"n_cpu_moe": "all"},
		{"load_mode": "auto"}, // Auto is no setting
		{"load_mode": "dio"},
		{"load_mode": 1},
	} {
		if _, err := CleanOptions(bad); err == nil {
			t.Errorf("%v accepted", bad)
		}
	}
}

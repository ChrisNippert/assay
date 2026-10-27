// Package presets stores named model settings as TOML files.
package presets

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/BurntSushi/toml"
)

// ErrNotFound is returned when a preset doesn't exist.
var ErrNotFound = errors.New("preset not found")

// Namespace prefixes the Ollama models ollamui creates for published presets.
// ollamui never touches models outside it.
const Namespace = "ollamui/"

// Preset is a base model plus settings.
type Preset struct {
	Name   string `toml:"-" json:"name"`
	Base   string `toml:"base" json:"base"`
	System string `toml:"system,omitempty" json:"system,omitempty"`
	Think  *bool  `toml:"think,omitempty" json:"think,omitempty"`
	// Compress is "Compress automatically" for chats with this preset; nil
	// follows the default in Settings.
	Compress  *bool          `toml:"compress,omitempty" json:"compress,omitempty"`
	Tools     *bool          `toml:"tools,omitempty" json:"tools,omitempty"`
	Published bool           `toml:"published" json:"published"`
	Options   map[string]any `toml:"options,omitempty" json:"options,omitempty"`
}

// OllamaName is the model name a published preset gets in Ollama.
func (p *Preset) OllamaName() string { return Namespace + p.Name }

var namePattern = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]{0,63}$`)

// ValidName reports whether name can be used as a preset (and Ollama model) name.
func ValidName(name string) bool { return namePattern.MatchString(name) }

// Store keeps presets in a directory, one file each.
type Store struct {
	dir string
	mu  sync.Mutex
}

// Open returns a store rooted at dir, creating it if needed.
func Open(dir string) (*Store, error) {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, err
	}
	return &Store{dir: dir}, nil
}

func (s *Store) path(name string) (string, error) {
	if !ValidName(name) {
		return "", ErrNotFound
	}
	return filepath.Join(s.dir, name+".toml"), nil
}

// Get loads one preset.
func (s *Store) Get(name string) (*Preset, error) {
	p, err := s.path(name)
	if err != nil {
		return nil, err
	}
	b, err := os.ReadFile(p)
	if errors.Is(err, os.ErrNotExist) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, err
	}
	var out Preset
	if _, err := toml.Decode(string(b), &out); err != nil {
		return nil, fmt.Errorf("preset %s: %w", name, err)
	}
	out.Name = name
	if out.Options, err = CleanOptions(out.Options); err != nil {
		return nil, fmt.Errorf("preset %s: %w", name, err)
	}
	return &out, nil
}

// List returns all presets sorted by name. Files that fail to parse are skipped.
func (s *Store) List() ([]*Preset, error) {
	entries, err := os.ReadDir(s.dir)
	if err != nil {
		return nil, err
	}
	var out []*Preset
	for _, e := range entries {
		name, ok := strings.CutSuffix(e.Name(), ".toml")
		if !ok || !ValidName(name) {
			continue
		}
		if p, err := s.Get(name); err == nil {
			out = append(out, p)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out, nil
}

// Save writes a preset atomically.
func (s *Store) Save(p *Preset) error {
	path, err := s.path(p.Name)
	if err != nil {
		return fmt.Errorf("invalid preset name %q", p.Name)
	}
	var buf bytes.Buffer
	if err := toml.NewEncoder(&buf).Encode(p); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	f, err := os.CreateTemp(s.dir, ".tmp-*")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	if _, err := f.Write(buf.Bytes()); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(f.Name(), path)
}

// Delete removes a preset file.
func (s *Store) Delete(name string) error {
	p, err := s.path(name)
	if err != nil {
		return err
	}
	if err := os.Remove(p); errors.Is(err, os.ErrNotExist) {
		return ErrNotFound
	} else {
		return err
	}
}

// Option kinds Ollama accepts that the UI exposes.
var (
	intOptions   = map[string]bool{"top_k": true, "num_ctx": true, "num_predict": true, "seed": true, "repeat_last_n": true, "num_gpu": true, "num_thread": true, "draft_max": true, "spec_ngram_max": true}
	floatOptions = map[string]bool{"temperature": true, "top_p": true, "min_p": true, "repeat_penalty": true, "presence_penalty": true, "frequency_penalty": true}
	// stringOptions are how llama.cpp loads a model; keep_alive is "-1"
	// (until unloaded) or a duration.
	stringOptions = map[string]func(string) bool{
		"cache_type": func(v string) bool { return v == "f16" || v == "q8_0" || v == "q4_0" },
		"flash_attn": func(v string) bool { return v == "on" || v == "off" || v == "auto" },
		// A model's name; the engine checks it's installed and compatible.
		"draft_model": func(v string) bool { return v != "" && len(v) < 300 },
		// What drafts for the model (auto: its MTP head if it has one), and
		// whether to guess from the conversation ("on" is ngram-mod).
		"draft_gpu":  func(v string) bool { return v == "auto" || v == "always" },
		"mmproj_cpu": func(v string) bool { return v == "on" || v == "off" },
		"vision":     func(v string) bool { return v == "on" || v == "off" },
		"speedup":    func(v string) bool { return v == "auto" || v == "off" || v == "mtp" },
		"spec_ngram": func(v string) bool {
			switch v {
			case "on", "off", "ngram-mod", "ngram-cache", "ngram-simple", "ngram-map-k", "ngram-map-k4v":
				return true
			}
			return false
		},
		"keep_alive": func(v string) bool {
			d, err := time.ParseDuration(v)
			return v == "-1" || (err == nil && d > 0)
		},
		// text is the default; json, schema and grammar constrain the reply.
		"reply_format": func(v string) bool { return v == "text" || v == "json" || v == "schema" || v == "grammar" },
		// How llama.cpp reads the file into memory (--load-mode). Auto is
		// no setting at all, so the option is absent.
		"load_mode": func(v string) bool { return v == "mmap" || v == "none" || v == "mlock" },
	}
	// maxReplyText caps a reply's JSON schema or grammar text.
	maxReplyText = 32 << 10
)

// CleanOptions validates model options and normalizes their types (ints as
// int64, floats as float64, stop as []string). Unknown keys are rejected so
// typos don't silently do nothing.
func CleanOptions(in map[string]any) (map[string]any, error) {
	if len(in) == 0 {
		return nil, nil
	}
	out := make(map[string]any, len(in))
	for k, v := range in {
		switch {
		case k == "n_cpu_moe":
			// -1 is all of them (--cpu-moe); otherwise the first N layers' experts.
			n, ok := toFloat(v)
			if !ok || n != float64(int64(n)) || n < -1 || n > 999 {
				return nil, fmt.Errorf("option n_cpu_moe must be -1 (all) or a count of layers")
			}
			out[k] = int64(n)
		case intOptions[k]:
			n, ok := toFloat(v)
			if !ok || n != float64(int64(n)) {
				return nil, fmt.Errorf("option %s must be a whole number", k)
			}
			out[k] = int64(n)
		case floatOptions[k]:
			n, ok := toFloat(v)
			if !ok {
				return nil, fmt.Errorf("option %s must be a number", k)
			}
			out[k] = n
		case stringOptions[k] != nil:
			str, ok := v.(string)
			if !ok || !stringOptions[k](str) {
				return nil, fmt.Errorf("option %s can't be %v", k, v)
			}
			out[k] = str
		case k == "stop":
			list, ok := toStrings(v)
			if !ok {
				return nil, fmt.Errorf("option stop must be a list of strings")
			}
			out[k] = list
		case k == "lora":
			list, err := cleanLoRA(v)
			if err != nil {
				return nil, err
			}
			if len(list) > 0 {
				out[k] = list
			}
		case k == "reply_schema":
			str, ok := v.(string)
			if !ok || len(str) > maxReplyText {
				return nil, fmt.Errorf("reply schema must be text of at most 32 KB")
			}
			if str != "" && !isJSONObject(str) {
				return nil, fmt.Errorf("reply schema must be a JSON object")
			}
			out[k] = str
		case k == "reply_grammar":
			str, ok := v.(string)
			if !ok || len(str) > maxReplyText {
				return nil, fmt.Errorf("reply grammar must be text of at most 32 KB")
			}
			out[k] = str
		default:
			return nil, fmt.Errorf("unknown option %q", k)
		}
	}
	return out, nil
}

// isJSONObject reports whether s is one JSON object.
func isJSONObject(s string) bool {
	var v any
	if err := json.Unmarshal([]byte(s), &v); err != nil {
		return false
	}
	_, ok := v.(map[string]any)
	return ok
}

// CheckReplyFormat reports a reply format that asks for a schema or grammar
// with none set, once the options are merged over a preset's.
func CheckReplyFormat(opts map[string]any) error {
	switch mode, _ := opts["reply_format"].(string); mode {
	case "schema":
		if s, _ := opts["reply_schema"].(string); s == "" {
			return fmt.Errorf("the reply format is JSON schema, but no schema is set")
		}
	case "grammar":
		if g, _ := opts["reply_grammar"].(string); g == "" {
			return fmt.Errorf("the reply format is grammar, but no grammar is set")
		}
	}
	return nil
}

// Merge returns base with overrides applied on top. Neither input is modified.
func Merge(base, overrides map[string]any) map[string]any {
	if len(base) == 0 && len(overrides) == 0 {
		return nil
	}
	out := maps.Clone(base)
	if out == nil {
		out = map[string]any{}
	}
	maps.Copy(out, overrides)
	return out
}

func toFloat(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case int64:
		return float64(n), true
	case int:
		return float64(n), true
	}
	return 0, false
}

func toStrings(v any) ([]string, bool) {
	switch l := v.(type) {
	case []string:
		return l, true
	case []any:
		out := make([]string, 0, len(l))
		for _, x := range l {
			s, ok := x.(string)
			if !ok {
				return nil, false
			}
			out = append(out, s)
		}
		return out, true
	}
	return nil, false
}

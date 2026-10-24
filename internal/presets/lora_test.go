package presets

import (
	"testing"
)

func TestCleanLoRA(t *testing.T) {
	opts, err := CleanOptions(map[string]any{"lora": []any{
		map[string]any{"name": "tuned", "scale": 0.5},
		map[string]any{"name": "style", "scale": int64(1)},
	}})
	if err != nil {
		t.Fatal(err)
	}
	list, ok := LoRAs(opts["lora"])
	if !ok || len(list) != 2 || list[0] != (LoRA{"tuned", 0.5}) || list[1] != (LoRA{"style", 1}) {
		t.Fatalf("cleaned lora = %#v", opts["lora"])
	}

	// An empty list is no setting at all.
	if opts, err := CleanOptions(map[string]any{"lora": []any{}}); err != nil || len(opts) != 0 {
		t.Fatalf("empty lora: %#v %v", opts, err)
	}

	bad := map[string]any{
		"scale too big":  []any{map[string]any{"name": "a", "scale": 2.5}},
		"scale negative": []any{map[string]any{"name": "a", "scale": -0.1}},
		"no name":        []any{map[string]any{"name": "", "scale": 1.0}},
		"no scale":       []any{map[string]any{"name": "a"}},
		"listed twice":   []any{map[string]any{"name": "a", "scale": 1.0}, map[string]any{"name": "a", "scale": 0.5}},
		"not a list":     "a",
		"not maps":       []any{"a"},
	}
	for what, v := range bad {
		if _, err := CleanOptions(map[string]any{"lora": v}); err == nil {
			t.Errorf("%s: accepted %#v", what, v)
		}
	}
	many := make([]any, maxLoRA+1)
	for i := range many {
		many[i] = map[string]any{"name": string(rune('a' + i)), "scale": 1.0}
	}
	if _, err := CleanOptions(map[string]any{"lora": many}); err == nil {
		t.Error("accepted too many adapters")
	}
}

func TestLoRAPresetRoundTrip(t *testing.T) {
	s, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	opts, err := CleanOptions(map[string]any{"lora": []any{map[string]any{"name": "tuned", "scale": 0.75}}, "temperature": 0.6})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Save(&Preset{Name: "coder", Base: "qwen3:8b", Options: opts}); err != nil {
		t.Fatal(err)
	}
	got, err := s.Get("coder")
	if err != nil {
		t.Fatal(err)
	}
	list, ok := LoRAs(got.Options["lora"])
	if !ok || len(list) != 1 || list[0] != (LoRA{"tuned", 0.75}) {
		t.Fatalf("after a TOML round trip: %#v", got.Options["lora"])
	}
}

package hf

import (
	"maps"
	"testing"
)

// Excerpts of real model cards, one per way they write settings.
func TestCardSettings(t *testing.T) {
	cases := []struct {
		name string
		card string
		want []CardProfile
	}{
		{"bullets (Gemma 4)", "### Recommended settings\n\nUse the following standardized sampling configuration across all use cases:\n\n* `temperature=1.0`  \n* `top_p=0.95`  \n* `top_k=64`\n\nMore text.",
			[]CardProfile{{"Use the following standardized sampling configuration across all use cases", map[string]any{"temperature": 1.0, "top_p": 0.95, "top_k": 64}}}},
		{"one line per profile, code skipped (Qwen3.5)", "> - Thinking mode for general tasks: `temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0`\n> - Instruct (or non-thinking) mode for general tasks: `temperature=0.7, top_p=0.8, top_k=20, min_p=0.0`\n\n```python\nresponse = client.chat(\n    temperature=0.3,\n    top_p=0.5,\n    max_tokens=81920,\n)\n```",
			[]CardProfile{
				{"Thinking mode for general tasks", map[string]any{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repeat_penalty": 1.0}},
				{"Instruct (or non-thinking) mode for general tasks", map[string]any{"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0}},
			}},
		{"table (Qwythos)", "## Sampling recommendations\n\n| Parameter | Value |\n|---|---|\n| `temperature` | 0.6 |\n| `top_p` | 0.95 |\n| `top_k` | 20 |\n\nAfter.",
			[]CardProfile{{"Sampling recommendations", map[string]any{"temperature": 0.6, "top_p": 0.95, "top_k": 20}}}},
		{"escaped equals (Muse)", "**Sampling Parameters:** Use the following configuration:\n\n* temperature \\= 1.0  \n* top\\_p \\= 0.95  \n",
			[]CardProfile{{"Sampling Parameters: Use the following configuration", map[string]any{"temperature": 1.0, "top_p": 0.95}}}},
		{"prose in a table cell (Nemotron)", "| **Recommended Sampling** | Temperature 1.0, top_p 0.95 in both thinking modes |",
			[]CardProfile{{"Recommended Sampling", map[string]any{"temperature": 1.0, "top_p": 0.95}}}},
		{"nothing", "This model is great. Use temperature wisely.", nil},
		{"insane values dropped", "temperature=7, top_p=3", nil},
	}
	for _, c := range cases {
		got := CardSettings(c.card)
		if len(got) != len(c.want) {
			t.Errorf("%s: got %d profiles %+v, want %d", c.name, len(got), got, len(c.want))
			continue
		}
		for i := range got {
			if got[i].Label != c.want[i].Label || !maps.Equal(got[i].Options, c.want[i].Options) {
				t.Errorf("%s #%d: got %q %v, want %q %v", c.name, i, got[i].Label, got[i].Options, c.want[i].Label, c.want[i].Options)
			}
		}
	}
}

package server

import (
	"strings"
	"testing"

	"assay/internal/ollama"
)

func TestPruneMessages(t *testing.T) {
	body := strings.Repeat("    x := 1\n", 20)
	code := func(first string) string { return "```go\n" + first + "\n" + body + "```" }
	msgs := []ollama.Message{
		{Role: "system", Content: code("func main() {")},
		{Role: "assistant", Content: "first try\n" + code("func main() {")},
		{Role: "user", Content: code("func other() {")},
		{Role: "assistant", Content: "fixed\n" + code("func main() {")},
		{Role: "user", Content: "a"}, {Role: "assistant", Content: "b"}, {Role: "user", Content: "c"},
	}
	out := pruneMessages(msgs, 4)
	if out[0].Content != msgs[0].Content {
		t.Error("system message changed")
	}
	if !strings.Contains(out[1].Content, pruneNote) || !strings.Contains(out[1].Content, "first try") {
		t.Errorf("superseded block not pruned: %q", out[1].Content)
	}
	if out[2].Content != msgs[2].Content {
		t.Error("unrelated block pruned")
	}
	for i := 3; i < len(msgs); i++ {
		if out[i].Content != msgs[i].Content {
			t.Errorf("message %d in the kept tail changed", i)
		}
	}
	if msgs[1].Content == out[1].Content {
		t.Error("input slice was modified or nothing pruned")
	}
}

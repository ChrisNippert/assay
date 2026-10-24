package hf

import (
	"slices"
	"testing"
)

func TestQuantsTernary(t *testing.T) {
	qs, _ := Quants("prism-ml/Ternary-Bonsai-2-27B-gguf", []File{
		{Type: "file", Path: "Ternary-Bonsai-2-27B-PQ2_0.gguf", Size: 7e9},
		{Type: "file", Path: "Ternary-Bonsai-2-27B-TQ1_0.gguf", Size: 6e9},
		{Type: "file", Path: "Ternary-Bonsai-2-27B-mmproj-BF16.gguf", Size: 9e8},
	})
	var names []string
	for _, q := range qs {
		names = append(names, q.Name)
	}
	if len(names) != 2 || !slices.Contains(names, "PQ2_0") || !slices.Contains(names, "TQ1_0") {
		t.Errorf("quants = %v", names)
	}
}

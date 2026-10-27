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

func TestQuantsLeaveMTPOut(t *testing.T) {
	files := []File{
		{Type: "file", Path: "gemma-4-12B-it-qat-UD-Q4_K_XL.gguf", Size: 7e9},
		{Type: "file", Path: "mmproj-F16.gguf", Size: 9e8},
		{Type: "file", Path: "mtp-gemma-4-12B-it.gguf", Size: 2.5e8},
	}
	qs, vision := Quants("unsloth/gemma-4-12B-it-qat-GGUF", files)
	if len(qs) != 1 || qs[0].Name != "UD-Q4_K_XL" || qs[0].Draft || !vision {
		t.Errorf("quants = %+v, vision %v", qs, vision)
	}
	if got := MTP(files); got != "mtp-gemma-4-12B-it.gguf" {
		t.Errorf("MTP = %q", got)
	}
	if got := MTP(files[:2]); got != "" {
		t.Errorf("MTP without one = %q", got)
	}
}

func TestMTPPicksOneHead(t *testing.T) {
	files := []File{
		{Type: "file", Path: "mtp/M-BF16.gguf"}, {Type: "file", Path: "mtp/M-Q8_0.gguf"}, {Type: "file", Path: "M-Q4_K_M.gguf"},
	}
	if got := MTP(files); got != "mtp/M-Q8_0.gguf" {
		t.Errorf("MTP = %q, want the Q8_0 head", got)
	}
	if !IsMTPFile("mtp/x.gguf") || !IsMTPFile("sub/mtp-x.GGUF") || IsMTPFile("x-mtp.gguf") || IsMTPFile("mtp-x.bin") {
		t.Error("IsMTPFile")
	}
}

func TestDraftKinds(t *testing.T) {
	qs, _ := Quants("o/M-GGUF", []File{
		{Type: "file", Path: "M-Q4_K_M.gguf", Size: 5},
		{Type: "file", Path: "dspark-M-Q8_0.gguf", Size: 1},
		{Type: "file", Path: "eagle3/M-eagle3-BF16.gguf", Size: 2},
		{Type: "file", Path: "M-DFlash-Q8_0.gguf", Size: 3},
	})
	kinds := map[string]string{}
	for _, q := range qs {
		kinds[q.Name] = q.Kind
	}
	if kinds["Q4_K_M"] != "" || kinds["DSpark Q8_0"] != "dspark" || kinds["EAGLE3 BF16"] != "eagle3" || kinds["DFlash Q8_0"] != "dflash" {
		t.Errorf("kinds = %v", kinds)
	}
}

func TestQuantWithWordsAfter(t *testing.T) {
	qs, _ := Quants("prism-ml/Ternary-Bonsai-2-27B-gguf-dev", []File{{Type: "file", Path: "Ternary-Bonsai-2-27B-Q2_0-prism-fork-required.gguf", Size: 7}})
	if len(qs) != 1 || qs[0].Name != "Q2_0" {
		t.Fatalf("quants = %+v", qs)
	}
	// A name ending in its quant still reads it from the end.
	qs, _ = Quants("o/M-GGUF", []File{{Type: "file", Path: "M-Q8_0-v2-Q4_K_M.gguf", Size: 7}})
	if len(qs) != 1 || qs[0].Name != "Q4_K_M" {
		t.Fatalf("quants = %+v", qs)
	}
}

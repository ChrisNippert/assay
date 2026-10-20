package server

import (
	"testing"

	"ollamui/internal/hw"
)

func TestChooseContext(t *testing.T) {
	// Qwen3.5-9B Q8_0 on a 12.9 GB card with ~2 GB used by the desktop:
	// measured 9.97 GB at 32k (fits) and 12.3 GB at 64k (spilled).
	qwen := hw.KVCache{PerToken: 32768}
	base := int64(8_860_000_000)
	budget := int64(12_868_124_672 - 2_000_000_000 - ctxMargin)

	cases := []struct {
		name     string
		budget   int64
		modelMax int
		loaded   int
		want     int
	}{
		{"fits 32k", budget, 262144, 0, 32768},
		{"capped by model", budget, 16384, 0, 16384},
		{"nothing fits", 8_000_000_000, 262144, 0, 0},
		{"loaded at 4k upgrades", budget, 262144, 4096, 32768},
		{"keeps loaded 32k when free memory dips", budget - 200<<20, 262144, 32768, 32768},
		{"drops loaded 32k when it really doesn't fit", budget - 2<<30, 262144, 32768, 0},
		{"lots of room", 40 << 30, 262144, 0, 262144},
	}
	for _, c := range cases {
		if got := chooseContext(base, qwen, c.budget, c.modelMax, c.loaded); got != c.want {
			t.Errorf("%s: got %d, want %d", c.name, got, c.want)
		}
	}
}

func TestSameModel(t *testing.T) {
	if !sameModel("gemma4", "gemma4:latest") || !sameModel("hf.co/a/B-GGUF:Q4", "hf.co/a/b-gguf:q4") || sameModel("a:1", "a:2") {
		t.Fatal("sameModel")
	}
}

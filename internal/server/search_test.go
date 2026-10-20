package server

import (
	"errors"
	"testing"

	"ollamui/internal/hf"
)

func TestSearchCache(t *testing.T) {
	var c searchCache
	calls := 0
	fetch := func() ([]hf.Repo, bool, error) {
		calls++
		return []hf.Repo{{ID: "a/b-GGUF"}}, false, nil
	}
	c.get(hf.SearchParams{Query: "Qwen", Author: "unsloth"}, fetch)
	repos, _, err := c.get(hf.SearchParams{Query: "qwen", Author: "Unsloth"}, fetch)
	if err != nil || len(repos) != 1 || calls != 1 {
		t.Fatalf("second search should hit the cache: calls=%d err=%v", calls, err)
	}
	c.get(hf.SearchParams{Query: "gemma"}, fetch)
	if calls != 2 {
		t.Fatalf("different query should fetch: calls=%d", calls)
	}

	// Errors aren't cached.
	failing := func() ([]hf.Repo, bool, error) { calls++; return nil, false, errors.New("down") }
	c.get(hf.SearchParams{Query: "x"}, failing)
	if _, _, err := c.get(hf.SearchParams{Query: "x"}, failing); err == nil || calls != 4 {
		t.Fatalf("errors must not be cached: calls=%d err=%v", calls, err)
	}

	// The cache stays bounded.
	for i := range searchCacheLimit + 5 {
		c.get(hf.SearchParams{Query: string(rune('a' + i))}, fetch)
	}
	if len(c.entries) > searchCacheLimit {
		t.Fatalf("cache grew to %d entries", len(c.entries))
	}
}

func TestOllamaDefaultContext(t *testing.T) {
	const gib = 1 << 30
	cases := []struct {
		vram     int64
		modelMax int
		want     int
	}{
		{12 * gib, 262144, 4096},
		{0, 0, 4096},
		{32 * gib, 262144, 32768},
		{80 * gib, 262144, 262144},
		{80 * gib, 40960, 40960},
		{12 * gib, 2048, 2048},
	}
	for _, c := range cases {
		if got := ollamaDefaultContext(c.vram, c.modelMax); got != c.want {
			t.Errorf("ollamaDefaultContext(%d GiB, %d) = %d, want %d", c.vram/gib, c.modelMax, got, c.want)
		}
	}
}

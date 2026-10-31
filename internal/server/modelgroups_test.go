package server

import (
	"reflect"
	"testing"

	"assay/internal/ollama"
)

func TestGroupModels(t *testing.T) {
	ms := []ollama.Model{
		{Name: "hf.co/unsloth/Qwen3.5-9B-GGUF:Q8_0", Digest: "aaa"},
		{Name: "gemma4:12b", Digest: "bbb"},
		{Name: "qwen3.5-9b:q8_0", Digest: "aaa"},
		{Name: "hf.co/x/y:Q4", Digest: "ccc"},
	}
	g := groupModels(ms)
	if len(g) != 3 {
		t.Fatalf("got %d groups", len(g))
	}
	if g[0].Primary.Name != "gemma4:12b" || g[1].Primary.Name != "hf.co/x/y:Q4" || g[2].Primary.Name != "qwen3.5-9b:q8_0" {
		t.Fatalf("order/primary: %+v", g)
	}
	if want := []string{"qwen3.5-9b:q8_0", "hf.co/unsloth/Qwen3.5-9B-GGUF:Q8_0"}; !reflect.DeepEqual(g[2].Names, want) {
		t.Fatalf("names %v", g[2].Names)
	}
	if g[2].HFRepo != "unsloth/Qwen3.5-9B-GGUF" || g[0].HFRepo != "" {
		t.Fatalf("repo %q", g[2].HFRepo)
	}
	p := &page{Models: ms, Loaded: []loadedModel{{Name: "hf.co/unsloth/Qwen3.5-9B-GGUF:Q8_0"}}}
	pg := p.ModelGroups()
	if pg[2].LoadedName != "hf.co/unsloth/Qwen3.5-9B-GGUF:Q8_0" || pg[2].EjectName() != pg[2].LoadedName || pg[0].Loaded() {
		t.Fatalf("loaded: %+v", pg)
	}
}

func TestDeleteMultipleNames(t *testing.T) {
	s, fake, _ := newTestServer(t)
	rec := do(s, "POST", "/api/models/delete", `{"names":["a:1","hf.co/o/r:Q4"]}`)
	if rec.Code != 204 || !reflect.DeepEqual(fake.deleted, []string{"a:1", "hf.co/o/r:Q4"}) {
		t.Fatalf("%d %v", rec.Code, fake.deleted)
	}
	rec = do(s, "POST", "/api/models/delete", `{}`)
	if rec.Code != 400 {
		t.Fatalf("empty: %d", rec.Code)
	}
}

func TestShortNameAndDiskUsed(t *testing.T) {
	ms := []ollama.Model{
		{Name: "hf.co/unsloth/Qwen3-8B-GGUF:Q4_K_M", Digest: "a", Size: 5e9},
		{Name: "qwen3:8b", Digest: "a", Size: 5e9},
		{Name: "hf.co/x/y:Q4", Digest: "b", Size: 2e9},
	}
	g := groupModels(ms)
	if g[0].ShortName() != "y:Q4" || g[0].PrimaryOwner() != "x" {
		t.Fatalf("short %q owner %q", g[0].ShortName(), g[0].PrimaryOwner())
	}
	if g[1].ShortName() != "qwen3:8b" || g[1].PrimaryOwner() != "" {
		t.Fatalf("plain name: %q", g[1].ShortName())
	}
	if got := (&page{Models: ms}).DiskUsed(); got != 7e9 {
		t.Fatalf("disk used %d", got)
	}
}

func TestParseParams(t *testing.T) {
	for in, want := range map[string]int64{"8.95B": 8_950_000_000, "671M": 671_000_000, "": 0, "huge": 0} {
		if got := parseParams(in); got != want {
			t.Errorf("parseParams(%q) = %d, want %d", in, got, want)
		}
	}
}

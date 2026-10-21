package hf

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestQuants(t *testing.T) {
	files := []File{
		{Type: "file", Path: "README.md", Size: 100},
		{Type: "directory", Path: "BF16"},
		{Type: "file", Path: "BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf", Size: 50},
		{Type: "file", Path: "BF16/Qwen3.8-27B-BF16-00002-of-00002.gguf", Size: 5},
		{Type: "file", Path: "MTP/mtp-Qwen3.8-27B-Q4_0.gguf", Size: 1},
		{Type: "file", Path: "Qwen3.8-27B-Q4_0.gguf", Size: 16},
		{Type: "file", Path: "Qwen3.8-27B-UD-Q4_K_XL.gguf", Size: 17},
		{Type: "file", Path: "Qwen3.8-27B-UD-IQ2_XXS.gguf", Size: 7},
		{Type: "file", Path: "imatrix_unsloth.gguf", Size: 1},
		{Type: "file", Path: "mmproj-F16.gguf", Size: 1},
		{Type: "file", Path: "Qwen_Qwen3-8B-Q6_K_L.gguf", Size: 9},
	}
	got, vision := Quants("unsloth/Qwen3.8-27B-GGUF", files)
	if !vision {
		t.Error("vision projector not detected")
	}
	want := []struct {
		name  string
		size  int64
		split bool
	}{
		{"UD-IQ2_XXS", 7, false},
		{"Q6_K_L", 9, false},
		{"Q4_0", 16, false},
		{"UD-Q4_K_XL", 17, false},
		{"BF16", 55, true},
	}
	if len(got) != len(want) {
		t.Fatalf("got %d quants: %+v", len(got), got)
	}
	for i, w := range want {
		if got[i].Name != w.name || got[i].Size != w.size || got[i].Split != w.split {
			t.Errorf("quant %d = %+v, want %+v", i, got[i], w)
		}
	}
}

func TestQuality(t *testing.T) {
	for name, want := range map[string]string{
		"UD-IQ1_M": "very lossy", "Q2_K": "lossy", "IQ3_XXS": "okay", "UD-Q4_K_XL": "good",
		"Q5_K_M": "very good", "Q6_K": "near full", "Q8_0": "near full", "BF16": "full", "MXFP4_MOE": "good",
	} {
		if got := (Quant{Name: name}).Quality(); got != want {
			t.Errorf("%s: %s, want %s", name, got, want)
		}
	}
}

func TestNames(t *testing.T) {
	if got := OllamaName("unsloth/Qwen3-8B-GGUF", "UD-Q4_K_XL"); got != "hf.co/unsloth/Qwen3-8B-GGUF:UD-Q4_K_XL" {
		t.Error(got)
	}
	if got := ShortName("bartowski/Qwen_Qwen3-8B-GGUF", "Q4_K_M"); got != "qwen_qwen3-8b:q4_k_m" {
		t.Error(got)
	}
}

func TestIsMoE(t *testing.T) {
	cases := []struct {
		arch, repo string
		want       bool
	}{
		{"qwen3moe", "unsloth/Qwen3-30B-A3B-GGUF", true},
		{"qwen3", "unsloth/Qwen3-8B-GGUF", false},
		{"lfm2moe", "x/LFM2-8B-A1B-GGUF", true},
		{"gptoss", "unsloth/gpt-oss-20b-GGUF", true},
		{"granitehybrid", "unsloth/granite-4.0-h-tiny-GGUF", false},
		{"", "someone/Thing-21B-A3.5B-GGUF", true},
		{"deepseek41", "unsloth/DeepSeek-V4.1-Flash-1017-GGUF", true},
		{"deepseek4", "antirez/deepseek-v4-flash-gguf", true},
	}
	for _, c := range cases {
		if got := IsMoE(c.arch, c.repo); got != c.want {
			t.Errorf("IsMoE(%q, %q) = %v", c.arch, c.repo, got)
		}
	}
}

func TestClient(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/models":
			if r.URL.Query().Get("filter") != "gguf" || r.URL.Query().Get("author") != "unsloth" {
				t.Errorf("query = %s", r.URL.RawQuery)
			}
			io.WriteString(w, `[{"id":"unsloth/Qwen3-8B-GGUF","downloads":5,"gated":false,"gguf":{"total":8190735360,"architecture":"qwen3","context_length":40960,"chat_template":"{{ huge }}"}}]`)
		case "/unsloth/Qwen3-8B-GGUF/resolve/main/params":
			io.WriteString(w, `{"temperature":0.6,"top_k":20}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	c := New(srv.URL, "")
	repos, _, err := c.Search(context.Background(), SearchParams{Query: "qwen3", Author: "unsloth"})
	if err != nil {
		t.Fatal(err)
	}
	if len(repos) != 1 || repos[0].Params() != 8190735360 || repos[0].Arch() != "qwen3" || repos[0].IsGated() {
		t.Fatalf("repos = %+v", repos)
	}
	p, err := c.RecommendedParams(context.Background(), "unsloth/Qwen3-8B-GGUF")
	if err != nil || p["temperature"] != 0.6 {
		t.Fatalf("params = %v, %v", p, err)
	}
	p, err = c.RecommendedParams(context.Background(), "bartowski/none")
	if err != nil || p != nil {
		t.Fatalf("missing params = %v, %v", p, err)
	}
	if _, err := c.Info(context.Background(), "../etc/passwd"); err != ErrNotFound {
		t.Fatalf("bad repo id: %v", err)
	}
}

func TestUnreachableIsReadable(t *testing.T) {
	srv := httptest.NewServer(http.NotFoundHandler())
	srv.Close() // nothing listens here any more
	_, _, err := New(srv.URL, "").Search(context.Background(), SearchParams{Query: "x"})
	if err == nil || err.Error() != "couldn't reach Hugging Face. Check your connection and try again" {
		t.Fatalf("err = %v", err)
	}
}

func TestSearchFollowsCursor(t *testing.T) {
	var srv *httptest.Server
	srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Query().Get("cursor") {
		case "":
			w.Header().Set("Link", "<"+srv.URL+`/api/models?filter=gguf&cursor=two>; rel="next"`)
			io.WriteString(w, `[{"id":"a/one-GGUF"},{"id":"a/two-GGUF"}]`)
		case "two":
			w.Header().Set("Link", "<"+srv.URL+`/api/models?filter=gguf&cursor=three>; rel="next"`)
			io.WriteString(w, `[{"id":"a/three-GGUF"},{"id":"a/four-GGUF"}]`)
		case "three":
			io.WriteString(w, `[{"id":"a/five-GGUF"}]`)
		}
	}))
	defer srv.Close()
	c := New(srv.URL, "")

	repos, truncated, err := c.Search(context.Background(), SearchParams{})
	if err != nil || len(repos) != 5 || truncated {
		t.Fatalf("all pages: %d repos, truncated=%v, err=%v", len(repos), truncated, err)
	}
	repos, truncated, err = c.Search(context.Background(), SearchParams{Max: 3})
	if err != nil || len(repos) != 3 || !truncated || repos[2].ID != "a/three-GGUF" {
		t.Fatalf("capped: %+v, truncated=%v, err=%v", repos, truncated, err)
	}
}

func TestSearchIgnoresForeignNextLink(t *testing.T) {
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		w.Header().Set("Link", `<https://evil.example/api/models?cursor=x>; rel="next"`)
		io.WriteString(w, `[{"id":"a/one-GGUF"}]`)
	}))
	defer srv.Close()
	repos, _, err := New(srv.URL, "secret").Search(context.Background(), SearchParams{})
	if err != nil || len(repos) != 1 || calls != 1 {
		t.Fatalf("repos=%d calls=%d err=%v", len(repos), calls, err)
	}
}

func TestIsChatModel(t *testing.T) {
	for tag, want := range map[string]bool{"": true, "text-generation": true, "image-text-to-text": true, "text-to-image": false, "image-to-image": false} {
		if got := (Repo{PipelineTag: tag}).IsChatModel(); got != want {
			t.Errorf("IsChatModel(%q) = %v", tag, got)
		}
	}
}

func TestDraftFiles(t *testing.T) {
	// unsloth/DeepSeek-V4.1-Flash-1017-GGUF ships DSpark drafts next to the
	// real quants; a 7 GB "Q8_0" must not pass for the 280 GB model.
	files := []File{
		{Type: "file", Path: "UD-Q2_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q2_K_XL-00001-of-00002.gguf", Size: 100e9},
		{Type: "file", Path: "UD-Q2_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q2_K_XL-00002-of-00002.gguf", Size: 100e9},
		{Type: "file", Path: "dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf", Size: 7e9},
		{Type: "file", Path: "dspark/dspark-DeepSeek-V4.1-Flash-1017-BF16.gguf", Size: 8e9},
		{Type: "file", Path: "DeepSeek-V4.1-Flash-1017-DSpark-draft-Q2K-Q4K.gguf", Size: 5e9},
	}
	quants, _ := Quants("unsloth/DeepSeek-V4.1-Flash-1017-GGUF", files)
	if len(quants) != 4 {
		t.Fatalf("got %d quants: %+v", len(quants), quants)
	}
	if q := quants[0]; q.Name != "UD-Q2_K_XL" || q.Draft || q.Size != 200e9 {
		t.Errorf("first should be the real quant, got %+v", q)
	}
	for _, q := range quants[1:] {
		if !q.Draft {
			t.Errorf("%s should be a draft", q.Files[0])
		}
	}
	if quants[3].Name != "DSpark BF16" {
		t.Errorf("draft name = %q, want DSpark BF16", quants[3].Name)
	}
	if quants[1].Name != "DeepSeek-V4.1-Flash-1017-DSpark-draft-Q2K-Q4K" {
		t.Errorf("a draft without a quant suffix is named by its file, got %q", quants[1].Name)
	}
}

func TestDraftRepos(t *testing.T) {
	cases := []struct {
		repo  Repo
		draft bool
		of    string
	}{
		{Repo{ID: "deepseek-ai/DeepSeek-V4-Flash-DSpark"}, true, ""},
		{Repo{ID: "z-lab/Qwen3.8-27B-DFlash2-GGUF", Tags: []string{"gguf", "draft-model", "base_model:Qwen/Qwen3.8-27B"}}, true, "Qwen/Qwen3.8-27B"},
		{Repo{ID: "AtomicChat/Qwen3.5-9B-DFlash-GGUF", Tags: []string{"base_model:z-lab/Qwen3.5-9B-DFlash", "base_model:quantized:z-lab/Qwen3.5-9B-DFlash"}}, true, ""},
		{Repo{ID: "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"}, true, ""},
		{Repo{ID: "RWKV/v5-Eagle-7B"}, false, ""},
		{Repo{ID: "unsloth/DeepSeek-V4.1-Flash-1017-GGUF", Tags: []string{"speculative-decoding"}}, false, ""},
		// The architecture beats the name: full models with DSpark heads...
		{Repo{ID: "Solstice-AI/Qwen3.8-27B-UltraOptimised-DSpark-MTP", GGUF: &GGUF{Architecture: "qwen35"}}, false, ""},
		{Repo{ID: "twaggs88/DeepSeek-V4-Flash-REAP25-DSpark-ds4-GGUF", GGUF: &GGUF{Architecture: "deepseek4"}}, false, ""},
		// ...and drafts whose names don't say so.
		{Repo{ID: "someone/Speedy-Helper-GGUF", GGUF: &GGUF{Architecture: "deepseek41-dflash-draft"}}, true, ""},
	}
	for _, c := range cases {
		if got := c.repo.IsDraft(); got != c.draft {
			t.Errorf("%s IsDraft = %v", c.repo.ID, got)
		}
		if got := c.repo.DraftFor(); got != c.of {
			t.Errorf("%s DraftFor = %q, want %q", c.repo.ID, got, c.of)
		}
	}
}

func TestDraftFileIgnoresRepoName(t *testing.T) {
	repo := "Solstice-AI/Qwen3.8-27B-UltraOptimised-DSpark-MTP-GGUF"
	if IsDraftFile(repo, "Qwen3.8-27B-UltraOptimised-DSpark-MTP-Q4_K_M.gguf") {
		t.Error("the model's own file was taken for a draft")
	}
	if !IsDraftFile(repo, "dspark/Qwen3.8-27B-UltraOptimised-DSpark-MTP-draft-Q8_0.gguf") {
		t.Error("a draft beside it was missed")
	}
}

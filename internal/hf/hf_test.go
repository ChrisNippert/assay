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
	got, vision := Quants(files)
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
	repos, err := c.Search(context.Background(), SearchParams{Query: "qwen3", Author: "unsloth"})
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
	_, err := New(srv.URL, "").Search(context.Background(), SearchParams{Query: "x"})
	if err == nil || err.Error() != "couldn't reach Hugging Face. Check your connection and try again" {
		t.Fatalf("err = %v", err)
	}
}

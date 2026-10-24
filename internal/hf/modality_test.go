package hf

import (
	"slices"
	"strings"
	"testing"
)

func TestModalities(t *testing.T) {
	for _, c := range []struct {
		name     string
		pipeline string
		tags     []string
		in, out  string // joined with "+"
		gen      bool
	}{
		{"untagged", "", nil, "text", "text", false},
		{"llm", "text-generation", []string{"gguf", "qwen3"}, "text", "text", false},
		{"vision", "image-text-to-text", nil, "text+image", "text", false},
		{"vision tag on llm", "text-generation", []string{"vision"}, "text+image", "text", false},
		{"captioner", "image-to-text", nil, "image", "text", false},
		{"video", "video-text-to-text", nil, "text+image+video", "text", false},
		{"audio in", "audio-text-to-text", nil, "text+audio", "text", false},
		{"voxtral-ish", "text-generation", []string{"audio"}, "text+audio", "text", false},
		{"omni", "any-to-any", []string{"omni"}, "text+image+video+audio", "text", false},
		{"any-to-any speech", "any-to-any", []string{"text-to-speech"}, "text+image+video+audio", "text+audio", false},
		{"asr", "automatic-speech-recognition", nil, "audio", "text", false},
		{"tts", "text-to-speech", []string{"audio"}, "text", "audio", true},
		{"text-to-audio", "text-to-audio", nil, "text", "audio", true},
		{"audio-to-audio", "audio-to-audio", nil, "audio", "audio", true},
		{"flux", "text-to-image", []string{"vision"}, "text", "image", true},
		{"img2img", "image-to-image", nil, "text+image", "image", true},
		{"t2v", "text-to-video", nil, "text", "video", true},
		{"i2v", "image-to-video", nil, "text+image", "video", true},
		{"embedding", "feature-extraction", nil, "text", "", false},
		{"unknown pipeline", "something-new", nil, "text", "text", false},
	} {
		r := Repo{PipelineTag: c.pipeline, Tags: c.tags}
		in, out := r.Modalities()
		if g := strings.Join(in, "+"); g != c.in {
			t.Errorf("%s: takes %q, want %q", c.name, g, c.in)
		}
		if g := strings.Join(out, "+"); g != c.out {
			t.Errorf("%s: makes %q, want %q", c.name, g, c.out)
		}
		if r.Generator() != c.gen || r.IsChatModel() == c.gen {
			t.Errorf("%s: Generator = %v, want %v", c.name, r.Generator(), c.gen)
		}
	}
}

func TestSearchPipelines(t *testing.T) {
	for _, c := range []struct {
		takes, makes []string
		want         []string
	}{
		{[]string{"text"}, []string{"image"}, []string{"image-to-image", "text-to-image"}},
		{[]string{"image"}, []string{"image"}, []string{"image-to-image"}},
		{[]string{"text"}, []string{"video"}, []string{"image-to-video", "text-to-video"}},
		{[]string{"text"}, []string{"audio"}, []string{"text-to-audio", "text-to-speech"}},
		{[]string{"audio"}, []string{"text"}, []string{"any-to-any", "audio-text-to-text", "automatic-speech-recognition"}},
		{[]string{"text"}, nil, nil},
	} {
		got := SearchPipelines(c.takes, c.makes)
		if c.want == nil {
			// Nothing narrows a plain text search, but wide ones still list the media pipelines.
			if !slices.Contains(got, "text-to-image") {
				t.Errorf("takes text: %v", got)
			}
			continue
		}
		if !slices.Equal(got, c.want) {
			t.Errorf("%v -> %v: got %v, want %v", c.takes, c.makes, got, c.want)
		}
	}
}

func TestCantRun(t *testing.T) {
	gg := func(arch string) *GGUF { return &GGUF{Architecture: arch} }
	cases := []struct {
		r    Repo
		want bool
	}{
		{Repo{PipelineTag: "text-to-image"}, true},
		{Repo{PipelineTag: "automatic-speech-recognition", GGUF: gg("whisper")}, true},
		{Repo{PipelineTag: "automatic-speech-recognition"}, true},
		{Repo{PipelineTag: "automatic-speech-recognition", GGUF: gg("qwen3asr")}, false},
		{Repo{PipelineTag: "text-generation", GGUF: gg("qwen3")}, false},
	}
	for _, c := range cases {
		if got := c.r.CantRun() != ""; got != c.want {
			t.Errorf("%+v: CantRun = %v", c.r, got)
		}
	}
}

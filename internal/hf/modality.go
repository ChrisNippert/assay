package hf

import (
	"slices"
	"strings"
)

// Modalities lists the kinds of data a model can take or make, in display order.
var Modalities = []string{"text", "image", "video", "audio"}

type inOut struct{ in, out []string }

// pipelineModalities maps Hugging Face pipeline tags to what the model
// takes and makes. Embedding and classification pipelines make nothing a
// person reads, so their output is empty.
var pipelineModalities = map[string]inOut{
	"text-generation":                {[]string{"text"}, []string{"text"}},
	"text2text-generation":           {[]string{"text"}, []string{"text"}},
	"conversational":                 {[]string{"text"}, []string{"text"}},
	"image-text-to-text":             {[]string{"text", "image"}, []string{"text"}},
	"image-to-text":                  {[]string{"image"}, []string{"text"}},
	"visual-question-answering":      {[]string{"text", "image"}, []string{"text"}},
	"video-text-to-text":             {[]string{"text", "image", "video"}, []string{"text"}},
	"audio-text-to-text":             {[]string{"text", "audio"}, []string{"text"}},
	"any-to-any":                     {[]string{"text", "image", "video", "audio"}, []string{"text"}},
	"automatic-speech-recognition":   {[]string{"audio"}, []string{"text"}},
	"text-to-speech":                 {[]string{"text"}, []string{"audio"}},
	"text-to-audio":                  {[]string{"text"}, []string{"audio"}},
	"audio-to-audio":                 {[]string{"audio"}, []string{"audio"}},
	"text-to-image":                  {[]string{"text"}, []string{"image"}},
	"image-to-image":                 {[]string{"text", "image"}, []string{"image"}},
	"unconditional-image-generation": {nil, []string{"image"}},
	"text-to-video":                  {[]string{"text"}, []string{"video"}},
	"image-to-video":                 {[]string{"text", "image"}, []string{"video"}},
	"feature-extraction":             {[]string{"text"}, nil},
	"sentence-similarity":            {[]string{"text"}, nil},
	"text-ranking":                   {[]string{"text"}, nil},
	"text-classification":            {[]string{"text"}, nil},
	"fill-mask":                      {[]string{"text"}, nil},
	"image-feature-extraction":       {[]string{"image"}, nil},
	"visual-document-retrieval":      {[]string{"text", "image"}, nil},
}

// plainText are the pipelines the ordinary search already finds.
var plainText = map[string]bool{
	"text-generation": true, "text2text-generation": true, "conversational": true,
	"unconditional-image-generation": true, "visual-question-answering": true,
}

// SearchPipelines lists the pipeline tags whose models take all of takes
// and make all of makes, to ask Hugging Face for them directly: a search
// for "whisper" or "flux" is otherwise drowned by language models. Plain
// text pipelines and ones that make nothing are left out.
func SearchPipelines(takes, makes []string) []string {
	var out []string
	for p, m := range pipelineModalities {
		if len(m.out) == 0 || plainText[p] {
			continue
		}
		if containsAll(m.in, takes) && containsAll(m.out, makes) {
			out = append(out, p)
		}
	}
	slices.Sort(out)
	return out
}

func containsAll(have, want []string) bool {
	for _, w := range want {
		if !slices.Contains(have, w) {
			return false
		}
	}
	return true
}

// Modalities reports what the model takes and makes, from its pipeline tag
// and then its tags. A repo with no pipeline tag is taken for a text model.
func (r Repo) Modalities() (in, out []string) {
	m, ok := pipelineModalities[r.PipelineTag]
	if !ok {
		m = pipelineModalities["text-generation"]
	}
	ins, outs := map[string]bool{}, map[string]bool{}
	for _, mod := range m.in {
		ins[mod] = true
	}
	for _, mod := range m.out {
		outs[mod] = true
	}
	tags := map[string]bool{}
	for _, t := range r.Tags {
		tags[strings.ToLower(t)] = true
	}
	if r.PipelineTag == "any-to-any" {
		// Say what comes out besides text only when the tags do.
		if tags["text-to-speech"] || tags["tts"] || tags["speech-generation"] || tags["audio-generation"] {
			outs["audio"] = true
		}
		if tags["text-to-image"] || tags["image-generation"] {
			outs["image"] = true
		}
	}
	// Tags can add inputs, but only to a model that writes text: a TTS model
	// tagged "audio" is not listening.
	if outs["text"] {
		if tags["vision"] || tags["multimodal"] || tags["image-text-to-text"] || tags["omni"] {
			ins["image"] = true
		}
		if tags["audio"] || tags["omni"] || tags["audio-text-to-text"] || tags["speech"] {
			ins["audio"] = true
		}
		if tags["video-text-to-text"] || tags["video"] {
			ins["video"] = true
		}
		if ins["video"] {
			ins["image"] = true
		}
	}
	return ordered(ins), ordered(outs)
}

// ordered returns the set in the fixed order text, image, video, audio.
func ordered(set map[string]bool) []string {
	out := []string{}
	for _, m := range Modalities {
		if set[m] {
			out = append(out, m)
		}
	}
	return out
}

// Generator reports whether the repo makes images, video or audio and no
// text: FLUX, Wan, Kokoro and the like, which llama.cpp can't run.
func (r Repo) Generator() bool {
	_, out := r.Modalities()
	return len(out) > 0 && !slices.Contains(out, "text")
}

// CantRun says why llama.cpp can't run the repo's GGUFs, or "" if it can:
// generators, and speech recognizers in whisper.cpp's format (HF lists
// their architecture as "whisper", or none), which only look like GGUFs.
// Speech models built on an LLM (Voxtral, Qwen-ASR) run through an mmproj.
func (r Repo) CantRun() string {
	switch {
	case r.Generator():
		return "Assay can’t run image, video or audio generators yet"
	case r.PipelineTag == "automatic-speech-recognition" && (r.Arch() == "whisper" || r.Arch() == ""):
		return "Whisper models run in whisper.cpp, not llama.cpp, so Assay can’t run them"
	}
	return ""
}

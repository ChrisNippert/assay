package server

import (
	"regexp"
	"slices"
	"strings"
)

var (
	sizeToken    = regexp.MustCompile(`^[ea]?\d+(\.\d+)?[bm]$`)
	versionToken = regexp.MustCompile(`^\d+(\.\d+)*v$`)
	sizeTag      = regexp.MustCompile(`^\d+(\.\d+)?[bm]$`)
	moeSize      = regexp.MustCompile(`^a\d+(\.\d+)?b$`)
)

var acronyms = map[string]string{"glm": "GLM", "gpt": "GPT", "llm": "LLM", "oss": "OSS", "vl": "VL", "ocr": "OCR", "tts": "TTS"}

// prettyModel turns a model name into something readable for the UI and
// pulls out tags worth a chip: "gemma-4-12b-it-qat:ud-q4_k_xl" becomes
// "Gemma 4 12B" with tags ["QAT"]. The raw name stays the identifier
// everywhere else; this is for display only. A name it can't make sense of
// comes back unchanged.
func prettyModel(name string) (title string, tags []string) {
	base := strings.TrimPrefix(name, "hf.co/")
	if i := strings.LastIndex(base, "/"); i >= 0 {
		base = base[i+1:]
	}
	base, tail, _ := strings.Cut(base, ":")
	var words []string
	add := func(t string) {
		if !slices.Contains(tags, t) {
			tags = append(tags, t)
		}
	}
	for _, tok := range strings.FieldsFunc(base, func(r rune) bool { return r == '-' || r == '_' || r == ' ' }) {
		low := strings.ToLower(tok)
		switch {
		case low == "gguf" || low == "it" || low == "ggml":
			continue
		case low == "qat":
			add("QAT")
			continue
		case low == "moe":
			add("MoE")
			continue
		case moeSize.MatchString(low):
			add("MoE") // "30b-a3b": three billion active
		case low == "vision":
			add("Vision")
			continue
		}
		words = append(words, prettyWord(tok, low))
	}
	if sizeTag.MatchString(strings.ToLower(tail)) {
		words = append(words, strings.ToUpper(tail)) // "qwen3-embedding:0.6b"
	}
	title = strings.Join(words, " ")
	if title == "" {
		return name, tags
	}
	return title, tags
}

func prettyWord(tok, low string) string {
	switch {
	case acronyms[low] != "":
		return acronyms[low]
	case sizeToken.MatchString(low) || versionToken.MatchString(low):
		return strings.ToUpper(tok)
	case tok != low:
		return tok // already has its own capitals ("Qwen3.5", "K2")
	}
	return strings.ToUpper(tok[:1]) + tok[1:]
}

package hf

import (
	"path"
	"regexp"
	"sort"
	"strings"
)

// Quant is one downloadable quantization of a model. Large quants are often
// split across several files.
type Quant struct {
	Name  string   // e.g. "UD-Q4_K_XL", "Q8_0", "BF16"
	Files []string // repo paths
	Size  int64
	Split bool
	// Draft marks a small draft model shipped next to the real one, for
	// speculative decoding (DSpark, DFlash, EAGLE). It can't chat by itself.
	Draft bool
	// Kind is what sort of draft it is: "dspark", "dflash", "eagle3" or
	// "draft" (a plain small model). Empty unless Draft.
	Kind string
}

// Bits is a rough bits-per-weight class used for quality labels and sorting.
func (q Quant) Bits() int {
	n := strings.TrimPrefix(strings.ToUpper(q.Name), "UD-")
	switch {
	case strings.HasPrefix(n, "BF16"), strings.HasPrefix(n, "F16"):
		return 16
	case strings.HasPrefix(n, "F32"):
		return 32
	case strings.HasPrefix(n, "MXFP4"):
		return 4
	}
	n = strings.TrimPrefix(n, "I")
	if len(n) >= 2 && n[0] == 'Q' && n[1] >= '1' && n[1] <= '8' {
		return int(n[1] - '0')
	}
	return 0
}

// Quality is a plain-language label for how close a quant stays to the
// full model. It's a rule of thumb from bits per weight, not a measurement.
func (q Quant) Quality() string {
	switch b := q.Bits(); {
	case b >= 16:
		return "full"
	case b >= 6:
		return "near full"
	case b == 5:
		return "very good"
	case b == 4:
		return "good"
	case b == 3:
		return "okay"
	case b == 2:
		return "lossy"
	case b == 1:
		return "very lossy"
	}
	return "unknown"
}

var (
	splitSuffix = regexp.MustCompile(`-\d{5}-of-\d{5}$`)
	quantSuffix = regexp.MustCompile(`(?i)[-._]((?:UD-)?(?:[IPT]?Q\d(?:_[A-Z0-9]+)*|BF16|F16|F32|MXFP4(?:_[A-Z0-9]+)*))$`)
)

// draftPattern matches the names speculative-decoding draft models go by.
var draftPattern = regexp.MustCompile(`(?i)(^|[^a-z])(dspark|dflash\d*|eagle\d+|drafter|draft)([^a-z]|$)`)

// IsDraftFile reports whether a path in repo is a draft model's file, e.g.
// "dspark-Model-Q8_0.gguf", "dspark/Model-BF16.gguf" or
// "Model-DSpark-draft-Q2K-Q4K.gguf". The repo's own model name is ignored,
// so a full model called "...-DSpark-MTP" isn't taken for its draft.
func IsDraftFile(repo, p string) bool {
	_, name, _ := strings.Cut(repo, "/")
	name = strings.TrimSuffix(strings.TrimSuffix(name, "-GGUF"), "-gguf")
	if name != "" {
		p = regexp.MustCompile(`(?i)`+regexp.QuoteMeta(name)).ReplaceAllString(p, "")
	}
	return draftPattern.MatchString(p)
}

// draftKind names the kind of draft a path holds, e.g. "DSpark".
func draftKind(p string) string {
	m := draftPattern.FindStringSubmatch(p)
	if m == nil {
		return "Draft"
	}
	switch w := strings.ToLower(m[2]); {
	case w == "dspark":
		return "DSpark"
	case strings.HasPrefix(w, "dflash"):
		return "DFlash" + w[len("dflash"):]
	case strings.HasPrefix(w, "eagle"):
		return strings.ToUpper(w)
	}
	return "Draft"
}

// DraftSlug names the kind of draft a path holds in lowercase: "dspark",
// "dflash", "eagle3", or "draft" for a plain small draft model.
func DraftSlug(p string) string {
	m := draftPattern.FindStringSubmatch(p)
	if m == nil {
		return "draft"
	}
	switch w := strings.ToLower(m[2]); {
	case w == "dspark":
		return "dspark"
	case strings.HasPrefix(w, "dflash"):
		return "dflash"
	case strings.HasPrefix(w, "eagle"):
		return "eagle3"
	}
	return "draft"
}

// IsMTPFile reports whether a repo path is a multi-token-prediction head:
// "mtp-Model.gguf" or anything under "mtp/".
func IsMTPFile(p string) bool {
	lower := strings.ToLower(p)
	return strings.HasSuffix(lower, ".gguf") && (strings.HasPrefix(strings.ToLower(path.Base(p)), "mtp-") || strings.HasPrefix(lower, "mtp/"))
}

// MTP picks the repo's multi-token-prediction head to go with its quants,
// or "". Unsloth's Gemma 4 repos ship one (mtp-<model>.gguf) next to the
// quants; llama.cpp loads it as the draft model with --spec-type draft-mtp.
// With several, Q8_0, then F16, then BF16 are preferred, else the first.
func MTP(files []File) string {
	var all []string
	for _, f := range files {
		if f.Type == "file" && IsMTPFile(f.Path) {
			all = append(all, f.Path)
		}
	}
	sort.Strings(all)
	for _, want := range []string{"q8_0", "f16", "bf16"} {
		for _, p := range all {
			if strings.Contains(strings.ToLower(path.Base(p)), want) && !(want == "f16" && strings.Contains(strings.ToLower(path.Base(p)), "bf16")) {
				return p
			}
		}
	}
	if len(all) > 0 {
		return all[0]
	}
	return ""
}

// Quants groups a repo's GGUF files into quants. It skips vision projector
// (mmproj), imatrix and multi-token-prediction (MTP) files (see MTP), and
// reports whether a vision projector was present. Draft models for speculative
// decoding come back as quants with Draft set, named by their file, so they
// never pass for the real model's quant.
func Quants(repo string, files []File) (quants []Quant, hasVision bool) {
	byName := map[string]*Quant{}
	for _, f := range files {
		if f.Type != "file" || !strings.HasSuffix(strings.ToLower(f.Path), ".gguf") {
			continue
		}
		base := strings.TrimSuffix(path.Base(f.Path), path.Ext(f.Path))
		lower := strings.ToLower(base)
		switch {
		case isProjector(lower):
			hasVision = true
			continue
		case strings.Contains(lower, "imatrix"), IsMTPFile(f.Path):
			continue
		}
		split := splitSuffix.MatchString(base)
		base = splitSuffix.ReplaceAllString(base, "")
		draft := IsDraftFile(repo, f.Path)
		var name string
		if m := quantSuffix.FindStringSubmatch(base); m != nil {
			name = strings.ToUpper(m[1])
		} else if !draft {
			continue
		}
		key := name
		if draft {
			key = "draft:" + base // one entry per draft file (or split set)
			if name == "" {
				name = base
			} else {
				name = draftKind(f.Path) + " " + name // "DSpark Q8_0"
			}
		}
		q := byName[key]
		if q == nil {
			q = &Quant{Name: name, Draft: draft}
			if draft {
				q.Kind = DraftSlug(f.Path)
			}
			byName[key] = q
		}
		q.Files = append(q.Files, f.Path)
		q.Size += f.Size
		q.Split = q.Split || split || len(q.Files) > 1
	}
	for _, q := range byName {
		sort.Strings(q.Files) // "-00001-of-" first, as llama.cpp needs
		quants = append(quants, *q)
	}
	sort.Slice(quants, func(i, j int) bool {
		if quants[i].Draft != quants[j].Draft {
			return !quants[i].Draft // the real quants first
		}
		if quants[i].Size != quants[j].Size {
			return quants[i].Size < quants[j].Size
		}
		return quants[i].Name < quants[j].Name
	})
	return quants, hasVision
}

// isProjector reports whether a lowercased file name is a vision
// projector: "mmproj-F16.gguf", or "Model.BF16-mmproj.gguf" as some
// uploaders name it.
func isProjector(lower string) bool { return strings.Contains(lower, "mmproj") }

// Projector picks the vision projector (mmproj) to go with a quant, or
// "": F16 when there's a choice, as Ollama does, then BF16, F32, any.
func Projector(files []File) string {
	var all []string
	for _, f := range files {
		base := strings.ToLower(path.Base(f.Path))
		if f.Type == "file" && isProjector(base) && strings.HasSuffix(base, ".gguf") {
			all = append(all, f.Path)
		}
	}
	sort.Strings(all)
	for _, want := range []string{"f16", "bf16", "f32"} {
		for _, p := range all {
			b := strings.ToLower(path.Base(p))
			if strings.Contains(strings.ReplaceAll(b, "bf16", "_"), want) || (want == "bf16" && strings.Contains(b, "bf16")) {
				return p
			}
		}
	}
	if len(all) > 0 {
		return all[0]
	}
	return ""
}

// OllamaName is the name `ollama pull` uses to fetch a quant from this repo.
func OllamaName(repo, quant string) string { return "hf.co/" + repo + ":" + quant }

// ShortName suggests a friendlier local name, e.g.
// "unsloth/Qwen3-8B-GGUF" + "UD-Q4_K_XL" -> "qwen3-8b:ud-q4_k_xl".
func ShortName(repo, quant string) string {
	_, name, _ := strings.Cut(repo, "/")
	name = strings.TrimSuffix(strings.TrimSuffix(name, "-GGUF"), "-gguf")
	name = strings.ToLower(name)
	name = strings.Map(func(r rune) rune {
		if r >= 'a' && r <= 'z' || r >= '0' && r <= '9' || r == '.' || r == '-' || r == '_' {
			return r
		}
		return '-'
	}, name)
	return name + ":" + strings.ToLower(quant)
}

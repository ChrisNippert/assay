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
	quantSuffix = regexp.MustCompile(`(?i)[-._]((?:UD-)?(?:I?Q\d(?:_[A-Z0-9]+)*|BF16|F16|F32|MXFP4(?:_[A-Z0-9]+)*))$`)
)

// Quants groups a repo's GGUF files into quants. It skips vision projector
// (mmproj), imatrix and speculative-decoding (MTP) files, and reports
// whether a vision projector was present.
func Quants(files []File) (quants []Quant, hasVision bool) {
	byName := map[string]*Quant{}
	for _, f := range files {
		if f.Type != "file" || !strings.HasSuffix(strings.ToLower(f.Path), ".gguf") {
			continue
		}
		base := strings.TrimSuffix(path.Base(f.Path), path.Ext(f.Path))
		lower := strings.ToLower(base)
		switch {
		case strings.HasPrefix(lower, "mmproj"):
			hasVision = true
			continue
		case strings.Contains(lower, "imatrix"), strings.HasPrefix(lower, "mtp-"),
			strings.HasPrefix(strings.ToLower(f.Path), "mtp/"):
			continue
		}
		split := splitSuffix.MatchString(base)
		base = splitSuffix.ReplaceAllString(base, "")
		m := quantSuffix.FindStringSubmatch(base)
		if m == nil {
			continue
		}
		name := strings.ToUpper(m[1])
		q := byName[name]
		if q == nil {
			q = &Quant{Name: name}
			byName[name] = q
		}
		q.Files = append(q.Files, f.Path)
		q.Size += f.Size
		q.Split = q.Split || split || len(q.Files) > 1
	}
	for _, q := range byName {
		quants = append(quants, *q)
	}
	sort.Slice(quants, func(i, j int) bool {
		if quants[i].Size != quants[j].Size {
			return quants[i].Size < quants[j].Size
		}
		return quants[i].Name < quants[j].Name
	})
	return quants, hasVision
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

package server

import (
	"sort"
	"strconv"
	"strings"

	"ollamui/internal/bench"
	"ollamui/internal/hw"
	"ollamui/internal/ollama"
)

// modelGroup is one installed model, which may have several names. Ollama
// lists a pulled hf.co model and its short alias separately, though they
// share the same blobs (same digest).
type modelGroup struct {
	Primary ollama.Model
	Names   []string // all names, Primary.Name first
	// HFRepo is "owner/repo" if any name is hf.co/owner/repo:quant.
	HFRepo string
	// LoadedName is the name Ollama reports as in memory, or "".
	LoadedName string
	// Fit is how it fits in memory; Bench its last speed test, if any.
	Fit   hw.Fit
	Bench *bench.Result
}

// Speed is the measured writing speed for the UI, or "".
func (g modelGroup) Speed() string {
	if g.Bench == nil || g.Bench.GenTPS == 0 {
		return ""
	}
	return strconv.FormatFloat(g.Bench.GenTPS, 'f', 0, 64) + " tok/s"
}

// Others lists the names besides the primary one.
func (g modelGroup) Others() []string { return g.Names[1:] }

// Loaded reports whether any of the group's names is in memory.
func (g modelGroup) Loaded() bool { return g.LoadedName != "" }

// EjectName is the name to unload: the loaded one, else the primary.
func (g modelGroup) EjectName() string {
	if g.LoadedName != "" {
		return g.LoadedName
	}
	return g.Primary.Name
}

// NamesJoined is the space-separated name list for data attributes.
func (g modelGroup) NamesJoined() string { return strings.Join(g.Names, " ") }

// groupModels merges models sharing a digest. The primary name is the
// shortest name not starting with "hf.co/", else the first name. Groups are
// sorted by primary name.
func groupModels(models []ollama.Model) []modelGroup {
	idx := map[string]int{}
	var groups []modelGroup
	for _, m := range models {
		if m.Digest == "" {
			groups = append(groups, modelGroup{Primary: m, Names: []string{m.Name}})
			continue
		}
		i, ok := idx[m.Digest]
		if !ok {
			idx[m.Digest] = len(groups)
			groups = append(groups, modelGroup{Primary: m, Names: []string{m.Name}})
			continue
		}
		groups[i].Names = append(groups[i].Names, m.Name)
	}
	repo := templateFuncs["hfRepo"].(func(string) string)
	for gi := range groups {
		g := &groups[gi]
		best := ""
		for _, n := range g.Names {
			if strings.HasPrefix(n, "hf.co/") {
				continue
			}
			if best == "" || len(n) < len(best) {
				best = n
			}
		}
		if best == "" {
			best = g.Names[0]
		}
		for _, m := range models {
			if m.Name == best && (m.Digest == g.Primary.Digest || m.Digest == "") {
				g.Primary = m
				break
			}
		}
		rest := []string{best}
		for _, n := range g.Names {
			if n != best {
				rest = append(rest, n)
			}
		}
		g.Names = rest
		for _, n := range g.Names {
			if r := repo(n); r != "" {
				g.HFRepo = r
				break
			}
		}
	}
	sort.SliceStable(groups, func(i, j int) bool { return groups[i].Primary.Name < groups[j].Primary.Name })
	return groups
}

// ModelGroups groups the installed models by digest, with load state,
// how well each fits and its measured speed.
func (p *page) ModelGroups() []modelGroup {
	if p.groups != nil {
		return p.groups
	}
	groups := groupModels(p.Models)
	for i := range groups {
		g := &groups[i]
		for _, n := range g.Names {
			if p.IsLoaded(n) {
				g.LoadedName = n
				break
			}
		}
		if p.bench != nil {
			if r, ok := p.bench.Get(g.Primary.Digest); ok {
				g.Bench = &r
			}
		}
		g.Fit = fitOf(g.Primary, g.Bench, p.memory)
	}
	p.groups = groups
	return groups
}

// fitCtx is the context the installed-model fit is judged at: enough for a
// normal chat. Auto context then grows it as far as memory allows.
const fitCtx = 8192

// fitOf says how an installed model fits in memory: measured, if a speed
// test ran, otherwise estimated from its size.
func fitOf(m ollama.Model, b *bench.Result, mem hw.Memory) hw.Fit {
	f := hw.Classify(hw.Estimate(m.Size, parseParams(m.Details.ParameterSize), fitCtx, false), mem)
	if b != nil && mem.VRAM > 0 {
		if b.GPUPct >= 100 && f.Level != hw.FitGPU && f.Level != hw.FitTight {
			f.Level, f.Label = hw.FitGPU, "Fits on GPU"
		} else if b.GPUPct < 100 {
			f.Level, f.Label, f.Percent = hw.FitPartial, "Spills to RAM", 100
		}
	}
	return f
}

// parseParams reads Ollama's parameter size, e.g. "8.95B" or "671M".
func parseParams(s string) int64 {
	mult := 1.0
	switch {
	case strings.HasSuffix(s, "B"):
		mult = 1e9
	case strings.HasSuffix(s, "M"):
		mult = 1e6
	case strings.HasSuffix(s, "K"):
		mult = 1e3
	}
	v, err := strconv.ParseFloat(strings.TrimRight(s, "BMK"), 64)
	if err != nil {
		return 0
	}
	return int64(v * mult)
}

// Has reports whether name is one of the group's names.
func (g modelGroup) Has(name string) bool {
	for _, n := range g.Names {
		if n == name {
			return true
		}
	}
	return false
}

// Owner is the owner part of HFRepo.
func (g modelGroup) Owner() string {
	o, _, _ := strings.Cut(g.HFRepo, "/")
	return o
}

// ShortName is the primary name without a leading "hf.co/owner/" prefix.
func (g modelGroup) ShortName() string {
	n := g.Primary.Name
	if rest, ok := strings.CutPrefix(n, "hf.co/"); ok {
		if _, tail, found := strings.Cut(rest, "/"); found {
			return tail
		}
	}
	return n
}

// PrimaryOwner is the HF owner when the primary name is an hf.co name.
func (g modelGroup) PrimaryOwner() string {
	rest, ok := strings.CutPrefix(g.Primary.Name, "hf.co/")
	if !ok {
		return ""
	}
	owner, _, _ := strings.Cut(rest, "/")
	return owner
}

// DiskUsed is the total size of all models on disk. Names sharing a digest
// share blobs, so each group counts once.
func (p *page) DiskUsed() int64 {
	var n int64
	for _, g := range groupModels(p.Models) {
		n += g.Primary.Size
	}
	return n
}

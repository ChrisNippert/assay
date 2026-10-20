package server

import (
	"sort"
	"strings"

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

// ModelGroups groups the installed models by digest, with load state.
func (p *page) ModelGroups() []modelGroup {
	groups := groupModels(p.Models)
	for i := range groups {
		for _, n := range groups[i].Names {
			if p.IsLoaded(n) {
				groups[i].LoadedName = n
				break
			}
		}
	}
	return groups
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

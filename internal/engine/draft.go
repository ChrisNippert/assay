package engine

import (
	"fmt"
	"sort"
	"strings"
)

// Draft is an installed model that can draft for another. Kind is "" for an
// ordinary small model, else what the downloaded draft is ("dspark",
// "dflash", "eagle3").
type Draft struct {
	Name string `json:"name"`
	Size int64  `json:"size"`
	Kind string `json:"kind,omitempty"`
	// For is the repo of the model a downloaded draft was made for.
	For string `json:"for,omitempty"`
}

// Drafts lists the installed models that can speed up model by speculative
// decoding: smaller, with the same vocabulary, smallest first. Embedding
// models and the model itself are left out.
func (e *Engine) Drafts(name string) ([]Draft, error) {
	m, err := e.store.Get(name)
	if err != nil {
		return nil, err
	}
	var out []Draft
	for _, d := range e.store.List() {
		if e.draftOK(m, d) == nil {
			out = append(out, Draft{Name: d.Name, Size: d.Size, Kind: d.Draft, For: d.DraftFor})
		}
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].Size < out[j].Size || out[i].Size == out[j].Size && out[i].Name < out[j].Name
	})
	return out, nil
}

// DraftModels lists the downloaded drafts (DSpark, DFlash, EAGLE3), which
// the model lists leave out because they can't chat.
func (e *Engine) DraftModels() []Draft {
	var out []Draft
	for _, d := range e.store.List() {
		if d.Draft != "" {
			out = append(out, Draft{Name: d.Name, Size: d.Size, Kind: d.Draft, For: d.DraftFor})
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// checkDraft says why the model called draft can't draft for m, or nil.
func (e *Engine) checkDraft(m *Model, draft string) error {
	d, err := e.store.Get(draft)
	if err != nil {
		return fmt.Errorf("draft model %s isn't installed", draft)
	}
	return e.draftOK(m, d)
}

func (e *Engine) draftOK(m, d *Model) error {
	if d.Name == m.Name {
		return fmt.Errorf("a model can't draft for itself")
	}
	if m.Draft != "" {
		return fmt.Errorf("%s only drafts for other models", m.Name)
	}
	if d.Draft != "" {
		// Made for one model, with its own vocabulary tricks (EAGLE3 and
		// DFlash drafts read the target's hidden state), so the size and
		// vocabulary rules for ordinary drafts don't apply.
		// A draft that doesn't say what it's for is matched by name:
		// "Qwen3-8B-EAGLE3-Speculator" is for "Qwen3-8B".
		core := modelCore(m.Repo)
		switch {
		case d.DraftFor != "" && modelCore(d.DraftFor) != core:
			return fmt.Errorf("draft model %s was made for %s, not %s", d.Name, d.DraftFor, m.Name)
		case d.DraftFor == "" && (core == "" || !strings.Contains(modelCore(d.Repo), core)):
			return fmt.Errorf("draft model %s doesn't say which model it's for, and it isn't named for %s", d.Name, m.Name)
		}
		return nil
	}
	if d.Size >= m.Size {
		return fmt.Errorf("draft model %s isn't smaller than %s", d.Name, m.Name)
	}
	tf, err := e.gguf(m)
	if err != nil {
		return err
	}
	df, err := e.gguf(d)
	if err != nil {
		return err
	}
	if df.IsEmbedding() || df.IsProjector() {
		return fmt.Errorf("%s can't write text", d.Name)
	}
	if !tf.SameVocab(df) {
		return fmt.Errorf("draft model %s has a different vocabulary than %s", d.Name, m.Name)
	}
	return nil
}

// modelCore reduces a repo id to its model name, so a draft made for
// "Qwen/Qwen3.5-9B" matches the quants in "unsloth/Qwen3.5-9B-GGUF".
func modelCore(repo string) string {
	_, name, ok := strings.Cut(repo, "/")
	if !ok {
		name = repo
	}
	return strings.TrimSuffix(strings.ToLower(name), "-gguf")
}

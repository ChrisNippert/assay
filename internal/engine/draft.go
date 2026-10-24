package engine

import (
	"fmt"
	"sort"
)

// Draft is an installed model that can draft for another.
type Draft struct {
	Name string `json:"name"`
	Size int64  `json:"size"`
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
			out = append(out, Draft{Name: d.Name, Size: d.Size})
		}
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].Size < out[j].Size || out[i].Size == out[j].Size && out[i].Name < out[j].Name
	})
	return out, nil
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

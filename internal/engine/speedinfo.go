package engine

import (
	"context"

	"ollamui/internal/hf"
)

// SpeedInfo is what the Tune panel needs to know about a model's speed-ups.
type SpeedInfo struct {
	// MTP is whether the model has an MTP head to draft with.
	MTP bool `json:"mtp"`
	// Name is the full name to download again for the head: set for a model
	// downloaded here without one. Whether its repo has one is MTPAvailable.
	Name string `json:"name,omitempty"`
}

// Speed describes a model's speed-up options.
func (e *Engine) Speed(name string) (SpeedInfo, error) {
	m, err := e.store.Get(name)
	if err != nil {
		return SpeedInfo{}, err
	}
	s := SpeedInfo{MTP: m.MTP != ""}
	if m.MTP == "" && m.Source == "hf" && m.Repo != "" && m.Draft == "" && e.HF != nil {
		s.Name = m.Name
	}
	return s, nil
}

// MTPAvailable asks Hugging Face whether the repo a downloaded model came
// from has an MTP head, and how big it is.
func (e *Engine) MTPAvailable(ctx context.Context, name string) (int64, bool) {
	m, err := e.store.Get(name)
	if err != nil || m.MTP != "" || m.Repo == "" || e.HF == nil {
		return 0, false
	}
	files, err := e.HF.Files(ctx, m.Repo)
	if err != nil {
		return 0, false
	}
	p := hf.MTP(files)
	for _, f := range files {
		if p != "" && f.Path == p {
			return f.Size, true
		}
	}
	return 0, false
}

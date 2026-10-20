package server

import (
	"context"
	"errors"
	"net/http"
	"time"

	"ollamui/internal/ollama"
	"ollamui/internal/presets"
)

func (s *Server) handlePresetsPage(w http.ResponseWriter, r *http.Request) {
	p := s.basePage(r.Context(), "presets", "Presets")
	s.render(w, "presets", p)
}

func (s *Server) handlePresetList(w http.ResponseWriter, r *http.Request) {
	list, err := s.presets.List()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if list == nil {
		list = []*presets.Preset{}
	}
	writeJSON(w, http.StatusOK, list)
}

func (s *Server) handlePresetSave(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	if !presets.ValidName(name) {
		writeError(w, http.StatusBadRequest, "preset names use lowercase letters, numbers, '.', '-' and '_' (up to 64)")
		return
	}
	var p presets.Preset
	if !readJSON(w, r, &p) {
		return
	}
	p.Name = name
	if p.Base == "" {
		writeError(w, http.StatusBadRequest, "base model is required")
		return
	}
	if p.Base == p.OllamaName() {
		writeError(w, http.StatusBadRequest, "a preset can't use itself as its base")
		return
	}
	var err error
	if p.Options, err = presets.CleanOptions(p.Options); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()
	if _, err := s.ollama.Show(ctx, p.Base); err != nil {
		if ollama.IsNotFound(err) {
			writeError(w, http.StatusBadRequest, "base model "+p.Base+" isn't installed")
		} else {
			writeError(w, ollamaStatus(err), err.Error())
		}
		return
	}

	prev, _ := s.presets.Get(name)
	if err := s.presets.Save(&p); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if err := s.syncPreset(r.Context(), &p, prev); err != nil {
		writeError(w, http.StatusBadGateway, "Saved, but updating "+p.OllamaName()+" in Ollama failed: "+err.Error())
		return
	}
	writeJSON(w, http.StatusOK, &p)
}

// syncPreset makes Ollama match a preset: a published preset becomes (or is
// refreshed as) the model ollamui/<name>; an unpublished one is removed.
func (s *Server) syncPreset(ctx context.Context, p, prev *presets.Preset) error {
	if p.Published {
		req := ollama.CreateRequest{Model: p.OllamaName(), From: p.Base, Parameters: p.Options, System: p.System}
		return s.ollama.Create(ctx, req, func(ollama.Status) error { return nil })
	}
	if prev != nil && prev.Published {
		if err := s.ollama.Delete(ctx, p.OllamaName()); err != nil && !ollama.IsNotFound(err) {
			return err
		}
	}
	return nil
}

func (s *Server) handlePresetDelete(w http.ResponseWriter, r *http.Request) {
	p, err := s.presets.Get(r.PathValue("name"))
	if errors.Is(err, presets.ErrNotFound) {
		writeError(w, http.StatusNotFound, "preset not found")
		return
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if p.Published {
		if err := s.ollama.Delete(r.Context(), p.OllamaName()); err != nil && !ollama.IsNotFound(err) {
			writeError(w, ollamaStatus(err), "couldn't remove "+p.OllamaName()+" from Ollama: "+err.Error())
			return
		}
	}
	if err := s.presets.Delete(p.Name); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// modelDefaults is what the Tune panel shows as "default" for a model.
type modelDefaults struct {
	Options       map[string]any `json:"options"`
	System        string         `json:"system,omitempty"`
	ContextLength int            `json:"contextLength,omitempty"` // the most the model supports
	OllamaContext int            `json:"ollamaContext"`           // what Ollama uses if num_ctx isn't set
	Auto          autoContext    `json:"auto"`                    // what ollamui picks if num_ctx isn't set
	Thinking      bool           `json:"thinking"`
}

// ollamaDefaultContext is the context length Ollama picks when a request
// doesn't set num_ctx: it scales with VRAM (4k under 24 GiB, 32k under
// 48 GiB, 256k above), capped at what the model supports. An
// OLLAMA_CONTEXT_LENGTH set on the Ollama server overrides this; we can't see
// that from here.
func ollamaDefaultContext(vram int64, modelMax int) int {
	const gib = 1 << 30
	ctx := 262144
	switch {
	case vram < 24*gib:
		ctx = 4096
	case vram < 48*gib:
		ctx = 32768
	}
	if modelMax > 0 && modelMax < ctx {
		ctx = modelMax
	}
	return ctx
}

func (s *Server) handleModelDefaults(w http.ResponseWriter, r *http.Request) {
	name := r.URL.Query().Get("name")
	if name == "" {
		writeError(w, http.StatusBadRequest, "name is required")
		return
	}
	show, err := s.ollama.Show(r.Context(), name)
	if err != nil {
		writeError(w, ollamaStatus(err), err.Error())
		return
	}
	// Keep only settings the Tune panel knows about.
	opts := map[string]any{}
	for k, v := range ollama.ParseParameters(show.Parameters) {
		if _, err := presets.CleanOptions(map[string]any{k: v}); err == nil {
			opts[k] = v
		}
	}
	writeJSON(w, http.StatusOK, modelDefaults{
		Options:       opts,
		System:        show.System,
		ContextLength: show.ContextLength(),
		OllamaContext: ollamaDefaultContext(s.memory.VRAM, show.ContextLength()),
		Auto:          s.pickContext(r.Context(), name, false),
		Thinking:      show.HasCapability("thinking"),
	})
}

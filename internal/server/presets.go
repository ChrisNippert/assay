package server

import (
	"context"
	"errors"
	"net/http"
	"runtime"
	"time"

	"ollamui/internal/engine"
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
	if p.Base == p.Name {
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

	p.Published = false // presets are listed in the /v1 API; nothing to publish
	if err := s.presets.Save(&p); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, &p)
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
	Layers        int            `json:"layers,omitempty"` // what num_gpu counts; 0 if unknown
	// Saved are the user's own settings for the model, and Engine what
	// applies when neither they nor a chat set one; both only with an
	// engine that keeps settings per model.
	Saved  map[string]any  `json:"saved,omitempty"`
	Engine *engineDefaults `json:"engine,omitempty"`
	// Drafts are the installed models that could draft for this one.
	Drafts []engine.Draft `json:"drafts,omitempty"`
	// Speed is what the model has to speed it up (an MTP head).
	Speed *engine.SpeedInfo `json:"speed,omitempty"`
	CPUs  int               `json:"cpus,omitempty"`
}

type engineDefaults struct {
	CacheType string `json:"cache_type"`
	FlashAttn string `json:"flash_attn"`
	KeepAlive string `json:"keep_alive"`
}

// modelSettings is a backend that keeps settings per model (the engine).
type modelSettings interface {
	Settings(name string) (map[string]any, error)
	SetSettings(name string, s map[string]any) error
	CacheType() string
	FlashAttn() string
	KeepAlive() time.Duration
}

// handleModelSettings saves a model's own settings: PUT {name, options}.
func (s *Server) handleModelSettings(w http.ResponseWriter, r *http.Request) {
	ms, ok := s.ollama.(modelSettings)
	if !ok {
		writeError(w, http.StatusNotImplemented, "settings per model need the llama.cpp engine")
		return
	}
	var req struct {
		Name    string         `json:"name"`
		Options map[string]any `json:"options"`
	}
	if !readJSON(w, r, &req) {
		return
	}
	if req.Name == "" {
		writeError(w, http.StatusBadRequest, "name is required")
		return
	}
	opts, err := presets.CleanOptions(req.Options)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if err := ms.SetSettings(req.Name, opts); err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"saved": opts})
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
	d := modelDefaults{
		Options:       opts,
		System:        show.System,
		ContextLength: show.ContextLength(),
		OllamaContext: ollamaDefaultContext(s.memory.VRAM, show.ContextLength()),
		Auto:          s.defaultsContext(r.Context(), name),
		Thinking:      show.HasCapability("thinking"),
		Layers:        show.Layers(),
	}
	if ms, ok := s.ollama.(modelSettings); ok {
		d.Saved, _ = ms.Settings(name)
		d.CPUs = runtime.NumCPU()
		if dr, ok := s.ollama.(interface {
			Drafts(string) ([]engine.Draft, error)
		}); ok {
			d.Drafts, _ = dr.Drafts(name)
		}
		if sp, ok := s.ollama.(interface {
			Speed(string) (engine.SpeedInfo, error)
		}); ok {
			if si, err := sp.Speed(name); err == nil {
				d.Speed = &si
			}
		}
		d.Engine = &engineDefaults{CacheType: ms.CacheType(), FlashAttn: ms.FlashAttn(), KeepAlive: ms.KeepAlive().String()}
	}
	writeJSON(w, http.StatusOK, d)
}

// handleModelMTP says whether the repo a model came from has an MTP head
// it lacks: GET ?name= -> {available, size}. It asks Hugging Face, so the
// Tune panel calls it only when the model has none.
func (s *Server) handleModelMTP(w http.ResponseWriter, r *http.Request) {
	out := map[string]any{"available": false}
	if m, ok := s.ollama.(interface {
		MTPAvailable(context.Context, string) (int64, bool)
	}); ok {
		ctx, cancel := context.WithTimeout(r.Context(), 8*time.Second)
		defer cancel()
		if size, ok := m.MTPAvailable(ctx, r.URL.Query().Get("name")); ok {
			out["available"], out["size"] = true, size
		}
	}
	writeJSON(w, http.StatusOK, out)
}

// defaultsContext is the automatic context the Tune panel shows.
func (s *Server) defaultsContext(ctx context.Context, name string) autoContext {
	if _, ok := s.ollama.(fitter); ok {
		return s.fittedContext(ctx, name)
	}
	return s.pickContext(ctx, name, false)
}

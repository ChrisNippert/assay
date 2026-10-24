package server

import (
	"net/http"

	"ollamui/internal/engine"
)

// adapterBackend is a backend that keeps LoRA adapters (the engine).
type adapterBackend interface {
	Adapters() []engine.Adapter
	AdaptersFor(model string) ([]engine.Adapter, error)
	AddAdapter(path, name string) (engine.Adapter, error)
}

// adapterRow is one adapter in the Library, with where it came from.
type adapterRow struct {
	engine.Adapter
	Link *linkNote
}

// adapterRows lists the library's adapters for the Library page.
func (s *Server) adapterRows() []adapterRow {
	ab, ok := s.ollama.(adapterBackend)
	if !ok {
		return nil
	}
	var out []adapterRow
	for _, a := range ab.Adapters() {
		out = append(out, adapterRow{Adapter: a, Link: s.linkNoteOf(a.Name)})
	}
	return out
}

// handleAdapterAdd adds a LoRA adapter file: POST {path, name}.
func (s *Server) handleAdapterAdd(w http.ResponseWriter, r *http.Request) {
	ab, ok := s.ollama.(adapterBackend)
	if !ok {
		writeError(w, http.StatusNotImplemented, "LoRA adapters need the llama.cpp engine")
		return
	}
	var req struct {
		Path string `json:"path"`
		Name string `json:"name"`
	}
	if !readJSON(w, r, &req) {
		return
	}
	a, err := ab.AddAdapter(req.Path, req.Name)
	if err != nil {
		writeError(w, http.StatusBadRequest, capitalize(err.Error()))
		return
	}
	writeJSON(w, http.StatusCreated, a)
}

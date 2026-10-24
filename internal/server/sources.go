package server

import (
	"net/http"

	"ollamui/internal/engine"
)

// modelSources is a backend that can find models other apps downloaded
// and link them in (the engine).
type modelSources interface {
	Found(refresh bool) []engine.Source
	ImportFound(id, dir string, names []string) (int, error)
	AddFolder(dir string) error
	RemoveFolder(dir string) error
}

// linkNote is the small line under a linked model in the Library.
type linkNote struct {
	Text    string
	Title   string // the path it was linked from
	Missing bool
}

// linkNoteOf describes where a model's files come from, if they're linked
// from another app, and whether they're still there.
func (s *Server) linkNoteOf(name string) *linkNote {
	mf, ok := s.ollama.(modelFolder)
	if !ok {
		return nil
	}
	st := mf.Store()
	m, err := st.Get(name)
	if err != nil || m.Origin == "" {
		return nil
	}
	n := &linkNote{Text: "linked from " + engine.SourceTitle(m.Source), Title: m.Origin}
	if m.Link == "symlink" {
		n.Text += " (symlink, on another disk)"
	}
	if len(st.Missing(m)) > 0 {
		n.Missing = true
		n.Text = "missing: it was linked from " + engine.SourceTitle(m.Source) + ", which no longer has it"
	}
	return n
}

// handleSources lists what other apps have that isn't in the library yet.
// ?refresh=1 rescans instead of using the last scan.
func (s *Server) handleSources(w http.ResponseWriter, r *http.Request) {
	ms, ok := s.ollama.(modelSources)
	if !ok {
		writeJSON(w, http.StatusOK, map[string]any{"sources": []engine.Source{}})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"sources": ms.Found(r.URL.Query().Get("refresh") != "")})
}

// handleSourcesImport links models in: POST {source, dir, names}. No names
// means every model the source lists.
func (s *Server) handleSourcesImport(w http.ResponseWriter, r *http.Request) {
	ms, ok := s.ollama.(modelSources)
	if !ok {
		writeError(w, http.StatusNotImplemented, "linking models needs the llama.cpp engine")
		return
	}
	var req struct {
		Source string   `json:"source"`
		Dir    string   `json:"dir"`
		Names  []string `json:"names"`
	}
	if !readJSON(w, r, &req) {
		return
	}
	n, err := ms.ImportFound(req.Source, req.Dir, req.Names)
	if err != nil {
		writeError(w, http.StatusBadRequest, capitalize(err.Error()))
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"imported": n})
}

// handleSourcesFolder adds (POST) or removes (DELETE) a folder to look in.
func (s *Server) handleSourcesFolder(w http.ResponseWriter, r *http.Request) {
	ms, ok := s.ollama.(modelSources)
	if !ok {
		writeError(w, http.StatusNotImplemented, "linking models needs the llama.cpp engine")
		return
	}
	var req struct {
		Path string `json:"path"`
	}
	if !readJSON(w, r, &req) {
		return
	}
	var err error
	if r.Method == http.MethodDelete {
		err = ms.RemoveFolder(req.Path)
	} else {
		err = ms.AddFolder(req.Path)
	}
	if err != nil {
		writeError(w, http.StatusBadRequest, capitalize(err.Error()))
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

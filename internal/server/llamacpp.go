package server

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"time"

	"assay/internal/llamacpp"
)

// The llama.cpp install API behind Settings → Models. Without the llama.cpp
// engine there is nothing to install, and these answer 404.

func (s *Server) llamaOrNotFound(w http.ResponseWriter) *llamacpp.Manager {
	if s.llama == nil {
		writeError(w, http.StatusNotFound, "llama.cpp isn't the engine in use")
	}
	return s.llama
}

func (s *Server) handleLlamaStatus(w http.ResponseWriter, r *http.Request) {
	if m := s.llamaOrNotFound(w); m != nil {
		ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
		defer cancel()
		writeJSON(w, http.StatusOK, m.Status(ctx))
	}
}

// handleLlamaCheck asks GitHub for the newest release (?fresh=1 skips the
// ten-minute cache).
func (s *Server) handleLlamaCheck(w http.ResponseWriter, r *http.Request) {
	m := s.llamaOrNotFound(w)
	if m == nil {
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 20*time.Second)
	defer cancel()
	c, err := m.Check(ctx, r.URL.Query().Get("fresh") == "1")
	if err != nil {
		var rl *llamacpp.RateLimitError
		status := http.StatusBadGateway
		if errors.As(err, &rl) {
			status = http.StatusTooManyRequests
		}
		writeError(w, status, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, struct {
		*llamacpp.Check
		// Recommended is the build ID that suits this machine's GPUs.
		Recommended string `json:"recommended,omitempty"`
	}{c, s.recommendBuild(c)})
}

// handleLlamaInstall starts an install or update. With models running from
// the install being replaced it answers 409 and their names; sending
// force stops them.
func (s *Server) handleLlamaInstall(w http.ResponseWriter, r *http.Request) {
	m := s.llamaOrNotFound(w)
	if m == nil {
		return
	}
	var req struct {
		Force bool   `json:"force"`
		Build string `json:"build"` // build ID; empty keeps the installed one, else the default
	}
	if !readJSON(w, r, &req) {
		return
	}
	job, err := m.Install(req.Force, req.Build)
	var running *llamacpp.RunningError
	if errors.As(err, &running) {
		writeJSON(w, http.StatusConflict, map[string]any{"error": err.Error(), "running": running.Models})
		return
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusAccepted, job)
}

func (s *Server) handleLlamaCancel(w http.ResponseWriter, r *http.Request) {
	m := s.llamaOrNotFound(w)
	if m == nil {
		return
	}
	if !m.Cancel() {
		writeError(w, http.StatusNotFound, "no install is running")
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// handleLlamaUse picks the installed release over the user's own build, or
// back. Models already running keep the build they started with.
func (s *Server) handleLlamaUse(w http.ResponseWriter, r *http.Request) {
	m := s.llamaOrNotFound(w)
	if m == nil {
		return
	}
	var req struct {
		Release bool `json:"release"`
	}
	if !readJSON(w, r, &req) {
		return
	}
	if req.Release && m.Inst.Binary() == "" {
		writeError(w, http.StatusConflict, "no release is installed")
		return
	}
	if err := m.SetPreferRelease(req.Release); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// handleLlamaCustom uses a llama-server the user points at (a file, or a
// folder holding one); an empty path goes back to the usual choice.
func (s *Server) handleLlamaCustom(w http.ResponseWriter, r *http.Request) {
	m := s.llamaOrNotFound(w)
	if m == nil {
		return
	}
	var req struct {
		Path string `json:"path"`
	}
	if !readJSON(w, r, &req) {
		return
	}
	if err := m.SetCustom(strings.TrimSpace(req.Path)); err != nil {
		writeError(w, http.StatusBadRequest, capitalize(err.Error()))
		return
	}
	writeJSON(w, http.StatusOK, m.Status(r.Context()))
}

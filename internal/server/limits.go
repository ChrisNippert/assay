package server

import (
	"net/http"

	"assay/internal/limits"
)

func (s *Server) handleLimits(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, s.limitSt.Get())
}

func (s *Server) handleLimitsSave(w http.ResponseWriter, r *http.Request) {
	var l limits.Limits
	if !readJSON(w, r, &l) {
		return
	}
	if err := s.limitSt.Set(l); err != nil {
		writeError(w, http.StatusBadRequest, capitalize(err.Error()))
		return
	}
	writeJSON(w, http.StatusOK, s.limitSt.Get())
}

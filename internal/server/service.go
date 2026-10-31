package server

import (
	"net/http"

	"ollamui/internal/service"
)

// handleService reports the start-on-boot entry. Only the machine itself
// may look at or change it: it names paths and edits that machine's
// service manager.
func (s *Server) handleService(w http.ResponseWriter, r *http.Request) {
	if !fromLoopback(r) || s.service == nil {
		writeJSON(w, http.StatusOK, service.Status{})
		return
	}
	writeJSON(w, http.StatusOK, s.service.Status(r.Context()))
}

// handleServiceSet turns it on (running the app with the flags this
// instance was started with) or off: PUT {enabled}. Neither starts nor
// stops anything, since this instance already holds the port.
func (s *Server) handleServiceSet(w http.ResponseWriter, r *http.Request) {
	if !fromLoopback(r) || s.service == nil {
		writeError(w, http.StatusForbidden, "start on boot can only be changed from the machine running ollamui")
		return
	}
	var req struct {
		Enabled bool `json:"enabled"`
	}
	if !readJSON(w, r, &req) {
		return
	}
	var err error
	if req.Enabled {
		_, err = s.service.Install(r.Context(), s.serviceArgs, false)
	} else {
		err = s.service.Uninstall(r.Context(), false)
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, capitalize(err.Error()))
		return
	}
	writeJSON(w, http.StatusOK, s.service.Status(r.Context()))
}

// handleUpdate says whether a newer release exists.
func (s *Server) handleUpdate(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, s.updates.Get(r.Context()))
}

// handleUpdateSet turns the daily check on or off: PUT {check}.
func (s *Server) handleUpdateSet(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Check bool `json:"check"`
	}
	if !readJSON(w, r, &req) {
		return
	}
	s.updates.SetCheck(req.Check)
	writeJSON(w, http.StatusOK, s.updates.Get(r.Context()))
}

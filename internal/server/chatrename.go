package server

import (
	"errors"
	"net/http"
	"strings"
	"unicode/utf8"

	"assay/internal/chats"
	"assay/internal/projects"
)

const maxTitleLen = 200

// handleRenameChat changes a chat: {"title": "..."} renames it and
// {"project": "<id>"} moves it into a project ("" takes it out).
func (s *Server) handleRenameChat(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Title   *string `json:"title"`
		Project *string `json:"project"`
	}
	if !readJSON(w, r, &req) {
		return
	}
	id := r.PathValue("id")
	if req.Project != nil {
		if err := s.handleChatMove(id, *req.Project); err != nil {
			switch {
			case errors.Is(err, chats.ErrNotFound):
				writeError(w, http.StatusNotFound, "chat not found")
			case errors.Is(err, projects.ErrNotFound):
				writeError(w, http.StatusNotFound, "project not found")
			default:
				writeError(w, http.StatusInternalServerError, err.Error())
			}
			return
		}
		if req.Title == nil {
			writeJSON(w, http.StatusOK, map[string]string{"id": id, "project": *req.Project})
			return
		}
	}
	if req.Title == nil {
		writeError(w, http.StatusBadRequest, "nothing to change")
		return
	}
	title := strings.Join(strings.Fields(*req.Title), " ")
	if title == "" {
		writeError(w, http.StatusBadRequest, "title can't be empty")
		return
	}
	if utf8.RuneCountInString(title) > maxTitleLen {
		title = string([]rune(title)[:maxTitleLen])
	}
	if err := s.chats.Rename(id, title); err != nil {
		if errors.Is(err, chats.ErrNotFound) {
			writeError(w, http.StatusNotFound, "chat not found")
		} else {
			writeError(w, http.StatusInternalServerError, err.Error())
		}
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"id": id, "title": title})
}

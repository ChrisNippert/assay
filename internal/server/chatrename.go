package server

import (
	"errors"
	"net/http"
	"strings"
	"unicode/utf8"

	"ollamui/internal/chats"
)

const maxTitleLen = 200

// handleRenameChat sets a chat's title from {"title": "..."} and returns it.
func (s *Server) handleRenameChat(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Title string `json:"title"`
	}
	if !readJSON(w, r, &req) {
		return
	}
	title := strings.Join(strings.Fields(req.Title), " ")
	if title == "" {
		writeError(w, http.StatusBadRequest, "title can't be empty")
		return
	}
	if utf8.RuneCountInString(title) > maxTitleLen {
		title = string([]rune(title)[:maxTitleLen])
	}
	id := r.PathValue("id")
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

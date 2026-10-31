package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"time"

	"assay/internal/chats"
	"assay/internal/tools"
)

// trimKeep is how many of the chat's last messages are never left out.
const trimKeep = 4

// forgetTool lets the model leave earlier messages out of later prompts, with
// a note in their place. Message numbers count the chat's messages from 1.
// The saved messages stay as they are: the trim is chat metadata, applied when
// a prompt is built, and the user can undo it.
func forgetTool(c *chats.Chat) tools.Tool {
	return tools.Tool{
		Name:        "forget",
		Description: fmt.Sprintf("Leave earlier messages out of the prompts from now on, once you no longer need them. Give message_ids (numbers counting the chat's messages from 1), or from and to for a range. note says briefly what they were and replaces each one. The last %d messages can't be left out.", trimKeep),
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"message_ids": map[string]any{"type": "array", "items": map[string]any{"type": "integer"}, "description": "Message numbers to leave out"},
				"from":        map[string]any{"type": "integer", "description": "First number of a range"},
				"to":          map[string]any{"type": "integer", "description": "Last number of a range (default: from)"},
				"note":        map[string]any{"type": "string", "description": "Short note that replaces the messages"},
			},
			"required": []string{"note"},
		},
		Run: func(ctx context.Context, args json.RawMessage) tools.Result {
			return runForget(c, args)
		},
	}
}

func forgetFail(msg string) tools.Result {
	return tools.Result{Text: "Error: " + msg, Label: "Forget · failed", Err: true}
}

// runForget adds a trim for the numbered messages to c. Numbers out of range
// or among the last messages are left out and reported; if none is left, it
// fails.
func runForget(c *chats.Chat, args json.RawMessage) tools.Result {
	var a struct {
		IDs  []int  `json:"message_ids"`
		From int    `json:"from"`
		To   int    `json:"to"`
		Note string `json:"note"`
	}
	if err := json.Unmarshal(args, &a); err != nil {
		return forgetFail("the arguments are not valid JSON")
	}
	note := strings.Join(strings.Fields(a.Note), " ")
	if note == "" {
		return forgetFail("note is required: say briefly what the messages were")
	}
	total := len(c.Messages)
	nums := a.IDs
	if len(nums) == 0 {
		if a.From == 0 {
			return forgetFail("give message_ids, or from and to")
		}
		to := a.To
		if to == 0 {
			to = a.From
		}
		if to < a.From {
			return forgetFail("to is before from")
		}
		for n := a.From; n <= min(to, total); n++ {
			nums = append(nums, n)
		}
	}

	var idx []int
	var why []string
	seen := map[int]bool{}
	for _, n := range nums {
		switch {
		case n < 1 || n > total:
			why = append(why, fmt.Sprintf("%d is not a message", n))
		case n > total-trimKeep:
			why = append(why, fmt.Sprintf("%d is one of the last %d", n, trimKeep))
		case !seen[n]:
			seen[n] = true
			idx = append(idx, n-1)
		}
	}
	if len(idx) == 0 {
		msg := "no message can be left out"
		if len(why) > 0 {
			msg += ": " + strings.Join(why, "; ")
		}
		return forgetFail(msg)
	}
	slices.Sort(idx)
	n := len(idx)
	c.Trims = append(c.Trims, chats.Trim{
		ID:       fmt.Sprintf("t%d", time.Now().UnixNano()),
		Messages: idx,
		Note:     note,
		Created:  time.Now(),
	})
	text := fmt.Sprintf("Left %d %s out of later prompts; each now reads [trimmed: %s].", n, messagesWord(n), note)
	if len(why) > 0 {
		text += " Not left out: " + strings.Join(why, "; ") + "."
	}
	return tools.Result{Text: text, Label: fmt.Sprintf("trimmed %d %s", n, messagesWord(n))}
}

func messagesWord(n int) string {
	if n == 1 {
		return "message"
	}
	return "messages"
}

// trimmedMessages returns msgs as a prompt sends them: each trimmed message
// (but not one of the last trimKeep) becomes "[trimmed: note]", keeping its
// role, with its thinking, tool calls and attachments dropped. The later of
// two trims of a message gives the note. msgs is not changed.
func trimmedMessages(msgs []chats.Message, trims []chats.Trim) []chats.Message {
	if len(trims) == 0 {
		return msgs
	}
	notes := map[int]string{}
	for _, t := range trims {
		for _, i := range t.Messages {
			notes[i] = t.Note
		}
	}
	out := slices.Clone(msgs)
	for i, note := range notes {
		if i < 0 || i >= len(out)-trimKeep {
			continue
		}
		out[i] = chats.Message{Role: out[i].Role, Content: "[trimmed: " + note + "]", Created: out[i].Created}
	}
	return out
}

// handleUndoTrim puts the messages of a trim back into later prompts:
// DELETE /api/chats/{id}/trims/{trim}.
func (s *Server) handleUndoTrim(w http.ResponseWriter, r *http.Request) {
	c, err := s.chats.Get(r.PathValue("id"))
	if err != nil {
		writeError(w, http.StatusNotFound, "chat not found")
		return
	}
	if !c.UndoTrim(r.PathValue("trim")) {
		writeError(w, http.StatusNotFound, "trim not found")
		return
	}
	if err := s.chats.Save(c); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	s.forgetChat(c.ID)
	w.WriteHeader(http.StatusNoContent)
}

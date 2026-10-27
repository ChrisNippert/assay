package server

import (
	"context"
	"net/http"
	"sync"
)

// A run is one chat reply being generated in the background. It outlives the
// HTTP request that started it: the browser can leave and come back, and the
// events so far are replayed to whoever attaches.
type run struct {
	cancel context.CancelFunc
	ctx    context.Context

	mu     sync.Mutex
	events []event
	done   bool
	wake   chan struct{} // closed (and replaced) on every change
}

// beginRun registers a run for the chat, or returns nil if it already has one.
func (s *Server) beginRun(r *http.Request, chatID string) *run {
	s.runsMu.Lock()
	defer s.runsMu.Unlock()
	if s.runs == nil {
		s.runs = map[string]*run{}
	}
	if s.runs[chatID] != nil {
		return nil
	}
	ctx, cancel := context.WithCancel(context.WithoutCancel(r.Context()))
	ru := &run{ctx: ctx, cancel: cancel, wake: make(chan struct{})}
	s.runs[chatID] = ru
	return ru
}

func (s *Server) getRun(chatID string) *run {
	s.runsMu.Lock()
	defer s.runsMu.Unlock()
	return s.runs[chatID]
}

// endRun removes the chat's run and marks it finished, with a last event.
func (s *Server) endRun(chatID string, ru *run, last *event) {
	s.runsMu.Lock()
	if s.runs[chatID] == ru {
		delete(s.runs, chatID)
	}
	s.runsMu.Unlock()
	ru.cancel()
	ru.mu.Lock()
	if last != nil {
		ru.events = append(ru.events, *last)
	}
	ru.done = true
	close(ru.wake)
	ru.wake = make(chan struct{})
	ru.mu.Unlock()
}

// runningChats lists the chats with a reply in progress.
func (s *Server) runningChats() []string {
	s.runsMu.Lock()
	defer s.runsMu.Unlock()
	ids := make([]string, 0, len(s.runs))
	for id := range s.runs {
		ids = append(ids, id)
	}
	return ids
}

func (ru *run) emit(e event) error {
	ru.mu.Lock()
	ru.events = append(ru.events, e)
	close(ru.wake)
	ru.wake = make(chan struct{})
	ru.mu.Unlock()
	return nil
}

// follow replays the run's events from the start, then streams new ones until
// it finishes or ctx ends (the client left; the run goes on).
func (ru *run) follow(ctx context.Context, emit func(event) error) {
	i := 0
	for {
		ru.mu.Lock()
		evs := append([]event(nil), ru.events[i:]...)
		done, wake := ru.done, ru.wake
		ru.mu.Unlock()
		for _, e := range evs {
			if emit(e) != nil {
				return
			}
		}
		i += len(evs)
		if len(evs) > 0 {
			continue
		}
		if done {
			return
		}
		select {
		case <-wake:
		case <-ctx.Done():
			return
		}
	}
}

func (s *Server) handleChatStream(w http.ResponseWriter, r *http.Request) {
	ru := s.getRun(r.PathValue("id"))
	if ru == nil {
		w.WriteHeader(http.StatusNoContent)
		return
	}
	ru.follow(r.Context(), streamEvents(w))
}

func (s *Server) handleChatStop(w http.ResponseWriter, r *http.Request) {
	ru := s.getRun(r.PathValue("id"))
	if ru == nil {
		writeError(w, http.StatusNotFound, "no reply is running in this chat")
		return
	}
	ru.cancel()
	w.WriteHeader(http.StatusNoContent)
}

func (s *Server) handleChatsRunning(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"running": s.runningChats()})
}

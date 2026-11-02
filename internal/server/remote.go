package server

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"slices"
	"strings"
	"sync"
	"time"

	"assay/internal/evals"
	"assay/internal/remote"
)

// apiEvalWorkers is how many questions an API model answers at once: the
// GPU isn't involved, so they needn't wait their turn.
const apiEvalWorkers = 4

// runAPIEvalItem asks an API model every question of one set, a few at a
// time.
func (s *Server) runAPIEvalItem(ctx context.Context, i int, set *evals.Set, redo bool) error {
	s.evalJob.mu.Lock()
	it, think := s.evalJob.Items[i], s.evalJob.Think
	s.evalJob.mu.Unlock()
	m, ep, ok := s.remote.Lookup(it.Model)
	if !ok {
		return errors.New("the API model is gone")
	}
	if think == evals.ThinkOff {
		m.Reasoning = "off"
	}
	return s.runEvalItemWith(ctx, i, set, redo, m.Digest(ep), apiEvalWorkers, func(ctx context.Context, q evals.Question, maxTok int) (evals.Answer, error) {
		system, user := set.Messages(q)
		return askAPIMsgs(ctx, ep, m, chatMsgs(system, user), maxTok)
	})
}

// runEvalItemWith asks every question of one set through ask, up to
// workers at a time, grading and saving after each answer like
// runEvalItem so a stopped run picks up where it left off.
func (s *Server) runEvalItemWith(ctx context.Context, i int, set *evals.Set, redo bool, digest string, workers int, ask func(ctx context.Context, q evals.Question, maxTok int) (evals.Answer, error)) error {
	j := &s.evalJob
	j.mu.Lock()
	it, think, maxTok, judge := j.Items[i], j.Think, j.MaxTokens, j.Judge
	j.mu.Unlock()
	maxTok = min(max(maxTok, set.MaxTokens), 65536)
	res := &evals.Result{Model: it.Model, Digest: digest, Set: set.ID, SetHash: set.Hash, Sampling: evals.Sampling, Think: think, MaxTokens: maxTok, Total: len(set.Questions)}
	if old, ok := s.evalRes.Get(res.Digest, set.ID, think); ok && old.Current(set.Hash) && !redo {
		res.Answers = old.Answers
		if old.MaxTokens > 0 {
			res.MaxTokens = old.MaxTokens
		}
	}
	answered := map[string]bool{}
	for _, a := range res.Answers {
		answered[a.ID] = true
	}
	j.set(i, func(it *evalItem) { it.Since, it.From = time.Now(), len(res.Answers) })

	todo := make(chan evals.Question)
	var mu sync.Mutex
	var firstErr error
	ctx, stop := context.WithCancel(ctx)
	defer stop()
	fail := func(err error) {
		mu.Lock()
		if firstErr == nil {
			firstErr = err
		}
		mu.Unlock()
		stop()
	}
	var wg sync.WaitGroup
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for q := range todo {
				a, err := ask(ctx, q, res.MaxTokens)
				a.ID = q.ID
				if ctx.Err() != nil {
					return
				}
				if err != nil {
					a.Error = err.Error()
					if strings.HasPrefix(a.Error, "the API refused the key") || strings.HasPrefix(a.Error, "out of credit") {
						fail(errors.New(a.Error)) // every other question would fail the same way
						return
					}
				} else {
					a.Got, a.Correct = set.Grade(q, a.Reply)
					if err := s.gradeAnswer(ctx, set, q, &a); err != nil {
						fail(err)
						return
					}
				}
				mu.Lock()
				res.Answers = append(res.Answers, a)
				res.Recount()
				res.At = time.Now()
				err = s.evalRes.Put(res)
				n, c := len(res.Answers), res.Correct
				mu.Unlock()
				if err != nil {
					fail(errors.New("couldn't save the result: " + err.Error()))
					return
				}
				j.set(i, func(it *evalItem) { it.Answered, it.Correct = n, c })
			}
		}()
	}
feed:
	for _, q := range set.Questions {
		if answered[q.ID] {
			continue
		}
		select {
		case todo <- q:
		case <-ctx.Done():
			break feed
		}
	}
	close(todo)
	wg.Wait()
	if firstErr != nil {
		return firstErr
	}
	if ctx.Err() != nil {
		return ctx.Err()
	}
	return s.judgeEvalItem(ctx, i, set, res, judge)
}

// askAPI sends one prompt to an API model with the eval sampling, the
// model's own system prompt first.
func askAPI(ctx context.Context, ep remote.Endpoint, m remote.Model, system, user string, maxTok int) (evals.Answer, error) {
	return askAPIMsgs(ctx, ep, m, chatMsgs(system, user), maxTok)
}

// askAPIMsgs is askAPI with a whole conversation; the model's own system
// prompt goes first.
func askAPIMsgs(ctx context.Context, ep remote.Endpoint, m remote.Model, msgs []remote.Message, maxTok int) (evals.Answer, error) {
	var a evals.Answer
	if m.System != "" {
		if len(msgs) > 0 && msgs[0].Role == "system" {
			msgs = append([]remote.Message{{Role: "system", Content: m.System + "\n\n" + msgs[0].Content}}, msgs[1:]...)
		} else {
			msgs = append([]remote.Message{{Role: "system", Content: m.System}}, msgs...)
		}
	}
	sampling := evals.SamplingOptions()
	delete(sampling, "min_p") // few APIs take it
	var content, thinking strings.Builder
	errLooped := errors.New("looped")
	chunks := 0
	t := time.Now()
	u, err := remote.Chat(ctx, ep, m, remote.Request{Messages: msgs, MaxTokens: maxTok, Sampling: sampling}, func(c, r string) error {
		content.WriteString(c)
		thinking.WriteString(r)
		if chunks++; chunks%64 == 0 && (evals.Looping(thinking.String()) || evals.Looping(content.String())) {
			return errLooped
		}
		return nil
	})
	a.Looped = errors.Is(err, errLooped)
	if a.Looped {
		err = nil
	}
	a.Ms = time.Since(t).Milliseconds()
	a.Reply, a.Thinking = content.String(), thinking.String()
	a.Tokens, a.Cut, a.Cost = u.Tokens, u.Length, u.Cost
	if a.Tokens == 0 {
		a.Tokens = chunks // about one token per streamed chunk
	}
	a.Clip()
	return a, err
}

// API model settings. Keys never go back to the browser: it sees only
// whether one is set and its last four characters.

type endpointJSON struct {
	ID         string `json:"id"`
	Name       string `json:"name"`
	URL        string `json:"url"`
	HasKey     bool   `json:"hasKey"`
	KeyHint    string `json:"keyHint,omitempty"`
	OpenRouter bool   `json:"openRouter"`
}

type remoteModelJSON struct {
	remote.Model
	Name   string `json:"name"`
	Digest string `json:"digest"`
}

func (s *Server) handleRemote(w http.ResponseWriter, r *http.Request) {
	out := struct {
		Endpoints []endpointJSON    `json:"endpoints"`
		Models    []remoteModelJSON `json:"models"`
		Default   string            `json:"defaultUrl"`
	}{Endpoints: []endpointJSON{}, Models: []remoteModelJSON{}, Default: remote.OpenRouterURL}
	for _, e := range s.remote.Endpoints() {
		ej := endpointJSON{ID: e.ID, Name: e.Name, URL: e.URL, HasKey: e.Key != "", OpenRouter: e.OpenRouter()}
		if len(e.Key) > 8 {
			ej.KeyHint = "…" + e.Key[len(e.Key)-4:]
		}
		out.Endpoints = append(out.Endpoints, ej)
	}
	for _, m := range s.remote.Models() {
		mj := remoteModelJSON{Model: m, Name: m.Name()}
		if ep, ok := s.remote.Endpoint(m.Endpoint); ok {
			mj.Digest = m.Digest(ep)
		}
		out.Models = append(out.Models, mj)
	}
	writeJSON(w, http.StatusOK, out)
}

func (s *Server) handleRemoteEndpointPut(w http.ResponseWriter, r *http.Request) {
	var e remote.Endpoint
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 64<<10)).Decode(&e); err != nil {
		writeError(w, http.StatusBadRequest, "bad request")
		return
	}
	e.ID = r.PathValue("id")
	if e.ID == "new" {
		e.ID = remote.Slug(e.Name, func(id string) bool { _, ok := s.remote.Endpoint(id); return ok || id == "new" })
	}
	if err := s.remote.PutEndpoint(e); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"id": e.ID})
}

func (s *Server) handleRemoteEndpointDelete(w http.ResponseWriter, r *http.Request) {
	if err := s.remote.DeleteEndpoint(r.PathValue("id")); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// handleRemoteList lists the models an endpoint offers, for the picker.
func (s *Server) handleRemoteList(w http.ResponseWriter, r *http.Request) {
	ep, ok := s.remote.Endpoint(r.PathValue("id"))
	if !ok {
		writeError(w, http.StatusNotFound, "no such endpoint")
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()
	list, err := remote.List(ctx, ep)
	if err != nil {
		writeError(w, http.StatusBadGateway, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, list)
}

func (s *Server) handleRemoteModelPut(w http.ResponseWriter, r *http.Request) {
	var m remote.Model
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 256<<10)).Decode(&m); err != nil {
		writeError(w, http.StatusBadRequest, "bad request")
		return
	}
	m.ID = r.PathValue("id")
	if m.ID == "new" {
		name := m.Title
		if name == "" {
			name = m.Model[strings.LastIndex(m.Model, "/")+1:]
		}
		m.ID = remote.Slug(name, func(id string) bool {
			return id == "new" || slices.ContainsFunc(s.remote.Models(), func(x remote.Model) bool { return x.ID == id })
		})
	}
	if err := s.remote.PutModel(m); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"id": m.ID, "name": m.Name()})
}

func (s *Server) handleRemoteModelDelete(w http.ResponseWriter, r *http.Request) {
	if err := s.remote.DeleteModel(r.PathValue("id")); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

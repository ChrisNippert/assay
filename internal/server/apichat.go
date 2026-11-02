package server

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"assay/internal/ollama"
	"assay/internal/remote"
)

// API models in chat, Compare and presets: their names start with "api:",
// chat goes to the API instead of llama-server, and there's nothing to load,
// size or measure on this computer.

func isAPIModel(name string) bool { return strings.HasPrefix(name, remote.Prefix) }

// notLocal says a chat model isn't a model file here: an API model or a
// team. Nothing to load, size or tokenize for it.
func notLocal(name string) bool {
	return strings.HasPrefix(name, remote.Prefix) || strings.HasPrefix(name, remote.TeamPrefix)
}

// chat sends a chat request to the model's backend: the API for an API
// model, llama-server otherwise.
func (s *Server) chat(ctx context.Context, req ollama.ChatRequest, fn func(ollama.ChatResponse) error) error {
	if t, ok := s.remote.Team(req.Model); ok {
		return s.chatTeam(ctx, t, req, fn)
	}
	if m, ep, ok := s.remote.Lookup(req.Model); ok {
		return remote.ChatOllama(ctx, ep, m, req, func(r ollama.ChatResponse) error {
			if r.Done {
				s.spend.Add(req.Model, r.Cost, r.EvalCount)
			}
			return fn(r)
		})
	}
	return s.ollama.Chat(ctx, req, fn)
}

// show is Show for any model; for an API model it's made up from the
// endpoint's model list: context length, and tools and vision unless the
// list says otherwise.
func (s *Server) show(ctx context.Context, name string) (*ollama.ShowResponse, error) {
	if t, ok := s.remote.Team(name); ok {
		// A team writes text; no tools or images (its members see text).
		lead, err := s.member(ctx, t.Lead)
		if err != nil {
			return nil, err
		}
		sh := &ollama.ShowResponse{Details: ollama.Details{Family: "team"}, ModelInfo: map[string]any{}, Capabilities: []string{"completion"}}
		if lsh, err := s.show(ctx, lead.name); err == nil && lsh.HasCapability("thinking") {
			sh.Capabilities = append(sh.Capabilities, "thinking")
		}
		return sh, nil
	}
	m, _, ok := s.remote.Lookup(name)
	if !ok {
		return s.ollama.Show(ctx, name)
	}
	info := s.apiInfo(ctx, name)
	sh := &ollama.ShowResponse{Details: ollama.Details{Family: "api"}, ModelInfo: map[string]any{}, Capabilities: []string{"completion", "thinking"}}
	if !info.NoTools {
		sh.Capabilities = append(sh.Capabilities, "tools")
	}
	if !info.NoVision {
		sh.Capabilities = append(sh.Capabilities, "vision")
	}
	if info.Context > 0 {
		sh.ModelInfo["api.context_length"] = float64(info.Context)
	}
	sh.System = m.System
	return sh, nil
}

// apiLists caches each endpoint's model list for an hour.
var apiLists sync.Map // endpoint URL → apiList

type apiList struct {
	at    time.Time
	items []remote.Listed
}

// apiInfo is what the endpoint's list says about an API model (zero if it
// can't be had).
func (s *Server) apiInfo(ctx context.Context, name string) remote.Listed {
	m, ep, ok := s.remote.Lookup(name)
	if !ok {
		return remote.Listed{}
	}
	var items []remote.Listed
	if v, ok := apiLists.Load(ep.URL); ok && time.Since(v.(apiList).at) < time.Hour {
		items = v.(apiList).items
	} else {
		cctx, cancel := context.WithTimeout(ctx, 10*time.Second)
		defer cancel()
		list, err := remote.List(cctx, ep)
		if err != nil {
			return remote.Listed{}
		}
		apiLists.Store(ep.URL, apiList{time.Now(), list})
		items = list
	}
	for _, l := range items {
		if l.ID == m.Model {
			return l
		}
	}
	return remote.Listed{}
}

// apiOption is an API model as a model picker shows it.
type apiOption struct {
	Name, Title, Model, Endpoint, Meta string
}

func (s *Server) apiOptions() []apiOption {
	var out []apiOption
	for _, m := range s.remote.Models() {
		ep, ok := s.remote.Endpoint(m.Endpoint)
		if !ok {
			continue
		}
		meta := m.Model
		if m.System != "" {
			meta += " · own prompt"
		}
		out = append(out, apiOption{Name: m.Name(), Title: m.Title, Model: m.Model, Endpoint: ep.Name, Meta: meta})
	}
	return out
}

// teamOptions are the teams as a model picker shows them.
func (s *Server) teamOptions(ctx context.Context) []apiOption {
	var out []apiOption
	for _, t := range s.remote.Teams() {
		lead, err1 := s.member(ctx, t.Lead)
		worker, err2 := s.member(ctx, t.Worker)
		if err1 != nil || err2 != nil {
			continue
		}
		out = append(out, apiOption{Name: t.Name(), Title: t.Title, Endpoint: "team", Meta: teamQuant(t, lead.title, worker.title)})
	}
	return out
}

// money formats US dollars: cents above a dollar, more places below.
func money(v float64) string {
	switch {
	case v >= 1:
		return fmt.Sprintf("$%.2f", v)
	case v >= 0.01:
		return fmt.Sprintf("$%.3f", v)
	case v >= 0.0001:
		return fmt.Sprintf("$%.4f", v)
	}
	return "<$0.0001"
}

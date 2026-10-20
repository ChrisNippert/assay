// Package server serves the ollamui web UI and its JSON API.
package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"html/template"
	"log/slog"
	"net"
	"net/http"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"ollamui/internal/chats"
	"ollamui/internal/hf"
	"ollamui/internal/hw"
	"ollamui/internal/ollama"
	"ollamui/internal/presets"
	"ollamui/internal/pulls"
	"ollamui/internal/scores"
	"ollamui/internal/voice"
	"ollamui/web"
)

// Options configures a Server.
type Options struct {
	Ollama  *ollama.Client
	Chats   *chats.Store
	Presets *presets.Store
	HF      *hf.Client
	Pulls   *pulls.Manager
	// Scores is optional; without it nothing is rated.
	Scores *scores.Store
	// Voice is the speech service for voice mode; nil turns it off.
	Voice *voice.Client
	// VoiceSettings are voice mode's preferences. Required.
	VoiceSettings *voice.SettingsStore
	// VoiceLauncher, if set, starts the speech service on demand.
	VoiceLauncher *voice.Launcher
	Memory        hw.Memory
	// AllowHosts are extra hostnames (besides localhost and IP addresses)
	// the UI may be reached by, e.g. a Tailscale machine name.
	AllowHosts []string
	Logger     *slog.Logger
}

// Server is the HTTP handler for the whole app.
type Server struct {
	ollama  *ollama.Client
	chats   *chats.Store
	presets *presets.Store
	hf      *hf.Client
	pulls   *pulls.Manager
	scores  *scores.Store
	voice   *voice.Client
	voiceSt *voice.SettingsStore
	speech  *voice.Launcher
	search  searchCache
	env     envCache
	ctxBase sync.Map // model name → measured weight memory, for picking context
	// thinkers remembers, per model, whether it thought when left to decide.
	thinkers sync.Map
	// otherVRAM is the last measured VRAM used outside Ollama (desktop etc.).
	otherVRAM atomic.Int64
	// vramUsed reports GPU memory in use; swapped out in tests.
	vramUsed func() int64
	memory   hw.Memory
	log      *slog.Logger
	pages    map[string]*template.Template
	handler  http.Handler
}

// New builds the server and its routes.
func New(opts Options) (*Server, error) {
	s := &Server{
		ollama: opts.Ollama, chats: opts.Chats, presets: opts.Presets,
		hf: opts.HF, pulls: opts.Pulls, scores: opts.Scores, voice: opts.Voice, voiceSt: opts.VoiceSettings, speech: opts.VoiceLauncher, memory: opts.Memory, log: opts.Logger,
		vramUsed: hw.VRAMUsed,
	}
	if s.voiceSt == nil {
		s.voiceSt, _ = voice.OpenSettings("") // in memory
	}
	s.env.read = func(ctx context.Context) ollama.ServerEnv { return ollama.ReadServerEnv(ctx, s.ollama.Port()) }
	if s.log == nil {
		s.log = slog.Default()
	}
	s.pages = map[string]*template.Template{}
	for _, name := range []string{"chat", "library", "discover", "model", "presets", "settings"} {
		t, err := template.New("").Funcs(templateFuncs).ParseFS(web.FS, "templates/layout.html", "templates/"+name+".html")
		if err != nil {
			return nil, fmt.Errorf("parse %s template: %w", name, err)
		}
		s.pages[name] = t
	}

	mux := http.NewServeMux()
	mux.Handle("GET /static/", http.FileServerFS(web.FS))
	mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "/chat", http.StatusFound)
	})
	mux.HandleFunc("GET /chat", s.handleChatPage)
	mux.HandleFunc("GET /chat/{id}", s.handleChatPage)
	mux.HandleFunc("GET /library", s.handleLibrary)
	mux.HandleFunc("GET /discover", s.handleDiscover)
	mux.HandleFunc("GET /discover/{owner}/{name}", s.handleModelPage)
	mux.HandleFunc("GET /presets", s.handlePresetsPage)
	mux.HandleFunc("GET /settings", s.handleSettingsPage)
	mux.HandleFunc("GET /api/ollama/env", s.handleOllamaEnv)

	mux.HandleFunc("GET /api/loaded", s.handleLoaded)
	mux.HandleFunc("GET /api/models/defaults", s.handleModelDefaults)
	mux.HandleFunc("POST /api/models/load", s.handleLoad)
	mux.HandleFunc("POST /api/models/eject", s.handleEject)
	mux.HandleFunc("POST /api/models/delete", s.handleDeleteModel)
	mux.HandleFunc("POST /api/chat", s.handleChatSend)
	mux.HandleFunc("DELETE /api/chats/{id}", s.handleDeleteChat)
	mux.HandleFunc("GET /api/voice", s.handleVoiceStatus)
	mux.HandleFunc("POST /api/voice/transcribe", s.handleTranscribe)
	mux.HandleFunc("POST /api/voice/speak", s.handleSpeak)
	mux.HandleFunc("GET /api/voice/settings", s.handleVoiceSettings)
	mux.HandleFunc("PUT /api/voice/settings", s.handleVoiceSettingsSave)

	mux.HandleFunc("GET /api/hf/search", s.handleSearch)
	mux.HandleFunc("GET /api/pulls", s.handlePullList)
	mux.HandleFunc("POST /api/pulls", s.handlePullStart)
	mux.HandleFunc("POST /api/pulls/{id}/cancel", s.handlePullCancel)

	mux.HandleFunc("GET /api/presets", s.handlePresetList)
	mux.HandleFunc("PUT /api/presets/{name}", s.handlePresetSave)
	mux.HandleFunc("DELETE /api/presets/{name}", s.handlePresetDelete)

	s.handler = hostGuard(opts.AllowHosts, http.NewCrossOriginProtection().Handler(mux))
	return s, nil
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) { s.handler.ServeHTTP(w, r) }

// hostGuard only answers requests addressed to localhost, an IP literal or
// an explicitly allowed name. That stops DNS-rebinding attacks, where a
// website you visit points its own domain at 127.0.0.1 to reach this UI.
func hostGuard(extra []string, next http.Handler) http.Handler {
	allowed := map[string]bool{"localhost": true}
	for _, h := range extra {
		allowed[strings.ToLower(strings.TrimSpace(h))] = true
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		host := r.Host
		if h, _, err := net.SplitHostPort(host); err == nil {
			host = h
		}
		host = strings.ToLower(strings.Trim(host, "[]"))
		if net.ParseIP(host) != nil || allowed[host] {
			next.ServeHTTP(w, r)
			return
		}
		http.Error(w, fmt.Sprintf("Host %q is not allowed. Restart ollamui with -allow-host %s to permit it.", host, host), http.StatusForbidden)
	})
}

// page is the data every template gets.
type page struct {
	Title     string
	Active    string
	ChatID    string
	Chats     []chats.Summary
	Models    []ollama.Model
	Loaded    []loadedModel
	Presets   []*presets.Preset
	OllamaURL string
	OllamaErr string
	Data      any
}

// LoadedNames lists the models currently in memory.
func (p *page) LoadedNames() []string {
	out := make([]string, 0, len(p.Loaded))
	for _, m := range p.Loaded {
		out = append(out, m.Name)
	}
	return out
}

// IsLoaded reports whether a model is currently in memory.
func (p *page) IsLoaded(name string) bool {
	for _, m := range p.Loaded {
		if m.Name == name {
			return true
		}
	}
	return false
}

func (s *Server) basePage(ctx context.Context, active, title string) *page {
	p := &page{Active: active, Title: title, OllamaURL: s.ollama.URL()}
	if list, err := s.chats.List(); err == nil {
		p.Chats = list
	} else {
		s.log.Error("list chats", "err", err)
	}
	if list, err := s.presets.List(); err == nil {
		p.Presets = list
	} else {
		s.log.Error("list presets", "err", err)
	}
	ctx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	models, err := s.ollama.List(ctx)
	if err != nil {
		p.OllamaErr = err.Error()
		return p
	}
	sort.Slice(models, func(i, j int) bool { return models[i].Name < models[j].Name })
	p.Models = models
	p.Loaded, _ = s.loaded(ctx)
	return p
}

func (s *Server) render(w http.ResponseWriter, name string, p *page) {
	var buf bytes.Buffer
	if err := s.pages[name].ExecuteTemplate(&buf, "layout", p); err != nil {
		s.log.Error("render", "page", name, "err", err)
		http.Error(w, "template error", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	buf.WriteTo(w)
}

// loadedModel is a model in memory, as the UI shows it.
type loadedModel struct {
	Name      string    `json:"name"`
	Size      int64     `json:"size"`
	SizeVRAM  int64     `json:"sizeVram"`
	Pinned    bool      `json:"pinned"`
	ExpiresAt time.Time `json:"expiresAt"`
	Context   int       `json:"context"`
}

// pinnedAfter is how far away an expiry has to be for us to treat the model
// as loaded "until ejected". Ollama reports keep_alive -1 as a date
// centuries out.
const pinnedAfter = 365 * 24 * time.Hour

func (s *Server) loaded(ctx context.Context) ([]loadedModel, error) {
	running, err := s.ollama.Running(ctx)
	if err != nil {
		return nil, err
	}
	out := make([]loadedModel, 0, len(running))
	for _, m := range running {
		out = append(out, loadedModel{
			Name:      m.Name,
			Size:      m.Size,
			SizeVRAM:  m.SizeVRAM,
			Pinned:    time.Until(m.ExpiresAt) > pinnedAfter,
			ExpiresAt: m.ExpiresAt,
			Context:   m.ContextLength,
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out, nil
}

// --- JSON helpers ---

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

// ollamaStatus maps an Ollama error to the status we return to the browser.
func ollamaStatus(err error) int {
	var apiErr *ollama.APIError
	if errors.As(err, &apiErr) && apiErr.StatusCode >= 400 && apiErr.StatusCode < 500 {
		return apiErr.StatusCode
	}
	return http.StatusBadGateway
}

func readJSON(w http.ResponseWriter, r *http.Request, v any) bool {
	if !strings.HasPrefix(r.Header.Get("Content-Type"), "application/json") {
		writeError(w, http.StatusUnsupportedMediaType, "expected application/json")
		return false
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20)).Decode(v); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return false
	}
	return true
}

// --- template helpers ---

var templateFuncs = template.FuncMap{
	// hfRepo turns "hf.co/owner/repo:quant" into "owner/repo" ("" for other names).
	"hfRepo": func(name string) string {
		rest, ok := strings.CutPrefix(name, "hf.co/")
		if !ok {
			return ""
		}
		rest, _, _ = strings.Cut(rest, ":")
		if strings.Count(rest, "/") != 1 {
			return ""
		}
		return rest
	},
	"bytes": formatBytes,
	"kib":   func(n int) int { return (n + 512) / 1024 },
	// compact formats counts like 1329922 as "1.3M".
	"compact": func(n int) string {
		switch {
		case n >= 1e6:
			return strconv.FormatFloat(float64(n)/1e6, 'f', 1, 64) + "M"
		case n >= 1e3:
			return strconv.FormatFloat(float64(n)/1e3, 'f', 1, 64) + "k"
		}
		return strconv.Itoa(n)
	},
	"ago": ago,
	"json": func(v any) (string, error) {
		b, err := json.Marshal(v)
		return string(b), err
	},
	"deref": func(b *bool) bool { return b != nil && *b },
	"lower": func(s string) string {
		if s == "" {
			return s
		}
		return strings.ToLower(s[:1]) + s[1:]
	},
	"hasChoice": func(list []string, v string) bool { return slices.Contains(list, v) },
	"billions": func(n int64) string {
		if n <= 0 {
			return ""
		}
		return fmt.Sprintf("%.1fB", float64(n)/1e9)
	},
	"gpuPct": func(m loadedModel) int {
		if m.Size == 0 {
			return 100
		}
		return int(m.SizeVRAM * 100 / m.Size)
	},
}

func formatBytes(n int64) string {
	switch {
	case n >= 1e9:
		return fmt.Sprintf("%.1f GB", float64(n)/1e9)
	case n >= 1e6:
		return fmt.Sprintf("%.0f MB", float64(n)/1e6)
	default:
		return fmt.Sprintf("%d KB", n/1e3)
	}
}

func ago(t time.Time) string {
	d := time.Since(t)
	switch {
	case d < time.Minute:
		return "just now"
	case d < time.Hour:
		return fmt.Sprintf("%dm ago", int(d.Minutes()))
	case d < 24*time.Hour:
		return fmt.Sprintf("%dh ago", int(d.Hours()))
	case d < 60*24*time.Hour:
		return fmt.Sprintf("%dd ago", int(d.Hours()/24))
	default:
		return t.Format("2 Jan 2006")
	}
}

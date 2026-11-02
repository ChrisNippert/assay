// Package server serves the Assay web UI and its JSON API.
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
	"os"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"assay/internal/bench"
	"assay/internal/chats"
	"assay/internal/evals"
	"assay/internal/hf"
	"assay/internal/hw"
	"assay/internal/lean"
	"assay/internal/limits"
	"assay/internal/llamacpp"
	"assay/internal/ollama"
	"assay/internal/presets"
	"assay/internal/projects"
	"assay/internal/pulls"
	"assay/internal/remote"
	"assay/internal/scores"
	"assay/internal/service"
	"assay/internal/tools"
	"assay/internal/update"
	"assay/internal/voice"
	"assay/web"
)

// Backend runs the models: the llama.cpp engine (a fake in tests).
type Backend = backend

type backend interface {
	Version(ctx context.Context) (string, error)
	List(ctx context.Context) ([]ollama.Model, error)
	Running(ctx context.Context) ([]ollama.RunningModel, error)
	Show(ctx context.Context, model string) (*ollama.ShowResponse, error)
	Load(ctx context.Context, model, keepAlive string, options map[string]any) error
	Unload(ctx context.Context, model string) error
	Delete(ctx context.Context, model string) error
	Chat(ctx context.Context, req ollama.ChatRequest, fn func(ollama.ChatResponse) error) error
	Embed(ctx context.Context, model string, input []string) ([][]float32, error)
}

// chatForgetter is a backend that keeps per-chat state (saved caches) to
// drop when a chat is deleted or edited.
type chatForgetter interface{ ForgetChat(chat string) }

// forgetChat drops the backend's state for a chat, if it keeps any.
func (s *Server) forgetChat(id string) {
	if f, ok := s.ollama.(chatForgetter); ok {
		f.ForgetChat(id)
	}
}

// fitter is a backend that sizes the context itself (llama.cpp's --fit):
// Assay's own auto context then stays out of the way.
type fitter interface{ CacheType() string }

// Options configures a Server.
type Options struct {
	Ollama backend
	Chats  *chats.Store
	// ChatSettings holds the chat preferences (automatic compression); nil
	// keeps them in memory.
	ChatSettings *chats.SettingsStore
	Presets      *presets.Store
	HF           *hf.Client
	Pulls        *pulls.Manager
	// LlamaCpp installs llama.cpp releases (Settings); nil turns that off.
	LlamaCpp *llamacpp.Manager
	// Scores is optional; without it nothing is rated.
	Scores *scores.Store
	// Bench keeps speed tests; without it they aren't saved.
	Bench *bench.Store
	// Evals holds benchmark sets and EvalResults their runs; nil keeps only
	// the built-in sets and no results.
	Evals       *evals.Library
	EvalResults *evals.Results
	// Lean installs and finds the Lean prover for proof sets; nil: only a
	// lean on PATH is used.
	Lean *lean.Installer
	// Projects and RAGSettings hold projects, their files and the RAG
	// preferences. Nil keeps them in a temporary folder / in memory.
	Projects    *projects.Store
	RAGSettings *projects.SettingsStore
	// Tools holds the tool settings; nil keeps them in memory.
	Tools *tools.Store
	// Limits holds the resource limits; nil keeps them in memory.
	Limits *limits.Store
	// Remote holds the API models the benchmarks can run; nil keeps them
	// in memory.
	Remote *remote.Store
	// Voice is the speech service for voice mode; nil turns it off.
	Voice *voice.Client
	// VoiceSettings are voice mode's preferences. Required.
	VoiceSettings *voice.SettingsStore
	// VoiceLauncher, if set, starts the speech service on demand.
	VoiceLauncher *voice.Launcher
	Memory        hw.Memory
	// Version is this build's version; Updates looks for a newer one (nil:
	// never). Service manages start-on-boot (nil: not offered) and
	// ServiceArgs are the flags it runs the app with.
	Version     string
	Updates     *update.Checker
	Service     service.Service
	ServiceArgs []string
	// SetupFile remembers that first-run setup was finished or skipped; empty keeps that in memory.
	SetupFile string
	// AllowHosts are extra hostnames (besides localhost and IP addresses)
	// the UI may be reached by, e.g. a Tailscale machine name.
	AllowHosts []string
	Logger     *slog.Logger
}

// Server is the HTTP handler for the whole app.
type Server struct {
	ollama  backend
	chats   *chats.Store
	chatSt  *chats.SettingsStore
	presets *presets.Store
	hf      *hf.Client
	pulls   *pulls.Manager
	move    moveJob // the models folder move, if any
	llama   *llamacpp.Manager
	scores  *scores.Store
	bench   *bench.Store
	// projects, their RAG settings, and files waiting to be indexed.
	projects   *projects.Store
	ragSt      *projects.SettingsStore
	indexQueue chan indexJob
	// toolSt holds the tool settings; fetcher and searcher run web tools.
	toolSt  *tools.Store
	limitSt *limits.Store
	remote  *remote.Store
	version string
	updates *update.Checker
	service service.Service
	// serviceArgs are the flags start-on-boot runs the app with.
	serviceArgs []string
	fetcher     *tools.Fetcher
	searcher    *tools.Searcher
	voice       *voice.Client
	voiceSt     *voice.SettingsStore
	speech      *voice.Launcher
	search      searchCache
	ctxBase     sync.Map // model name → measured weight memory, for picking context
	// ctxCap is model name → largest context worth trying, learned when a
	// bigger one spilled onto the CPU.
	ctxCap sync.Map
	// manualGPU is model name → true while chats set its GPU layers by hand.
	manualGPU sync.Map
	// thinkers remembers, per model, whether it thought when left to decide.
	thinkers sync.Map
	// libScores caches each repo's intelligence score for the Library.
	libScores sync.Map
	// measure is the Library's measure-all run; benchFn runs one speed test
	// (nil: runBench; tests swap it).
	measure measureJob
	// evalLib, evalRes and evalJob are the benchmarks (see evals.go);
	// evalFn runs one model × set (nil: runEvalItem; tests swap it).
	evalLib  *evals.Library
	evalRes  *evals.Results
	evalJob  evalJob
	leanInst *lean.Installer
	evalFn   func(ctx context.Context, item int, redo bool) error
	benchFn  func(ctx context.Context, name string, keepPinned bool) (bench.Result, error)
	// otherVRAM is the last measured VRAM used outside Ollama (desktop etc.).
	otherVRAM atomic.Int64
	// vramUsed reports GPU memory in use; swapped out in tests.
	vramUsed func() int64
	memory   hw.Memory
	setup    *setupState
	// sugg are the first-model suggestions, kept for a few minutes.
	suggMu  sync.Mutex
	sugg    []suggestion
	suggAt  time.Time
	log     *slog.Logger
	pages   map[string]*template.Template
	handler http.Handler
	// runs are the replies being written, by chat id (see runs.go).
	runsMu sync.Mutex
	runs   map[string]*run
}

// New builds the server and its routes.
func New(opts Options) (*Server, error) {
	s := &Server{
		ollama: opts.Ollama, chats: opts.Chats, chatSt: opts.ChatSettings, presets: opts.Presets,
		hf: opts.HF, pulls: opts.Pulls, llama: opts.LlamaCpp, scores: opts.Scores, bench: opts.Bench, projects: opts.Projects, ragSt: opts.RAGSettings, toolSt: opts.Tools, limitSt: opts.Limits, remote: opts.Remote, version: opts.Version, updates: opts.Updates, service: opts.Service, serviceArgs: opts.ServiceArgs, indexQueue: make(chan indexJob, 256), voice: opts.Voice, voiceSt: opts.VoiceSettings, speech: opts.VoiceLauncher, memory: opts.Memory, log: opts.Logger, setup: openSetupState(opts.SetupFile),
		vramUsed: hw.VRAMUsed,
	}
	if s.bench == nil {
		s.bench = bench.Open("")
	}
	s.evalLib, s.evalRes = opts.Evals, opts.EvalResults
	if s.evalLib == nil {
		var err error
		if s.evalLib, err = evals.OpenLibrary(""); err != nil {
			return nil, err
		}
	}
	if s.evalRes == nil {
		s.evalRes = evals.OpenResults("")
	}
	if s.leanInst = opts.Lean; s.leanInst == nil {
		s.leanInst = &lean.Installer{}
	}
	if s.projects == nil {
		dir, err := os.MkdirTemp("", "assay-projects-")
		if err != nil {
			return nil, err
		}
		if s.projects, err = projects.Open(dir); err != nil {
			return nil, err
		}
	}
	if s.ragSt == nil {
		s.ragSt, _ = projects.OpenSettings("")
	}
	if s.remote == nil {
		s.remote, _ = remote.Open("")
	}
	if s.limitSt == nil {
		s.limitSt, _ = limits.Open("")
	}
	if s.updates == nil {
		s.updates = update.Open("", "", s.version) // no repo: never asks
	}
	if s.toolSt == nil {
		s.toolSt, _ = tools.OpenStore("")
	}
	s.fetcher, s.searcher = tools.NewFetcher(), tools.NewSearcher()
	if s.voiceSt == nil {
		s.voiceSt, _ = voice.OpenSettings("") // in memory
	}
	if s.chatSt == nil {
		s.chatSt, _ = chats.OpenSettings("") // in memory
	}
	if s.log == nil {
		s.log = slog.Default()
	}
	s.pages = map[string]*template.Template{}
	for _, name := range []string{"chat", "library", "discover", "model", "presets", "settings", "compare", "project", "setup", "benchmarks"} {
		t, err := template.New("").Funcs(templateFuncs).ParseFS(web.FS, "templates/layout.html", "templates/"+name+".html")
		if err != nil {
			return nil, fmt.Errorf("parse %s template: %w", name, err)
		}
		s.pages[name] = t
	}

	mux := http.NewServeMux()
	mux.Handle("GET /static/", http.FileServerFS(web.FS))
	// The service worker must sit at the root to control "/". Serve it, and the
	// manifest, with explicit types so they never depend on the OS mime database.
	mux.Handle("GET /sw.js", staticFile("static/sw.js", "application/javascript", "no-cache"))
	mux.Handle("GET /static/manifest.webmanifest", staticFile("static/manifest.webmanifest", "application/manifest+json", ""))
	mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, r *http.Request) {
		u := "/chat"
		if r.URL.RawQuery != "" {
			u += "?" + r.URL.RawQuery
		}
		http.Redirect(w, r, u, http.StatusFound)
	})
	mux.HandleFunc("GET /chat", s.handleChatPage)
	mux.HandleFunc("GET /chat/{id}", s.handleChatPage)
	mux.HandleFunc("GET /library", s.handleLibrary)
	mux.HandleFunc("GET /discover", s.handleDiscover)
	mux.HandleFunc("GET /discover/{owner}/{name}", s.handleModelPage)
	mux.HandleFunc("GET /presets", s.handlePresetsPage)
	mux.HandleFunc("GET /settings", s.handleSettingsPage)
	mux.HandleFunc("GET /setup", s.handleSetupPage)
	mux.HandleFunc("GET /api/setup", s.handleSetup)
	mux.HandleFunc("GET /api/setup/models", s.handleSetupModels)
	mux.HandleFunc("POST /api/setup/done", s.handleSetupDone)
	mux.HandleFunc("GET /compare", s.handleComparePage)
	mux.HandleFunc("GET /benchmarks", s.handleBenchmarksPage)
	mux.HandleFunc("GET /api/evals/sets", s.handleEvalSets)
	mux.HandleFunc("GET /api/evals/sets/{id}", s.handleEvalSet)
	mux.HandleFunc("PUT /api/evals/sets/{id}", s.handleEvalSetSave)
	mux.HandleFunc("DELETE /api/evals/sets/{id}", s.handleEvalSetDelete)
	mux.HandleFunc("GET /api/evals/results", s.handleEvalResults)
	mux.HandleFunc("GET /api/evals/result", s.handleEvalResult)
	mux.HandleFunc("DELETE /api/evals/result", s.handleEvalResultDelete)
	mux.HandleFunc("GET /api/evals/run", s.handleEvalRunState)
	mux.HandleFunc("POST /api/evals/run", s.handleEvalRunStart)
	mux.HandleFunc("DELETE /api/evals/run", s.handleEvalRunCancel)
	mux.HandleFunc("GET /api/remote", s.handleRemote)
	mux.HandleFunc("PUT /api/remote/endpoints/{id}", s.handleRemoteEndpointPut)
	mux.HandleFunc("DELETE /api/remote/endpoints/{id}", s.handleRemoteEndpointDelete)
	mux.HandleFunc("GET /api/remote/endpoints/{id}/models", s.handleRemoteList)
	mux.HandleFunc("PUT /api/remote/models/{id}", s.handleRemoteModelPut)
	mux.HandleFunc("DELETE /api/remote/models/{id}", s.handleRemoteModelDelete)
	mux.HandleFunc("GET /api/teams", s.handleTeams)
	mux.HandleFunc("PUT /api/teams/{id}", s.handleTeamPut)
	mux.HandleFunc("DELETE /api/teams/{id}", s.handleTeamDelete)
	mux.HandleFunc("GET /api/lean", s.handleLeanStatus)
	mux.HandleFunc("POST /api/lean/install", s.handleLeanInstall)
	mux.HandleFunc("DELETE /api/lean/install", s.handleLeanCancel)
	mux.HandleFunc("GET /projects/{id}", s.handleProjectPage)
	mux.HandleFunc("POST /api/projects", s.handleProjectCreate)
	mux.HandleFunc("GET /api/projects/{id}", s.handleProjectGet)
	mux.HandleFunc("PATCH /api/projects/{id}", s.handleProjectUpdate)
	mux.HandleFunc("DELETE /api/projects/{id}", s.handleProjectDelete)
	mux.HandleFunc("POST /api/projects/{id}/files", s.handleProjectUpload)
	mux.HandleFunc("DELETE /api/projects/{id}/files/{name}", s.handleProjectFileDelete)
	mux.HandleFunc("POST /api/projects/{id}/reindex", s.handleProjectReindex)
	mux.HandleFunc("GET /api/knowledge/settings", s.handleRAGSettings)
	mux.HandleFunc("PUT /api/knowledge/settings", s.handleRAGSettingsSave)
	mux.HandleFunc("GET /api/tools/settings", s.handleToolSettings)
	mux.HandleFunc("PUT /api/tools/settings", s.handleToolSettingsSave)
	mux.HandleFunc("GET /api/limits", s.handleLimits)
	mux.HandleFunc("PUT /api/limits", s.handleLimitsSave)
	mux.HandleFunc("POST /api/bench", s.handleBench)
	mux.HandleFunc("GET /api/library/scores", s.handleLibScores)
	mux.HandleFunc("GET /api/library/nearby", s.handleLibNearby)
	mux.HandleFunc("GET /api/library/measure", s.handleMeasureState)
	mux.HandleFunc("POST /api/library/measure", s.handleMeasureStart)
	mux.HandleFunc("DELETE /api/library/measure", s.handleMeasureCancel)

	mux.HandleFunc("GET /api/loaded", s.handleLoaded)
	mux.HandleFunc("GET /api/gpu", s.handleGPU)
	mux.HandleFunc("GET /api/models/defaults", s.handleModelDefaults)
	mux.HandleFunc("PUT /api/models/settings", s.handleModelSettings)
	mux.HandleFunc("GET /api/models/recommended", s.handleModelRecommended)
	mux.HandleFunc("POST /api/models/recommended", s.handleModelUseRecommended)
	mux.HandleFunc("GET /api/models/mtp", s.handleModelMTP)
	mux.HandleFunc("GET /api/pick", s.handlePickInfo)
	mux.HandleFunc("POST /api/pick", s.handlePick)
	mux.HandleFunc("GET /api/fs", s.handleFS)
	mux.HandleFunc("GET /api/service", s.handleService)
	mux.HandleFunc("PUT /api/service", s.handleServiceSet)
	mux.HandleFunc("GET /api/update", s.handleUpdate)
	mux.HandleFunc("PUT /api/update", s.handleUpdateSet)
	mux.HandleFunc("GET /api/models/folder", s.handleFolder)
	mux.HandleFunc("POST /api/models/folder", s.handleFolderMove)
	mux.HandleFunc("DELETE /api/models/folder", s.handleFolderCancel)
	mux.HandleFunc("POST /api/models/folder/check", s.handleFolderCheck)
	mux.HandleFunc("GET /api/models/sources", s.handleSources)
	mux.HandleFunc("POST /api/models/sources/import", s.handleSourcesImport)
	mux.HandleFunc("POST /api/models/adapters", s.handleAdapterAdd)
	mux.HandleFunc("POST /api/models/sources/folder", s.handleSourcesFolder)
	mux.HandleFunc("DELETE /api/models/sources/folder", s.handleSourcesFolder)
	mux.HandleFunc("POST /api/models/load", s.handleLoad)
	mux.HandleFunc("POST /api/models/eject", s.handleEject)
	mux.HandleFunc("POST /api/models/star", s.handleStar)
	mux.HandleFunc("POST /api/models/delete", s.handleDeleteModel)
	mux.HandleFunc("POST /api/chat", s.handleChatSend)
	mux.HandleFunc("GET /api/chats/running", s.handleChatsRunning)
	mux.HandleFunc("GET /api/chats/{id}/stream", s.handleChatStream)
	mux.HandleFunc("POST /api/chats/{id}/stop", s.handleChatStop)
	mux.HandleFunc("POST /api/attachments", s.handleAttachmentUpload)
	mux.HandleFunc("GET /api/chats/{id}/files/{file}", s.handleChatFile)
	mux.HandleFunc("GET /api/vision", s.handleVision)
	mux.HandleFunc("DELETE /api/chats/{id}", s.handleDeleteChat)
	mux.HandleFunc("PATCH /api/chats/{id}", s.handleRenameChat)
	mux.HandleFunc("POST /api/chats/{id}/compress", s.handleCompress)
	mux.HandleFunc("DELETE /api/chats/{id}/compression", s.handleUncompress)
	mux.HandleFunc("DELETE /api/chats/{id}/trims/{trim}", s.handleUndoTrim)
	mux.HandleFunc("GET /api/chats/settings", s.handleChatSettings)
	mux.HandleFunc("PUT /api/chats/settings", s.handleChatSettings)
	mux.HandleFunc("GET /api/voice", s.handleVoiceStatus)
	mux.HandleFunc("POST /api/voice/transcribe", s.handleTranscribe)
	mux.HandleFunc("POST /api/voice/speak", s.handleSpeak)
	mux.HandleFunc("GET /api/voice/settings", s.handleVoiceSettings)
	mux.HandleFunc("PUT /api/voice/settings", s.handleVoiceSettingsSave)
	mux.HandleFunc("POST /api/voice/remove", s.handleVoiceRemove)

	mux.HandleFunc("GET /api/hf/search", s.handleSearch)
	mux.HandleFunc("GET /api/pulls", s.handlePullList)
	mux.HandleFunc("POST /api/pulls", s.handlePullStart)
	mux.HandleFunc("POST /api/pulls/{id}/cancel", s.handlePullCancel)

	mux.HandleFunc("GET /v1/models", s.handleV1Models)
	mux.HandleFunc("POST /v1/chat/completions", s.handleV1Chat)
	mux.HandleFunc("POST /v1/embeddings", s.handleV1Embeddings)
	mux.HandleFunc("GET /api/llamacpp", s.handleLlamaStatus)
	mux.HandleFunc("GET /api/llamacpp/check", s.handleLlamaCheck)
	mux.HandleFunc("POST /api/llamacpp/install", s.handleLlamaInstall)
	mux.HandleFunc("POST /api/llamacpp/cancel", s.handleLlamaCancel)
	mux.HandleFunc("PUT /api/llamacpp/use", s.handleLlamaUse)
	mux.HandleFunc("PUT /api/llamacpp/custom", s.handleLlamaCustom)

	mux.HandleFunc("GET /api/presets", s.handlePresetList)
	mux.HandleFunc("PUT /api/presets/{name}", s.handlePresetSave)
	mux.HandleFunc("DELETE /api/presets/{name}", s.handlePresetDelete)

	s.handler = hostGuard(opts.AllowHosts, http.NewCrossOriginProtection().Handler(mux))
	return s, nil
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) { s.handler.ServeHTTP(w, r) }

// staticFile serves one embedded file with a fixed Content-Type and, when
// cacheControl is non-empty, that Cache-Control header.
func staticFile(name, contentType, cacheControl string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		b, err := web.FS.ReadFile(name)
		if err != nil {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", contentType)
		if cacheControl != "" {
			w.Header().Set("Cache-Control", cacheControl)
		}
		w.Write(b)
	}
}

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
		http.Error(w, fmt.Sprintf("Host %q is not allowed. Restart Assay with -allow-host %s to permit it.", host, host), http.StatusForbidden)
	})
}

// page is the data every template gets.
type page struct {
	Title  string
	Active string
	ChatID string
	// ProjectID is the project the page is about or in, for the sidebar.
	ProjectID string
	Projects  []*projects.Project
	Chats     []chats.Summary
	running   []string
	Models    []ollama.Model
	Loaded    []loadedModel
	// APIModels are the API models, offered next to the local ones.
	APIModels []apiOption
	Presets   []*presets.Preset
	ModelsErr string // couldn't list the models
	// Starred is the default model (see chats.Settings.DefaultModel).
	Starred string
	Data    any

	// For ModelGroups' fit and speed.
	memory hw.Memory
	bench  *bench.Store
	groups []modelGroup
	// linkNote says where a linked model comes from (the Library); may be nil.
	linkNote func(name string) *linkNote
	// Adapters are the LoRA adapters in the library (the Library page).
	Adapters []adapterRow
}

// LoadedNames lists the models currently in memory.
func (p *page) LoadedNames() []string {
	out := make([]string, 0, len(p.Loaded))
	for _, m := range p.Loaded {
		out = append(out, m.Name)
	}
	return out
}

// IsRunning reports whether the chat has a reply being written.
func (p *page) IsRunning(id string) bool { return slices.Contains(p.running, id) }

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
	p := &page{running: s.runningChats(), Active: active, Title: title, memory: s.memory, bench: s.bench, Starred: s.chatSt.Get().DefaultModel}
	if list, err := s.chats.List(); err == nil {
		p.Chats = list
	} else {
		s.log.Error("list chats", "err", err)
	}
	if list, err := s.projects.List(); err == nil {
		p.Projects = list
	} else {
		s.log.Error("list projects", "err", err)
	}
	if list, err := s.presets.List(); err == nil {
		p.Presets = list
	} else {
		s.log.Error("list presets", "err", err)
	}
	p.APIModels = s.apiOptions()
	ctx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	models, err := s.ollama.List(ctx)
	if err != nil {
		p.ModelsErr = err.Error()
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
	Loading   bool      `json:"loading,omitempty"`
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
	if l, ok := s.ollama.(interface{ Loading() string }); ok {
		if name := l.Loading(); name != "" && !slices.ContainsFunc(out, func(m loadedModel) bool { return m.Name == name }) {
			out = append(out, loadedModel{Name: name, Loading: true})
		}
	}
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
	"filesize": func(n int64) string {
		switch {
		case n >= 1<<20:
			return fmt.Sprintf("%.1f MB", float64(n)/(1<<20))
		case n >= 1024:
			return fmt.Sprintf("%d KB", n/1024)
		}
		return fmt.Sprintf("%d B", n)
	},
	"join": strings.Join,
	"kib":  func(n int) int { return (n + 512) / 1024 },
	// secs formats milliseconds as seconds with one decimal: 11.2.
	"secs": func(ms int64) string { return strconv.FormatFloat(float64(ms)/1000, 'f', 1, 64) },
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
	// pretty indents a JSON string (tool arguments); anything else stays as is.
	"pretty": func(s string) string {
		var v any
		if json.Unmarshal([]byte(s), &v) != nil {
			return s
		}
		b, _ := json.MarshalIndent(v, "", "  ")
		return string(b)
	},
	"lower": func(s string) string {
		if s == "" {
			return s
		}
		return strings.ToLower(s[:1]) + s[1:]
	},
	"prettyTitle": func(name string) string { t, _ := prettyModel(name); return t },
	"hasChoice":   func(list []string, v string) bool { return slices.Contains(list, v) },
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

// Run does the server's background work until ctx ends.
func (s *Server) Run(ctx context.Context) {
	s.runIndexer(ctx)
}

package server

import (
	"context"
	"fmt"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"

	"ollamui/internal/engine"
	"ollamui/internal/hw"
	"ollamui/internal/limits"
)

// kvFactor scales f16 cache estimates to the engine's default K/V cache
// type (one slot per model).
func (s *Server) kvFactor(ctx context.Context) float64 {
	if f, ok := s.ollama.(fitter); ok {
		return hw.KVTypeFactor(f.CacheType())
	}
	return 1
}

// settingsData is what the settings page renders.
type settingsData struct {
	Version   string
	Memory    hw.Memory
	Engine    *enginePage // nil with Ollama
	Voice     voicePage
	Knowledge knowledgePage
	// CompressAuto is the default for chats' "Compress automatically".
	CompressAuto bool
	Tools        toolsPage
	Limits       limits.Limits
	// AppVersion is this build; Local says the request came from this
	// machine, which alone may set start-on-boot (when Service is set).
	AppVersion string
	Local      bool
	Service    bool
}

// enginePage is the Models section with llama.cpp.
type enginePage struct {
	Version, Server, Backend string
	// What a model without its own settings gets.
	CacheType, FlashAttn, KeepAlive string
	Saved                           []savedModel
	Drafts                          []engine.Draft // downloaded drafts, which no model list shows
	// Install is set when ollamui can install llama.cpp (the page's script
	// fills in the rest from /api/llamacpp).
	Install bool
	Source  string // flag, own, release, path or none
}

type savedModel struct {
	Name, Summary string
}

// engineInfo is the llama.cpp engine as the settings page sees it.
type engineInfo interface {
	modelSettings
	ServerPath() string
	Backend() string
	SavedSettings() map[string]map[string]any
}

func (s *Server) enginePage(ctx context.Context, e engineInfo) *enginePage {
	p := &enginePage{Server: e.ServerPath(), Backend: e.Backend(), CacheType: e.CacheType(), FlashAttn: e.FlashAttn(), KeepAlive: keepAliveText(e.KeepAlive().String())}
	p.Version, _ = s.ollama.Version(ctx)
	if s.llama != nil {
		p.Install = true
		_, p.Source = s.llama.Resolve()
	}
	for name, opts := range e.SavedSettings() {
		p.Saved = append(p.Saved, savedModel{Name: name, Summary: settingsSummary(opts)})
	}
	sort.Slice(p.Saved, func(i, j int) bool { return p.Saved[i].Name < p.Saved[j].Name })
	if d, ok := e.(interface{ DraftModels() []engine.Draft }); ok {
		p.Drafts = d.DraftModels()
	}
	return p
}

// settingsSummary says what a model's settings set, in a few words each.
func settingsSummary(o map[string]any) string {
	keys := make([]string, 0, len(o))
	for k := range o {
		keys = append(keys, k)
	}
	order := map[string]int{"num_ctx": 1, "num_gpu": 2, "keep_alive": 3, "cache_type": 4, "flash_attn": 5, "speedup": 6, "spec_ngram": 7, "draft_model": 8}
	sort.Slice(keys, func(i, j int) bool {
		oi, oj := order[keys[i]], order[keys[j]]
		if oi == 0 {
			oi = 99
		}
		if oj == 0 {
			oj = 99
		}
		return oi < oj || oi == oj && keys[i] < keys[j]
	})
	var parts []string
	for _, k := range keys {
		v := fmt.Sprint(o[k])
		switch k {
		case "num_ctx":
			if n, err := strconv.Atoi(v); err == nil {
				v = strconv.Itoa(n/1024) + "k"
			}
			parts = append(parts, v+" context")
		case "num_gpu":
			parts = append(parts, v+" GPU layers")
		case "keep_alive":
			parts = append(parts, "stays loaded "+keepAliveText(v))
		case "cache_type":
			parts = append(parts, v+" KV cache")
		case "flash_attn":
			parts = append(parts, "flash attention "+v)
		case "speedup":
			parts = append(parts, map[string]string{"auto": "speculative decoding auto", "off": "no speculative decoding", "mtp": "drafts with its MTP head"}[v])
		case "spec_ngram":
			if v == "off" {
				parts = append(parts, "no n-gram guessing")
			} else {
				parts = append(parts, "n-gram guessing")
			}
		case "draft_model":
			parts = append(parts, "drafts with "+v)
		case "num_thread":
			parts = append(parts, v+" CPU threads")
		case "load_mode":
			parts = append(parts, map[string]string{"none": "read into RAM", "mmap": "memory-mapped", "mlock": "locked in RAM"}[v])
		case "n_cpu_moe":
			if v == "-1" {
				parts = append(parts, "all experts on the CPU")
			} else {
				parts = append(parts, "experts of "+v+" layers on the CPU")
			}
		default:
			parts = append(parts, strings.ReplaceAll(k, "_", " ")+" "+v)
		}
	}
	return strings.Join(parts, " · ")
}

func keepAliveText(v string) string {
	switch v {
	case "-1":
		return "until unloaded"
	case "5m0s", "5m":
		return "5 minutes"
	case "30m0s", "30m":
		return "30 minutes"
	case "1h0m0s", "1h":
		return "1 hour"
	case "4h0m0s", "4h":
		return "4 hours"
	}
	return v
}

func (s *Server) handleSettingsPage(w http.ResponseWriter, r *http.Request) {
	p := s.basePage(r.Context(), "settings", "Settings")
	ctx, cancel := context.WithTimeout(r.Context(), 3*time.Second)
	defer cancel()
	d := settingsData{Memory: s.memory, CompressAuto: s.chatSt.Get().CompressAuto}
	d.Version, _ = s.ollama.Version(ctx)
	if e, ok := s.ollama.(engineInfo); ok {
		d.Engine = s.enginePage(ctx, e)
	}
	d.Voice = s.voicePage(ctx)
	d.Knowledge = s.knowledgePage(ctx, r.URL.Query().Get("pulled"))
	d.Tools = s.toolsPage()
	d.Limits = s.limitSt.Get()
	d.AppVersion, d.Local, d.Service = s.version, fromLoopback(r), s.service != nil
	p.Data = d
	s.render(w, "settings", p)
}

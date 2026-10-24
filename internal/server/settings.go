package server

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"ollamui/internal/hw"
	"ollamui/internal/ollama"
)

// envCache remembers the Ollama server's environment for a little while;
// reading it runs systemctl, and auto context asks on every chat.
type envCache struct {
	mu   sync.Mutex
	env  ollama.ServerEnv
	at   time.Time
	read func(context.Context) ollama.ServerEnv
}

const envCacheTTL = 30 * time.Second

// ollamaEnv returns the Ollama server's environment. It's only meaningful
// when Ollama runs on this machine; for a remote server Vars is empty.
func (s *Server) ollamaEnv(ctx context.Context) ollama.ServerEnv {
	c := &s.env
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.at.IsZero() && time.Since(c.at) < envCacheTTL {
		return c.env
	}
	if !s.ollamaIsLocal() || c.read == nil {
		c.env = ollama.ServerEnv{Vars: map[string]string{}}
	} else {
		c.env = c.read(ctx)
	}
	c.at = time.Now()
	return c.env
}

// ollamaIsLocal reports whether the Ollama URL points at this machine.
func (s *Server) ollamaIsLocal() bool {
	host := s.ollama.Host()
	if host == "localhost" {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// kvFactor scales f16 cache estimates to the server's K/V cache type, and
// to the number of parallel requests: Ollama gives each its own cache.
func (s *Server) kvFactor(ctx context.Context) float64 {
	if f, ok := s.ollama.(fitter); ok {
		return hw.KVTypeFactor(f.CacheType()) // one slot per model
	}
	return hw.KVTypeFactor(s.ollamaEnv(ctx).KVCacheType()) * float64(s.ollamaParallel(ctx))
}

// serverSetting is one row on the settings page: an Ollama environment
// variable, what it's set to, and what it means.
type serverSetting struct {
	Key     string            // environment variable
	Title   string            // what the UI calls it
	Help    string            // one-line explanation
	Value   string            // raw value, "" if unset
	Shown   string            // value as the UI shows it (default spelled out if unset)
	Choices []string          // values the editor offers; nil for free text
	Labels  map[string]string // how some choices read, e.g. "1" → "On"
	Default string            // what Ollama does when unset
}

// settingsData is what the settings page renders.
type settingsData struct {
	Version   string
	Source    string // where the environment was read from
	Local     bool
	KVType    string // effective K/V cache type
	Settings  []serverSetting
	Memory    hw.Memory
	Engine    *enginePage // nil with Ollama
	Voice     voicePage
	Knowledge knowledgePage
}

// enginePage is the Models section with llama.cpp.
type enginePage struct {
	Version, Server, Backend string
	// What a model without its own settings gets.
	CacheType, FlashAttn, KeepAlive string
	Saved                           []savedModel
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
	for name, opts := range e.SavedSettings() {
		p.Saved = append(p.Saved, savedModel{Name: name, Summary: settingsSummary(opts)})
	}
	sort.Slice(p.Saved, func(i, j int) bool { return p.Saved[i].Name < p.Saved[j].Name })
	return p
}

// settingsSummary says what a model's settings set, in a few words each.
func settingsSummary(o map[string]any) string {
	keys := make([]string, 0, len(o))
	for k := range o {
		keys = append(keys, k)
	}
	order := map[string]int{"num_ctx": 1, "num_gpu": 2, "keep_alive": 3, "cache_type": 4, "flash_attn": 5, "draft_model": 6}
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
			parts = append(parts, v+" cache")
		case "flash_attn":
			parts = append(parts, "flash attention "+v)
		case "draft_model":
			parts = append(parts, "drafts with "+v)
		case "num_thread":
			parts = append(parts, v+" CPU threads")
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

// knownSettings are the server variables worth showing, in display order.
var knownSettings = []serverSetting{
	{Key: "OLLAMA_KV_CACHE_TYPE", Title: "K/V cache type", Help: "q8_0 halves the context cache with almost no quality loss, so twice the context fits. q4_0 quarters it but costs some quality. Needs flash attention.", Choices: []string{"f16", "q8_0", "q4_0"}, Default: "f16"},
	{Key: "OLLAMA_FLASH_ATTENTION", Title: "Flash attention", Help: "Faster and smaller attention. Required for a quantized K/V cache.", Choices: []string{"1", "0"}, Default: "chosen per model"},
	{Key: "OLLAMA_CONTEXT_LENGTH", Title: "Default context", Help: "Used when neither the request nor the model sets num_ctx. ollamui picks its own for chats.", Choices: []string{"4096", "8192", "16384", "32768", "65536", "131072"}, Default: "4k–256k by VRAM"},
	{Key: "OLLAMA_KEEP_ALIVE", Title: "Keep alive", Help: "How long an idle model stays in memory.", Choices: []string{"5m", "30m", "1h", "-1"}, Default: "5m"},
	{Key: "OLLAMA_MAX_LOADED_MODELS", Title: "Max loaded models", Help: "How many models may sit in memory at once, per GPU.", Choices: []string{"1", "2", "3"}, Default: "3 per GPU"},
	{Key: "OLLAMA_NUM_PARALLEL", Title: "Parallel requests", Help: "Requests a model serves at once. Each one needs its own context cache.", Choices: []string{"1", "2", "4"}, Default: "1"},
	{Key: "OLLAMA_GPU_OVERHEAD", Title: "Reserved VRAM", Help: "Bytes per GPU Ollama leaves free for other apps.", Default: "none"},
	{Key: "OLLAMA_HOST", Title: "Listen address", Help: "0.0.0.0 makes Ollama reachable from other machines; 127.0.0.1 keeps it local.", Default: "127.0.0.1:11434"},
	{Key: "OLLAMA_ORIGINS", Title: "Allowed origins", Help: "Web pages allowed to call Ollama from a browser.", Default: "localhost only"},
	{Key: "OLLAMA_MODELS", Title: "Models directory", Help: "Where downloaded models live.", Default: "the service's home"},
}

var choiceLabels = map[string]map[string]string{
	"OLLAMA_FLASH_ATTENTION": {"1": "On", "0": "Off", "true": "On", "false": "Off"},
	"OLLAMA_KEEP_ALIVE":      {"-1": "Forever"},
}

func (s *Server) handleSettingsPage(w http.ResponseWriter, r *http.Request) {
	p := s.basePage(r.Context(), "settings", "Settings")
	ctx, cancel := context.WithTimeout(r.Context(), 3*time.Second)
	defer cancel()
	s.env.mu.Lock()
	s.env.at = time.Time{} // always show fresh values here
	s.env.mu.Unlock()
	env := s.ollamaEnv(ctx)
	d := settingsData{Source: env.Source, Local: s.ollamaIsLocal(), KVType: env.KVCacheType(), Memory: s.memory}
	d.Version, _ = s.ollama.Version(ctx)
	for _, st := range knownSettings {
		st.Value = env.Vars[st.Key]
		st.Labels = choiceLabels[st.Key]
		st.Shown = st.Value
		if l, ok := choiceLabels[st.Key][st.Value]; ok {
			st.Shown = l
		}
		if st.Value == "" {
			st.Shown = "Default (" + st.Default + ")"
		}
		d.Settings = append(d.Settings, st)
	}
	if e, ok := s.ollama.(engineInfo); ok {
		d.Engine = s.enginePage(ctx, e)
	}
	d.Voice = s.voicePage(ctx)
	d.Knowledge = s.knowledgePage(ctx, r.URL.Query().Get("pulled"))
	p.Data = d
	s.render(w, "settings", p)
}

// handleSettingsJSON returns the server environment for scripts in the page.
func (s *Server) handleOllamaEnv(w http.ResponseWriter, r *http.Request) {
	env := s.ollamaEnv(r.Context())
	writeJSON(w, http.StatusOK, map[string]any{"vars": env.Vars, "source": env.Source, "kvType": env.KVCacheType()})
}

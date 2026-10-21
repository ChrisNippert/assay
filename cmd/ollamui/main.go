// Command ollamui serves a web UI for Ollama.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"ollamui/internal/bench"
	"ollamui/internal/chats"
	"ollamui/internal/engine"
	"ollamui/internal/hf"
	"ollamui/internal/hw"
	"ollamui/internal/ollama"
	"ollamui/internal/presets"
	"ollamui/internal/projects"
	"ollamui/internal/pulls"
	"ollamui/internal/scores"
	"ollamui/internal/server"
	"ollamui/internal/voice"
	voiceserver "ollamui/tools/voice"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "ollamui:", err)
		os.Exit(1)
	}
}

func run() error {
	listen := flag.String("listen", "127.0.0.1:8484", "address to serve the UI on")
	ollamaURL := flag.String("ollama", ollama.ResolveHost(os.Getenv("OLLAMA_HOST")), "Ollama server URL (with -engine ollama, and for importing and pulling models meanwhile)")
	engineName := flag.String("engine", "llama.cpp", `what runs the models: "llama.cpp" or "ollama"`)
	llamaServer := flag.String("llama-server", os.Getenv("OLLAMUI_LLAMA_SERVER"), "llama.cpp's llama-server binary (default: ~/.local/share/ollamui/llama.cpp/llama-server, then PATH)")
	cacheType := flag.String("cache-type", "q8_0", "K/V cache type for llama.cpp: f16, q8_0 or q4_0")
	ollamaModels := flag.String("ollama-models", filepath.Join(os.Getenv("HOME"), ".ollama", "models"), "Ollama's models folder, imported (by hard link) into ollamui's own")
	dataDir := flag.String("data", defaultDir("XDG_DATA_HOME", ".local/share"), "where chats are stored")
	configDir := flag.String("config", defaultDir("XDG_CONFIG_HOME", ".config"), "where presets and settings are stored")
	voiceURL := flag.String("voice", envOr("OLLAMUI_VOICE", "http://127.0.0.1:8485"), "speech service for voice mode (OpenAI audio API); empty turns voice off")
	sttModel := flag.String("stt-model", "", "speech-to-text model name to ask the speech service for")
	voiceLaunch := flag.Bool("voice-launch", true, "start the bundled speech service (with uv) when voice mode is used, and stop it when idle")
	voiceIdle := flag.Duration("voice-idle", 10*time.Minute, "stop the speech service ollamui started after this long without voice use (0 never)")
	ttsVoice := flag.String("tts-voice", "", "voice to ask the speech service for")
	debug := flag.Bool("debug", false, "log extra detail, e.g. how context lengths are picked")
	var allowHosts []string
	flag.Func("allow-host", "extra hostname the UI may be reached by, e.g. a Tailscale name (repeatable)", func(v string) error {
		allowHosts = append(allowHosts, v)
		return nil
	})
	flag.Parse()

	level := slog.LevelInfo
	if *debug {
		level = slog.LevelDebug
	}
	log := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: level}))

	client, err := ollama.New(*ollamaURL)
	if err != nil {
		return err
	}
	store, err := chats.Open(filepath.Join(*dataDir, "chats"))
	if err != nil {
		return fmt.Errorf("open chat store: %w", err)
	}
	presetStore, err := presets.Open(filepath.Join(*configDir, "presets"))
	if err != nil {
		return fmt.Errorf("open preset store: %w", err)
	}
	mem := hw.Detect()
	log.Info("memory", "gpu", mem.GPU, "vram_gb", float64(mem.VRAM)/1e9, "ram_gb", float64(mem.RAM)/1e9)
	scoreStore := scores.Open(scores.DefaultURL, filepath.Join(*dataDir, "scores.json"), log)
	var speech *voice.Client
	if *voiceURL != "" {
		if speech, err = voice.New(*voiceURL); err != nil {
			return err
		}
		speech.STTModel, speech.TTSVoice = *sttModel, *ttsVoice
	}
	var launcher *voice.Launcher
	if speech != nil && *voiceLaunch {
		launcher = voice.NewLauncher(speech, voiceserver.Script, log)
		launcher.Idle = *voiceIdle
	}
	voiceSettings, err := voice.OpenSettings(filepath.Join(*configDir, "voice.json"))
	if err != nil {
		return fmt.Errorf("voice settings: %w", err)
	}
	projectStore, err := projects.Open(filepath.Join(*dataDir, "projects"))
	if err != nil {
		return fmt.Errorf("open project store: %w", err)
	}
	ragSettings, err := projects.OpenSettings(filepath.Join(*configDir, "knowledge.json"))
	if err != nil {
		return fmt.Errorf("knowledge settings: %w", err)
	}
	// The engine: llama.cpp runs the models ollamui keeps; Ollama's models
	// are imported by hard link, so nothing is downloaded twice.
	hfClient := hf.New(hf.DefaultBaseURL, os.Getenv("HF_TOKEN"))
	var models server.Backend = client
	var puller pulls.Puller = client
	var eng *engine.Engine
	if *engineName == "llama.cpp" {
		store, err := engine.OpenStore(filepath.Join(*dataDir, "models"))
		if err != nil {
			return fmt.Errorf("model store: %w", err)
		}
		if n, err := store.ImportOllama(*ollamaModels); err != nil {
			log.Warn("import Ollama models", "err", err)
		} else if n > 0 {
			log.Info("imported Ollama models", "count", n)
		}
		bin := findLlamaServer(*llamaServer, *dataDir)
		if bin == "" {
			log.Warn("llama-server not found: models can't run until llama.cpp is installed (see Settings)")
		}
		eng = engine.New(store, engine.Options{Server: bin, CacheType: *cacheType, FlashAttn: "on", Log: log})
		eng.OllamaDir = *ollamaModels
		eng.HF = hfClient
		models = eng
		puller = enginePuller{eng, client}
	} else if *engineName != "ollama" {
		return fmt.Errorf("unknown engine %q", *engineName)
	}
	srv, err := server.New(server.Options{
		Ollama:        models,
		Chats:         store,
		Presets:       presetStore,
		Projects:      projectStore,
		RAGSettings:   ragSettings,
		HF:            hfClient,
		Pulls:         pulls.New(puller, log),
		Scores:        scoreStore,
		Bench:         bench.Open(filepath.Join(*dataDir, "bench")),
		Voice:         speech,
		VoiceLauncher: launcher,
		VoiceSettings: voiceSettings,
		Memory:        mem,
		AllowHosts:    allowHosts,
		Logger:        log,
	})
	if err != nil {
		return err
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	go scoreStore.Run(ctx)
	go srv.Run(ctx) // background work: indexing project files
	engineDone := make(chan struct{})
	go func() {
		defer close(engineDone)
		if eng != nil {
			eng.Run(ctx) // unloads idle models, and all of them on exit
		}
	}()
	launcherDone := make(chan struct{})
	go func() {
		defer close(launcherDone)
		if launcher != nil {
			launcher.Run(ctx) // stops the speech service when ollamui exits
		}
	}()

	checkCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
	if v, err := client.Version(checkCtx); err != nil {
		log.Warn("can't reach Ollama yet; the UI will show an error until it's up", "url", client.URL(), "err", err)
	} else {
		log.Info("connected to Ollama", "url", client.URL(), "version", v)
	}
	cancel()

	if !isLoopback(*listen) {
		log.Warn("listening beyond this machine: anyone who can reach this address can chat, load and delete models", "listen", *listen)
	}

	hs := &http.Server{
		Addr:              *listen,
		Handler:           srv,
		ReadHeaderTimeout: 10 * time.Second,
		// No WriteTimeout: chat replies stream for as long as the model talks.
	}
	errc := make(chan error, 1)
	go func() { errc <- hs.ListenAndServe() }()
	log.Info("ollamui ready", "url", "http://"+displayAddr(*listen))

	select {
	case err := <-errc:
		return err
	case <-ctx.Done():
	}
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	<-launcherDone
	<-engineDone
	if err := hs.Shutdown(shutdownCtx); err != nil && !errors.Is(err, context.DeadlineExceeded) {
		return err
	}
	return nil
}

func envOr(key, fallback string) string {
	if v, ok := os.LookupEnv(key); ok {
		return v
	}
	return fallback
}

// defaultDir follows the XDG convention: $env/ollamui, else ~/fallback/ollamui.
func defaultDir(env, fallback string) string {
	if d := os.Getenv(env); d != "" {
		return filepath.Join(d, "ollamui")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "ollamui-" + filepath.Base(fallback)
	}
	return filepath.Join(home, fallback, "ollamui")
}

func isLoopback(addr string) bool {
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		return false
	}
	if strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// displayAddr makes a listen address clickable ("127.0.0.1:8484" or ":8484"
// becomes "localhost:8484").
func displayAddr(addr string) string {
	host, port, err := net.SplitHostPort(addr)
	if err != nil {
		return addr
	}
	if host == "" || host == "0.0.0.0" || host == "::" || host == "127.0.0.1" {
		host = "localhost"
	}
	return net.JoinHostPort(host, port)
}

// findLlamaServer looks for llama-server: the flag, ollamui's own copy,
// then PATH.
func findLlamaServer(flagPath, dataDir string) string {
	for _, p := range []string{flagPath, filepath.Join(dataDir, "llama.cpp", "llama-server")} {
		if p == "" {
			continue
		}
		if info, err := os.Stat(p); err == nil && !info.IsDir() {
			return p
		}
	}
	if p, err := exec.LookPath("llama-server"); err == nil {
		return p
	}
	return ""
}

// enginePuller downloads Hugging Face models with the engine itself and
// anything else (Ollama library names, e.g. the embedding models) through
// Ollama, then imports it.
type enginePuller struct {
	eng    *engine.Engine
	ollama *ollama.Client
}

func (p enginePuller) Pull(ctx context.Context, model string, fn func(ollama.PullProgress) error) error {
	if _, _, ok := engine.ParseHFName(model); ok {
		return p.eng.Pull(ctx, model, fn)
	}
	if err := p.ollama.Pull(ctx, model, fn); err != nil {
		return err
	}
	p.eng.Import()
	return nil
}

func (p enginePuller) Copy(ctx context.Context, src, dst string) error {
	if p.eng.Has(src) {
		return p.eng.Copy(ctx, src, dst)
	}
	if err := p.ollama.Copy(ctx, src, dst); err != nil {
		return err
	}
	p.eng.Import()
	return nil
}

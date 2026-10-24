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
	"runtime"
	"strings"
	"syscall"
	"time"

	"ollamui/internal/bench"
	"ollamui/internal/chats"
	"ollamui/internal/engine"
	"ollamui/internal/hf"
	"ollamui/internal/hw"
	"ollamui/internal/llamacpp"
	"ollamui/internal/ollama"
	"ollamui/internal/presets"
	"ollamui/internal/projects"
	"ollamui/internal/pulls"
	"ollamui/internal/scores"
	"ollamui/internal/server"
	"ollamui/internal/voice"
	voiceserver "ollamui/tools/voice"
)

// version is set at build time (tools/release/build.sh).
var version = "dev"

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "ollamui:", err)
		// Started by double-clicking, the console window would close
		// before anyone could read why.
		if runtime.GOOS == "windows" {
			fmt.Fprintln(os.Stderr, "Press Enter to close.")
			fmt.Scanln()
		}
		os.Exit(1)
	}
}

func run() error {
	listen := flag.String("listen", "127.0.0.1:8484", "address to serve the UI on")
	llamaServer := flag.String("llama-server", os.Getenv("OLLAMUI_LLAMA_SERVER"), "llama.cpp's llama-server binary (default: ~/.local/share/ollamui/llama.cpp/llama-server, then the release installed from Settings, then PATH)")
	cacheType := flag.String("cache-type", "q8_0", "K/V cache type for llama.cpp: f16, q8_0 or q4_0")
	ollamaModels := flag.String("ollama-models", defaultOllamaModels(), "Ollama's models folder, imported (by hard link) into ollamui's own")
	dataDir := flag.String("data", defaultDir("XDG_DATA_HOME", ".local/share", "LOCALAPPDATA"), "where chats are stored")
	modelsFlag := flag.String("models", "", "where models are stored (default: the folder chosen in Settings, else <data>/models); overrides the saved choice")
	configDir := flag.String("config", defaultDir("XDG_CONFIG_HOME", ".config", "APPDATA"), "where presets and settings are stored")
	showVersion := flag.Bool("version", false, "print the version and exit")
	openUI := flag.Bool("open", runtime.GOOS == "windows", "open the UI in the browser once it's ready")
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
	if *showVersion {
		fmt.Println("ollamui", version)
		return nil
	}

	level := slog.LevelInfo
	if *debug {
		level = slog.LevelDebug
	}
	log := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: level}))

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
	folder := engine.FolderSetting{File: filepath.Join(*configDir, "models.json"), Pinned: *modelsFlag != ""}
	modelsDir := *modelsFlag
	if modelsDir == "" {
		modelsDir = engine.LoadFolder(folder.File)
		// A disk that isn't mounted must not get a fresh empty store.
		if _, err := os.Stat(modelsDir); modelsDir != "" && err != nil {
			log.Warn("the saved models folder isn't there (is its disk mounted?); using the default", "dir", modelsDir, "err", err)
			modelsDir = ""
		}
	}
	if modelsDir == "" {
		modelsDir = filepath.Join(*dataDir, "models")
	}
	modelStore, err := engine.OpenStore(modelsDir)
	if err != nil {
		return fmt.Errorf("model store: %w", err)
	}
	if n, err := modelStore.ImportOllama(*ollamaModels); err != nil {
		log.Warn("import Ollama models", "err", err)
	} else if n > 0 {
		log.Info("imported Ollama models", "count", n)
	}
	llama := llamacpp.NewManager(*dataDir, *llamaServer, nil, log)
	bin, _ := llama.Resolve()
	if bin == "" {
		log.Warn("llama-server not found: models can't run until llama.cpp is installed (see Settings)")
	}
	eng := engine.New(modelStore, engine.Options{Server: bin, CacheType: *cacheType, FlashAttn: "on", KVDir: filepath.Join(*dataDir, "kvcache"), Log: log})
	llama.Eng = eng
	eng.OllamaDir = *ollamaModels
	eng.Folder = folder
	eng.Finder = engine.NewFinder(filepath.Join(*configDir, "sources.json"))
	eng.HF = hfClient
	srv, err := server.New(server.Options{
		Ollama:        eng,
		Chats:         store,
		Presets:       presetStore,
		Projects:      projectStore,
		RAGSettings:   ragSettings,
		HF:            hfClient,
		Pulls:         pulls.New(enginePuller{eng}, log),
		LlamaCpp:      llama,
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
		eng.Run(ctx) // unloads idle models, and all of them on exit
	}()
	launcherDone := make(chan struct{})
	go func() {
		defer close(launcherDone)
		if launcher != nil {
			launcher.Run(ctx) // stops the speech service when ollamui exits
		}
	}()

	if !isLoopback(*listen) {
		log.Warn("listening beyond this machine: anyone who can reach this address can chat, load and delete models", "listen", *listen)
	}

	hs := &http.Server{
		Addr:              *listen,
		Handler:           srv,
		ReadHeaderTimeout: 10 * time.Second,
		// No WriteTimeout: chat replies stream for as long as the model talks.
	}
	ln, err := net.Listen("tcp", *listen)
	if err != nil {
		// Most likely ollamui is already running (a second double-click):
		// show that one instead.
		if *openUI && addrInUse(err) {
			log.Info("ollamui is already running; opening it", "listen", *listen)
			return openBrowser("http://" + displayAddr(*listen))
		}
		return err
	}
	errc := make(chan error, 1)
	go func() { errc <- hs.Serve(ln) }()
	url := "http://" + displayAddr(*listen)
	log.Info("ollamui ready", "url", url)
	if *openUI {
		if err := openBrowser(url); err != nil {
			log.Warn("couldn't open the browser", "url", url, "err", err)
		}
	}

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

// defaultDir follows the XDG convention: $env/ollamui, else
// ~/fallback/ollamui. On Windows it's %winEnv%\ollamui (LOCALAPPDATA for
// data, APPDATA for settings).
func defaultDir(env, fallback, winEnv string) string {
	if runtime.GOOS == "windows" {
		if d := os.Getenv(winEnv); d != "" {
			return filepath.Join(d, "ollamui")
		}
	}
	if d := os.Getenv(env); d != "" {
		return filepath.Join(d, "ollamui")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "ollamui-" + filepath.Base(fallback)
	}
	return filepath.Join(home, fallback, "ollamui")
}

// defaultOllamaModels is where Ollama keeps its models: $OLLAMA_MODELS, else
// ~/.ollama/models (on Windows too, under the user's profile).
func defaultOllamaModels() string {
	if d := os.Getenv("OLLAMA_MODELS"); d != "" {
		return d
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".ollama", "models")
}

// addrInUse reports whether listening failed because the address is taken
// (WSAEADDRINUSE, 10048, on Windows).
func addrInUse(err error) bool {
	var errno syscall.Errno
	return errors.As(err, &errno) && (errno == syscall.EADDRINUSE || errno == 10048)
}

// openBrowser shows url in the default browser.
func openBrowser(url string) error {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "windows":
		cmd = exec.Command("rundll32", "url.dll,FileProtocolHandler", url)
	case "darwin":
		cmd = exec.Command("open", url)
	default:
		cmd = exec.Command("xdg-open", url)
	}
	return cmd.Start()
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

// enginePuller downloads models from Hugging Face. Ollama library names
// (no hf.co/ prefix) can't be fetched without Ollama.
type enginePuller struct {
	eng *engine.Engine
}

func (p enginePuller) Pull(ctx context.Context, model string, fn func(ollama.PullProgress) error) error {
	if _, _, ok := engine.ParseHFName(model); !ok {
		return fmt.Errorf("%s isn't a Hugging Face model: use a name like hf.co/<owner>/<repo>:<quant>", model)
	}
	return p.eng.Pull(ctx, model, fn)
}

func (p enginePuller) Copy(ctx context.Context, src, dst string) error {
	return p.eng.Copy(ctx, src, dst)
}

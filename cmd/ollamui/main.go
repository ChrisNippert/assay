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
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"ollamui/internal/chats"
	"ollamui/internal/hf"
	"ollamui/internal/hw"
	"ollamui/internal/ollama"
	"ollamui/internal/presets"
	"ollamui/internal/pulls"
	"ollamui/internal/scores"
	"ollamui/internal/server"
	"ollamui/internal/voice"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "ollamui:", err)
		os.Exit(1)
	}
}

func run() error {
	listen := flag.String("listen", "127.0.0.1:8484", "address to serve the UI on")
	ollamaURL := flag.String("ollama", ollama.ResolveHost(os.Getenv("OLLAMA_HOST")), "Ollama server URL")
	dataDir := flag.String("data", defaultDir("XDG_DATA_HOME", ".local/share"), "where chats are stored")
	configDir := flag.String("config", defaultDir("XDG_CONFIG_HOME", ".config"), "where presets are stored")
	voiceURL := flag.String("voice", envOr("OLLAMUI_VOICE", "http://127.0.0.1:8485"), "speech service for voice mode (OpenAI audio API); empty turns voice off")
	sttModel := flag.String("stt-model", "", "speech-to-text model name to ask the speech service for")
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
	srv, err := server.New(server.Options{
		Ollama:     client,
		Chats:      store,
		Presets:    presetStore,
		HF:         hf.New(hf.DefaultBaseURL, os.Getenv("HF_TOKEN")),
		Pulls:      pulls.New(client, log),
		Scores:     scoreStore,
		Voice:      speech,
		Memory:     mem,
		AllowHosts: allowHosts,
		Logger:     log,
	})
	if err != nil {
		return err
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	go scoreStore.Run(ctx)

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

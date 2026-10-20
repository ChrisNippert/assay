package ollama

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"unicode"
)

// ServerEnv is the environment the Ollama server runs with, as far as we can
// tell from the outside. Most of Ollama's tuning (KV cache type, flash
// attention, default context) is only settable this way.
type ServerEnv struct {
	Vars   map[string]string
	Source string // where Vars came from, e.g. "systemd (ollama.service)"; "" if unknown
}

// ReadServerEnv reads the OLLAMA_* variables of the Ollama server on port.
// It prefers the running process (/proc/<pid>/environ, readable when it runs
// as the same user), which also covers a hand-started "ollama serve", and
// falls back to the systemd service's configuration, which `systemctl show`
// gives anyone. Without either, Vars is empty.
func ReadServerEnv(ctx context.Context, port string) ServerEnv {
	if env, ok := processEnv(port); ok {
		return env
	}
	return systemdEnv(ctx)
}

// processEnv finds an "ollama serve" process listening on port and reads
// its environment.
func processEnv(port string) (ServerEnv, bool) {
	dirs, _ := filepath.Glob("/proc/[0-9]*")
	for _, dir := range dirs {
		cmd, err := os.ReadFile(dir + "/cmdline")
		if err != nil {
			continue
		}
		args := strings.Split(strings.TrimRight(string(cmd), "\x00"), "\x00")
		if len(args) < 2 || filepath.Base(args[0]) != "ollama" || args[1] != "serve" {
			continue
		}
		raw, err := os.ReadFile(dir + "/environ")
		if err != nil {
			continue // another user's process
		}
		vars := map[string]string{}
		for _, kv := range strings.Split(string(raw), "\x00") {
			k, v, ok := strings.Cut(kv, "=")
			if ok && (strings.HasPrefix(k, "OLLAMA_") || strings.HasPrefix(k, "LLAMA_")) {
				vars[k] = v
			}
		}
		if hostPort(vars["OLLAMA_HOST"]) != port {
			continue
		}
		return ServerEnv{Vars: vars, Source: "the running Ollama (pid " + filepath.Base(dir) + ")"}, true
	}
	return ServerEnv{}, false
}

// hostPort is the port an OLLAMA_HOST value listens on.
func hostPort(host string) string {
	if host == "" {
		return "11434"
	}
	host = strings.TrimPrefix(strings.TrimPrefix(host, "http://"), "https://")
	if i := strings.LastIndex(host, ":"); i >= 0 && !strings.HasSuffix(host, "]") {
		return host[i+1:]
	}
	return "11434"
}

func systemdEnv(ctx context.Context) ServerEnv {
	for _, scope := range [][]string{nil, {"--user"}} {
		args := append(append([]string{}, scope...), "show", "ollama", "-p", "Environment,LoadState", "--value")
		out, err := exec.CommandContext(ctx, "systemctl", args...).Output()
		if err != nil {
			continue
		}
		// --value prints one property per line, in systemd's own order.
		envLine, loaded := "", false
		for _, line := range strings.Split(string(out), "\n") {
			if strings.TrimSpace(line) == "loaded" {
				loaded = true
			} else if strings.Contains(line, "=") {
				envLine = line
			}
		}
		if !loaded {
			continue
		}
		src := "systemd (ollama.service)"
		if scope != nil {
			src = "systemd user service (ollama.service)"
		}
		return ServerEnv{Vars: ParseEnvironment(envLine), Source: src}
	}
	return ServerEnv{Vars: map[string]string{}}
}

// ParseEnvironment parses systemd's Environment= value: space-separated
// KEY=value words, any of which may be double-quoted as a whole
// ("KEY=a b"). Only OLLAMA_* and LLAMA_* variables are kept.
func ParseEnvironment(s string) map[string]string {
	out := map[string]string{}
	var word strings.Builder
	inQuote := false
	flush := func() {
		k, v, ok := strings.Cut(word.String(), "=")
		word.Reset()
		if ok && (strings.HasPrefix(k, "OLLAMA_") || strings.HasPrefix(k, "LLAMA_")) {
			out[k] = v
		}
	}
	for _, r := range s {
		switch {
		case r == '"':
			inQuote = !inQuote
		case unicode.IsSpace(r) && !inQuote:
			flush()
		default:
			word.WriteRune(r)
		}
	}
	flush()
	return out
}

// KVCacheType is the K/V cache type the server will use. A quantized cache
// needs flash attention, so it falls back to f16 when that's turned off.
func (e ServerEnv) KVCacheType() string {
	t := strings.ToLower(strings.TrimSpace(e.Vars["OLLAMA_KV_CACHE_TYPE"]))
	if t == "" || !e.FlashAttention() {
		return "f16"
	}
	return t
}

// FlashAttention reports whether flash attention is on. Unset means Ollama
// decides per model; recent versions turn it on wherever it's supported.
func (e ServerEnv) FlashAttention() bool {
	switch strings.ToLower(strings.TrimSpace(e.Vars["OLLAMA_FLASH_ATTENTION"])) {
	case "0", "false", "no", "off":
		return false
	}
	return true
}

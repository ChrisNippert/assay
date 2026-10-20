package ollama

import (
	"context"
	"os/exec"
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

// ReadServerEnv reads the OLLAMA_* variables of the systemd ollama service.
// It needs no root: `systemctl show` is readable by anyone. On systems
// without systemd, or where Ollama isn't a service, Vars is empty.
func ReadServerEnv(ctx context.Context) ServerEnv {
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

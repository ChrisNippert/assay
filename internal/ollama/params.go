package ollama

import (
	"strconv"
	"strings"
)

// ParseParameters turns the "parameters" text from /api/show, e.g.
//
//	temperature                    0.6
//	stop                           "<|im_end|>"
//
// into a map. Numbers become float64; repeated keys (stop) become []string.
func ParseParameters(text string) map[string]any {
	out := map[string]any{}
	for _, line := range strings.Split(text, "\n") {
		key, val, ok := strings.Cut(strings.TrimSpace(line), " ")
		if !ok {
			continue
		}
		val = strings.TrimSpace(val)
		if s, err := strconv.Unquote(val); err == nil {
			val = s
		} else if n, err := strconv.ParseFloat(val, 64); err == nil && key != "stop" {
			out[key] = n
			continue
		}
		if key == "stop" {
			list, _ := out[key].([]string)
			out[key] = append(list, val)
			continue
		}
		out[key] = val
	}
	return out
}

// ContextLength returns the trained context length from /api/show's
// model_info (the "<arch>.context_length" key), or 0 if absent.
func (s *ShowResponse) ContextLength() int {
	for k, v := range s.ModelInfo {
		if strings.HasSuffix(k, ".context_length") {
			if n, ok := v.(float64); ok {
				return int(n)
			}
		}
	}
	return 0
}

// HasCapability reports whether /api/show lists a capability such as
// "thinking", "vision" or "tools".
func (s *ShowResponse) HasCapability(c string) bool {
	for _, x := range s.Capabilities {
		if x == c {
			return true
		}
	}
	return false
}

package server

import (
	"encoding/json"
	"strconv"
	"strings"

	"assay/internal/ollama"
)

// Gemma 4 writes tool calls as <|tool_call>call:name{key:<|"|>text<|"|>,n:3}<tool_call|>.
// llama-server usually turns that into a real tool call, but sometimes
// (seen with multi-line values) leaves it in the reply text, where the user
// sees the raw markup and the call never runs. textToolCalls reads such
// calls out of the text and returns the text without them.
const (
	tcOpen  = "<|tool_call>"
	tcClose = "<tool_call|>"
	tcQuote = `<|"|>`
)

func textToolCalls(text string) (calls []ollama.ToolCall, rest string) {
	var keep strings.Builder
	for {
		i := strings.Index(text, tcOpen)
		if i < 0 {
			keep.WriteString(text)
			break
		}
		j := strings.Index(text[i:], tcClose)
		if j < 0 {
			keep.WriteString(text)
			break
		}
		call, ok := parseTextCall(text[i+len(tcOpen) : i+j])
		if !ok {
			keep.WriteString(text[:i+j+len(tcClose)])
		} else {
			keep.WriteString(text[:i])
			calls = append(calls, call)
		}
		text = text[i+j+len(tcClose):]
	}
	return calls, strings.TrimSpace(keep.String())
}

// parseTextCall reads `call:name{key:value,...}`.
func parseTextCall(s string) (ollama.ToolCall, bool) {
	var tc ollama.ToolCall
	s = strings.TrimSpace(s)
	s, ok := strings.CutPrefix(s, "call:")
	if !ok {
		return tc, false
	}
	b := strings.IndexByte(s, '{')
	if b <= 0 || !strings.HasSuffix(s, "}") {
		return tc, false
	}
	name, body := strings.TrimSpace(s[:b]), s[b+1:len(s)-1]
	args := map[string]any{}
	for {
		body = strings.TrimLeft(body, " \n\t,")
		if body == "" {
			break
		}
		c := strings.IndexByte(body, ':')
		if c <= 0 {
			return tc, false
		}
		key := strings.Trim(strings.TrimSpace(body[:c]), `"`)
		body = body[c+1:]
		if v, ok := strings.CutPrefix(body, tcQuote); ok {
			e := strings.Index(v, tcQuote)
			if e < 0 {
				return tc, false
			}
			args[key], body = v[:e], v[e+len(tcQuote):]
			continue
		}
		e := strings.IndexByte(body, ',')
		if e < 0 {
			e = len(body)
		}
		raw := strings.TrimSpace(body[:e])
		body = body[e:]
		switch {
		case raw == "true" || raw == "false":
			args[key] = raw == "true"
		case raw == "null":
			args[key] = nil
		default:
			if f, err := strconv.ParseFloat(raw, 64); err == nil {
				args[key] = f
			} else {
				args[key] = strings.Trim(raw, `"`)
			}
		}
	}
	tc.Function.Name = name
	tc.Function.Arguments, _ = json.Marshal(args)
	return tc, name != ""
}

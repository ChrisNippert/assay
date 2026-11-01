package hf

import (
	"context"
	"errors"
	"fmt"
	"io"
	"maps"
	"net/http"
	"regexp"
	"strconv"
	"strings"
)

// Card fetches a repo's model card (README.md); "" if it has none.
func (c *Client) Card(ctx context.Context, repo string) (string, error) {
	if !ValidRepo(repo) {
		return "", ErrNotFound
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.base+"/"+repo+"/resolve/main/README.md", nil)
	if err != nil {
		return "", err
	}
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return "", ctx.Err()
		}
		return "", errors.New("couldn't reach Hugging Face. Check your connection and try again")
	}
	defer resp.Body.Close()
	switch {
	case resp.StatusCode == http.StatusNotFound:
		return "", nil
	case resp.StatusCode >= 300:
		return "", fmt.Errorf("hugging face: %s", resp.Status)
	}
	b, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	return string(b), err
}

// CardProfile is one set of recommended sampling settings found in a model
// card, with the card's words for when to use it ("Thinking mode for
// general tasks").
type CardProfile struct {
	Label   string         `json:"label"`
	Options map[string]any `json:"options"`
}

// SamplingKeys are the options a recommendation may set.
var SamplingKeys = []string{"temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty", "repeat_penalty"}

var (
	// A setting's name, then =, :, a table bar or just a space, then a number.
	cardPair = regexp.MustCompile(`(?i)\b(temperature|temp|top[_ -]?p|top[_ -]?k|min[_ -]?p|presence[_ -]?penalty|frequency[_ -]?penalty|repetition[_ -]?penalty|repeat[_ -]?penalty)\s*(?:=|:|\|)?\s*(-?[0-9]*\.?[0-9]+)\b`)
	cardJunk = regexp.MustCompile("[#>*`_|]+")
	// cardTidy drops markdown that sits between a name and its value:
	// escapes (temperature \= 1.0, Top\_P), code ticks and bold.
	cardTidy = strings.NewReplacer(`\=`, "=", `\_`, "_", `\*`, "", "`", "", "**", "")
)

// cardKey turns a card's spelling into an option name and checks the value
// is sane for it.
func cardKey(k string, v float64) (string, any, bool) {
	k = strings.ToLower(strings.NewReplacer(" ", "_", "-", "_").Replace(k))
	switch k {
	case "temperature", "temp":
		return "temperature", v, v >= 0 && v <= 2
	case "top_p", "topp":
		return "top_p", v, v > 0 && v <= 1
	case "top_k", "topk":
		return "top_k", int(v), v >= 0 && v <= 1000 && v == float64(int(v))
	case "min_p", "minp":
		return "min_p", v, v >= 0 && v < 1
	case "presence_penalty":
		return "presence_penalty", v, v >= -2 && v <= 2
	case "frequency_penalty":
		return "frequency_penalty", v, v >= -2 && v <= 2
	case "repetition_penalty", "repeat_penalty":
		return "repeat_penalty", v, v >= 0.5 && v <= 2
	}
	return "", nil, false
}

// cardLabel tidies a line of prose into a short label.
func cardLabel(s string) string {
	s = cardJunk.ReplaceAllString(s, " ")
	s = strings.Join(strings.Fields(s), " ")
	s = strings.TrimLeft(s, "-•0123456789. ")
	s = strings.TrimRight(s, ":,;.- ")
	if strings.Contains(s, "http") || len(s) > 90 {
		return ""
	}
	return s
}

// CardSettings finds recommended sampling settings in a model card. A line
// with two or more settings ("temperature=0.6, top_p=0.95, top_k=20") is
// one profile, labelled by the text before them; consecutive lines with
// one setting each (a bulleted list) make one profile, labelled by the line
// above. Code blocks are skipped (they repeat the prose as API calls).
// Duplicates are dropped; at most eight profiles.
func CardSettings(card string) []CardProfile {
	var out []CardProfile
	add := func(label string, opts map[string]any) {
		if len(opts) < 2 {
			return
		}
		for _, p := range out {
			if maps.Equal(p.Options, opts) {
				return
			}
		}
		if label == "" {
			label = "Recommended"
		}
		out = append(out, CardProfile{Label: label, Options: opts})
	}
	var run map[string]any
	runLabel, prev := "", ""
	flush := func() {
		if run != nil {
			add(runLabel, run)
		}
		run = nil
	}
	inCode := false
	for _, line := range strings.Split(card, "\n") {
		if t := strings.TrimSpace(line); strings.HasPrefix(t, "```") || strings.HasPrefix(t, "~~~") {
			inCode = !inCode
			flush()
			continue
		}
		if inCode {
			continue
		}
		line = cardTidy.Replace(line)
		t := strings.TrimSpace(line)
		ms := cardPair.FindAllStringSubmatchIndex(line, -1)
		opts := map[string]any{}
		for _, m := range ms {
			v, err := strconv.ParseFloat(line[m[4]:m[5]], 64)
			if err != nil {
				continue
			}
			if k, val, ok := cardKey(line[m[2]:m[3]], v); ok {
				opts[k] = val
			}
		}
		switch {
		case len(opts) >= 2:
			flush()
			label := cardLabel(line[:ms[0][0]])
			if label == "" {
				label = prev
			}
			add(label, opts)
		case len(opts) == 1:
			if run == nil {
				run, runLabel = map[string]any{}, prev
			}
			for k, v := range opts {
				if _, dup := run[k]; dup {
					flush()
					run, runLabel = map[string]any{}, prev
				}
				run[k] = v
			}
		case t != "":
			flush()
			if strings.HasPrefix(t, "|") {
				break // a table's header or rule: the label stays the text above the table
			}
			if l := cardLabel(t); l != "" {
				prev = l
			}
		}
		if len(out) >= 8 {
			break
		}
	}
	flush()
	if len(out) > 8 {
		out = out[:8]
	}
	return out
}

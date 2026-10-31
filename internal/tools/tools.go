// Package tools holds the built-in functions a chat model may call: web
// search, fetching a page, a calculator, searching saved chats and a
// project's files, and the time. Each is a JSON schema the model sees plus
// a Go function that runs it.
package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"
	"unicode/utf8"

	"assay/internal/chats"
)

// Search backends for web_search.
const (
	SearchOff        = "off"
	SearchSearXNG    = "searxng"
	SearchDuckDuckGo = "duckduckgo"
)

// Settings are the preferences shared by every chat.
type Settings struct {
	// Search is the web_search backend; "" and "off" turn the tool off.
	Search string `json:"search"`
	// SearxURL is the SearXNG instance, e.g. http://localhost:8888.
	SearxURL string `json:"searxUrl"`
	// Off lists the tools switched off. New tools start on.
	Off []string `json:"off"`
}

// Validate checks values a browser could send wrong.
func (s Settings) Validate() error {
	switch s.Search {
	case "", SearchOff, SearchDuckDuckGo:
	case SearchSearXNG:
		if _, err := searxBase(s.SearxURL); err != nil {
			return err
		}
	default:
		return fmt.Errorf("unknown search backend %q", s.Search)
	}
	for _, n := range s.Off {
		if _, ok := infoOf(n); !ok {
			return fmt.Errorf("unknown tool %q", n)
		}
	}
	return nil
}

// Info describes a tool for the Settings page.
type Info struct {
	Name  string
	Title string
	Desc  string
}

// Catalog lists every tool, in the order the Settings page shows them.
var Catalog = []Info{
	{"web_search", "Search the web", "Looks things up with the search backend chosen above"},
	{"fetch_url", "Read a web page", "Downloads a page and reads its text; never reaches local or private addresses"},
	{"calculator", "Calculator", "Exact arithmetic so the model doesn’t do sums in its head"},
	{"search_chats", "Search saved chats", "Finds things you discussed in earlier chats"},
	{"search_project", "Search project files", "Looks through the files of the chat’s project"},
	{"current_time", "Current time", "Today’s date and the time"},
}

func infoOf(name string) (Info, bool) {
	for _, i := range Catalog {
		if i.Name == name {
			return i, true
		}
	}
	return Info{}, false
}

// On reports whether a tool is switched on in the settings. web_search also
// needs a backend.
func (s Settings) On(name string) bool {
	if slices.Contains(s.Off, name) {
		return false
	}
	if name == "web_search" {
		return s.Search == SearchSearXNG && s.SearxURL != "" || s.Search == SearchDuckDuckGo
	}
	return true
}

// Result is what a tool call produced.
type Result struct {
	// Text goes back to the model.
	Text string
	// Label is the one-line summary shown in the chat.
	Label string
	// Err means the call failed; Text then says why.
	Err bool
}

// Tool is one function the model may call.
type Tool struct {
	Name        string
	Description string
	// Parameters is the JSON schema of the arguments.
	Parameters map[string]any
	Run        func(ctx context.Context, args json.RawMessage) Result
}

// Passage is one hit from a project's files.
type Passage struct {
	File    string
	Page    int
	Heading string
	Text    string
}

// ChatStore is the part of the chat store search_chats reads.
type ChatStore interface {
	List() ([]chats.Summary, error)
	Get(id string) (*chats.Chat, error)
}

// Deps is what the tools need from the app.
type Deps struct {
	Settings Settings
	Chats    ChatStore
	// ExceptChat is the chat being answered, left out of search_chats.
	ExceptChat string
	// Project searches the chat's project; nil when it has none.
	Project func(ctx context.Context, query string) ([]Passage, error)
	// Fetcher and Searcher can be replaced in tests.
	Fetcher  *Fetcher
	Searcher *Searcher
	Now      func() time.Time
}

// Build returns the tools that are switched on and usable here.
func Build(d Deps) []Tool {
	if d.Fetcher == nil {
		d.Fetcher = NewFetcher()
	}
	if d.Searcher == nil {
		d.Searcher = NewSearcher()
	}
	if d.Now == nil {
		d.Now = time.Now
	}
	var out []Tool
	for _, i := range Catalog {
		if !d.Settings.On(i.Name) {
			continue
		}
		switch i.Name {
		case "web_search":
			out = append(out, webSearchTool(d.Searcher, d.Settings))
		case "fetch_url":
			out = append(out, fetchTool(d.Fetcher))
		case "calculator":
			out = append(out, calculatorTool())
		case "search_chats":
			if d.Chats != nil {
				out = append(out, searchChatsTool(d.Chats, d.ExceptChat))
			}
		case "search_project":
			if d.Project != nil {
				out = append(out, searchProjectTool(d.Project))
			}
		case "current_time":
			out = append(out, timeTool(d.Now))
		}
	}
	return out
}

// Run calls the named tool; an unknown name or bad arguments come back as
// an error result for the model to read.
func Run(ctx context.Context, set []Tool, name string, args json.RawMessage) Result {
	for _, t := range set {
		if t.Name != name {
			continue
		}
		if len(args) == 0 {
			args = json.RawMessage("{}")
		}
		return t.Run(ctx, args)
	}
	var names []string
	for _, t := range set {
		names = append(names, t.Name)
	}
	return fail("Called "+name, "There is no tool called "+name+". Available tools: "+strings.Join(names, ", ")+".")
}

func fail(label, msg string) Result {
	return Result{Text: "Error: " + msg, Label: label + " · failed", Err: true}
}

// strArg reads a required string argument.
func strArg(args json.RawMessage, key string) (string, error) {
	var m map[string]any
	if err := json.Unmarshal(args, &m); err != nil {
		return "", errors.New("the arguments must be a JSON object like {\"" + key + "\": \"...\"}")
	}
	v, _ := m[key].(string)
	v = strings.TrimSpace(v)
	if v == "" {
		return "", errors.New("missing argument \"" + key + "\"")
	}
	return v, nil
}

func objSchema(key, desc string) map[string]any {
	return map[string]any{
		"type":       "object",
		"properties": map[string]any{key: map[string]any{"type": "string", "description": desc}},
		"required":   []string{key},
	}
}

// Definitions turns tools into the request's `tools` field.
func Definitions(set []Tool) []map[string]any {
	var out []map[string]any
	for _, t := range set {
		out = append(out, map[string]any{"type": "function", "function": map[string]any{
			"name": t.Name, "description": t.Description, "parameters": t.Parameters,
		}})
	}
	return out
}

// Clip cuts s to at most n bytes at a rune boundary and says so.
func Clip(s string, n int) string {
	if len(s) <= n {
		return s
	}
	cut := n
	for cut > 0 && !utf8.RuneStart(s[cut]) {
		cut--
	}
	return s[:cut] + fmt.Sprintf("\n\n[Cut off: showing the first %d of %d characters.]", utf8.RuneCountInString(s[:cut]), utf8.RuneCountInString(s))
}

// short shortens a label's quoted part.
func short(s string, n int) string {
	s = strings.Join(strings.Fields(s), " ")
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return strings.TrimSpace(string(r[:n])) + "…"
}

func plural(n int, one, many string) string {
	if n == 1 {
		return "1 " + one
	}
	return fmt.Sprintf("%d %s", n, many)
}

func timeTool(now func() time.Time) Tool {
	return Tool{
		Name:        "current_time",
		Description: "Get the current date and time. Use it when the answer depends on today's date or the time of day; you can't know them yourself.",
		Parameters:  map[string]any{"type": "object", "properties": map[string]any{}},
		Run: func(ctx context.Context, _ json.RawMessage) Result {
			t := now()
			name, off := t.Zone()
			text := fmt.Sprintf("%s (%s, UTC%+03d:%02d)", t.Format("Monday 2 January 2006, 15:04"), name, off/3600, abs(off%3600)/60)
			return Result{Text: text, Label: "Checked the time · " + t.Format("15:04")}
		},
	}
}

func abs(n int) int {
	if n < 0 {
		return -n
	}
	return n
}

package server

import (
	"context"
	"errors"
	"encoding/json"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"time"

	"ollamui/internal/chats"
	"ollamui/internal/ollama"
	"ollamui/internal/tools"
)

// maxToolSteps is how many times a reply may call tools before the model is
// asked to answer with what it has.
const maxToolSteps = 8

// toolsSystem is the one line that tells the model about its tools. Their
// schemas carry the detail; a long prompt makes thinking models think longer.
const toolsSystem = "You can use tools when they help. When your answer relies on a web page or search result, mention its URL."

// toolsPage is the Tools section of Settings.
type toolsPage struct {
	Settings tools.Settings
	Catalog  []toolRow
}

type toolRow struct {
	tools.Info
	On bool
}

func (s *Server) toolsPage() toolsPage {
	set := s.toolSt.Get()
	p := toolsPage{Settings: set}
	if p.Settings.Search == "" {
		p.Settings.Search = tools.SearchOff
	}
	for _, i := range tools.Catalog {
		p.Catalog = append(p.Catalog, toolRow{Info: i, On: !slices.Contains(set.Off, i.Name)})
	}
	return p
}

func (s *Server) handleToolSettings(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, s.toolSt.Get())
}

func (s *Server) handleToolSettingsSave(w http.ResponseWriter, r *http.Request) {
	var set tools.Settings
	if !readJSON(w, r, &set) {
		return
	}
	set.SearxURL = strings.TrimSpace(set.SearxURL)
	if err := s.toolSt.Set(set); err != nil {
		writeError(w, http.StatusBadRequest, capitalize(err.Error()))
		return
	}
	writeJSON(w, http.StatusOK, s.toolSt.Get())
}

// turnTools picks the tools for a reply: none unless the chat wants them
// (its own switch, then the preset's, then on) and the model can call them.
func (s *Server) turnTools(ctx context.Context, c *chats.Chat, want *bool, model string) []tools.Tool {
	if want != nil && !*want {
		return nil
	}
	cctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	if show, err := s.ollama.Show(cctx, model); err != nil || !show.HasCapability("tools") {
		return nil
	}
	d := tools.Deps{Settings: s.toolSt.Get(), Chats: s.chats, ExceptChat: c.ID, Fetcher: s.fetcher, Searcher: s.searcher}
	if c.Project != "" {
		d.Project = s.projectSearch(c.Project)
	}
	return tools.Build(d)
}

// projectSearch is search_project for a project, or nil if it has no
// indexed files.
func (s *Server) projectSearch(project string) func(context.Context, string) ([]tools.Passage, error) {
	pr, err := s.projects.Get(project)
	if err != nil || pr.Ready() == 0 {
		return nil
	}
	return func(ctx context.Context, q string) ([]tools.Passage, error) {
		var out []tools.Passage
		for _, x := range s.retrieve(ctx, project, q) {
			out = append(out, tools.Passage{File: x.File, Page: x.Page, Heading: x.Heading, Text: x.Text})
		}
		return out, nil
	}
}

// toolDefs converts tools to the request's tool list.
func toolDefs(set []tools.Tool) []ollama.Tool {
	var out []ollama.Tool
	for _, t := range set {
		var d ollama.Tool
		d.Type = "function"
		d.Function.Name, d.Function.Description, d.Function.Parameters = t.Name, t.Description, t.Parameters
		out = append(out, d)
	}
	return out
}

// toolResultLimit caps what one call adds to the chat file and the prompt.
const toolResultLimit = 12000

// chatLoop streams the model's reply to t.req. When the model asks for tool
// calls it runs them, adds the calls and their results to the request and
// asks again, up to maxToolSteps times. Thinking and text stream out as they
// arrive; each call is announced ("tool") and then reported ("toolresult").
// Everything made so far is in t.calls even if the browser aborts.
func (s *Server) chatLoop(ctx context.Context, t *turn, emit func(event) error) (string, string, *chats.Stats, error) {
	var stats *chats.Stats
	var err error
	for step := 0; ; step++ {
		req := t.req
		if step >= maxToolSteps {
			req.Tools = nil // out of steps: answer with what there is
		}
		var c, th strings.Builder
		var final ollama.ChatResponse
		loop := newLoopCheck()
		err = s.ollama.Chat(ctx, req, func(resp ollama.ChatResponse) error {
			if x := resp.Message.Thinking; x != "" {
				th.WriteString(x)
				if loop.add(x) {
					return errLooping
				}
				if err := emit(event{Type: "thinking", Text: x}); err != nil {
					return err
				}
			}
			if x := resp.Message.Content; x != "" {
				c.WriteString(x)
				if err := emit(event{Type: "content", Text: x}); err != nil {
					return err
				}
			}
			if resp.Done {
				final = resp
				if t.auto.Fit {
					t.fitted = s.fittedContext(ctx, req.Model).Ctx
				}
			}
			return nil
		})
		if err == nil {
			stats = mergeStats(stats, t.stats(final))
		}
		calls := final.Message.ToolCalls
		if err != nil || len(calls) == 0 || len(req.Tools) == 0 {
			return c.String(), th.String(), stats, err
		}

		// The model asked for tools: run them, record them, go around again.
		asst := ollama.Message{Role: "assistant", Content: c.String(), Thinking: th.String(), ToolCalls: calls}
		var results []ollama.Message
		for i := range calls {
			if calls[i].ID == "" {
				calls[i].ID = "call_" + strconv.Itoa(len(t.calls)+1)
			}
			call := chats.ToolCall{Step: step, ID: calls[i].ID, Name: calls[i].Function.Name, Args: string(calls[i].Function.Arguments)}
			if i == 0 {
				call.Said, call.Think = c.String(), th.String()
			}
			t.calls = append(t.calls, call)
			idx := len(t.calls) - 1
			if err = emit(event{Type: "tool", Call: &call}); err != nil {
				return "", "", stats, err
			}
			res := tools.Run(ctx, t.tools, call.Name, calls[i].Function.Arguments)
			if ctx.Err() != nil {
				t.calls = t.calls[:idx] // never ran to the end
				if i == 0 {             // what it said first was kept on this call
					return c.String(), th.String(), stats, ctx.Err() // keep what it said first
				}
				return "", "", stats, ctx.Err()
			}
			text := tools.Clip(res.Text, toolResultLimit)
			t.calls[idx].Result, t.calls[idx].Label, t.calls[idx].Error = text, res.Label, res.Err
			results = append(results, ollama.Message{Role: "tool", Content: text, ToolCallID: call.ID})
			if err = emit(event{Type: "toolresult", Call: &t.calls[idx]}); err != nil {
				return "", "", stats, err
			}
		}
		t.req.Messages = append(slices.Clone(t.req.Messages), append([]ollama.Message{asst}, results...)...)
	}
}

// mergeStats folds one step's numbers into the reply's: tokens and time add
// up, speed and context are the latest step's.
func mergeStats(acc, st *chats.Stats) *chats.Stats {
	if acc == nil {
		return st
	}
	st.OutputTokens += acc.OutputTokens
	st.TotalMS += acc.TotalMS
	st.LoadMS += acc.LoadMS
	return st
}

// toolMessages turns a saved reply back into the messages the model saw:
// for each step the assistant's tool calls and the results, then the
// final text.
func toolMessages(m chats.Message) []ollama.Message {
	var out []ollama.Message
	for i := 0; i < len(m.Calls); {
		step := m.Calls[i].Step
		asst := ollama.Message{Role: "assistant", Content: m.Calls[i].Said, Thinking: m.Calls[i].Think}
		var results []ollama.Message
		for ; i < len(m.Calls) && m.Calls[i].Step == step; i++ {
			c := m.Calls[i]
			var tc ollama.ToolCall
			tc.ID, tc.Function.Name = c.ID, c.Name
			tc.Function.Arguments = json.RawMessage(c.Args)
			if !json.Valid(tc.Function.Arguments) {
				tc.Function.Arguments, _ = json.Marshal(c.Args)
			}
			asst.ToolCalls = append(asst.ToolCalls, tc)
			results = append(results, ollama.Message{Role: "tool", Content: c.Result, ToolCallID: c.ID})
		}
		out = append(out, asst)
		out = append(out, results...)
	}
	return out
}

// errLooping stops a reply whose thinking goes round in circles: small
// reasoning models can repeat the same few lines until the context is full
// (K2 Horizon spent 32k tokens, 5 minutes, saying the same thing 30 times).
var errLooping = errors.New("the model was going in circles in its thinking, so ollamui stopped it. Try again, turn Thinking off, or raise the repeat penalty in Tune")

// loopCheck counts the long lines of a stream; one seen loopRepeats times
// is a loop. Short lines ("Wait.", "}") repeat honestly.
type loopCheck struct {
	seen    map[string]int
	partial strings.Builder
}

const (
	loopLineMin = 40
	loopRepeats = 8
)

func newLoopCheck() *loopCheck { return &loopCheck{seen: map[string]int{}} }

// add takes the next piece of the stream and reports a loop.
func (l *loopCheck) add(x string) bool {
	l.partial.WriteString(x)
	text := l.partial.String()
	i := strings.LastIndexByte(text, '\n')
	if i < 0 {
		return false
	}
	l.partial.Reset()
	l.partial.WriteString(text[i+1:])
	for _, line := range strings.Split(text[:i], "\n") {
		line = strings.TrimSpace(line)
		if len(line) < loopLineMin {
			continue
		}
		l.seen[line]++
		if l.seen[line] >= loopRepeats {
			return true
		}
	}
	return false
}

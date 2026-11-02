package server

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"slices"
	"strings"
	"sync"
	"time"

	"assay/internal/ollama"
	"assay/internal/remote"
)

// Teams in an editor: a request that brings tools (an editor's: reading
// and editing files, running commands) has the lead work with them as well
// as with its helpers. Assay can't run the editor's tools, so when the lead
// calls one the request ends there with the call, as a model's would; the
// editor runs it and sends the conversation back with the result. Helpers
// get no editor tools: they work on what the lead puts in their task.
//
// What the lead did in between (its helpers' tasks and results) isn't in
// what the editor keeps, so Assay keeps the lead's own transcript and finds
// it again by the last assistant message the editor sends back. If it's
// gone (Assay restarted), the lead goes on from the editor's history.

const editorLead = `You're working for the user in their code editor, with the editor's tools (reading and editing files, running commands and so on); use them as you normally would.

You also lead helper models: the spawn_agents tool hands tasks to them. Each helper works on its own, in its own context, without the editor's tools or files, and sends back only a short result, so put everything one needs in its task, including any code it should work on. Use helpers for self-contained work worth doing side by side or outside your own context: writing or reviewing a function, checking an approach, going through many items. Do the work in the editor yourself.`

const editorAgentLimit = `Helpers' work on each of your turns shares a budget of about %s tokens; their results say how much is left.`

const editorLoopLimit = `You can hand out at most %d tasks at once, and %d round%s of tasks on each of your turns.`

const editorPipeline = `You're working for the user in their code editor, with the editor's tools (reading and editing files, running commands and so on); use them as you normally would. Helper models have already worked on parts of the user's latest request, and their results come with it. They may contain mistakes, so check them before relying on them.`

// leadTranscripts are the leads' transcripts, by the assistant message
// that ended each request; a few dozen, the oldest dropped first.
var leadTranscripts = struct {
	sync.Mutex
	m map[string]leadTranscript
}{m: map[string]leadTranscript{}}

type leadTranscript struct {
	at   time.Time
	msgs []ollama.Message
}

const maxTranscripts = 64

// transcriptKey names an assistant message: by its first tool call, or by
// its text. Empty messages have none.
func transcriptKey(team string, m ollama.Message) string {
	if len(m.ToolCalls) > 0 {
		return team + "\x00call:" + m.ToolCalls[0].ID
	}
	text := strings.TrimSpace(m.Content)
	if text == "" {
		return ""
	}
	h := sha256.Sum256([]byte(text))
	return team + "\x00text:" + hex.EncodeToString(h[:12])
}

func keepTranscript(key string, msgs []ollama.Message) {
	if key == "" {
		return
	}
	lt := &leadTranscripts
	lt.Lock()
	defer lt.Unlock()
	lt.m[key] = leadTranscript{at: time.Now(), msgs: msgs}
	for len(lt.m) > maxTranscripts {
		oldest := ""
		for k, v := range lt.m {
			if oldest == "" || v.at.Before(lt.m[oldest].at) {
				oldest = k
			}
		}
		delete(lt.m, oldest)
	}
}

// findTranscript is the lead's transcript up to the last assistant message
// of conv it knows, then the rest of conv.
func findTranscript(team string, conv []ollama.Message) ([]ollama.Message, bool) {
	lt := &leadTranscripts
	lt.Lock()
	defer lt.Unlock()
	for i := len(conv) - 1; i >= 0; i-- {
		if conv[i].Role != "assistant" {
			continue
		}
		if t, ok := lt.m[transcriptKey(team, conv[i])]; ok {
			return append(slices.Clone(t.msgs), conv[i+1:]...), true
		}
	}
	return nil, false
}

// editorSystem is the lead's system message in an editor.
func editorSystem(t remote.Team, editor string) string {
	sys := editorPipeline
	switch t.Strategy {
	case remote.Agent:
		sys = editorLead + "\n\n" + fmt.Sprintf(editorAgentLimit, kTokens(t.Budget))
	case remote.Loop:
		sys = editorLead + "\n\n" + fmt.Sprintf(editorLoopLimit, t.Size, t.Rounds, plural(t.Rounds))
	}
	if t.Notes != "" {
		sys += "\n\n" + t.Notes
	}
	if editor != "" {
		sys += "\n\nThe editor's instructions:\n\n" + editor
	}
	return sys
}

// chatTeamEditor has a team answer a request that brings tools.
func (s *Server) chatTeamEditor(ctx context.Context, t remote.Team, req ollama.ChatRequest, fn func(ollama.ChatResponse) error) error {
	lead, err := s.member(ctx, t.Lead)
	if err != nil {
		return fmt.Errorf("the team's lead: %w", err)
	}
	worker, err := s.member(ctx, t.Worker)
	if err != nil {
		return fmt.Errorf("the team's workers: %w", err)
	}
	if sh, err := s.show(ctx, lead.name); err != nil || !sh.HasCapability("tools") {
		return fmt.Errorf("%s leads this team and can't call tools, which the editor needs; give the team a lead that can", lead.title)
	}
	var sys []string
	var conv []ollama.Message
	for _, m := range req.Messages {
		if m.Role == "system" {
			sys = append(sys, m.Content)
			continue
		}
		m.Thinking = ""
		conv = append(conv, m)
	}
	msgs, ok := findTranscript(t.Name(), conv)
	if !ok {
		msgs = append([]ollama.Message{{Role: "system", Content: editorSystem(t, strings.Join(sys, "\n\n"))}}, conv...)
	}
	var gpu sync.Mutex
	loaded, numCtx := "", 0
	tr := &teamRun{s: s, t: t, lead: lead, worker: worker, maxTok: intOf(req.Options["num_predict"]), gpu: &gpu, loaded: &loaded, numCtx: &numCtx, chat: &req, emit: fn, live: true, tog: s.together(ctx, t, lead, worker)}
	start := time.Now()
	if t.Strategy == remote.Pipeline && msgs[len(msgs)-1].Role == "user" {
		if msgs, err = tr.editorParts(ctx, msgs); err != nil {
			return err
		}
	}
	msgs, last, err := tr.editorLoop(ctx, msgs, req.Tools)
	if err != nil {
		return err
	}
	// The editor gets only its own tools' calls.
	var calls []ollama.ToolCall
	for _, c := range last.ToolCalls {
		if c.Function.Name != "spawn_agents" {
			calls = append(calls, c)
		}
	}
	key := ollama.Message{Role: "assistant", Content: tr.said.String(), ToolCalls: calls}
	keepTranscript(transcriptKey(t.Name(), key), msgs)
	d := time.Since(start)
	return fn(ollama.ChatResponse{Model: req.Model, CreatedAt: time.Now(), Done: true, DoneReason: "stop", Message: ollama.Message{Role: "assistant", ToolCalls: calls}, EvalCount: tr.tokens, EvalDuration: int64(d), TotalDuration: int64(d), Cost: tr.cost, Progress: tr.finalSteps()})
}

// editorParts is the pipeline's plan and parts for the latest request,
// done before the lead takes it up; their results go with the request.
func (tr *teamRun) editorParts(ctx context.Context, msgs []ollama.Message) ([]ollama.Message, error) {
	// The planner and helpers see the conversation as text.
	var hist []remote.Message
	for _, m := range msgs[1:] {
		text := strings.TrimSpace(m.Content)
		for _, c := range m.ToolCalls {
			text += fmt.Sprintf("\n[calls %s %s]", c.Function.Name, c.Function.Arguments)
		}
		if text = strings.TrimSpace(text); text != "" {
			hist = append(hist, remote.Message{Role: m.Role, Content: text})
		}
	}
	convo := workerContext(hist)
	plan, err := tr.call(ctx, tr.lead, chatMsgs(fmt.Sprintf(pipelinePlan, tr.t.Size), convo), false, tr.begin(true, "Planning", "", tr.lead))
	if err != nil {
		return nil, err
	}
	ps := parts(plan.Reply, tr.t.Size)
	if len(ps) == 0 {
		return msgs, nil
	}
	tr.note("## Plan (%s)\n\n%s\n\n", tr.lead.title, strings.TrimSpace(plan.Reply))
	results := tr.work(ctx, ps, func(k int) string { return fmt.Sprintf("Part %d", k+1) }, func(p string) []remote.Message {
		return chatMsgs(pipelineWorker, convo+"\n\n---\n\nYour part:\n\n"+p)
	})
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	var b strings.Builder
	for k, p := range ps {
		tr.note("## Part %d (%s): %s\n\n%s\n\n", k+1, tr.worker.title, p, results[k])
		fmt.Fprintf(&b, "### Part %d: %s\n\n%s\n\n", k+1, p, results[k])
	}
	msgs = slices.Clone(msgs)
	last := &msgs[len(msgs)-1]
	last.Content += "\n\n---\n\nYour helpers' work on parts of this:\n\n" + strings.TrimSpace(b.String())
	return msgs, nil
}

// editorLoop has the lead work until it answers or calls the editor's
// tools, running its helpers in between. It returns the transcript and the
// lead's last message.
func (tr *teamRun) editorLoop(ctx context.Context, msgs []ollama.Message, tools []ollama.Tool) ([]ollama.Message, ollama.Message, error) {
	start := time.Now()
	spawn := tr.t.Strategy != remote.Pipeline
	atOnce, maxRounds := agentMaxAtOnce, agentMaxRounds
	if tr.t.Strategy == remote.Loop {
		atOnce, maxRounds = tr.t.Size, tr.t.Rounds
	}
	for round := 1; ; round++ {
		offer := tools
		if spawn && round <= maxRounds && (tr.t.Strategy != remote.Agent || tr.spent(start) == "") {
			offer = append(slices.Clone(tools), spawnTool)
		}
		m, _, err := tr.leadCall(ctx, msgs, offer, tr.begin(true, fmt.Sprintf("Turn %d", round), "", tr.lead))
		if err != nil {
			return nil, m, err
		}
		m.Thinking = "" // the lead's own thinking stays out of its context
		var spawns []int
		for i := range m.ToolCalls {
			if m.ToolCalls[i].ID == "" {
				b := make([]byte, 8)
				rand.Read(b)
				m.ToolCalls[i].ID = "call_" + hex.EncodeToString(b)
			}
			if m.ToolCalls[i].Function.Name == "spawn_agents" {
				spawns = append(spawns, i)
			}
		}
		msgs = append(msgs, m)
		if len(spawns) == 0 {
			return msgs, m, nil
		}

		// Run every task asked for together, up to the ceiling; each spawn
		// call gets its own results.
		var tasks []string
		asked := make([][]string, len(spawns))
		bad := make([]string, len(spawns))
		for j, i := range spawns {
			asked[j], bad[j] = spawnTasks(m.ToolCalls[i])
			tasks = append(tasks, asked[j]...)
		}
		run := tasks[:min(len(tasks), atOnce)]
		results := tr.work(ctx, run, func(k int) string { return fmt.Sprintf("Turn %d, task %d", round, k+1) }, func(task string) []remote.Message { return chatMsgs(agentWorker, task) })
		if ctx.Err() != nil {
			return nil, m, ctx.Err()
		}
		for k, task := range run {
			tr.note("\n\n## Task %d (%s)\n\n%s\n\n**Result:**\n\n%s\n\n", k+1, tr.worker.title, task, results[k])
		}
		n := 0
		for j, i := range spawns {
			var b strings.Builder
			b.WriteString(bad[j])
			for range asked[j] {
				if n < len(results) {
					fmt.Fprintf(&b, "<result task=\"%d\">\n%s\n</result>\n\n", n+1, results[n])
				} else {
					fmt.Fprintf(&b, "<result task=\"%d\">\n(not run: at most %d tasks run at once; send it again)\n</result>\n\n", n+1, atOnce)
				}
				n++
			}
			text := strings.TrimSpace(b.String())
			if j == len(spawns)-1 && tr.t.Strategy == remote.Agent {
				text += "\n\n" + tr.budgetLine(start)
			}
			msgs = append(msgs, ollama.Message{Role: "tool", Content: text, ToolCallID: m.ToolCalls[i].ID})
		}
		if len(spawns) < len(m.ToolCalls) {
			// It called the editor's tools too: those go out now.
			return msgs, m, nil
		}
	}
}

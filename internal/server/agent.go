package server

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"strings"
	"time"

	"assay/internal/chats"
	"assay/internal/evals"
	"assay/internal/ollama"
	"assay/internal/remote"
)

// The Agent strategy: the lead decides how much to delegate. It spawns
// helpers with a spawn_agents tool (or <task> blocks, for a lead without
// tool calling) as often and as many at once as it likes; helpers work in
// their own context and send back only a short result, so the lead's
// context stays small. A budget of tokens, cost and time is the only limit;
// when it runs out the lead has to answer with what it has.

// Hard ceilings, so a confused model can't run away inside a big budget.
const (
	agentMaxAtOnce = 16
	agentMaxRounds = 30
)

const agentLead = `You can hand work to helper models with the spawn_agents tool. Each helper works on its own, in its own context, and sends back only a short result, so delegating keeps your own context small and lets work happen in parallel. Helpers see only the task you write, not this conversation, so put everything each one needs in its task.

Delegate when it helps: separate parts of a big job, research or checking, a draft to review, many items to go through. Don't for things you can answer well yourself. You can call spawn_agents as many times as you need, with as many tasks at once as are useful, and you'll see each round's results before deciding what to do next.

You have a budget of about %s tokens for this answer, shared by you and your helpers; each round's results say how much is left. When you're done, write your final answer to the original request in exactly the form it asks for. If it asks for code, give the whole program in one code block.`

const agentLeadText = `You can hand work to helper models. To do so, write each task in its own block and end your message:

<task>
the full description of one task
</task>

Each helper works on its own, in its own context, and sends back only a short result, so delegating keeps your own context small and lets work happen in parallel. Helpers see only the task you write, not this conversation, so put everything each one needs in its task.

Delegate when it helps: separate parts of a big job, research or checking, a draft to review, many items to go through. Don't for things you can answer well yourself. You can hand out work as many times as you need, as many tasks at once as are useful, and you'll see each round's results before deciding what to do next.

You have a budget of about %s tokens for this answer, shared by you and your helpers; each round's results say how much is left. When you're done, write your final answer to the original request in exactly the form it asks for, with no <task> blocks. If it asks for code, give the whole program in one code block.`

const agentWorker = `You're a helper doing one task for a lead model, which reads your reply next to other helpers'. Work it out carefully, then reply with only what the lead needs: the answer, the code, the facts you found or the problems you spotted. Don't restate the task or show your working unless the task asks for it, and be as short as the result allows.`

// spawnTool is the lead's tool for handing out work.
var spawnTool = func() ollama.Tool {
	var t ollama.Tool
	t.Type = "function"
	t.Function.Name = "spawn_agents"
	t.Function.Description = "Hand tasks to helper models that work on them at the same time, each in its own context, and get back each one's short result. A helper sees only its task text, so include everything it needs."
	t.Function.Parameters = map[string]any{
		"type": "object",
		"properties": map[string]any{
			"tasks": map[string]any{"type": "array", "items": map[string]any{"type": "string"}, "description": "One complete task description per helper."},
		},
		"required": []string{"tasks"},
	}
	return t
}()

// spawnTasks reads a spawn_agents call's tasks, or says what's wrong with
// its arguments.
func spawnTasks(c ollama.ToolCall) (tasks []string, bad string) {
	var args struct {
		Tasks []string `json:"tasks"`
	}
	if json.Unmarshal(c.Function.Arguments, &args) != nil {
		return nil, `The arguments should be {"tasks": ["…", "…"]}.`
	}
	for _, t := range args.Tasks {
		if t = strings.TrimSpace(t); t != "" {
			tasks = append(tasks, t)
		}
	}
	return tasks, ""
}

// kTokens writes a token count briefly: 150k, 1.2M.
func kTokens(n int) string {
	switch {
	case n >= 1_000_000:
		return strings.TrimSuffix(fmt.Sprintf("%.1f", float64(n)/1e6), ".0") + "M"
	case n >= 1000:
		return fmt.Sprintf("%dk", (n+500)/1000)
	}
	return fmt.Sprint(n)
}

// spent says whether the budget is used up, and which part.
func (tr *teamRun) spent(start time.Time) string {
	tr.mu.Lock()
	tokens, cost := tr.tokens, tr.cost
	tr.mu.Unlock()
	switch {
	case tokens >= tr.t.Budget:
		return "the token budget is used up"
	case tr.t.MaxCost > 0 && cost >= tr.t.MaxCost:
		return "the cost cap is reached"
	case time.Since(start) >= time.Duration(tr.t.Minutes)*time.Minute:
		return "the time limit is reached"
	}
	return ""
}

// budgetLine is what the lead is told after each round.
func (tr *teamRun) budgetLine(start time.Time) string {
	tr.mu.Lock()
	tokens, cost := tr.tokens, tr.cost
	tr.mu.Unlock()
	line := fmt.Sprintf("[Budget: %s of %s tokens used", kTokens(tokens), kTokens(tr.t.Budget))
	if tr.t.MaxCost > 0 {
		line += fmt.Sprintf(", $%.3f of $%.2f", cost, tr.t.MaxCost)
	}
	return line + fmt.Sprintf(", %d of %d minutes.]", int(time.Since(start).Minutes()), tr.t.Minutes)
}

// agent runs the Agent strategy on a conversation.
func (tr *teamRun) agent(ctx context.Context, system string, hist []remote.Message) (evals.Answer, error) {
	start := time.Now()
	tools := false
	if sh, err := tr.s.show(ctx, tr.lead.name); err == nil && sh.HasCapability("tools") {
		tools = true
	}
	base := agentLeadText
	if tools {
		base = agentLead
	}
	msgs := []ollama.Message{{Role: "system", Content: tr.leadSystem(fmt.Sprintf(base, kTokens(tr.t.Budget)), system)}}
	for _, m := range hist {
		msgs = append(msgs, ollama.Message{Role: m.Role, Content: m.Content})
	}
	for round := 1; ; round++ {
		why := tr.spent(start)
		if why == "" && round > agentMaxRounds {
			why = "that was the last round allowed"
		}
		last := why != ""
		if last {
			msgs = append(msgs, ollama.Message{Role: "user", Content: "No more helpers: " + why + ". Write your final answer now from what you have."})
		}
		step := fmt.Sprintf("Round %d", round)
		if last {
			step = "Answer"
		}
		id := tr.begin(true, step, "", tr.lead)
		var offer []ollama.Tool
		if tools && !last {
			offer = []ollama.Tool{spawnTool}
		}
		m, a, err := tr.leadCall(ctx, msgs, offer, id)
		if err != nil {
			return a, err
		}
		// What it asked for, per tool call (or all as one, from text).
		type ask struct {
			call  ollama.ToolCall
			tasks []string
			bad   string
		}
		var asks []ask
		for i, c := range m.ToolCalls {
			if c.ID == "" {
				c.ID = fmt.Sprintf("call_%d_%d", round, i+1)
				m.ToolCalls[i].ID = c.ID
			}
			k := ask{call: c}
			if c.Function.Name != "spawn_agents" {
				k.bad = "There's no tool called " + c.Function.Name + "; the only one is spawn_agents."
			} else {
				k.tasks, k.bad = spawnTasks(c)
			}
			asks = append(asks, k)
		}
		if !tools {
			var k ask
			for _, x := range taskRe.FindAllStringSubmatch(a.Reply, -1) {
				if t := strings.TrimSpace(x[1]); t != "" {
					k.tasks = append(k.tasks, t)
				}
			}
			if len(k.tasks) > 0 {
				asks = append(asks, k)
			}
		}
		if len(asks) == 0 || last {
			tr.progress(id, func(st *chats.TeamStep) { st.Step = "Answer" })
			a.Reply = strings.TrimSpace(taskRe.ReplaceAllString(a.Reply, ""))
			return a, nil
		}

		// Run every task asked for this round together, up to the ceiling.
		var tasks []string
		for _, k := range asks {
			tasks = append(tasks, k.tasks...)
		}
		run := tasks[:min(len(tasks), agentMaxAtOnce)]
		if strings.TrimSpace(a.Reply) != "" {
			tr.note("## Round %d (%s)\n\n%s\n\n", round, tr.lead.title, strings.TrimSpace(a.Reply))
		}
		results := tr.work(ctx, run, func(k int) string { return fmt.Sprintf("Round %d, task %d", round, k+1) }, func(task string) []remote.Message { return chatMsgs(agentWorker, task) })
		if ctx.Err() != nil {
			return evals.Answer{}, ctx.Err()
		}
		for k, task := range run {
			tr.note("## Round %d, task %d (%s)\n\n%s\n\n**Result:**\n\n%s\n\n", round, k+1, tr.worker.title, task, results[k])
		}
		result := func(n int) string {
			if n < len(results) {
				return results[n]
			}
			return fmt.Sprintf("(not run: at most %d tasks run at once; send it again in another round)", agentMaxAtOnce)
		}
		budget := tr.budgetLine(start)
		m.Thinking = "" // the lead's own thinking stays out of its context
		msgs = append(msgs, m)
		if tools {
			n := 0
			for i, k := range asks {
				var b strings.Builder
				if k.bad != "" {
					b.WriteString(k.bad)
				}
				for j := range k.tasks {
					fmt.Fprintf(&b, "<result task=\"%d\">\n%s\n</result>\n\n", j+1, result(n))
					n++
				}
				text := strings.TrimSpace(b.String())
				if i == len(asks)-1 {
					text += "\n\n" + budget
				}
				msgs = append(msgs, ollama.Message{Role: "tool", Content: text, ToolCallID: k.call.ID})
			}
		} else {
			var b strings.Builder
			for j := range tasks {
				fmt.Fprintf(&b, "<result task=\"%d\">\n%s\n</result>\n\n", j+1, result(j))
			}
			msgs = append(msgs, ollama.Message{Role: "user", Content: "Results of your tasks:\n\n" + b.String() + budget + "\n\nHand out more tasks or write your final answer."})
		}
	}
}

// leadCall asks the lead with the whole conversation and, if offered, the
// spawn tool. It goes through s.chat in chats and benchmarks alike (local
// leads in a benchmark are loaded the way the other calls load them), and
// shows the lead's context size on its step.
func (tr *teamRun) leadCall(ctx context.Context, msgs []ollama.Message, tools []ollama.Tool, id int) (ollama.Message, evals.Answer, error) {
	mb := tr.lead
	opts := map[string]any{}
	var think any
	if tr.chat != nil {
		for _, k := range []string{"temperature", "top_p", "top_k", "min_p", "seed"} {
			if v, ok := tr.chat.Options[k]; ok {
				opts[k] = v
			}
		}
		think = tr.chat.Think
	} else {
		opts = evals.SamplingOptions()
		if mb.api {
			delete(opts, "min_p") // few APIs take it
		}
		if tr.think == evals.ThinkOff {
			think = false
		}
	}
	if tr.maxTok > 0 {
		opts["num_predict"] = tr.maxTok
	}
	var a evals.Answer
	if tr.tog != nil {
		maps.Copy(opts, tr.tog.opts(mb))
		if tr.chat == nil {
			if err := tr.s.evalWait(ctx); err != nil {
				return ollama.Message{}, a, err
			}
		}
	} else if !mb.api {
		tr.gpu.Lock()
		defer tr.gpu.Unlock()
		if tr.chat == nil {
			if err := tr.s.evalWait(ctx); err != nil {
				return ollama.Message{}, a, err
			}
			if *tr.loaded != mb.name {
				auto, err := tr.s.loadAuto(ctx, mb.name, "10m")
				if err != nil {
					return ollama.Message{}, a, err
				}
				*tr.loaded, *tr.numCtx = mb.name, auto.Ctx
			}
			if *tr.numCtx > 0 {
				opts["num_ctx"] = *tr.numCtx
			}
		}
	}
	req := ollama.ChatRequest{Model: mb.name, Messages: msgs, Options: opts, Think: think, Tools: tools}
	var content, thinking strings.Builder
	var final ollama.ChatResponse
	t0 := time.Now()
	pieces := 0
	err := tr.s.chat(ctx, req, func(r ollama.ChatResponse) error {
		if r.Done {
			final = r
			return nil
		}
		content.WriteString(r.Message.Content)
		thinking.WriteString(r.Message.Thinking)
		if tr.live && (r.Message.Content != "" || r.Message.Thinking != "") {
			tr.said.WriteString(r.Message.Content)
			if err := tr.emit(ollama.ChatResponse{Model: tr.chat.Model, CreatedAt: time.Now(), Message: ollama.Message{Role: "assistant", Content: r.Message.Content, Thinking: r.Message.Thinking}}); err != nil {
				return err
			}
		}
		if r.Message.Content != "" || r.Message.Thinking != "" {
			pieces++
			tr.progress(id, func(st *chats.TeamStep) {
				st.Tokens, st.Ms = pieces, time.Since(t0).Milliseconds()
				if content.Len() > 0 {
					st.Tail = stepTail(content.String())
				} else {
					st.Tail = stepTail(thinking.String())
				}
			})
		}
		return nil
	})
	a.Reply, a.Thinking, a.Ms = content.String(), thinking.String(), time.Since(t0).Milliseconds()
	a.Tokens, a.Cost, a.Cut = final.EvalCount, final.Cost, final.DoneReason == "length"
	if a.Tokens == 0 {
		a.Tokens = pieces
	}
	calls := final.Message.ToolCalls
	if len(calls) == 0 && len(tools) > 0 && strings.Contains(a.Reply, tcOpen) {
		calls, a.Reply = textToolCalls(a.Reply)
	}
	tr.mu.Lock()
	tr.tokens += a.Tokens
	tr.cost += a.Cost
	tr.mu.Unlock()
	ctxTok := final.PromptEvalCount + final.EvalCount
	tr.progress(id, func(st *chats.TeamStep) {
		st.State, st.Tokens, st.Ms, st.Tail, st.Ctx = "done", a.Tokens, a.Ms, "", ctxTok
		if err != nil {
			st.State = "failed"
		}
	})
	return ollama.Message{Role: "assistant", Content: a.Reply, Thinking: a.Thinking, ToolCalls: calls}, a, err
}

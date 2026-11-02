package server

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"assay/internal/evals"
	"assay/internal/remote"
)

// Teams: an orchestrator–worker setup run on the benchmarks as one entry.
// The lead and the workers are any benchmark models, local or API; local
// calls take turns on the GPU, API calls run side by side.

// teamPrompts versions the prompts below; changing them starts fresh
// results.
const teamPrompts = "1"

// member is one model a team calls.
type member struct {
	name, title, digest string
	api                 bool
	m                   remote.Model
	ep                  remote.Endpoint
}

// member resolves a benchmark model name, local or "api:…".
func (s *Server) member(ctx context.Context, name string) (member, error) {
	if m, ep, ok := s.remote.Lookup(name); ok {
		return member{name: name, title: m.Title, digest: m.Digest(ep), api: true, m: m, ep: ep}, nil
	}
	if strings.HasPrefix(name, remote.Prefix) {
		return member{}, errors.New("no such API model: " + name)
	}
	models, err := s.ollama.List(ctx)
	if err != nil {
		return member{}, err
	}
	for _, g := range groupModels(models) {
		if g.Has(name) {
			return member{name: g.Primary.Name, title: groupTitle(g), digest: g.Primary.Digest}, nil
		}
	}
	return member{}, errors.New("no such model: " + name)
}

// teamDigest keys a team's results: it changes with anything that changes
// its answers, members included.
func teamDigest(t remote.Team, lead, worker member) string {
	h := sha256.Sum256([]byte(strings.Join([]string{teamPrompts, t.Strategy, strconv.Itoa(t.Size), strconv.Itoa(t.Rounds), t.Notes, lead.digest, worker.digest}, "\x00")))
	return "team-" + hex.EncodeToString(h[:12])
}

// teamQuant describes a team in a few words, for its row.
func teamQuant(t remote.Team, lead, worker string) string {
	how := "Plan, work, combine"
	if t.Strategy == remote.Loop {
		how = fmt.Sprintf("Free loop, %d round%s", t.Rounds, plural(t.Rounds))
	}
	return fmt.Sprintf("%s · %s → %s ×%d", how, lead, worker, t.Size)
}

func plural(n int) string {
	if n == 1 {
		return ""
	}
	return "s"
}

// teamRun is one team answering one question.
type teamRun struct {
	s            *Server
	t            remote.Team
	lead, worker member
	think        string
	maxTok       int
	// gpu makes local calls take turns; loaded is the local model last
	// loaded for this run and numCtx its context.
	gpu    *sync.Mutex
	loaded *string
	numCtx *int

	mu     sync.Mutex
	tokens int
	cost   float64
	log    strings.Builder
}

func (tr *teamRun) note(format string, args ...any) {
	tr.mu.Lock()
	fmt.Fprintf(&tr.log, format, args...)
	tr.mu.Unlock()
}

// call asks one member and counts what it cost.
func (tr *teamRun) call(ctx context.Context, mb member, msgs []remote.Message) (evals.Answer, error) {
	var a evals.Answer
	var err error
	if mb.api {
		m := mb.m
		if tr.think == evals.ThinkOff {
			m.Reasoning = "off"
		}
		a, err = tr.s.askAPIMsgs(ctx, mb.ep, m, msgs, tr.maxTok)
	} else {
		tr.gpu.Lock()
		if err = tr.s.evalWait(ctx); err == nil && *tr.loaded != mb.name {
			var auto autoContext
			if auto, err = tr.s.loadAuto(ctx, mb.name, "10m"); err == nil {
				*tr.loaded, *tr.numCtx = mb.name, auto.Ctx
			}
		}
		if err == nil {
			a, err = tr.s.askModelMsgs(ctx, mb.name, msgs, tr.think, tr.maxTok, *tr.numCtx)
		}
		tr.gpu.Unlock()
	}
	tr.mu.Lock()
	tr.tokens += a.Tokens
	tr.cost += a.Cost
	tr.mu.Unlock()
	return a, err
}

// work runs tasks on the worker side by side (local ones take turns) and
// returns their replies in order.
func (tr *teamRun) work(ctx context.Context, tasks []string, msgs func(task string) []remote.Message) []string {
	out := make([]string, len(tasks))
	var wg sync.WaitGroup
	for k, task := range tasks {
		wg.Add(1)
		go func() {
			defer wg.Done()
			a, err := tr.call(ctx, tr.worker, msgs(task))
			reply := strings.TrimSpace(a.Reply)
			if err != nil {
				reply = "(this helper failed: " + err.Error() + ")"
			} else if reply == "" {
				reply = "(this helper wrote nothing)"
			}
			out[k] = reply
		}()
	}
	wg.Wait()
	return out
}

const pipelinePlan = `You lead a small team of helper models. Read the task, then split it into at most %d parts that helpers can work on at the same time, each without seeing the others' work. Each helper sees the whole task plus the part you give it, so describe each part precisely: what to produce, and any decisions it must follow so the parts fit together (names, signatures, conventions, formulas). If the task is small or can't be split, give a single part.

Reply with only a JSON array of strings, one per part, and nothing else.`

const pipelineWorker = `You're one of several helpers, each working on one part of a bigger task. Do only your part, carefully and completely, and show the result plainly (working, code or numbers). The lead will combine the parts and check them.`

const pipelineCombine = `Your helpers worked on parts of this task. Their results follow. They may contain mistakes, so check them before relying on them, and fix or redo anything that's wrong.

%s

Now write your complete final answer to the task, in exactly the form it asks for. If it asks for code, give the whole program in one code block, not pieces.`

const loopLead = `You lead a small team of helper models and can hand them work. To hand out work, write each task in its own block:

<task>
the full description of one task
</task>

Helpers see only the task text you write, not the original task or each other's work, so include everything they need. You can hand out at most %d tasks at once and you get %d round%s of handing out work; after each round you'll see the results and can hand out more, or answer.

When you're ready, write your final answer to the original task in exactly the form it asks for, with no <task> blocks. If it asks for code, give the whole program in one code block. Doing small tasks yourself is fine.`

const loopWorker = `You're a helper doing one task for a lead model. Do it carefully and completely, and show the result plainly (working, code or numbers).`

var taskRe = regexp.MustCompile(`(?s)<task>(.*?)</task>`)

// parts reads the lead's plan: a JSON array, or failing that a list.
func parts(plan string, max int) []string {
	var out []string
	if i, j := strings.Index(plan, "["), strings.LastIndex(plan, "]"); i >= 0 && j > i {
		var arr []any
		if json.Unmarshal([]byte(plan[i:j+1]), &arr) == nil {
			for _, v := range arr {
				switch v := v.(type) {
				case string:
					out = append(out, v)
				default:
					b, _ := json.Marshal(v)
					out = append(out, string(b))
				}
			}
		}
	}
	if len(out) == 0 {
		item := regexp.MustCompile(`(?m)^\s*(?:[-*•]|\d+[.)])\s+(.+)$`)
		for _, m := range item.FindAllStringSubmatch(plan, -1) {
			out = append(out, m[1])
		}
	}
	out = slices.DeleteFunc(out, func(p string) bool { return strings.TrimSpace(p) == "" })
	return out[:min(len(out), max)]
}

// answer runs the team on one question.
func (tr *teamRun) answer(ctx context.Context, set *evals.Set, q evals.Question) (evals.Answer, error) {
	t0 := time.Now()
	system, user := set.Messages(q)
	var final evals.Answer
	var err error
	if tr.t.Strategy == remote.Loop {
		final, err = tr.loop(ctx, system, user)
	} else {
		final, err = tr.pipeline(ctx, system, user)
	}
	final.Thinking = strings.TrimSpace(tr.log.String())
	final.Tokens, final.Cost, final.Ms = tr.tokens, tr.cost, time.Since(t0).Milliseconds()
	final.Clip()
	return final, err
}

func (tr *teamRun) leadSystem(base, setSystem string) string {
	sys := base
	if tr.t.Notes != "" {
		sys += "\n\n" + tr.t.Notes
	}
	if setSystem != "" {
		sys += "\n\nThe task comes with these instructions for the final answer:\n\n" + setSystem
	}
	return sys
}

func (tr *teamRun) pipeline(ctx context.Context, system, user string) (evals.Answer, error) {
	plan, err := tr.call(ctx, tr.lead, chatMsgs(tr.leadSystem(fmt.Sprintf(pipelinePlan, tr.t.Size), system), user))
	if err != nil {
		return plan, err
	}
	ps := parts(plan.Reply, tr.t.Size)
	if len(ps) == 0 {
		ps = []string{"Do the whole task."}
	}
	tr.note("## Plan (%s)\n\n%s\n\n", tr.lead.title, strings.TrimSpace(plan.Reply))
	results := tr.work(ctx, ps, func(p string) []remote.Message {
		return chatMsgs(pipelineWorker, "The whole task, for context:\n\n"+user+"\n\n---\n\nYour part:\n\n"+p)
	})
	var b strings.Builder
	for k, p := range ps {
		tr.note("## Part %d (%s): %s\n\n%s\n\n", k+1, tr.worker.title, p, results[k])
		fmt.Fprintf(&b, "### Part %d: %s\n\n%s\n\n", k+1, p, results[k])
	}
	if ctx.Err() != nil {
		return evals.Answer{}, ctx.Err()
	}
	final, err := tr.call(ctx, tr.lead, chatMsgs(system, user+"\n\n---\n\n"+fmt.Sprintf(pipelineCombine, strings.TrimSpace(b.String()))))
	if final.Thinking != "" {
		tr.note("## Final answer, thinking (%s)\n\n%s\n\n", tr.lead.title, strings.TrimSpace(final.Thinking))
	}
	return final, err
}

func (tr *teamRun) loop(ctx context.Context, system, user string) (evals.Answer, error) {
	msgs := chatMsgs(tr.leadSystem(fmt.Sprintf(loopLead, tr.t.Size, tr.t.Rounds, plural(tr.t.Rounds)), system), user)
	for round := 1; ; round++ {
		a, err := tr.call(ctx, tr.lead, msgs)
		if err != nil {
			return a, err
		}
		var tasks []string
		for _, m := range taskRe.FindAllStringSubmatch(a.Reply, -1) {
			if t := strings.TrimSpace(m[1]); t != "" {
				tasks = append(tasks, t)
			}
		}
		if len(tasks) == 0 {
			return a, nil
		}
		msgs = append(msgs, remote.Message{Role: "assistant", Content: a.Reply})
		if round > tr.t.Rounds {
			// It kept handing out work past its last round: answer with
			// what it wrote outside the blocks.
			a.Reply = strings.TrimSpace(taskRe.ReplaceAllString(a.Reply, ""))
			return a, nil
		}
		tasks = tasks[:min(len(tasks), tr.t.Size)]
		tr.note("## Round %d (%s)\n\n%s\n\n", round, tr.lead.title, strings.TrimSpace(a.Reply))
		results := tr.work(ctx, tasks, func(task string) []remote.Message { return chatMsgs(loopWorker, task) })
		if ctx.Err() != nil {
			return evals.Answer{}, ctx.Err()
		}
		var b strings.Builder
		for k, task := range tasks {
			tr.note("## Round %d, task %d (%s)\n\n%s\n\n**Result:**\n\n%s\n\n", round, k+1, tr.worker.title, task, results[k])
			fmt.Fprintf(&b, "<result task=\"%d\">\n%s\n</result>\n\n", k+1, results[k])
		}
		next := "Results of your tasks:\n\n" + b.String()
		if round == tr.t.Rounds {
			next += "That was your last round of handing out work. Write your final answer now, with no <task> blocks."
		} else {
			next += fmt.Sprintf("Hand out more tasks (%d round%s left) or write your final answer.", tr.t.Rounds-round, plural(tr.t.Rounds-round))
		}
		msgs = append(msgs, remote.Message{Role: "user", Content: next})
	}
}

// runTeamEvalItem runs a team on every question of one set: four
// questions at once when every member is an API model, else one at a
// time so local models aren't swapped back and forth.
func (s *Server) runTeamEvalItem(ctx context.Context, i int, set *evals.Set, redo bool) error {
	s.evalJob.mu.Lock()
	it, think := s.evalJob.Items[i], s.evalJob.Think
	s.evalJob.mu.Unlock()
	t, ok := s.remote.Team(it.Model)
	if !ok {
		return errors.New("the team is gone")
	}
	lead, err := s.member(ctx, t.Lead)
	if err != nil {
		return fmt.Errorf("lead: %w", err)
	}
	worker, err := s.member(ctx, t.Worker)
	if err != nil {
		return fmt.Errorf("worker: %w", err)
	}
	workers := 1
	if lead.api && worker.api {
		workers = apiEvalWorkers
	}
	var gpu sync.Mutex
	loaded, numCtx := "", 0
	return s.runEvalItemWith(ctx, i, set, redo, teamDigest(t, lead, worker), workers, func(ctx context.Context, q evals.Question, maxTok int) (evals.Answer, error) {
		tr := &teamRun{s: s, t: t, lead: lead, worker: worker, think: think, maxTok: maxTok, gpu: &gpu, loaded: &loaded, numCtx: &numCtx}
		return tr.answer(ctx, set, q)
	})
}

// teamRows adds a row per team to the results.
func (s *Server) teamRows(ctx context.Context, byDigest map[string][]*evals.Result, hashes map[string]string, suite []string) []evalRow {
	var rows []evalRow
	for _, t := range s.remote.Teams() {
		lead, err1 := s.member(ctx, t.Lead)
		worker, err2 := s.member(ctx, t.Worker)
		if err1 != nil || err2 != nil {
			continue
		}
		row := evalRow{Name: t.Name(), Title: t.Title, Quant: teamQuant(t, lead.title, worker.title), Digest: teamDigest(t, lead, worker), Team: true}
		fillRow(&row, byDigest[row.Digest], hashes, suite)
		rows = append(rows, row)
	}
	return rows
}

// Team settings.

func (s *Server) handleTeams(w http.ResponseWriter, r *http.Request) {
	type teamJSON struct {
		remote.Team
		Name  string `json:"name"`
		Quant string `json:"quant"`
		Error string `json:"error,omitempty"`
	}
	out := []teamJSON{}
	for _, t := range s.remote.Teams() {
		tj := teamJSON{Team: t, Name: t.Name()}
		lead, err1 := s.member(r.Context(), t.Lead)
		worker, err2 := s.member(r.Context(), t.Worker)
		if err := errors.Join(err1, err2); err != nil {
			tj.Error = err.Error()
		} else {
			tj.Quant = teamQuant(t, lead.title, worker.title)
		}
		out = append(out, tj)
	}
	writeJSON(w, http.StatusOK, out)
}

func (s *Server) handleTeamPut(w http.ResponseWriter, r *http.Request) {
	var t remote.Team
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 256<<10)).Decode(&t); err != nil {
		writeError(w, http.StatusBadRequest, "bad request")
		return
	}
	t.ID = r.PathValue("id")
	if t.ID == "new" {
		t.ID = remote.Slug(t.Title, func(id string) bool {
			return id == "new" || slices.ContainsFunc(s.remote.Teams(), func(x remote.Team) bool { return x.ID == id })
		})
	}
	for _, name := range []string{t.Lead, t.Worker} {
		if _, err := s.member(r.Context(), name); err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}
	}
	if err := s.remote.PutTeam(t); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"id": t.ID, "name": t.Name()})
}

func (s *Server) handleTeamDelete(w http.ResponseWriter, r *http.Request) {
	if err := s.remote.DeleteTeam(r.PathValue("id")); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

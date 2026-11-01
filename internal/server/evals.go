package server

import (
	"context"
	"errors"
	"io"
	"io/fs"
	"math"
	"net/http"
	"slices"
	"sort"
	"strings"
	"sync"
	"time"

	"assay/internal/evals"
	"assay/internal/lean"
	"assay/internal/ollama"
	"assay/internal/scores"
)

// Benchmarks: your own question sets, run on installed models one question
// at a time through the engine, and an estimate of the Intelligence Index
// from a line fitted through the models that have a published score.

// Reply limits per question.
const (
	evalTokensThink   = 8192
	evalTokensNoThink = 2048
)

type evalItem struct {
	Model    string `json:"model"`
	Title    string `json:"title"`
	Set      string `json:"set"`
	SetName  string `json:"setName"`
	State    string `json:"state"` // pending, running, done, skipped, failed
	Reason   string `json:"reason,omitempty"`
	Answered int    `json:"answered"`
	Correct  int    `json:"correct"`
	Total    int    `json:"total"`
	// Judging is how many answers the judge has left; Waiting how many
	// still wait for one after the run (no judge was picked).
	Judging int `json:"judging,omitempty"`
	Waiting int `json:"waiting,omitempty"`
	// Since is when this item started running and From how many answers
	// it had then, so the page can show the pace and the time left.
	Since time.Time `json:"since,omitzero"`
	From  int       `json:"from,omitempty"`
}

type evalJob struct {
	mu        sync.Mutex
	running   bool
	Items     []evalItem
	Think     string
	MaxTokens int
	Judge     string
	Waiting   string
	Cancelled bool
	cancel    context.CancelFunc
}

type evalState struct {
	Running   bool       `json:"running"`
	Items     []evalItem `json:"items"`
	Think     string     `json:"think"`
	Judge     string     `json:"judge,omitempty"`
	Waiting   string     `json:"waiting,omitempty"`
	Cancelled bool       `json:"cancelled,omitempty"`
	// Done and Total count questions across all items.
	Done  int `json:"done"`
	Total int `json:"total"`
}

func (j *evalJob) state() evalState {
	j.mu.Lock()
	defer j.mu.Unlock()
	st := evalState{Running: j.running, Items: slices.Clone(j.Items), Think: j.Think, Judge: j.Judge, Waiting: j.Waiting, Cancelled: j.Cancelled}
	if st.Items == nil {
		st.Items = []evalItem{}
	}
	for _, it := range j.Items {
		if it.State == itemSkipped {
			continue
		}
		st.Total += it.Total
		if it.State == itemFailed {
			st.Done += it.Total
		} else {
			st.Done += it.Answered
		}
	}
	return st
}

func (j *evalJob) set(i int, f func(*evalItem)) {
	j.mu.Lock()
	f(&j.Items[i])
	j.mu.Unlock()
}

type evalRunRequest struct {
	Models []string `json:"models"`
	Sets   []string `json:"sets"`
	Think  string   `json:"think"`
	// MaxTokens limits each reply; 0 picks by think mode.
	MaxTokens int `json:"maxTokens"`
	// Redo runs sets again that already have a finished result.
	Redo bool `json:"redo"`
	// Judge grades judge questions (proofs); required when a set has them.
	Judge string `json:"judge"`
}

func (s *Server) handleEvalRunStart(w http.ResponseWriter, r *http.Request) {
	var req evalRunRequest
	if !readJSON(w, r, &req) {
		return
	}
	st, err := s.startEvals(r.Context(), req)
	if err != nil {
		var ce *conflictError
		if errors.As(err, &ce) {
			writeError(w, http.StatusConflict, err.Error())
			return
		}
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	writeJSON(w, http.StatusAccepted, st)
}

func (s *Server) handleEvalRunState(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	writeJSON(w, http.StatusOK, s.evalJob.state())
}

func (s *Server) handleEvalRunCancel(w http.ResponseWriter, r *http.Request) {
	s.evalJob.mu.Lock()
	if s.evalJob.running && s.evalJob.cancel != nil {
		s.evalJob.cancel()
	}
	s.evalJob.mu.Unlock()
	writeJSON(w, http.StatusOK, s.evalJob.state())
}

func (s *Server) startEvals(ctx context.Context, req evalRunRequest) (evalState, error) {
	if req.Think != evals.ThinkDefault && req.Think != evals.ThinkOff {
		return evalState{}, errors.New(`think should be "" or "off"`)
	}
	if len(req.Models) == 0 || len(req.Sets) == 0 {
		return evalState{}, errors.New("pick at least one model and one set")
	}
	if req.MaxTokens <= 0 {
		req.MaxTokens = evalTokensThink
		if req.Think == evals.ThinkOff {
			req.MaxTokens = evalTokensNoThink
		}
	}
	req.MaxTokens = min(req.MaxTokens, 65536)
	j := &s.evalJob
	j.mu.Lock()
	busy := j.running
	j.mu.Unlock()
	if busy {
		return evalState{}, &conflictError{"benchmarks are already running"}
	}
	models, err := s.ollama.List(ctx)
	if err != nil {
		return evalState{}, err
	}
	groups := groupModels(models)
	var sets []*evals.Set
	for _, id := range req.Sets {
		set, err := s.evalLib.Get(id)
		if err != nil {
			return evalState{}, errors.New("no such set: " + id)
		}
		if set.NeedsLean() && s.leanInst.Binary() == "" {
			return evalState{}, errors.New(set.Name + " is checked by Lean, which isn't installed yet. Install it from the Benchmarks page first.")
		}
		if set.NeedsCode() && !evals.HavePython() {
			return evalState{}, errors.New(set.Name + " runs the model's code with Python 3, and none was found on PATH")
		}
		sets = append(sets, set)
	}
	if req.Judge != "" {
		gi := slices.IndexFunc(groups, func(g modelGroup) bool { return g.Has(req.Judge) })
		if gi < 0 {
			return evalState{}, errors.New("no such judge model: " + req.Judge)
		}
		req.Judge = groups[gi].Primary.Name
	}
	var items []evalItem
	for _, name := range req.Models {
		gi := slices.IndexFunc(groups, func(g modelGroup) bool { return g.Has(name) })
		if gi < 0 {
			return evalState{}, errors.New("no such model: " + name)
		}
		g := groups[gi]
		title := groupTitle(g)
		for _, set := range sets {
			if slices.ContainsFunc(items, func(it evalItem) bool { return it.Model == g.Primary.Name && it.Set == set.ID }) {
				continue
			}
			it := evalItem{Model: g.Primary.Name, Title: title, Set: set.ID, SetName: set.Name, State: itemPending, Total: len(set.Questions)}
			if isEmbeddingName(g.Primary.Name) {
				it.State, it.Reason = itemSkipped, "It's an embedding model; it doesn't write text."
			} else if old, ok := s.evalRes.Get(g.Primary.Digest, set.ID, req.Think); ok && old.Current(set.Hash) {
				if old.Done() && !req.Redo {
					it.State, it.Reason = itemSkipped, "Already done."
					it.Answered, it.Correct = len(old.Answers), old.Correct
				} else if !req.Redo {
					it.Answered, it.Correct = len(old.Answers), old.Correct // resumes
				}
			}
			items = append(items, it)
		}
	}
	if !slices.ContainsFunc(items, func(it evalItem) bool { return it.State == itemPending }) {
		return evalState{}, &conflictError{"nothing to run: every pick is done already (choose Run again to redo)"}
	}
	jctx, cancel := context.WithCancel(context.Background())
	j.mu.Lock()
	j.running, j.Cancelled, j.Waiting, j.Items, j.cancel = true, false, "", items, cancel
	j.Think, j.MaxTokens, j.Judge = req.Think, req.MaxTokens, req.Judge
	j.mu.Unlock()
	go s.runEvals(jctx, req.Redo)
	return j.state(), nil
}

func groupTitle(g modelGroup) string {
	title, _ := prettyModel(g.Primary.Name)
	if g.HFRepo != "" && g.PrimaryOwner() != "" {
		title, _ = prettyModel(g.ShortName())
	}
	return title
}

func (s *Server) runEvals(ctx context.Context, redo bool) {
	j := &s.evalJob
	defer func() {
		j.mu.Lock()
		j.running, j.Waiting = false, ""
		j.Cancelled = ctx.Err() != nil
		j.mu.Unlock()
	}()
	run := s.evalFn
	if run == nil {
		run = s.runEvalItem
	}
	for i := range j.Items {
		if ctx.Err() != nil {
			return
		}
		if j.Items[i].State != itemPending {
			continue
		}
		j.set(i, func(it *evalItem) { it.State = itemRunning })
		err := run(ctx, i, redo)
		switch {
		case ctx.Err() != nil:
			j.set(i, func(it *evalItem) { it.State = itemPending })
		case err != nil:
			s.log.Warn("benchmark", "model", j.Items[i].Model, "set", j.Items[i].Set, "err", err)
			j.set(i, func(it *evalItem) { it.State, it.Reason = itemFailed, err.Error() })
		default:
			j.set(i, func(it *evalItem) { it.State = itemDone })
		}
	}
}

// evalWait holds a run back while a chat reply is being written or a
// speed test runs; both need the GPU more than a benchmark does.
func (s *Server) evalWait(ctx context.Context) error {
	for {
		busy := ""
		if len(s.runningChats()) > 0 {
			busy = "Waiting for a chat reply to finish"
		} else if !benchMu.TryLock() {
			busy = "Waiting for a speed test to finish"
		} else {
			benchMu.Unlock()
		}
		s.evalJob.mu.Lock()
		s.evalJob.Waiting = busy
		s.evalJob.mu.Unlock()
		if busy == "" {
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(time.Second):
		}
	}
}

// runEvalItem asks one model every question of one set, saving after each
// answer so a stopped run picks up where it left off.
func (s *Server) runEvalItem(ctx context.Context, i int, redo bool) error {
	j := &s.evalJob
	j.mu.Lock()
	it, think, maxTok, judge := j.Items[i], j.Think, j.MaxTokens, j.Judge
	j.mu.Unlock()
	set, err := s.evalLib.Get(it.Set)
	if err != nil {
		return errors.New("the set is gone")
	}
	models, err := s.ollama.List(ctx)
	if err != nil {
		return err
	}
	mi := slices.IndexFunc(models, func(m ollama.Model) bool { return sameModel(m.Name, it.Model) })
	if mi < 0 {
		return errors.New("the model is gone")
	}
	maxTok = min(max(maxTok, set.MaxTokens), 65536) // hard sets ask for room to reason
	res := &evals.Result{Model: models[mi].Name, Digest: models[mi].Digest, Set: set.ID, SetHash: set.Hash, Sampling: evals.Sampling, Think: think, MaxTokens: maxTok, Total: len(set.Questions)}
	if old, ok := s.evalRes.Get(res.Digest, set.ID, think); ok && old.Current(set.Hash) && !redo {
		res.Answers = old.Answers
		if old.MaxTokens > 0 {
			res.MaxTokens = old.MaxTokens
		}
	}
	answered := map[string]bool{}
	for _, a := range res.Answers {
		answered[a.ID] = true
	}
	j.set(i, func(it *evalItem) { it.Since, it.From = time.Now(), len(res.Answers) })
	if err := s.evalWait(ctx); err != nil {
		return err
	}
	auto, err := s.loadAuto(ctx, it.Model, "10m")
	if err != nil {
		return err
	}
	res.Ctx = auto.Ctx
	for _, q := range set.Questions {
		if answered[q.ID] {
			continue
		}
		if err := s.evalWait(ctx); err != nil {
			return err
		}
		a, err := s.askEval(ctx, it.Model, set, q, think, res.MaxTokens, auto.Ctx)
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if err != nil {
			a.Error = err.Error()
		} else {
			switch set.CheckOf(q) {
			case evals.CheckCode:
				if err := gradeCode(ctx, q, &a); ctx.Err() != nil {
					return ctx.Err()
				} else if err != nil {
					return err // no Python: the rest would fail the same way
				}
			case evals.CheckJudge:
				a.Pending = true
			case evals.CheckLean:
				if err := s.gradeLean(ctx, q, &a); ctx.Err() != nil {
					return ctx.Err()
				} else if err != nil {
					return err
				}
			}
		}
		res.Answers = append(res.Answers, a)
		res.Recount()
		res.At = time.Now()
		if err := s.evalRes.Put(res); err != nil {
			return errors.New("couldn't save the result: " + err.Error())
		}
		j.set(i, func(it *evalItem) { it.Answered, it.Correct = len(res.Answers), res.Correct })
	}
	return s.judgeEvalItem(ctx, i, set, res, judge)
}

// gradeCode runs the reply's code against the question's tests.
func gradeCode(ctx context.Context, q evals.Question, a *evals.Answer) error {
	code, ok := evals.ExtractCode(a.Reply)
	if !ok {
		a.Output = "No Python code block in the reply."
		return nil
	}
	pass, out, err := evals.RunCode(ctx, code, q.Tests)
	if err != nil {
		return err
	}
	a.Correct, a.Output = pass, out
	return nil
}

// gradeLean checks the reply's Lean proof of the question's statement.
func (s *Server) gradeLean(ctx context.Context, q evals.Question, a *evals.Answer) error {
	bin := s.leanInst.Binary()
	if bin == "" {
		return errors.New("Lean isn't installed")
	}
	code, ok := lean.ExtractCode(evals.StripThinking(a.Reply))
	if !ok {
		a.Output = "No ```lean code block in the reply."
		return nil
	}
	r, err := lean.Check(ctx, bin, q.Preamble, q.Statement, code)
	if err != nil {
		return err
	}
	a.Correct, a.Output = r.OK, r.Output
	if r.OK && a.Output == "" {
		a.Output = "Lean accepted the proof."
	}
	return nil
}

// judgeEvalItem has the judge grade the answers waiting for one. Without a
// judge they stay waiting and the item says so.
func (s *Server) judgeEvalItem(ctx context.Context, i int, set *evals.Set, res *evals.Result, judge string) error {
	j := &s.evalJob
	var waiting []int
	for k, a := range res.Answers {
		if a.Pending {
			waiting = append(waiting, k)
		}
	}
	if len(waiting) == 0 {
		return nil
	}
	if judge == "" {
		j.set(i, func(it *evalItem) { it.Waiting = len(waiting) })
		return nil
	}
	byID := map[string]evals.Question{}
	for _, q := range set.Questions {
		byID[q.ID] = q
	}
	if err := s.evalWait(ctx); err != nil {
		return err
	}
	auto, err := s.loadAuto(ctx, judge, "10m")
	if err != nil {
		return errors.New("couldn't load the judge: " + err.Error())
	}
	for n, k := range waiting {
		j.set(i, func(it *evalItem) { it.Judging = len(waiting) - n })
		if err := s.evalWait(ctx); err != nil {
			return err
		}
		a := &res.Answers[k]
		system, user := evals.JudgeMessages(byID[a.ID], a.Reply)
		verdict, err := s.askModel(ctx, judge, system, user, evals.ThinkDefault, evalTokensThink, auto.Ctx)
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if err != nil {
			return errors.New("the judge failed: " + err.Error())
		}
		pass, ok := evals.ParseVerdict(verdict.Reply)
		a.Judge, a.JudgeModel, a.Pending, a.Correct = evals.StripThinking(verdict.Reply), judge, false, pass && ok
		if !ok {
			a.Judge += "\n\n(The judge gave no verdict; counted as a fail.)"
		}
		res.Recount()
		if err := s.evalRes.Put(res); err != nil {
			return errors.New("couldn't save the result: " + err.Error())
		}
		j.set(i, func(it *evalItem) { it.Correct = res.Correct })
	}
	j.set(i, func(it *evalItem) { it.Judging = 0 })
	return nil
}

// askEval asks one question at temperature 0 and grades what it can.
func (s *Server) askEval(ctx context.Context, model string, set *evals.Set, q evals.Question, think string, maxTok, numCtx int) (evals.Answer, error) {
	system, user := set.Messages(q)
	a, err := s.askModel(ctx, model, system, user, think, maxTok, numCtx)
	a.ID = q.ID
	if err != nil {
		return a, err
	}
	a.Got, a.Correct = set.Grade(q, a.Reply)
	return a, nil
}

// askModel sends one prompt with the eval sampling (evals.SamplingOptions)
// and collects the reply, stopping it early if the model loops.
func (s *Server) askModel(ctx context.Context, model, system, user, think string, maxTok, numCtx int) (evals.Answer, error) {
	var a evals.Answer
	var msgs []ollama.Message
	if system != "" {
		msgs = append(msgs, ollama.Message{Role: "system", Content: system})
	}
	msgs = append(msgs, ollama.Message{Role: "user", Content: user})
	opts := evals.SamplingOptions()
	opts["num_predict"] = maxTok
	if numCtx > 0 {
		opts["num_ctx"] = numCtx
	}
	stream := true
	req := ollama.ChatRequest{Model: model, Messages: msgs, Stream: &stream, Options: opts, KeepAlive: "10m"}
	if think == evals.ThinkOff {
		req.Think = false
	}
	var content, thinking strings.Builder
	var final ollama.ChatResponse
	t := time.Now()
	errLooped := errors.New("looped")
	chunks := 0
	run := func() error {
		content.Reset()
		thinking.Reset()
		chunks = 0
		return s.ollama.Chat(ctx, req, func(c ollama.ChatResponse) error {
			content.WriteString(c.Message.Content)
			thinking.WriteString(c.Message.Thinking)
			if c.Done {
				final = c
			}
			if chunks++; chunks%64 == 0 && (evals.Looping(thinking.String()) || evals.Looping(content.String())) {
				return errLooped
			}
			return nil
		})
	}
	err := run()
	looped := errors.Is(err, errLooped)
	if looped {
		err = nil
	}
	if err != nil && req.Think != nil && strings.Contains(err.Error(), "think") {
		req.Think = nil // models that can't turn thinking off say so
		err = run()
	}
	a.Ms = time.Since(t).Milliseconds()
	a.Reply, a.Thinking = content.String(), thinking.String()
	a.Tokens, a.Cut, a.Looped = final.EvalCount, final.DoneReason == "length", looped
	if looped {
		a.Tokens = chunks // about one token per streamed chunk
	}
	a.Clip()
	return a, err
}

// Sets.

type evalSetJSON struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Short       string `json:"short"`
	Description string `json:"description,omitempty"`
	Check       string `json:"check,omitempty"`
	Questions   int    `json:"questions"`
	BuiltIn     bool   `json:"builtIn"`
	Hash        string `json:"hash"`
	Judge       bool   `json:"judge,omitempty"` // has questions a judge model grades
	Code        bool   `json:"code,omitempty"`  // runs the model's code
	Lean        bool   `json:"lean,omitempty"`  // checked by Lean
}

func setJSON(st *evals.Set) evalSetJSON {
	return evalSetJSON{ID: st.ID, Name: st.Name, Short: st.ShortName(), Description: st.Description, Check: st.Check, Questions: len(st.Questions), BuiltIn: st.BuiltIn, Hash: st.Hash, Judge: st.NeedsJudge(), Code: st.NeedsCode(), Lean: st.NeedsLean()}
}

func (s *Server) evalSets() ([]evalSetJSON, map[string]string) {
	sets, errs := s.evalLib.List()
	out := make([]evalSetJSON, 0, len(sets))
	for _, st := range sets {
		out = append(out, setJSON(st))
	}
	return out, errs
}

func (s *Server) handleEvalSets(w http.ResponseWriter, r *http.Request) {
	sets, errs := s.evalSets()
	writeJSON(w, http.StatusOK, map[string]any{"sets": sets, "errors": errs, "checks": evals.Checks})
}

func (s *Server) handleEvalSet(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	raw, err := s.evalLib.Raw(id)
	if err != nil {
		writeError(w, http.StatusNotFound, "no such set")
		return
	}
	out := map[string]any{"id": id, "toml": string(raw)}
	if st, err := s.evalLib.Get(id); err == nil {
		out["set"] = st
	} else {
		out["error"] = err.Error()
	}
	writeJSON(w, http.StatusOK, out)
}

func (s *Server) handleEvalSetSave(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if !evals.ValidID(id) {
		writeError(w, http.StatusBadRequest, "the id should be lowercase letters, digits and dashes")
		return
	}
	raw, err := io.ReadAll(io.LimitReader(r.Body, 4<<20))
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	st, err := s.evalLib.Save(id, raw)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, setJSON(st))
}

func (s *Server) handleEvalSetDelete(w http.ResponseWriter, r *http.Request) {
	if err := s.evalLib.Delete(r.PathValue("id")); err != nil {
		code := http.StatusBadRequest
		if errors.Is(err, fs.ErrNotExist) {
			code = http.StatusNotFound
		}
		writeError(w, code, err.Error())
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// Results.

type evalCell struct {
	Score    float64   `json:"score"` // share right of those answered
	Correct  int       `json:"correct"`
	Answered int       `json:"answered"`
	Total    int       `json:"total"`
	Done     bool      `json:"done"`
	Stale    bool      `json:"stale,omitempty"` // the set changed since
	Cut      int       `json:"cut,omitempty"`   // answers that hit the token limit or looped
	Errors   int       `json:"errors,omitempty"`
	Pending  int       `json:"pending,omitempty"` // waiting for a judge
	At       time.Time `json:"at"`
	// For the speed stats: tokens written and time taken over all answers,
	// and the tokens of the right ones.
	Tokens      int   `json:"tokens"`
	Ms          int64 `json:"ms"`
	RightTokens int   `json:"rightTokens"`
}

type evalRow struct {
	Name   string              `json:"name"`
	Title  string              `json:"title"`
	Quant  string              `json:"quant,omitempty"`
	Digest string              `json:"digest"`
	Params float64             `json:"params,omitempty"` // billions
	Cells  map[string]evalCell `json:"cells"`
	// Overall is the mean score over the suite, when every suite set is done.
	Overall *float64 `json:"overall,omitempty"`
	// Index is the published Intelligence Index of the base model; Of names it.
	Index *float64 `json:"index,omitempty"`
	Of    string   `json:"of,omitempty"`
	// Estimate is the fitted index; Outside marks one beyond the fitted range.
	Estimate *float64 `json:"estimate,omitempty"`
	Outside  bool     `json:"outside,omitempty"`
	Anchor   bool     `json:"anchor,omitempty"`
}

type evalFitJSON struct {
	OK      bool    `json:"ok"`
	N       int     `json:"n"`
	A       float64 `json:"a"`
	B       float64 `json:"b"`
	R2      float64 `json:"r2"`
	SE      float64 `json:"se"`
	MinX    float64 `json:"minX"`
	MaxX    float64 `json:"maxX"`
	Need    int     `json:"need"`
	Comment string  `json:"comment"`
}

type evalResults struct {
	Sets     []evalSetJSON `json:"sets"`
	Suite    []string      `json:"suite"`
	Think    string        `json:"think"`
	Rows     []evalRow     `json:"rows"`
	Fit      evalFitJSON   `json:"fit"`
	ScoreURL string        `json:"scoreUrl"`
}

// evalResultsFor builds the results table for one think mode. suite names
// the sets the overall score and the fit use; empty means every built-in set.
func (s *Server) evalResultsFor(ctx context.Context, think string, suite []string) (evalResults, error) {
	out := evalResults{Think: think, ScoreURL: scores.AttributionURL}
	out.Sets, _ = s.evalSets()
	hashes := map[string]string{}
	for _, st := range out.Sets {
		hashes[st.ID] = st.Hash
	}
	suite = slices.DeleteFunc(slices.Clone(suite), func(id string) bool { return hashes[id] == "" })
	if len(suite) == 0 {
		for _, st := range out.Sets {
			if st.BuiltIn {
				suite = append(suite, st.ID)
			}
		}
	}
	out.Suite = suite
	models, err := s.ollama.List(ctx)
	if err != nil {
		return out, err
	}
	byDigest := map[string][]*evals.Result{}
	for _, r := range s.evalRes.All() {
		if r.Think == think {
			byDigest[r.Digest] = append(byDigest[r.Digest], r)
		}
	}
	var mu sync.Mutex
	var wg sync.WaitGroup
	sem := make(chan struct{}, 4)
	for _, g := range groupModels(models) {
		if isEmbeddingName(g.Primary.Name) {
			continue
		}
		row := evalRow{Name: g.Primary.Name, Title: groupTitle(g), Quant: quantOf(g), Digest: g.Primary.Digest, Params: float64(parseParams(g.Primary.Details.ParameterSize)) / 1e9, Cells: map[string]evalCell{}}
		for _, r := range byDigest[g.Primary.Digest] {
			c := evalCell{Score: r.Score(), Correct: r.Correct, Answered: len(r.Answers), Total: r.Total, Done: r.Done(), At: r.At, Pending: r.Pending()}
			c.Stale = hashes[r.Set] != "" && !r.Current(hashes[r.Set])
			for _, a := range r.Answers {
				c.Tokens += a.Tokens
				c.Ms += a.Ms
				if a.Correct {
					c.RightTokens += a.Tokens
				}
				if a.Cut || a.Looped {
					c.Cut++
				}
				if a.Error != "" {
					c.Errors++
				}
			}
			row.Cells[r.Set] = c
		}
		sum, all := 0.0, true
		for _, id := range suite {
			c, ok := row.Cells[id]
			if !ok || !c.Done || c.Stale {
				all = false
				break
			}
			sum += c.Score
		}
		if all {
			v := sum / float64(len(suite))
			row.Overall = &v
		}
		out.Rows = append(out.Rows, row)
		if g.HFRepo != "" && row.Overall != nil {
			i := len(out.Rows) - 1
			wg.Add(1)
			go func(repo string) {
				defer wg.Done()
				sem <- struct{}{}
				ls := s.libScoreOf(ctx, repo)
				<-sem
				if ls.ok {
					mu.Lock()
					sc := ls.score
					out.Rows[i].Index, out.Rows[i].Of = &sc, ls.of
					mu.Unlock()
				}
			}(g.HFRepo)
		}
	}
	wg.Wait()
	out.Fit = fitRows(out.Rows)
	sort.SliceStable(out.Rows, func(a, b int) bool {
		x, y := out.Rows[a], out.Rows[b]
		if (x.Overall == nil) != (y.Overall == nil) {
			return x.Overall != nil
		}
		if x.Overall != nil && *x.Overall != *y.Overall {
			return *x.Overall > *y.Overall
		}
		return len(x.Cells) > len(y.Cells)
	})
	return out, nil
}

// fitRows fits the line on the rows with both an overall score and an
// index, and fills in every row's estimate.
func fitRows(rows []evalRow) evalFitJSON {
	var pts []evals.Point
	seen := map[string]bool{}
	for i := range rows {
		r := &rows[i]
		if r.Overall != nil && r.Index != nil {
			r.Anchor = true
			// Several quants of one base model share its published score;
			// count the base model once, at its average.
			if seen[r.Of] {
				continue
			}
			seen[r.Of] = true
			x, n := 0.0, 0
			for _, o := range rows {
				if o.Overall != nil && o.Index != nil && o.Of == r.Of {
					x += *o.Overall
					n++
				}
			}
			pts = append(pts, evals.Point{X: x / float64(n), Y: *r.Index})
		}
	}
	f, ok := evals.FitLine(pts)
	out := evalFitJSON{OK: ok, N: len(pts), Need: evals.MinAnchors}
	if !ok {
		switch {
		case len(pts) < evals.MinAnchors:
			out.Comment = "needs more models with a published score"
		default:
			out.Comment = "the models with a published score all scored the same here"
		}
		return out
	}
	out.A, out.B, out.R2, out.SE, out.MinX, out.MaxX = f.A, f.B, f.R2, f.SE, f.MinX, f.MaxX
	switch {
	case f.B <= 0:
		out.Comment = "doesn't track the index: a better score here predicts a lower index"
	case f.R2 >= 0.8:
		out.Comment = "tracks the index well"
	case f.R2 >= 0.5:
		out.Comment = "tracks the index roughly"
	default:
		out.Comment = "tracks the index poorly"
	}
	for i := range rows {
		if rows[i].Overall != nil {
			v := math.Max(0, f.Predict(*rows[i].Overall))
			rows[i].Estimate, rows[i].Outside = &v, f.Outside(*rows[i].Overall)
		}
	}
	return out
}

func (s *Server) handleEvalResults(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	out, err := s.evalResultsFor(r.Context(), q.Get("think"), q["suite"])
	if err != nil {
		writeError(w, ollamaStatus(err), err.Error())
		return
	}
	w.Header().Set("Cache-Control", "no-store")
	writeJSON(w, http.StatusOK, out)
}

// handleEvalResult returns one run with its questions, for reading answers.
func (s *Server) handleEvalResult(w http.ResponseWriter, r *http.Request) {
	digest, setID, think := r.URL.Query().Get("digest"), r.URL.Query().Get("set"), r.URL.Query().Get("think")
	res, ok := s.evalRes.Get(digest, setID, think)
	if !ok {
		writeError(w, http.StatusNotFound, "no result")
		return
	}
	out := map[string]any{"result": res}
	if set, err := s.evalLib.Get(setID); err == nil {
		out["set"] = set
		switch {
		case res.SetHash != set.Hash:
			out["stale"] = "the set has changed since this run"
		case res.Sampling != evals.Sampling:
			out["stale"] = "run at temperature 0, before runs used temperature 1"
		}
	}
	writeJSON(w, http.StatusOK, out)
}

func (s *Server) handleEvalResultDelete(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	if err := s.evalRes.Delete(q.Get("digest"), q.Get("set"), q.Get("think")); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// The page.

type benchmarksData struct {
	Initial any // JSON-encoded into the page by html/template
}

func (s *Server) handleBenchmarksPage(w http.ResponseWriter, r *http.Request) {
	p := s.basePage(r.Context(), "benchmarks", "Benchmarks")
	think := r.URL.Query().Get("think")
	res, err := s.evalResultsFor(r.Context(), think, r.URL.Query()["suite"])
	if err != nil {
		s.log.Warn("benchmarks page", "err", err)
	}
	p.Data = &benchmarksData{Initial: map[string]any{"results": res, "run": s.evalJob.state(), "lean": s.leanInst.Status()}}
	s.render(w, "benchmarks", p)
}

// Lean, for proof sets.

func (s *Server) handleLeanStatus(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	writeJSON(w, http.StatusOK, s.leanInst.Status())
}

func (s *Server) handleLeanInstall(w http.ResponseWriter, r *http.Request) {
	if s.leanInst.Dir == "" {
		writeError(w, http.StatusBadRequest, "no folder to install Lean in")
		return
	}
	if err := s.leanInst.Start(); err != nil {
		writeError(w, http.StatusConflict, err.Error())
		return
	}
	writeJSON(w, http.StatusAccepted, s.leanInst.Status())
}

func (s *Server) handleLeanCancel(w http.ResponseWriter, r *http.Request) {
	s.leanInst.Cancel()
	writeJSON(w, http.StatusOK, s.leanInst.Status())
}

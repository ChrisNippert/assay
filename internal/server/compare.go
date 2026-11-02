package server

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"sync"
	"time"

	"assay/internal/bench"
	"assay/internal/hf"
	"assay/internal/ollama"
	"assay/internal/scores"
)

// maxCompare is how many models fit side by side.
const maxCompare = 4

// compareData is the Compare page: installed models side by side.
type compareData struct {
	Columns  []compareColumn
	Picked   []string // names in the URL, in order
	ScoreURL string
	// Best values per row, to highlight the winner when there are two or more.
	BestScore, BestGen, BestPrompt float64
	BestTTFT, BestLoad             int64
	BestCtx                        int
}

type compareColumn struct {
	Group   modelGroup
	Show    *ollama.ShowResponse
	Err     string
	Quality string // rule-of-thumb quant quality
	MaxCtx  int
	Auto    autoContext // the context that fits on the GPU
	// Score is the intelligence index of the model this was quantized from.
	Score    float64
	Rated    bool
	ScoreOf  string
	ScorePct int
	Bench    *bench.Result
}

// Caps lists the model's capabilities worth showing.
func (c compareColumn) Caps() []string {
	if c.Show == nil {
		return nil
	}
	var out []string
	for _, cp := range c.Show.Capabilities {
		switch cp {
		case "vision", "tools", "thinking", "audio", "embedding":
			out = append(out, cp)
		}
	}
	return out
}

func (s *Server) handleComparePage(w http.ResponseWriter, r *http.Request) {
	p := s.basePage(r.Context(), "compare", "Compare")
	data := &compareData{ScoreURL: scores.AttributionURL}
	p.Data = data
	groups := p.ModelGroups()
	find := func(name string) (modelGroup, bool) {
		for _, g := range groups {
			if slices.ContainsFunc(g.Names, func(n string) bool { return sameModel(n, name) }) {
				return g, true
			}
		}
		return modelGroup{}, false
	}
	for _, name := range r.URL.Query()["m"] {
		g, ok := find(name)
		if !ok || len(data.Columns) == maxCompare || slices.ContainsFunc(data.Columns, func(c compareColumn) bool { return c.Group.Primary.Digest == g.Primary.Digest }) {
			continue
		}
		data.Columns = append(data.Columns, compareColumn{Group: g})
		data.Picked = append(data.Picked, g.Primary.Name)
	}

	var wg sync.WaitGroup
	for i := range data.Columns {
		wg.Add(1)
		go func(c *compareColumn) {
			defer wg.Done()
			s.fillColumn(r.Context(), c)
		}(&data.Columns[i])
	}
	wg.Wait()
	data.best()
	s.render(w, "compare", p)
}

func (s *Server) fillColumn(ctx context.Context, c *compareColumn) {
	m := c.Group.Primary
	show, err := s.ollama.Show(ctx, m.Name)
	if err != nil {
		c.Err = err.Error()
		return
	}
	c.Show = show
	c.MaxCtx = show.ContextLength()
	if q := m.Details.QuantizationLevel; q != "" {
		c.Quality = hf.Quant{Name: q}.Quality()
	}
	c.Auto = s.pickContext(ctx, m.Name, false)
	if r, ok := s.bench.Get(m.Digest); ok {
		c.Bench = &r
		if c.Auto.Ctx == 0 && r.GPUPct == 100 && r.Ctx > 0 {
			c.Auto = autoContext{Ctx: r.Ctx, Reason: "measured in the speed test"}
		}
	}
	if c.Group.HFRepo != "" && s.scores != nil {
		ictx, cancel := context.WithTimeout(ctx, 5*time.Second)
		defer cancel()
		if repo, err := s.hf.Info(ictx, c.Group.HFRepo); err == nil {
			if sc, of := s.scoreFor(repo.Tags); sc != nil {
				c.Score, c.Rated, c.ScoreOf = *sc, true, of
				if top := s.scores.Max(); top > 0 {
					c.ScorePct = min(100, int(*sc/top*100+0.5))
				}
			}
		}
	}
}

// Secs formats milliseconds for the page.
func (compareColumn) Secs(ms int64) string {
	if ms < 1000 {
		return fmt.Sprintf("%d ms", ms)
	}
	return fmt.Sprintf("%.1f s", float64(ms)/1000)
}

// SecsNum and SecsUnit split Secs so the page can set the unit in muted text.
func (compareColumn) SecsNum(ms int64) string {
	if ms < 1000 {
		return fmt.Sprintf("%d", ms)
	}
	return fmt.Sprintf("%.1f", float64(ms)/1000)
}

func (compareColumn) SecsUnit(ms int64) string {
	if ms < 1000 {
		return "ms"
	}
	return "s"
}

// TitleText is the readable name for the column heading, TitleTags its chips
// (QAT, MoE…).
func (c compareColumn) TitleText() string   { t, _ := c.title(); return t }
func (c compareColumn) TitleTags() []string { _, t := c.title(); return t }

func (c compareColumn) title() (string, []string) {
	if c.Group.HFRepo != "" {
		return prettyModel(c.Group.ShortName())
	}
	return prettyModel(c.Group.Primary.Name)
}

// best finds the winning value of each row; with one column there's
// nothing to compare, so nothing is highlighted.
func (d *compareData) best() {
	if len(d.Columns) < 2 {
		return
	}
	for _, c := range d.Columns {
		if c.Rated {
			d.BestScore = max(d.BestScore, c.Score)
		}
		d.BestCtx = max(d.BestCtx, c.Auto.Ctx)
		if b := c.Bench; b != nil {
			d.BestGen = max(d.BestGen, b.GenTPS)
			d.BestPrompt = max(d.BestPrompt, b.PromptTPS)
			if d.BestTTFT == 0 || b.TTFTMs < d.BestTTFT {
				d.BestTTFT = b.TTFTMs
			}
			if d.BestLoad == 0 || b.LoadMs < d.BestLoad {
				d.BestLoad = b.LoadMs
			}
		}
	}
}

// benchMu runs one speed test at a time: two at once would share the GPU
// and measure each other.
var benchMu sync.Mutex

// handleBench runs a speed test on a model and saves the result.
func (s *Server) handleBench(w http.ResponseWriter, r *http.Request) {
	var req nameRequest
	if !readJSON(w, r, &req) {
		return
	}
	if req.Name == "" {
		writeError(w, http.StatusBadRequest, "name is required")
		return
	}
	if !benchMu.TryLock() {
		writeError(w, http.StatusConflict, "another speed test is running")
		return
	}
	defer benchMu.Unlock()
	res, err := s.runBench(r.Context(), req.Name, false)
	if err != nil {
		writeError(w, ollamaStatus(err), err.Error())
		return
	}
	if err := s.bench.Put(res); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, res)
}

// runBench measures a model the way chats use it: alone in memory, loaded
// fresh at the context Assay would pick, then a fixed prompt of about a thousand
// tokens and a reply capped at bench.GenTokens.
func (s *Server) runBench(ctx context.Context, name string, keepPinned bool) (bench.Result, error) {
	models, err := s.ollama.List(ctx)
	if err != nil {
		return bench.Result{}, err
	}
	i := slices.IndexFunc(models, func(m ollama.Model) bool { return sameModel(m.Name, name) })
	if i < 0 {
		return bench.Result{}, errors.New("no such model: " + name)
	}
	res := bench.Result{Model: models[i].Name, Digest: models[i].Digest, At: time.Now()}

	// Alone on the GPU, so models are measured the same way whatever
	// happened to be loaded.
	if err := s.unloadFor(ctx, keepPinned); err != nil {
		return res, err
	}
	// Settle the context first: this measures the model and, if the first
	// pick spills onto the CPU, finds the one that doesn't.
	auto, err := s.loadAuto(ctx, name, "5m")
	if err != nil {
		return res, err
	}
	opts := map[string]any{}
	if auto.Ctx > 0 {
		opts["num_ctx"] = auto.Ctx
	}
	// Unload again (measuring may have loaded it), so the load is timed and
	// the prompt isn't cached.
	if err := s.unloadFor(ctx, keepPinned); err != nil {
		return res, err
	}
	t := time.Now()
	if err := s.ollama.Load(ctx, name, "5m", opts); err != nil {
		return res, err
	}
	res.LoadMs = time.Since(t).Milliseconds()
	if running, err := s.ollama.Running(ctx); err == nil {
		for _, m := range running {
			if sameModel(m.Name, name) {
				res.Ctx = m.ContextLength
				if m.Size > 0 {
					res.GPUPct = int(m.SizeVRAM * 100 / m.Size)
				}
			}
		}
	}

	opts["num_predict"] = bench.GenTokens
	opts["temperature"] = 0
	opts["seed"] = 1
	stream := true
	chat := ollama.ChatRequest{
		Model:     name,
		Messages:  []ollama.Message{{Role: "user", Content: bench.Prompt}},
		Stream:    &stream,
		Think:     false, // thinking would spend the token budget unseen
		Options:   opts,
		KeepAlive: "5m",
	}
	var final ollama.ChatResponse
	t = time.Now()
	first := time.Duration(0)
	run := func() error {
		return s.chat(ctx, chat, func(c ollama.ChatResponse) error {
			if first == 0 && (c.Message.Content != "" || c.Message.Thinking != "") {
				first = time.Since(t)
			}
			if c.Done {
				final = c
			}
			return nil
		})
	}
	err = run()
	if err != nil && strings.Contains(err.Error(), "think") {
		chat.Think = nil // models that can't turn thinking off say so
		t, first = time.Now(), 0
		err = run()
	}
	if err != nil {
		return res, err
	}
	res.TTFTMs = first.Milliseconds()
	res.PromptTokens, res.GenTokens = final.PromptEvalCount, final.EvalCount
	if final.PromptEvalDuration > 0 {
		res.PromptTPS = float64(final.PromptEvalCount) / time.Duration(final.PromptEvalDuration).Seconds()
	}
	if final.EvalDuration > 0 {
		res.GenTPS = float64(final.EvalCount) / time.Duration(final.EvalDuration).Seconds()
	}
	return res, nil
}

// unloadAll unloads every model and waits until Ollama reports them gone;
// an unload request returns before the memory is free.
func (s *Server) unloadAll(ctx context.Context) error {
	for range 60 {
		running, err := s.ollama.Running(ctx)
		if err != nil {
			return err
		}
		if len(running) == 0 {
			return nil
		}
		for _, m := range running {
			s.ollama.Unload(ctx, m.Name)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(500 * time.Millisecond):
		}
	}
	return errors.New("Ollama didn't unload the other models")
}

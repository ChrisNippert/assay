package server

import (
	"bytes"
	"context"
	"errors"
	"net/http"
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"ollamui/internal/hf"
	"ollamui/internal/hw"
	"ollamui/internal/scores"
)

// --- Discover list ---

func (s *Server) handleDiscover(w http.ResponseWriter, r *http.Request) {
	p := s.basePage(r.Context(), "discover", "Discover")
	p.Data = s.memory
	s.render(w, "discover", p)
}

// searchResult is one row on the Discover page.
type searchResult struct {
	ID        string    `json:"id"`
	Author    string    `json:"author"`
	Name      string    `json:"name"`
	Params    int64     `json:"params"`
	Arch      string    `json:"arch"`
	Context   int       `json:"context"`
	MoE       bool      `json:"moe"`
	Vision    bool      `json:"vision"`
	Gated     bool      `json:"gated"`
	Downloads int       `json:"downloads"`
	Likes     int       `json:"likes"`
	Trending  float64   `json:"trending"`
	Updated   time.Time `json:"updated"`
	Installed bool      `json:"installed"`
	Q4Size    int64     `json:"q4Size"` // estimated size of a ~4.8 bits/weight quant
	Fit       hw.Fit    `json:"fit"`    // of that quant at the default context
	// Score is the base model's Artificial Analysis Intelligence Index, when
	// the repo is a quantization of a rated model; ScoreOf names that model.
	Score   *float64 `json:"score,omitempty"`
	ScoreOf string   `json:"scoreOf,omitempty"`
	// Draft marks a speculative-decoding draft model, which can't chat on
	// its own; DraftFor is the model it speeds up, when known.
	Draft    bool   `json:"draft,omitempty"`
	DraftFor string `json:"draftFor,omitempty"`
}

// discoverCtx is the context length fit estimates assume unless the user
// picks another.
const discoverCtx = 8192

// handleSearch returns every match for a search (up to hf.DefaultSearchMax),
// so the browser can sort and filter the full set without asking again.
func (s *Server) handleSearch(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	params := hf.SearchParams{
		Query:  strings.TrimSpace(q.Get("q")),
		Author: strings.TrimSpace(q.Get("author")),
	}
	repos, truncated, err := s.search.get(params, func() ([]hf.Repo, bool, error) {
		return s.hf.Search(r.Context(), params)
	})
	if err != nil {
		writeError(w, http.StatusBadGateway, err.Error())
		return
	}
	installed := s.installedRepos(r.Context())
	kvf := s.kvFactor(r.Context())
	out := make([]searchResult, 0, len(repos))
	for _, repo := range repos {
		if !repo.IsChatModel() {
			continue
		}
		params := repo.Params()
		moe := hf.IsMoE(repo.Arch(), repo.ID)
		res := searchResult{
			ID: repo.ID, Author: repo.Author(), Name: repo.Name(),
			Params: params, Arch: repo.Arch(), Context: repo.ContextLength(),
			MoE: moe, Vision: repo.Vision(), Gated: repo.IsGated(),
			Downloads: repo.Downloads, Likes: repo.Likes, Trending: repo.Trending,
			Updated:   repo.LastModified,
			Installed: installed[strings.ToLower(repo.ID)],
		}
		res.Score, res.ScoreOf = s.scoreFor(repo.Tags)
		res.Draft, res.DraftFor = repo.IsDraft(), repo.DraftFor()
		if params > 0 {
			res.Q4Size = params * 48 / 80 // ~4.8 bits per weight, typical of Q4_K_M
			res.Fit = hw.Classify(hw.EstimateKV(res.Q4Size, params, discoverCtx, moe, kvf), s.memory)
		}
		if res.Draft {
			// Fitting on the GPU says nothing useful about a draft, and it
			// has no score of its own.
			res.Fit = hw.Fit{Level: "draft", Label: "Draft model"}
			res.Score, res.ScoreOf = nil, ""
		}
		out = append(out, res)
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"results": out, "truncated": truncated, "max": hf.DefaultSearchMax,
		"memory": s.memory, "context": discoverCtx,
	})
}

// scoreFor rates a repo by the model it was quantized from. Finetunes and
// repos with no base model tag are unrated.
func (s *Server) scoreFor(tags []string) (*float64, string) {
	base := scores.BaseModel(tags)
	if base == "" {
		return nil, ""
	}
	e, ok := s.scores.Lookup(base)
	if !ok {
		return nil, ""
	}
	return &e.Score, base
}

// searchCache keeps recent search results for a few minutes, so going back
// to Discover or tweaking a search is instant and doesn't re-page through HF.
type searchCache struct {
	mu      sync.Mutex
	entries map[hf.SearchParams]searchEntry
}

type searchEntry struct {
	repos     []hf.Repo
	truncated bool
	at        time.Time
}

const (
	searchTTL        = 5 * time.Minute
	searchCacheLimit = 20
)

func (c *searchCache) get(key hf.SearchParams, fetch func() ([]hf.Repo, bool, error)) ([]hf.Repo, bool, error) {
	key.Query = strings.ToLower(key.Query)
	key.Author = strings.ToLower(key.Author)
	c.mu.Lock()
	e, ok := c.entries[key]
	c.mu.Unlock()
	if ok && time.Since(e.at) < searchTTL {
		return e.repos, e.truncated, nil
	}

	repos, truncated, err := fetch()
	if err != nil {
		return nil, false, err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.entries == nil {
		c.entries = map[hf.SearchParams]searchEntry{}
	}
	for k, v := range c.entries {
		if time.Since(v.at) >= searchTTL {
			delete(c.entries, k)
		}
	}
	if len(c.entries) >= searchCacheLimit {
		var oldest hf.SearchParams
		first := true
		for k, v := range c.entries {
			if first || v.at.Before(c.entries[oldest].at) {
				oldest, first = k, false
			}
		}
		delete(c.entries, oldest)
	}
	c.entries[key] = searchEntry{repos, truncated, time.Now()}
	return repos, truncated, nil
}

// installedRepos returns the lowercased HF repo IDs that have at least one
// quant pulled into Ollama.
func (s *Server) installedRepos(ctx context.Context) map[string]bool {
	ctx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	models, err := s.ollama.List(ctx)
	out := map[string]bool{}
	if err != nil {
		return out
	}
	for _, m := range models {
		name, ok := strings.CutPrefix(strings.ToLower(m.Name), "hf.co/")
		if !ok {
			continue
		}
		repo, _, _ := strings.Cut(name, ":")
		out[repo] = true
	}
	return out
}

// --- Model page ---

type quantRow struct {
	hf.Quant
	Tag       string // what to pull
	Alias     string // suggested short name
	Fit       hw.Fit
	Installed bool
	Best      bool
}

type setting struct {
	Key   string
	Value any
}

type modelPageData struct {
	Repo        *hf.Repo
	Quants      []quantRow
	MoE         bool
	Vision      bool
	Recommended []setting
	Ctx         int
	CtxOptions  []int
	KVType      string // server K/V cache type; the estimates assume it
	// Drafts are speculative-decoding draft models in the repo, kept apart
	// from the real quants. A repo that is itself a draft has only these.
	Drafts   []quantRow
	Draft    bool
	DraftFor string
	Memory      hw.Memory
	Score       float64 // of the base model, 0 if unrated
	ScoreOf     string
	ScorePct    int // Score as a percentage of the best known score
	ScoreURL    string
	Error       string
	Partial     bool // render only the detail block, for the Discover panel
}

var ctxChoices = []int{4096, 8192, 16384, 32768, 65536, 131072}

func (s *Server) handleModelPage(w http.ResponseWriter, r *http.Request) {
	repoID := r.PathValue("owner") + "/" + r.PathValue("name")
	p := s.basePage(r.Context(), "discover", repoID)
	data := &modelPageData{Memory: s.memory, Ctx: discoverCtx, Partial: r.URL.Query().Get("partial") == "1"}
	p.Data = data
	if !hf.ValidRepo(repoID) {
		data.Error = "That isn't a Hugging Face repo name."
		s.renderModel(w, p, data.Partial)
		return
	}

	// Fetch the three pieces in parallel; each is a separate HF request.
	var (
		wg               sync.WaitGroup
		repo             *hf.Repo
		files            []hf.File
		recommended      map[string]any
		infoErr, fileErr error
	)
	wg.Add(3)
	go func() { defer wg.Done(); repo, infoErr = s.hf.Info(r.Context(), repoID) }()
	go func() { defer wg.Done(); files, fileErr = s.hf.Files(r.Context(), repoID) }()
	go func() { defer wg.Done(); recommended, _ = s.hf.RecommendedParams(r.Context(), repoID) }()
	wg.Wait()

	switch {
	case errors.Is(infoErr, hf.ErrNotFound):
		w.WriteHeader(http.StatusNotFound)
		data.Error = "No such repo on Hugging Face (or it's private)."
		s.renderModel(w, p, data.Partial)
		return
	case infoErr != nil:
		data.Error = infoErr.Error()
		s.renderModel(w, p, data.Partial)
		return
	case fileErr != nil:
		data.Error = "Couldn't list files: " + fileErr.Error()
	}
	data.Repo = repo
	data.MoE = hf.IsMoE(repo.Arch(), repo.ID)
	data.ScoreURL = scores.AttributionURL
	if sc, of := s.scoreFor(repo.Tags); sc != nil && !repo.IsDraft() {
		data.Score, data.ScoreOf = *sc, of
		if top := s.scores.Max(); top > 0 {
			data.ScorePct = min(100, int(*sc/top*100+0.5))
		}
	}

	maxCtx := repo.ContextLength()
	for _, c := range ctxChoices {
		if maxCtx == 0 || c <= maxCtx {
			data.CtxOptions = append(data.CtxOptions, c)
		}
	}
	if c, err := strconv.Atoi(r.URL.Query().Get("ctx")); err == nil && slices.Contains(data.CtxOptions, c) {
		data.Ctx = c
	}

	installed := map[string]bool{}
	for _, m := range p.Models {
		installed[strings.ToLower(m.Name)] = true
	}
	kvf := s.kvFactor(r.Context())
	data.KVType = s.ollamaEnv(r.Context()).KVCacheType()
	quants, vision := hf.Quants(repo.ID, files)
	data.Vision = vision || repo.Vision()
	data.Draft, data.DraftFor = repo.IsDraft(), repo.DraftFor()
	for _, q := range quants {
		if q.Draft || data.Draft {
			data.Drafts = append(data.Drafts, quantRow{Quant: q})
			continue
		}
		tag := hf.OllamaName(repo.ID, q.Name)
		data.Quants = append(data.Quants, quantRow{
			Quant:     q,
			Tag:       tag,
			Alias:     hf.ShortName(repo.ID, q.Name),
			Fit:       hw.Classify(hw.EstimateKV(q.Size, repo.Params(), data.Ctx, data.MoE, kvf), s.memory),
			Installed: installed[strings.ToLower(tag)],
		})
	}
	markBest(data.Quants)

	keys := make([]string, 0, len(recommended))
	for k := range recommended {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		data.Recommended = append(data.Recommended, setting{k, recommended[k]})
	}
	s.renderModel(w, p, data.Partial)
}

// renderModel writes the full page, or just the "model-detail" block when
// the Discover panel asked for a fragment.
func (s *Server) renderModel(w http.ResponseWriter, p *page, partial bool) {
	if !partial {
		s.render(w, "model", p)
		return
	}
	var buf bytes.Buffer
	if err := s.pages["model"].ExecuteTemplate(&buf, "model-detail", p); err != nil {
		s.log.Error("render", "page", "model-detail", "err", err)
		http.Error(w, "template error", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	buf.WriteTo(w)
}

// markBest flags the largest quant that fits fully on the GPU. Without a
// GPU it picks the largest that fits in RAM. Split quants are skipped since
// Ollama may not be able to pull them.
func markBest(rows []quantRow) {
	best := -1
	for i, q := range rows {
		if q.Split || q.Bits() == 0 {
			continue
		}
		if q.Fit.Level == hw.FitGPU || q.Fit.Level == hw.FitCPU {
			best = i // rows are sorted by size, so the last match is the largest
		}
	}
	if best >= 0 {
		rows[best].Best = true
	}
}

// --- downloads ---

var modelNamePattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._/:-]{0,250}$`)

func (s *Server) handlePullList(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, s.pulls.List())
}

func (s *Server) handlePullStart(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Model string `json:"model"`
		Alias string `json:"alias"`
	}
	if !readJSON(w, r, &req) {
		return
	}
	if !modelNamePattern.MatchString(req.Model) || strings.Contains(req.Model, "..") {
		writeError(w, http.StatusBadRequest, "invalid model name")
		return
	}
	if req.Alias != "" && (!modelNamePattern.MatchString(req.Alias) || strings.Contains(req.Alias, "..")) {
		writeError(w, http.StatusBadRequest, "invalid short name: use letters, numbers, '.', '-', '_' and one ':'")
		return
	}
	writeJSON(w, http.StatusAccepted, s.pulls.Start(req.Model, req.Alias))
}

func (s *Server) handlePullCancel(w http.ResponseWriter, r *http.Request) {
	if !s.pulls.Cancel(r.PathValue("id")) {
		writeError(w, http.StatusNotFound, "no running download with that id")
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

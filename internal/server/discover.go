package server

import (
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
	Updated   time.Time `json:"updated"`
	Installed bool      `json:"installed"`
	Q4Size    int64     `json:"q4Size"` // estimated size of a ~4.8 bits/weight quant
	Fit       hw.Fit    `json:"fit"`    // of that quant at the default context
}

// discoverCtx is the context length fit estimates assume unless the user
// picks another.
const discoverCtx = 8192

func (s *Server) handleSearch(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	repos, err := s.hf.Search(r.Context(), hf.SearchParams{
		Query:  strings.TrimSpace(q.Get("q")),
		Author: strings.TrimSpace(q.Get("author")),
		Sort:   q.Get("sort"),
	})
	if err != nil {
		writeError(w, http.StatusBadGateway, err.Error())
		return
	}
	installed := s.installedRepos(r.Context())
	out := make([]searchResult, 0, len(repos))
	for _, repo := range repos {
		params := repo.Params()
		moe := hf.IsMoE(repo.Arch(), repo.ID)
		res := searchResult{
			ID: repo.ID, Author: repo.Author(), Name: repo.Name(),
			Params: params, Arch: repo.Arch(), Context: repo.ContextLength(),
			MoE: moe, Vision: repo.Vision(), Gated: repo.IsGated(),
			Downloads: repo.Downloads, Likes: repo.Likes, Updated: repo.LastModified,
			Installed: installed[strings.ToLower(repo.ID)],
		}
		if params > 0 {
			res.Q4Size = params * 48 / 80 // ~4.8 bits per weight, typical of Q4_K_M
			res.Fit = hw.Classify(hw.Estimate(res.Q4Size, params, discoverCtx, moe), s.memory)
		}
		out = append(out, res)
	}
	writeJSON(w, http.StatusOK, map[string]any{"results": out, "memory": s.memory, "context": discoverCtx})
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
	Memory      hw.Memory
	Error       string
}

var ctxChoices = []int{4096, 8192, 16384, 32768, 65536, 131072}

func (s *Server) handleModelPage(w http.ResponseWriter, r *http.Request) {
	repoID := r.PathValue("owner") + "/" + r.PathValue("name")
	p := s.basePage(r.Context(), "discover", repoID)
	data := &modelPageData{Memory: s.memory, Ctx: discoverCtx}
	p.Data = data
	if !hf.ValidRepo(repoID) {
		data.Error = "That isn't a Hugging Face repo name."
		s.render(w, "model", p)
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
		s.render(w, "model", p)
		return
	case infoErr != nil:
		data.Error = infoErr.Error()
		s.render(w, "model", p)
		return
	case fileErr != nil:
		data.Error = "Couldn't list files: " + fileErr.Error()
	}
	data.Repo = repo
	data.MoE = hf.IsMoE(repo.Arch(), repo.ID)

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
	quants, vision := hf.Quants(files)
	data.Vision = vision || repo.Vision()
	for _, q := range quants {
		tag := hf.OllamaName(repo.ID, q.Name)
		data.Quants = append(data.Quants, quantRow{
			Quant:     q,
			Tag:       tag,
			Alias:     hf.ShortName(repo.ID, q.Name),
			Fit:       hw.Classify(hw.Estimate(q.Size, repo.Params(), data.Ctx, data.MoE), s.memory),
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
	s.render(w, "model", p)
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

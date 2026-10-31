package server

import (
	"context"
	"math"
	"net/http"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"assay/internal/engine"
	"assay/internal/hw"
	"assay/internal/scores"
)

// libraryData is the Library page: the stat strip and one row per model.
type libraryData struct {
	Rows     []libRow
	Untested int // models without a speed test (embedding models don't count)
	Live     hw.Live
	HasGPU   bool
	// Disk is the models folder: bytes in use by the library, bytes free on
	// its disk (FreeKnown false when the OS can't say), and the percent of
	// that disk the library fills for the bar.
	DiskUsed, DiskFree int64
	FreeKnown          bool
	DiskPct            int
	ScoreURL           string
}

// libRow is one model in the Library table.
type libRow struct {
	Group modelGroup
	// Title is the readable name, Tags its chips (QAT, MoE…); the raw name
	// stays in Group.Primary.Name.
	Title string
	Tags  []string
	Quant string
	Size  int64
	// Speed is the measured writing speed, 0 when untested.
	Speed     float64
	SpeedBest bool
	Embedding bool
	MemNeed   int64
	FitLevel  string
	FitShort  string
	FitTitle  string
	FitPct    int
	// Score comes from the cache; when it isn't there yet the page asks
	// /api/library/scores and fills it in.
	Score float64
	Rated bool
}

// GB formats bytes as gigabytes with one decimal, no unit.
func (libRow) GB(n int64) string { return strconv.FormatFloat(float64(n)/1e9, 'f', 1, 64) }

// SpeedText is the speed as a whole number.
func (r libRow) SpeedText() string { return strconv.FormatFloat(r.Speed, 'f', 0, 64) }

// ScoreText is the score as a whole number.
func (r libRow) ScoreText() string { return strconv.FormatFloat(r.Score, 'f', 0, 64) }

// RawName is the identifier of the model, for URLs and data attributes.
func (r libRow) RawName() string { return r.Group.Primary.Name }

// Bytes helpers for the stat strip.
func (d *libraryData) GB(n int64) string {
	if n >= 100e9 {
		return strconv.FormatFloat(float64(n)/1e9, 'f', 0, 64)
	}
	return strconv.FormatFloat(float64(n)/1e9, 'f', 1, 64)
}

// Pct is used/total as a percentage, capped to 100.
func (d *libraryData) Pct(used, total int64) int {
	if total <= 0 {
		return 0
	}
	return int(math.Min(100, float64(used)*100/float64(total)+0.5))
}

func isEmbeddingName(name string) bool { return strings.Contains(strings.ToLower(name), "embed") }

var fitShort = map[string]string{
	hw.FitGPU: "GPU", hw.FitTight: "Tight", hw.FitPartial: "+ RAM", hw.FitCPU: "CPU", hw.FitNone: "Too big",
}

func (s *Server) handleLibrary(w http.ResponseWriter, r *http.Request) {
	p := s.basePage(r.Context(), "library", "Library")
	p.linkNote = s.linkNoteOf
	p.Adapters = s.adapterRows()
	p.Data = s.libraryData(p)
	s.render(w, "library", p)
}

func (s *Server) libraryData(p *page) *libraryData {
	d := &libraryData{ScoreURL: scores.AttributionURL, Live: readLiveGPU()}
	if d.Live.VRAMTotal == 0 {
		d.Live.VRAMTotal = s.memory.VRAM
	}
	if d.Live.RAMTotal == 0 {
		d.Live.RAMTotal = s.memory.RAM
	}
	d.HasGPU = d.Live.VRAMTotal > 0
	d.DiskUsed = p.DiskUsed()
	if mf, ok := s.ollama.(modelFolder); ok {
		if free, err := engine.DiskFree(mf.Store().Blob("")); err == nil {
			d.DiskFree, d.FreeKnown = free, true
			d.DiskPct = d.Pct(d.DiskUsed, d.DiskUsed+free)
		}
	}
	var best float64
	for _, g := range p.ModelGroups() {
		title, tags := prettyModel(g.Primary.Name)
		if g.HFRepo != "" && g.PrimaryOwner() != "" {
			// An hf.co primary has no short alias; its repo name reads best.
			title, tags = prettyModel(g.ShortName())
		}
		row := libRow{
			Group: g, Title: title, Tags: tags, Quant: g.Primary.Details.QuantizationLevel, Size: g.Primary.Size,
			Embedding: isEmbeddingName(g.Primary.Name),
			MemNeed:   g.Fit.Need, FitLevel: g.Fit.Level, FitShort: fitShort[g.Fit.Level], FitPct: g.Fit.Percent,
		}
		row.FitTitle = g.Fit.Label
		if g.Bench != nil && g.Bench.GenTPS > 0 {
			row.Speed = g.Bench.GenTPS
			best = max(best, row.Speed)
		} else if !row.Embedding {
			d.Untested++
		}
		if g.HFRepo != "" {
			if ls, ok := s.libScores.Load(strings.ToLower(g.HFRepo)); ok && ls.(libScore).ok {
				row.Score, row.Rated = ls.(libScore).score, true
			}
		}
		d.Rows = append(d.Rows, row)
	}
	for i := range d.Rows {
		d.Rows[i].SpeedBest = best > 0 && d.Rows[i].Speed == best
	}
	return d
}

// libScore is what we learned about a repo's intelligence score.
type libScore struct {
	score  float64
	of     string // the base model's repo the score belongs to
	ok     bool
	reason string // why not: "no-base", "unlisted", "offline"
	at     time.Time
}

const (
	libScoreTTL    = 24 * time.Hour
	libScoreErrTTL = time.Minute
)

// libScoreOf finds the score of a repo, asking Hugging Face for its base
// model once and remembering the answer.
func (s *Server) libScoreOf(ctx context.Context, repo string) libScore {
	key := strings.ToLower(repo)
	if v, ok := s.libScores.Load(key); ok {
		ls := v.(libScore)
		ttl := libScoreTTL
		if ls.reason == "offline" {
			ttl = libScoreErrTTL
		}
		if time.Since(ls.at) < ttl {
			return ls
		}
	}
	ls := libScore{at: time.Now()}
	switch {
	case s.scores == nil || s.hf == nil:
		ls.reason = "unlisted"
	default:
		ictx, cancel := context.WithTimeout(ctx, 6*time.Second)
		info, err := s.hf.Info(ictx, repo)
		cancel()
		if err != nil {
			ls.reason = "offline"
			break
		}
		base := scores.BaseModel(info.Tags)
		if base == "" {
			ls.reason = "no-base"
			break
		}
		ls.of = base
		if sc, _ := s.scoreFor(info.Tags); sc != nil {
			ls.score, ls.ok = *sc, true
		} else {
			ls.reason = "unlisted"
		}
	}
	s.libScores.Store(key, ls)
	return ls
}

type scoreJSON struct {
	Score  float64 `json:"score,omitempty"`
	Of     string  `json:"of,omitempty"`
	Pct    int     `json:"pct,omitempty"`
	Rated  bool    `json:"rated"`
	Reason string  `json:"reason,omitempty"`
	Repo   string  `json:"repo,omitempty"`
}

// handleLibScores answers {scores: {model name: …}, max, url} for every
// installed model, asking Hugging Face (a few at a time) for what isn't
// cached yet.
func (s *Server) handleLibScores(w http.ResponseWriter, r *http.Request) {
	models, err := s.ollama.List(r.Context())
	if err != nil {
		writeError(w, ollamaStatus(err), err.Error())
		return
	}
	groups := groupModels(models)
	var mu sync.Mutex
	out := make(map[string]scoreJSON, len(groups))
	var top float64
	if s.scores != nil {
		top = s.scores.Max()
	}
	var wg sync.WaitGroup
	sem := make(chan struct{}, 4)
	for _, g := range groups {
		g := g
		wg.Add(1)
		go func() {
			defer wg.Done()
			sj := scoreJSON{Repo: g.HFRepo}
			if g.HFRepo == "" {
				sj.Reason = "local"
			} else {
				sem <- struct{}{}
				ls := s.libScoreOf(r.Context(), g.HFRepo)
				<-sem
				sj.Rated, sj.Score, sj.Of, sj.Reason = ls.ok, ls.score, ls.of, ls.reason
				if ls.ok && top > 0 {
					sj.Pct = min(100, int(ls.score/top*100+0.5))
				}
			}
			mu.Lock()
			out[g.Primary.Name] = sj
			mu.Unlock()
		}()
	}
	wg.Wait()
	w.Header().Set("Cache-Control", "no-store")
	writeJSON(w, http.StatusOK, map[string]any{"scores": out, "max": top, "url": scores.AttributionURL})
}

var paramInName = regexp.MustCompile(`(?:^|[-_.])(\d+(?:\.\d+)?)b(?:$|[-_.])`)

// billionsIn reads a parameter count out of a model id ("qwen3-8b" is 8).
func billionsIn(id string) float64 {
	_, name, ok := strings.Cut(strings.ToLower(id), "/")
	if !ok {
		name = strings.ToLower(id)
	}
	if m := paramInName.FindStringSubmatch(name); m != nil {
		v, _ := strconv.ParseFloat(m[1], 64)
		return v
	}
	return 0
}

type nearEntry struct {
	Model     string  `json:"model"`
	Score     float64 `json:"score"`
	You       bool    `json:"you,omitempty"`
	Installed bool    `json:"installed,omitempty"`
}

// nearScores picks the scores around one in a list of similar-size models:
// two above, two below, and the model itself, best first. rank and n say
// where it sits among all of them.
func nearScores(all []nearEntry, mine nearEntry) (list []nearEntry, rank, n int) {
	above, below := []nearEntry{}, []nearEntry{}
	for _, e := range all {
		if e.Score > mine.Score {
			above = append(above, e)
		} else {
			below = append(below, e)
		}
	}
	sort.Slice(above, func(i, j int) bool { return above[i].Score < above[j].Score }) // nearest first
	sort.Slice(below, func(i, j int) bool { return below[i].Score > below[j].Score })
	rank, n = len(above)+1, len(all)+1
	a := above[:min(2, len(above))]
	b := below[:min(2, len(below))]
	for i := len(a) - 1; i >= 0; i-- {
		list = append(list, a[i])
	}
	list = append(list, mine)
	list = append(list, b...)
	return list, rank, n
}

// handleLibNearby shows a model's score among models of a similar size,
// from the cached score data and what's installed.
func (s *Server) handleLibNearby(w http.ResponseWriter, r *http.Request) {
	name := r.URL.Query().Get("name")
	models, err := s.ollama.List(r.Context())
	if err != nil {
		writeError(w, ollamaStatus(err), err.Error())
		return
	}
	var g *modelGroup
	groups := groupModels(models)
	for i := range groups {
		if groups[i].Has(name) {
			g = &groups[i]
			break
		}
	}
	if g == nil {
		writeError(w, http.StatusNotFound, "no such model: "+name)
		return
	}
	ls := libScore{}
	if g.HFRepo != "" {
		ls = s.libScoreOf(r.Context(), g.HFRepo)
	}
	resp := map[string]any{"url": scores.AttributionURL, "repo": g.HFRepo, "rated": ls.ok, "reason": ls.reason}
	if g.HFRepo == "" {
		resp["reason"] = "local"
	}
	resp["of"] = ls.of
	if !ls.ok {
		writeJSON(w, http.StatusOK, resp)
		return
	}
	params := float64(parseParams(g.Primary.Details.ParameterSize)) / 1e9
	if params == 0 {
		params = billionsIn(ls.of)
	}
	// Installed models' bases, to mark them in the list.
	installed := map[string]bool{}
	for _, o := range groups {
		if o.HFRepo == "" || o.HFRepo == g.HFRepo {
			continue
		}
		if v, ok := s.libScores.Load(strings.ToLower(o.HFRepo)); ok && v.(libScore).ok {
			installed[strings.ToLower(v.(libScore).of)] = true
		}
	}
	mine := nearEntry{Model: afterSlash(ls.of), Score: ls.score, You: true}
	var near []nearEntry
	if params > 0 {
		for _, e := range s.scores.All() {
			if strings.EqualFold(e.Model, ls.of) {
				continue
			}
			if b := billionsIn(e.Model); b >= params*0.65 && b <= params*1.55 {
				near = append(near, nearEntry{Model: afterSlash(e.Model), Score: e.Score, Installed: installed[strings.ToLower(e.Model)]})
			}
		}
	}
	list, rank, n := nearScores(near, mine)
	resp["score"], resp["of"], resp["params"], resp["near"], resp["rank"], resp["count"] = ls.score, ls.of, params, list, rank, n
	writeJSON(w, http.StatusOK, resp)
}

func afterSlash(id string) string {
	if _, n, ok := strings.Cut(id, "/"); ok {
		return n
	}
	return id
}

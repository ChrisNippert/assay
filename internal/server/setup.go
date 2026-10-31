package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"assay/internal/hf"
	"assay/internal/hw"
	"assay/internal/llamacpp"
)

// The guided first-run setup (/setup): what the computer can run, which
// llama.cpp to install, a first model. Nothing is installed by these
// endpoints except through the same calls Settings and Discover use
// (/api/llamacpp/install, /api/pulls), each started by an explicit click.

// setupState remembers that setup was finished or skipped, in a small file
// in the config folder (in memory when path is empty).
type setupState struct {
	mu   sync.Mutex
	path string
	done bool
}

func openSetupState(path string) *setupState {
	st := &setupState{path: path}
	if path != "" {
		if _, err := os.Stat(path); err == nil {
			st.done = true
		}
	}
	return st
}

func (st *setupState) Done() bool {
	st.mu.Lock()
	defer st.mu.Unlock()
	return st.done
}

func (st *setupState) Set(done bool) error {
	st.mu.Lock()
	defer st.mu.Unlock()
	st.done = done
	if st.path == "" {
		return nil
	}
	if !done {
		if err := os.Remove(st.path); err != nil && !os.IsNotExist(err) {
			return err
		}
		return nil
	}
	b, _ := json.Marshal(map[string]any{"done": true, "at": time.Now().UTC()})
	if err := os.MkdirAll(filepath.Dir(st.path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(st.path, b, 0o644)
}

// needsSetup is the first-run condition: setup hasn't been finished or
// skipped, and there is no llama-server to run models or no model to run.
func needsSetup(done, haveEngine bool, models int) bool {
	return !done && (!haveEngine || models == 0)
}

// firstRun says whether to send the browser to /setup. It needs llama.cpp
// as the engine, and stays quiet when the models can't be listed.
func (s *Server) firstRun(ctx context.Context) bool {
	if s.llama == nil || s.setup.Done() {
		return false
	}
	path, _ := s.llama.Resolve()
	ctx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	models, err := s.ollama.List(ctx)
	if err != nil {
		return false
	}
	return needsSetup(false, path != "", len(models))
}

func (s *Server) handleSetupPage(w http.ResponseWriter, r *http.Request) {
	s.render(w, "setup", s.basePage(r.Context(), "setup", "Setup"))
}

// systemInfo is "Your computer" in plain words.
type systemInfo struct {
	OS     string   `json:"os"`
	Arch   string   `json:"arch"`
	GPUs   []gpuRow `json:"gpus"`
	VRAM   int64    `json:"vram"`
	RAM    int64    `json:"ram"`
	Advice string   `json:"advice"`
}

type gpuRow struct {
	Name string `json:"name"`
	VRAM int64  `json:"vram"`
}

var osNames = map[string]string{"linux": "Linux", "windows": "Windows", "darwin": "macOS"}

func osName(goos string) string {
	if n, ok := osNames[goos]; ok {
		return n
	}
	return goos
}

func (s *Server) systemInfo() systemInfo {
	m := s.memory
	info := systemInfo{OS: osName(runtime.GOOS), Arch: runtime.GOARCH, VRAM: m.VRAM, RAM: m.RAM, GPUs: []gpuRow{}}
	for _, c := range m.Cards {
		info.GPUs = append(info.GPUs, gpuRow{Name: c.Name, VRAM: c.Total})
	}
	if len(info.GPUs) == 0 && m.GPU != "" {
		info.GPUs = append(info.GPUs, gpuRow{Name: m.GPU, VRAM: m.VRAM})
	}
	info.Advice = modelAdvice(m)
	return info
}

// reachSteps are model sizes (billions of parameters) the advice talks about.
var reachSteps = []int64{1, 2, 4, 8, 12, 14, 20, 27, 32, 70, 120}

// reach is the largest model size, in billions of parameters, that runs
// wholly on the GPU and the largest that runs at all (spilling into RAM),
// for a ~4.8 bits per weight quant at the default context.
func reach(m hw.Memory) (gpu, any int64) {
	for _, b := range reachSteps {
		params := b * 1_000_000_000
		size := params * 48 / 80
		switch hw.Classify(hw.Estimate(size, params, discoverCtx, false), m).Level {
		case hw.FitGPU:
			gpu, any = b, b
		case hw.FitTight, hw.FitPartial, hw.FitCPU:
			any = b
		}
	}
	return
}

func gb(n int64) string {
	g := float64(n) / 1e9
	if g >= 10 {
		return fmt.Sprintf("%.0f GB", g)
	}
	return fmt.Sprintf("%.1f GB", g)
}

// modelAdvice says in plain words what the machine can run.
func modelAdvice(m hw.Memory) string {
	gpu, anyB := reach(m)
	switch {
	case m.VRAM == 0 && anyB == 0:
		return "No graphics card was found and there isn't much memory, so even small models will be slow. Start with the smallest model offered."
	case m.VRAM == 0:
		return fmt.Sprintf("No supported graphics card was found, so models run on the processor from your %s of RAM. That works but is slow: models up to about %dB parameters fit, and the small ones (1–4B) feel the most responsive.", gb(m.RAM), anyB)
	case gpu == 0:
		return fmt.Sprintf("Your graphics card has %s of memory, which is little for language models. Small ones will mostly run from RAM, up to about %dB parameters, slowly.", gb(m.VRAM), anyB)
	case anyB > gpu:
		return fmt.Sprintf("Your %s of video memory runs models up to about %dB parameters entirely on the graphics card, which is the fast way. Larger ones, up to about %dB, spill into your %s of RAM and run slower.", gb(m.VRAM), gpu, anyB, gb(m.RAM))
	}
	return fmt.Sprintf("Your %s of video memory runs models up to about %dB parameters entirely on the graphics card, which is the fast way.", gb(m.VRAM), gpu)
}

// engineInfo is "Engine" for the setup page.
type setupEngine struct {
	// Managed is false when llama.cpp isn't the engine, and nothing here applies.
	Managed   bool            `json:"managed"`
	Found     bool            `json:"found"`
	Path      string          `json:"path,omitempty"`
	Source    string          `json:"source,omitempty"`
	Probe     *llamacpp.Probe `json:"probe,omitempty"`
	Supported bool            `json:"supported"`
	InstallTo string          `json:"installTo,omitempty"`
	License   string          `json:"license"`
	Project   string          `json:"project"`
	// ModelsDir is where downloaded models go.
	ModelsDir string `json:"modelsDir,omitempty"`
}

func (s *Server) setupEngine(ctx context.Context) setupEngine {
	e := setupEngine{License: "MIT", Project: "https://github.com/ggml-org/llama.cpp"}
	if f, ok := s.ollama.(modelFolder); ok {
		e.ModelsDir = f.Store().Dir()
	}
	if s.llama == nil {
		return e
	}
	e.Managed = true
	e.Path, e.Source = s.llama.Resolve()
	e.Found = e.Path != ""
	e.InstallTo = s.llama.Inst.Dir
	goos, goarch := runtime.GOOS, runtime.GOARCH
	e.Supported = llamacpp.Supported(goos, goarch)
	if e.Found {
		e.Probe = s.llama.Probe(ctx, e.Path)
	}
	return e
}

// handleSetup is everything the first two steps show, without asking GitHub
// or Hugging Face.
func (s *Server) handleSetup(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()
	names := []string{}
	if models, err := s.ollama.List(ctx); err == nil {
		for _, m := range models {
			names = append(names, m.Name)
		}
		sort.Strings(names)
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"done":   s.setup.Done(),
		"system": s.systemInfo(),
		"engine": s.setupEngine(ctx),
		"models": names,
	})
}

// recommendBuild is the build of c that suits this machine's GPUs.
func (s *Server) recommendBuild(c *llamacpp.Check) string {
	var names []string
	for _, g := range s.systemInfo().GPUs {
		names = append(names, g.Name)
	}
	var builds []llamacpp.Build
	for _, b := range c.Builds {
		builds = append(builds, llamacpp.Build{ID: b.ID})
	}
	return llamacpp.Recommend(builds, names)
}

// handleSetupDone records that setup was finished or skipped, or (with
// done false) that it should show again.
func (s *Server) handleSetupDone(w http.ResponseWriter, r *http.Request) {
	req := struct {
		Done bool `json:"done"`
	}{Done: true}
	if !readJSON(w, r, &req) {
		return
	}
	if err := s.setup.Set(req.Done); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// --- A first model ---

// candidate is a model worth suggesting to a newcomer, if its repo still
// exists with a quant we can pick.
type candidate struct {
	Repo   string
	Family string
	Params int64 // for the context estimate
}

// firstModels are well-known small instruct models, smallest first. The
// list is checked against Hugging Face when asked: a repo that is gone or
// has no usable quant is simply left out.
var firstModels = []candidate{
	{"unsloth/Qwen3.5-0.8B-GGUF", "Qwen", 800_000_000},
	{"unsloth/Qwen3.5-2B-GGUF", "Qwen", 2_000_000_000},
	{"unsloth/gemma-4-E2B-it-qat-GGUF", "Gemma", 2_000_000_000},
	{"unsloth/Qwen3.5-4B-GGUF", "Qwen", 4_000_000_000},
	{"unsloth/gemma-4-E4B-it-qat-GGUF", "Gemma", 4_000_000_000},
	{"unsloth/Qwen3.5-9B-GGUF", "Qwen", 9_000_000_000},
	{"unsloth/gemma-4-12B-it-qat-GGUF", "Gemma", 12_000_000_000},
}

// suggestion is one model offered in the setup's third step.
type suggestion struct {
	Repo   string   `json:"repo"`
	Name   string   `json:"name"`
	Family string   `json:"family"`
	Quant  string   `json:"quant"`
	Model  string   `json:"model"` // what to pull: hf.co/repo:quant
	Alias  string   `json:"alias"`
	Size   int64    `json:"size"`
	Files  []string `json:"files"`
	Fit    hw.Fit   `json:"fit"`
	Why    string   `json:"why"`
	URL    string   `json:"url"`
}

// quantPrefs are the quants to prefer, best first: good quality at about
// 4 bits per weight.
var quantPrefs = []string{"UD-Q4_K_XL", "Q4_K_M", "UD-Q4_K_M", "Q4_0", "IQ4_XS", "Q4_K_S", "Q5_K_M"}

// pickQuant chooses the quant to offer from a repo's.
func pickQuant(quants []hf.Quant) (hf.Quant, bool) {
	var real []hf.Quant
	for _, q := range quants {
		if !q.Draft {
			real = append(real, q)
		}
	}
	for _, want := range quantPrefs {
		for _, q := range real {
			if q.Name == want {
				return q, true
			}
		}
	}
	for _, q := range real { // sorted by size: the smallest of about 4 bits
		if q.Bits() == 4 {
			return q, true
		}
	}
	return hf.Quant{}, false
}

// rankSuggestions picks up to four from what fits comfortably (fully on the
// GPU, or in RAM without one): the most capable one, the best of the other
// family, and a small quick one. Failing that, the smallest that runs at all.
func rankSuggestions(all []suggestion) []suggestion {
	var ok []suggestion
	for _, c := range all {
		if c.Fit.Level == hw.FitGPU || c.Fit.Level == hw.FitCPU {
			ok = append(ok, c)
		}
	}
	if len(ok) == 0 {
		for _, c := range all {
			if c.Fit.Level == hw.FitTight || c.Fit.Level == hw.FitPartial {
				ok = append(ok, c)
			}
		}
		if len(ok) == 0 {
			return nil
		}
		sort.Slice(ok, func(i, j int) bool { return ok[i].Size < ok[j].Size })
		ok[0].Why = "Everything here is a stretch for this computer; this is the smallest, so it has the best chance of running well."
		return ok[:1]
	}
	sort.Slice(ok, func(i, j int) bool { return ok[i].Size > ok[j].Size })
	var out []suggestion
	used := map[string]bool{}
	add := func(c suggestion, why string) {
		if !used[c.Repo] {
			used[c.Repo] = true
			c.Why = why
			out = append(out, c)
		}
	}
	top := ok[0]
	where := "fits on your graphics card with room to spare"
	if top.Fit.Level == hw.FitCPU {
		where = "fits in your RAM"
	}
	add(top, "The most capable one that "+where+".")
	for _, c := range ok[1:] {
		if c.Family != top.Family {
			add(c, fmt.Sprintf("From a different family (%s), the best of its kind that fits, so you can compare.", c.Family))
			break
		}
	}
	for _, c := range ok[1:] {
		if c.Size*100 <= top.Size*45 && !used[c.Repo] {
			add(c, "Small and quick: replies start almost at once, good for trying things out.")
			break
		}
	}
	// Largest first among the rest, if there are fewer than three.
	for _, c := range ok[1:] {
		if len(out) >= 3 {
			break
		}
		add(c, "A smaller alternative that leaves more memory free.")
	}
	return out
}

// buildSuggestions looks the candidates up on Hugging Face (read only).
func (s *Server) buildSuggestions(ctx context.Context) []suggestion {
	kvf := s.kvFactor(ctx)
	res := make([]*suggestion, len(firstModels))
	var wg sync.WaitGroup
	for i, c := range firstModels {
		wg.Add(1)
		go func() {
			defer wg.Done()
			files, err := s.hf.Files(ctx, c.Repo)
			if err != nil {
				return
			}
			quants, _ := hf.Quants(c.Repo, files)
			q, ok := pickQuant(quants)
			if !ok {
				return
			}
			_, name, _ := strings.Cut(c.Repo, "/")
			name = strings.TrimSuffix(name, "-GGUF")
			res[i] = &suggestion{
				Repo: c.Repo, Name: name, Family: c.Family, Quant: q.Name,
				Model: hf.OllamaName(c.Repo, q.Name), Alias: hf.ShortName(c.Repo, q.Name),
				Size: q.Size, Files: q.Files,
				Fit: hw.Classify(hw.EstimateKV(q.Size, c.Params, discoverCtx, false, kvf), s.memory),
				URL: "https://huggingface.co/" + c.Repo,
			}
		}()
	}
	wg.Wait()
	var all []suggestion
	for _, r := range res {
		if r != nil {
			all = append(all, *r)
		}
	}
	return rankSuggestions(all)
}

// handleSetupModels suggests first models that fit this machine.
func (s *Server) handleSetupModels(w http.ResponseWriter, r *http.Request) {
	s.suggMu.Lock()
	cached, at := s.sugg, s.suggAt
	s.suggMu.Unlock()
	if cached == nil || time.Since(at) > 10*time.Minute || r.URL.Query().Get("fresh") == "1" {
		ctx, cancel := context.WithTimeout(r.Context(), 25*time.Second)
		defer cancel()
		list := s.buildSuggestions(ctx)
		if len(list) == 0 {
			writeError(w, http.StatusBadGateway, "couldn't find suitable models on Hugging Face right now (offline, or nothing fits this computer)")
			return
		}
		cached = list
		s.suggMu.Lock()
		s.sugg, s.suggAt = list, time.Now()
		s.suggMu.Unlock()
	}
	dir := ""
	if f, ok := s.ollama.(modelFolder); ok {
		dir = f.Store().Dir()
	}
	writeJSON(w, http.StatusOK, map[string]any{"models": cached, "modelsDir": dir})
}

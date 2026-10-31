package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"assay/internal/hf"
	"assay/internal/hw"
	"assay/internal/llamacpp"
)

func TestNeedsSetup(t *testing.T) {
	for _, c := range []struct {
		done, engine bool
		models       int
		want         bool
	}{
		{false, false, 0, true}, // brand new
		{false, true, 0, true},  // engine but nothing to run
		{false, false, 3, true}, // models but no engine
		{false, true, 3, false}, // everything there
		{true, false, 0, false}, // finished or skipped
		{true, true, 0, false},
	} {
		if got := needsSetup(c.done, c.engine, c.models); got != c.want {
			t.Errorf("done=%v engine=%v models=%d: got %v", c.done, c.engine, c.models, got)
		}
	}
}

func TestSetupStateFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config", "setup.json")
	st := openSetupState(path)
	if st.Done() {
		t.Fatal("done before anything happened")
	}
	if err := st.Set(true); err != nil {
		t.Fatal(err)
	}
	if !openSetupState(path).Done() {
		t.Error("the flag didn't survive a restart")
	}
	if err := st.Set(false); err != nil {
		t.Fatal(err)
	}
	if openSetupState(path).Done() {
		t.Error("clearing the flag didn't stick")
	}
	mem := openSetupState("")
	mem.Set(true)
	if !mem.Done() {
		t.Error("in-memory state")
	}
}

func TestPickQuant(t *testing.T) {
	q := func(names ...string) []hf.Quant {
		var out []hf.Quant
		for _, n := range names {
			out = append(out, hf.Quant{Name: n, Size: 1})
		}
		return out
	}
	for _, c := range []struct {
		have []hf.Quant
		want string
	}{
		{q("Q2_K", "Q4_K_M", "UD-Q4_K_XL", "Q8_0"), "UD-Q4_K_XL"},
		{q("Q3_K_M", "Q4_K_M", "Q8_0"), "Q4_K_M"},
		{q("Q4_0", "Q8_0"), "Q4_0"},
		{q("Q4_1", "Q8_0"), "Q4_1"}, // any 4-bit
		{[]hf.Quant{{Name: "Q4_K_M", Draft: true}, {Name: "Q4_0"}}, "Q4_0"},
		{q("BF16", "Q8_0"), ""},
	} {
		got, ok := pickQuant(c.have)
		if got.Name != c.want || ok != (c.want != "") {
			t.Errorf("%v: got %q %v, want %q", c.have, got.Name, ok, c.want)
		}
	}
}

func fitFor(mem hw.Memory, size, params int64) hw.Fit {
	return hw.Classify(hw.Estimate(size, params, discoverCtx, false), mem)
}

func TestRankSuggestions(t *testing.T) {
	cands := []struct {
		repo, family string
		size, params int64
	}{
		{"q08", "Qwen", 0.6e9, 0.8e9}, {"q2", "Qwen", 1.3e9, 2e9}, {"g2", "Gemma", 2.6e9, 2e9},
		{"q4", "Qwen", 2.9e9, 4e9}, {"g4", "Gemma", 4.2e9, 4e9}, {"q9", "Qwen", 5.6e9, 9e9}, {"g12", "Gemma", 6.7e9, 12e9},
	}
	build := func(mem hw.Memory) []suggestion {
		var all []suggestion
		for _, c := range cands {
			all = append(all, suggestion{Repo: c.repo, Family: c.family, Size: c.size, Fit: fitFor(mem, c.size, c.params)})
		}
		return rankSuggestions(all)
	}
	ids := func(l []suggestion) string {
		var s []string
		for _, x := range l {
			s = append(s, x.Repo)
		}
		return strings.Join(s, ",")
	}

	// 12 GB card: the biggest that fits, the best of the other family, a small one.
	got := build(hw.Memory{VRAM: 12e9, RAM: 48e9})
	if ids(got) != "g12,q9,q4" {
		t.Errorf("12 GB: %s", ids(got))
	}
	for _, g := range got {
		if g.Why == "" || g.Fit.Level != hw.FitGPU {
			t.Errorf("%s: why %q fit %q", g.Repo, g.Why, g.Fit.Level)
		}
	}
	// 4 GB card: only small ones fit on the GPU.
	got = build(hw.Memory{VRAM: 4e9, RAM: 16e9})
	for _, g := range got {
		if g.Fit.Level != hw.FitGPU {
			t.Errorf("4 GB offered %s that doesn't fit the GPU (%s)", g.Repo, g.Fit.Level)
		}
	}
	if len(got) == 0 || got[0].Repo == "g12" || got[0].Repo == "q9" {
		t.Errorf("4 GB: %s", ids(got))
	}
	// No GPU: fits in RAM counts, and says so.
	got = build(hw.Memory{RAM: 16e9})
	if len(got) == 0 || got[0].Repo != "g12" || !strings.Contains(got[0].Why, "RAM") {
		t.Errorf("CPU only: %s %+v", ids(got), got)
	}
	// Nothing comfortable: just the smallest that runs at all.
	got = build(hw.Memory{VRAM: 1.2e9, RAM: 2.4e9})
	if len(got) > 1 {
		t.Errorf("tiny machine: %s", ids(got))
	}
	// Nothing at all.
	if got = build(hw.Memory{VRAM: 0.3e9, RAM: 0.3e9}); got != nil {
		t.Errorf("hopeless machine: %s", ids(got))
	}
}

func TestModelAdvice(t *testing.T) {
	for _, c := range []struct {
		mem  hw.Memory
		want string
	}{
		{hw.Memory{VRAM: 12e9, RAM: 48e9}, "entirely on the graphics card"},
		{hw.Memory{VRAM: 12e9, RAM: 48e9}, "spill into your 48 GB of RAM"},
		{hw.Memory{RAM: 16e9}, "No supported graphics card"},
		{hw.Memory{RAM: 0.2e9}, "even small models will be slow"},
	} {
		if got := modelAdvice(c.mem); !strings.Contains(got, c.want) {
			t.Errorf("%+v: %q lacks %q", c.mem, got, c.want)
		}
	}
}

// --- handlers ---

// fakeLlamaGitHub serves a release list with Linux builds only.
func fakeLlamaGitHub(t *testing.T) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		asset := func(n string, size int) map[string]any {
			return map[string]any{"name": n, "browser_download_url": "http://invalid.invalid/" + n, "size": size}
		}
		json.NewEncoder(w).Encode([]any{map[string]any{"tag_name": "b500", "assets": []any{
			asset("llama-b500-bin-ubuntu-vulkan-x64.tar.gz", 36e6),
			asset("llama-b500-bin-ubuntu-x64.tar.gz", 23e6),
			asset("llama-b500-bin-ubuntu-rocm-10.0-x64.tar.gz", 250e6),
		}}})
	}))
	t.Cleanup(srv.Close)
	return srv
}

// withLlama gives s a llama.cpp manager that finds no llama-server and
// reads releases from a fake GitHub (never the real one), under temp dirs.
func withLlama(t *testing.T, s *Server) *llamacpp.Manager {
	t.Helper()
	t.Setenv("PATH", t.TempDir()) // no llama-server on the PATH
	gh := fakeLlamaGitHub(t)
	m := llamacpp.NewManager(t.TempDir(), "", nil, nil)
	m.Inst.API, m.Inst.HTTP, m.Inst.GOOS, m.Inst.GOARCH = gh.URL, gh.Client(), "linux", "amd64"
	s.llama = m
	return m
}

func getJSON(t *testing.T, s *Server, path string, out any) *httptest.ResponseRecorder {
	t.Helper()
	rec := httptest.NewRecorder()
	s.ServeHTTP(rec, httptest.NewRequest("GET", "http://127.0.0.1"+path, nil))
	if out != nil && rec.Code == http.StatusOK {
		if err := json.Unmarshal(rec.Body.Bytes(), out); err != nil {
			t.Fatalf("%s: %v: %s", path, err, rec.Body)
		}
	}
	return rec
}

func TestSetupInfoEndpoint(t *testing.T) {
	s, _, _ := newTestServer(t)
	withLlama(t, s)
	s.memory = hw.Memory{GPU: "AMD Radeon RX 6700 XT", VRAM: 12e9, RAM: 48e9, Cards: []hw.Card{{Name: "AMD Radeon RX 6700 XT", Total: 12e9}}}
	var got struct {
		Done   bool
		System systemInfo
		Engine setupEngine
		Models []string
	}
	getJSON(t, s, "/api/setup", &got)
	if got.Done || len(got.System.GPUs) != 1 || got.System.GPUs[0].Name != "AMD Radeon RX 6700 XT" || got.System.RAM != 48e9 || got.System.Advice == "" {
		t.Errorf("system: %+v", got)
	}
	if !got.Engine.Managed || got.Engine.Found || got.Engine.InstallTo == "" || got.Engine.License != "MIT" {
		t.Errorf("engine: %+v", got.Engine)
	}
	if len(got.Models) != 1 || got.Models[0] != "qwen3:8b" {
		t.Errorf("models: %v", got.Models)
	}

	// A llama-server the user has is reported with its path.
	bin := filepath.Join(t.TempDir(), "llama-server")
	os.WriteFile(bin, []byte("#!/bin/sh\necho 'version: 7 (abc)'\n"), 0o755)
	s.llama.Flag = bin
	getJSON(t, s, "/api/setup", &got)
	if !got.Engine.Found || got.Engine.Path != bin || got.Engine.Source != llamacpp.SourceFlag || got.Engine.Probe == nil || got.Engine.Probe.Version == "" {
		t.Errorf("engine with a binary: %+v", got.Engine)
	}
}

func TestLlamaCheckListsBuildsAndRecommends(t *testing.T) {
	s, _, _ := newTestServer(t)
	withLlama(t, s)
	var amd struct {
		Tag         string
		Recommended string
		Builds      []struct{ ID, Label, URL string }
	}
	s.memory = hw.Memory{VRAM: 12e9, RAM: 48e9, Cards: []hw.Card{{Name: "AMD Radeon RX 6700 XT", Total: 12e9}}}
	getJSON(t, s, "/api/llamacpp/check?fresh=1", &amd)
	if amd.Tag != "b500" || len(amd.Builds) != 3 || amd.Builds[0].ID != "vulkan" || amd.Recommended != "rocm" {
		t.Errorf("amd: %+v", amd)
	}
	s.memory = hw.Memory{VRAM: 8e9, RAM: 16e9, Cards: []hw.Card{{Name: "NVIDIA GeForce RTX 3070", Total: 8e9}}}
	var nv struct{ Recommended string }
	getJSON(t, s, "/api/llamacpp/check", &nv)
	if nv.Recommended != "vulkan" { // no CUDA build in this release
		t.Errorf("nvidia without cuda: %q", nv.Recommended)
	}
	s.memory = hw.Memory{RAM: 16e9}
	getJSON(t, s, "/api/llamacpp/check", &nv)
	if nv.Recommended != "cpu" {
		t.Errorf("no gpu: %q", nv.Recommended)
	}
}

func TestLlamaInstallTakesABuild(t *testing.T) {
	s, _, _ := newTestServer(t)
	m := withLlama(t, s)
	// A build the release doesn't have fails the job without downloading anything.
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "http://127.0.0.1/api/llamacpp/install", strings.NewReader(`{"build":"cuda"}`))
	req.Header.Set("Content-Type", "application/json")
	s.ServeHTTP(rec, req)
	if rec.Code != http.StatusAccepted {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	var job llamacpp.Job
	json.Unmarshal(rec.Body.Bytes(), &job)
	if job.Build != "cuda" {
		t.Errorf("job: %+v", job)
	}
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		var st llamacpp.Status
		getJSON(t, s, "/api/llamacpp", &st)
		if st.Job != nil && st.Job.State == llamacpp.Failed {
			if !strings.Contains(st.Job.Error, "this build") {
				t.Errorf("error: %q", st.Job.Error)
			}
			if m.Inst.Binary() != "" {
				t.Error("something got installed")
			}
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Error("the job never failed")
}

func TestSetupDoneAndFirstRunRedirect(t *testing.T) {
	s, _, _ := newTestServer(t)
	withLlama(t, s)
	s.setup = openSetupState(filepath.Join(t.TempDir(), "setup.json"))

	// No llama-server (the fake backend does list a model): /chat sends you to setup.
	rec := getJSON(t, s, "/chat", nil)
	if rec.Code != http.StatusFound || rec.Header().Get("Location") != "/setup" {
		t.Fatalf("first run: %d %s", rec.Code, rec.Header().Get("Location"))
	}
	// A saved chat link isn't hijacked, and /setup itself renders.
	if rec = getJSON(t, s, "/setup", nil); rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "su-steps") {
		t.Errorf("/setup: %d", rec.Code)
	}

	post := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "http://127.0.0.1/api/setup/done", strings.NewReader(`{}`))
	req.Header.Set("Content-Type", "application/json")
	s.ServeHTTP(post, req)
	if post.Code != http.StatusNoContent || !s.setup.Done() {
		t.Fatalf("done: %d", post.Code)
	}
	if rec = getJSON(t, s, "/chat", nil); rec.Code != http.StatusOK {
		t.Errorf("after setup: %d", rec.Code)
	}
	// Run again: the flag clears.
	post = httptest.NewRecorder()
	req = httptest.NewRequest("POST", "http://127.0.0.1/api/setup/done", strings.NewReader(`{"done":false}`))
	req.Header.Set("Content-Type", "application/json")
	s.ServeHTTP(post, req)
	if s.setup.Done() {
		t.Error("done=false didn't reopen setup")
	}
}

func TestNoRedirectWithEngineAndModels(t *testing.T) {
	s, _, _ := newTestServer(t)
	withLlama(t, s)
	bin := filepath.Join(t.TempDir(), "llama-server")
	os.WriteFile(bin, []byte("#!/bin/sh\n"), 0o755)
	s.llama.Flag = bin
	if rec := getJSON(t, s, "/chat", nil); rec.Code != http.StatusOK {
		t.Errorf("engine and a model present: %d", rec.Code)
	}
}

func TestNoSetupWithoutLlamaCpp(t *testing.T) {
	s, _, _ := newTestServer(t) // no manager: not the llama.cpp engine
	if s.firstRun(t.Context()) {
		t.Error("first run without the llama.cpp engine")
	}
}

func TestSetupModelsEndpoint(t *testing.T) {
	s, _, _ := newTestServer(t)
	// A Hugging Face that has two of the candidates (one with a usable quant)
	// and 404s the rest.
	trees := map[string]string{
		"unsloth/Qwen3.5-9B-GGUF":         `[{"type":"file","path":"Qwen3.5-9B-UD-Q4_K_XL.gguf","size":5600000000},{"type":"file","path":"Qwen3.5-9B-BF16.gguf","size":18000000000},{"type":"file","path":"mmproj-F16.gguf","size":900000000}]`,
		"unsloth/Qwen3.5-2B-GGUF":         `[{"type":"file","path":"Qwen3.5-2B-Q4_K_M.gguf","size":1300000000}]`,
		"unsloth/Qwen3.5-4B-GGUF":         `[{"type":"file","path":"Qwen3.5-4B-BF16.gguf","size":8400000000}]`, // no 4-bit: left out
		"unsloth/gemma-4-E4B-it-qat-GGUF": `[{"type":"file","path":"gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf","size":4200000000}]`,
	}
	hfSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		repo, ok := strings.CutSuffix(strings.TrimPrefix(r.URL.Path, "/api/models/"), "/tree/main")
		body, found := trees[repo]
		if !ok || !found {
			http.NotFound(w, r)
			return
		}
		fmt.Fprint(w, body)
	}))
	defer hfSrv.Close()
	s.hf = hf.New(hfSrv.URL, "")
	s.memory = hw.Memory{VRAM: 12e9, RAM: 48e9}

	var got struct {
		Models []suggestion
	}
	if rec := getJSON(t, s, "/api/setup/models", &got); rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	if len(got.Models) < 2 {
		t.Fatalf("models: %+v", got.Models)
	}
	top := got.Models[0]
	if top.Repo != "unsloth/Qwen3.5-9B-GGUF" || top.Quant != "UD-Q4_K_XL" || top.Model != "hf.co/unsloth/Qwen3.5-9B-GGUF:UD-Q4_K_XL" ||
		top.URL != "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF" || top.Size != 5600000000 || top.Why == "" || top.Name != "Qwen3.5-9B" {
		t.Errorf("top: %+v", top)
	}
	for _, m := range got.Models {
		if m.Repo == "unsloth/Qwen3.5-4B-GGUF" {
			t.Errorf("offered a repo without a 4-bit quant")
		}
	}

	// Nothing reachable: a clear error, not an empty list.
	hfSrv.Close()
	s.suggAt = time.Time{}
	if rec := getJSON(t, s, "/api/setup/models", nil); rec.Code != http.StatusBadGateway {
		t.Errorf("offline: %d", rec.Code)
	}
}

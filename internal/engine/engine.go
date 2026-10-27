// Package engine runs models with llama.cpp's llama-server: one process
// per loaded model, started on demand and stopped when idle, the way
// Ollama does it internally. It answers in the same shapes the Ollama
// client did (package ollama's types), so the rest of ollamui doesn't care
// which engine it talks to.
package engine

import (
	"bufio"
	"cmp"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"ollamui/internal/gguf"
	"ollamui/internal/hw"
	"ollamui/internal/limits"
	"ollamui/internal/ollama"
	"ollamui/internal/presets"
)

// Options are the engine-wide settings.
type Options struct {
	// Server is the llama-server binary.
	Server string
	// CacheType is the K/V cache type ("f16", "q8_0", "q4_0").
	CacheType string
	// FlashAttn is "on", "off" or "auto".
	FlashAttn string
	// KeepAlive is how long an idle model stays loaded by default.
	KeepAlive time.Duration
	// LoadTimeout bounds how long a model may take to load.
	LoadTimeout time.Duration
	// KVDir holds the saved caches of chats (see kvcache.go); "" turns that off.
	KVDir string
	// KVLimit caps KVDir in bytes; 0 is 2 GB.
	KVLimit int64
	Log     *slog.Logger
}

// Engine runs the models in a Store.
type Engine struct {
	// OllamaDir, if set, is an Ollama models folder whose new models are
	// imported now and then (models pulled through Ollama show up here).
	OllamaDir string
	// Limits are the app-wide resource limits (nil: none); Live reads GPU
	// memory use (nil: the real reader).
	Limits     *limits.Store
	Live       func() hw.Live
	lastImport time.Time
	// Finder looks for models in other apps; nil gets a default.
	Finder *Finder
	// HF downloads models (Pull); nil turns downloads off.
	HF HF

	store *Store
	// Folder says where the models folder is remembered.
	Folder FolderSetting
	// moving is set while the models folder moves: nothing starts.
	moving atomic.Bool
	opt    Options
	http   *http.Client

	mu    sync.Mutex
	procs map[string]*proc // by model name
	meta  map[string]*gguf.File
	// loading serialises starts, so two requests don't start one model twice.
	loading sync.Mutex
	// files are the downloads in progress, by destination (see lockFile).
	files map[string]*sync.Mutex
	// pending is the load in progress, if any (under mu), so Running can
	// show it and Unload can cancel it.
	pending *pendingLoad
}

// pendingLoad is a model being loaded.
type pendingLoad struct {
	model   *Model
	asked   string
	cancel  context.CancelFunc
	ejected bool
}

// errEjected is a load cancelled by ejecting the model.
var errEjected = errors.New("ejected while loading")

// New returns an engine for store.
func New(store *Store, opt Options) *Engine {
	if opt.CacheType == "" {
		opt.CacheType = "f16"
	}
	if opt.FlashAttn == "" {
		opt.FlashAttn = "auto"
	}
	if opt.KeepAlive == 0 {
		opt.KeepAlive = 5 * time.Minute
	}
	if opt.LoadTimeout == 0 {
		opt.LoadTimeout = 10 * time.Minute
	}
	if opt.Log == nil {
		opt.Log = slog.Default()
	}
	return &Engine{store: store, opt: opt, http: &http.Client{}, procs: map[string]*proc{}, meta: map[string]*gguf.File{}}
}

// Store is the engine's model store.
func (e *Engine) Store() *Store { return e.store }

// ---- processes ----

// DefaultCtx is the context a model loads with when nothing sets one.
const DefaultCtx = 32768

// cacheReuse is the shortest prompt chunk (tokens) llama-server tries to
// move into place from the cache when it isn't at the same position.
const cacheReuse = 256

// CacheTypes are the K/V cache types a model can load with.
var CacheTypes = map[string]bool{"f16": true, "q8_0": true, "q4_0": true}

// CPUMoE values besides a layer count: unset leaves the experts to
// llama.cpp's fitting; all puts every expert on the CPU (--cpu-moe).
const (
	cpuMoEUnset = -1
	cpuMoEAll   = -2
)

// loadOpts are the request options that need a (re)start to change.
type loadOpts struct {
	Ctx         int    // 0: let llama.cpp fit the largest that fits
	GPULayers   int    // -1: let llama.cpp decide
	CPUMoE      int    // cpuMoEUnset, cpuMoEAll, or how many layers keep their experts on the CPU
	LoadMode    string // --load-mode; "": llama.cpp's default (mmap)
	Threads     int    // 0: default
	CacheType   string // "": the engine's default
	FlashAttn   string // "": the engine's default
	Embedding   bool
	DraftGPU    bool   // keep the whole draft (MTP head or draft model) on the GPU
	NoMMProjGPU bool   // keep the vision projector in RAM
	NoVision    bool   // don't load the vision projector at all
	Draft       string // the draft model's name for speculative decoding, "": none
	DraftMax    int    // tokens drafted at a time; 0: llama.cpp's default
	NGramMax    int    // tokens n-gram guessing drafts at a time; 0: llama.cpp's default
	LoRA        []presets.LoRA
	// Speed and NGram are the speed-up settings as asked; planSpeed turns
	// them into MTP, Draft and NGram (the n-gram type, "": none) before a
	// server starts, so what runs is what compares.
	Speed string
	NGram string
	MTP   bool // draft with the model's own MTP head
}

// sameLoad reports whether a server started with o is exactly what want
// asks for.
func (o loadOpts) sameLoad(want loadOpts) bool {
	return reflect.DeepEqual(o, want)
}

// sameNames reports whether a server started with o can take want's
// adapter scales live: the same options, and the same adapters in the
// same order (only their scales may differ).
func (o loadOpts) sameNames(want loadOpts) bool {
	if len(o.LoRA) != len(want.LoRA) {
		return false
	}
	for i := range o.LoRA {
		if o.LoRA[i].Name != want.LoRA[i].Name {
			return false
		}
	}
	a, b := o, want
	a.LoRA, b.LoRA = nil, nil
	return reflect.DeepEqual(a, b)
}

func loadOptsFrom(o map[string]any) loadOpts {
	lo := loadOpts{GPULayers: -1, CPUMoE: cpuMoEUnset}
	if l, ok := presets.LoRAs(o["lora"]); ok {
		lo.LoRA = l
	}
	if v, ok := num(o["num_ctx"]); ok && v > 0 {
		lo.Ctx = int(v)
	}
	if v, ok := num(o["num_gpu"]); ok && v >= 0 {
		lo.GPULayers = int(v)
	}
	if v, ok := num(o["n_cpu_moe"]); ok && v == -1 {
		lo.CPUMoE = cpuMoEAll
	} else if ok && v >= 0 {
		lo.CPUMoE = int(v)
	}
	if v, _ := o["load_mode"].(string); v == "mmap" || v == "none" || v == "mlock" {
		lo.LoadMode = v
	}
	if v, ok := num(o["num_thread"]); ok && v > 0 {
		lo.Threads = int(v)
	}
	if v, _ := o["draft_model"].(string); v != "" {
		lo.Draft = v
	}
	if v, ok := num(o["spec_ngram_max"]); ok && v > 0 {
		lo.NGramMax = int(v)
	}
	if v, ok := num(o["draft_max"]); ok && v > 0 {
		lo.DraftMax = int(v)
	}
	lo.DraftGPU = o["draft_gpu"] == "always"
	lo.NoMMProjGPU = o["mmproj_cpu"] == "on"
	lo.NoVision = o["vision"] == "off"
	lo.Speed, _ = o["speedup"].(string)
	lo.NGram, _ = o["spec_ngram"].(string)
	if v, _ := o["cache_type"].(string); CacheTypes[v] {
		lo.CacheType = v
	}
	if v, _ := o["flash_attn"].(string); v == "on" || v == "off" || v == "auto" {
		lo.FlashAttn = v
	}
	return lo
}

// proc is one running llama-server.
type proc struct {
	model *Model
	asked string // the name it was loaded by, which Running reports
	opts  loadOpts
	cmd   *exec.Cmd
	base  string // http://127.0.0.1:port
	done  chan struct{}

	mu       sync.Mutex
	ctx      int   // the context it runs with
	layers   int   // layers on the GPU
	total    int   // layers in all
	vram     int64 // bytes on the GPU
	ram      int64 // bytes in system memory
	lastUsed time.Time
	expires  time.Time // zero while pinned
	pinned   bool
	busy     int    // requests in flight
	kvDir    string // where its saved chat caches go, "" if none
	chat     string // the chat its slot holds (see kvcache.go), "" if none
	// chatTokens is how many tokens that chat fills the slot with.
	chatTokens int
	// fresh is set until the first chat request, which may restore a saved cache.
	fresh   bool
	tail    []string
	err     error
	oom     string        // the first out-of-memory line it printed, if any
	logDone chan struct{} // closed once its output has all been read
	fitted  bool          // asked for a layer count that didn't fit; --fit chose
}

// beginChat says which chat the slot now serves and how many tokens it
// holds ("" for requests that aren't a chat's).
func (p *proc) beginChat(chat string, tokens int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.chat, p.chatTokens = chat, tokens
}

// takeFresh reports whether no chat has used the server yet, and marks it used.
func (p *proc) takeFresh() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	fresh := p.fresh
	p.fresh = false
	return fresh
}

func (p *proc) running() bool {
	select {
	case <-p.done:
		return false
	default:
		return true
	}
}

// touch marks the model used and sets how long it stays: keepAlive < 0
// pins it, 0 keeps the default.
func (p *proc) touch(keepAlive time.Duration, def time.Duration) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.lastUsed = time.Now()
	switch {
	case keepAlive < 0:
		p.pinned, p.expires = true, time.Time{}
	case keepAlive > 0:
		p.pinned, p.expires = false, time.Now().Add(keepAlive)
	case !p.pinned:
		p.expires = time.Now().Add(def)
	}
}

func freePort() (int, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port, nil
}

func (e *Engine) args(m *Model, o loadOpts, port int) []string {
	a := []string{
		"-m", e.store.Blob(m.Files[0]),
		"--host", "127.0.0.1", "--port", strconv.Itoa(port),
		"-np", "1", // more slots split the context and overfill VRAM
		"--jinja", "--no-webui", "-lv", "4",
		"-a", m.Name,
	}
	// Without a context, --fit would grow it to the model's maximum (262k
	// for Qwen3.5), which made writing 3x slower here (15 vs 46 tok/s on a
	// 2B). So ask for DefaultCtx and let --fit go lower if memory is short.
	ctx := o.Ctx
	if ctx == 0 {
		ctx = DefaultCtx
	}
	a = append(a, "-c", strconv.Itoa(ctx))
	switch {
	case o.GPULayers >= 0:
		a = append(a, "-ngl", strconv.Itoa(o.GPULayers))
	case o.CPUMoE != cpuMoEUnset:
		// Experts placed by hand: everything else goes on the GPU.
		a = append(a, "-ngl", "all")
	}
	switch {
	case o.CPUMoE == cpuMoEAll:
		a = append(a, "--cpu-moe")
	case o.CPUMoE >= 0:
		a = append(a, "--n-cpu-moe", strconv.Itoa(o.CPUMoE))
	}
	if o.LoadMode != "" {
		a = append(a, "--load-mode", o.LoadMode)
	}
	if o.Threads > 0 {
		a = append(a, "-t", strconv.Itoa(o.Threads))
	} else if e.Limits.Get().Gentle {
		a = append(a, "-t", strconv.Itoa(gentleThreads()))
	}
	a = append(a, e.gentleArgs(o)...)
	if t, ok := e.fitTarget(m, o); ok {
		a = append(a, "--fit-target", strconv.FormatInt(t, 10))
	}
	if !o.Embedding {
		// Each adapter is numbered in the order it's given (setScales relies on it).
		for _, u := range o.LoRA {
			if ad, err := e.store.Get(u.Name); err == nil && ad.Adapter != nil {
				a = append(a, "--lora", e.store.Blob(ad.Files[0]))
			}
		}
		if len(o.LoRA) > 0 {
			// Loaded unapplied, then scaled once it answers, so a scale can
			// change later without a restart.
			a = append(a, "--lora-init-without-apply")
		}
	}
	a = append(a, e.specArgs(m, o)...)
	if o.Embedding {
		a = append(a, "--embedding", "-b", "2048", "-ub", "2048")
	} else {
		if m.Projector != "" && !o.NoVision {
			a = append(a, "--mmproj", e.store.Blob(m.Projector))
			if o.NoMMProjGPU {
				a = append(a, "--no-mmproj-offload")
			}
		}
		fa, ct := cmp.Or(o.FlashAttn, e.opt.FlashAttn), cmp.Or(o.CacheType, e.opt.CacheType)
		if ct != "f16" && fa == "off" {
			fa = "on" // a quantized cache needs flash attention
		}
		a = append(a, "-fa", fa)
		if ct != "f16" {
			a = append(a, "-ctk", ct, "-ctv", ct)
		}
		// Reuse cached prompt chunks found after a changed part (an edited
		// system prompt, a message that moved), not just the common prefix.
		a = append(a, "--cache-reuse", strconv.Itoa(cacheReuse))
		if d := e.kvDir(m, o); d != "" {
			a = append(a, "--slot-save-path", d)
		}
	}
	return a
}

// ensure returns a running server for model with the request's options
// (over the model's own), starting or restarting one when needed. It
// reports whether it loaded the model.
func (e *Engine) ensure(ctx context.Context, name string, options map[string]any) (*proc, bool, error) {
	if e.moving.Load() {
		return nil, false, errMovingModels
	}
	m, err := e.store.Get(name)
	if err != nil {
		return nil, false, err
	}
	if m.Adapter != nil {
		return nil, false, fmt.Errorf("%s is a LoRA adapter, not a model", m.Name)
	}
	if err := e.missingError(m); err != nil {
		return nil, false, err
	}
	o := loadOptsFrom(m.options(options))
	if f, err := e.gguf(m); err == nil {
		o.Embedding = f.IsEmbedding()
		if f.ExpertCount() == 0 {
			// Dense models have no experts to place or load differently.
			o.CPUMoE, o.LoadMode = cpuMoEUnset, ""
		}
	}
	if o.Embedding {
		o.LoRA = nil // embedding models take no adapters
	}
	o = planSpeed(m, o)
	if o.Draft != "" && !o.Embedding {
		if err := e.checkDraft(m, o.Draft); err != nil {
			return nil, false, err
		}
	}
	if err := e.checkLoRA(m, o.LoRA); err != nil {
		return nil, false, err
	}
	e.loading.Lock()
	defer e.loading.Unlock()

	e.mu.Lock()
	p := e.procs[m.Name]
	e.mu.Unlock()
	if p != nil && p.running() && p.opts.sameLoad(o) {
		p.mu.Lock()
		p.asked = name
		p.mu.Unlock()
		return p, false, nil
	}
	if p != nil && p.running() && p.opts.sameNames(o) {
		// Only the adapter scales changed: set them on the running server.
		err := e.setScales(ctx, p, o.LoRA)
		if err == nil {
			p.mu.Lock()
			p.asked = name
			p.mu.Unlock()
			p.opts = o
			return p, false, nil
		}
		e.opt.Log.Warn("couldn't set adapter scales live; loading again", "model", m.Name, "err", err)
	}
	if p != nil {
		e.stop(p)
	}
	if !o.Embedding && o.GPULayers != 0 {
		e.makeRoom(m.Name)
	}
	// The load goes on if the request that asked for it goes away (the
	// user switched chats or pages): it's slow, and they'll want the model.
	// Only ejecting it stops it.
	lctx, cancel := context.WithCancel(context.WithoutCancel(ctx))
	pl := &pendingLoad{model: m, asked: name, cancel: cancel}
	e.mu.Lock()
	e.pending = pl
	e.mu.Unlock()
	defer func() {
		e.mu.Lock()
		if e.pending == pl {
			e.pending = nil
		}
		e.mu.Unlock()
		cancel()
	}()
	p, err = e.start(lctx, m, o)
	manual := o.GPULayers > 0 || o.CPUMoE != cpuMoEUnset
	if err != nil && manual && outOfMemory(err) && lctx.Err() == nil {
		// A fixed layer count (or experts placed by hand) turns llama.cpp's
		// --fit off, so too many layers for the GPU just crash. Load it the
		// way Auto would: --fit keeps what fits on the GPU and, for MoE
		// models, moves experts to RAM. The process keeps the asked-for
		// options, so the next request with the same settings doesn't
		// restart it.
		e.opt.Log.Warn("too many GPU layers or experts on the GPU for free memory; loading with automatic fitting instead", "model", m.Name, "asked", o.GPULayers, "n_cpu_moe", o.CPUMoE)
		auto := o
		auto.GPULayers, auto.CPUMoE = -1, cpuMoEUnset
		if p, err = e.start(lctx, m, auto); err == nil {
			p.opts = o
			p.fitted = true
		}
	}
	e.mu.Lock()
	ejected := pl.ejected
	e.mu.Unlock()
	if err != nil && ejected {
		err = fmt.Errorf("%s was %w", m.Name, errEjected)
	}
	if err == nil {
		p.asked = name
	}
	return p, err == nil, err
}

// errGPUMemory is a load that ran out of GPU memory.
var errGPUMemory = errors.New("not enough GPU memory")

// oomRe matches llama.cpp's allocation failures (CUDA, ROCm, Vulkan, Metal).
var oomRe = regexp.MustCompile(`(?i)out of memory|failed to allocate|unable to allocate|ErrorOutOfDeviceMemory`)

func outOfMemory(err error) bool { return errors.Is(err, errGPUMemory) }

// makeRoom stops other models that use the GPU, so the new one gets it
// whole, like Ollama does. Pinned models, CPU-only models and embedding
// models stay; llama.cpp fits the new one around them.
func (e *Engine) makeRoom(except string) {
	e.mu.Lock()
	var victims []*proc
	for name, p := range e.procs {
		p.mu.Lock()
		keep := name == except || p.pinned || p.opts.Embedding || p.opts.GPULayers == 0 || p.vram == 0 || p.busy > 0
		p.mu.Unlock()
		if !keep {
			victims = append(victims, p)
		}
	}
	e.mu.Unlock()
	for _, p := range victims {
		e.opt.Log.Info("unloading to make room", "model", p.model.Name)
		e.stop(p)
	}
}

var (
	bufferRe  = regexp.MustCompile(`(\S+) (?:model|KV|RS|compute|output) buffer size =\s+([\d.]+) MiB`)
	offloadRe = regexp.MustCompile(`offloaded (\d+)/(\d+) layers to GPU`)
	nctxRe    = regexp.MustCompile(`n_ctx_slot = (\d+)|llama_context: n_ctx\s+= (\d+)`)
	gpuDevRe  = regexp.MustCompile(`^(Vulkan|ROCm|CUDA|Metal|SYCL|OpenCL)\d+$`)
)

func (e *Engine) start(ctx context.Context, m *Model, o loadOpts) (*proc, error) {
	bin := e.ServerPath()
	if bin == "" {
		return nil, errors.New("llama.cpp isn't set up: no llama-server found")
	}
	port, err := freePort()
	if err != nil {
		return nil, err
	}
	kvDir := e.kvDir(m, o)
	if kvDir != "" {
		if err := os.MkdirAll(kvDir, 0o755); err != nil {
			e.opt.Log.Warn("chat caches folder", "err", err)
			kvDir = ""
		}
	}
	cmd := exec.Command(bin, e.args(m, o, port)...)
	// Its shared libraries sit beside it, wherever that is.
	cmd.Env = append(os.Environ(), "LD_LIBRARY_PATH="+strings.TrimSuffix(filepath.Dir(bin)+":"+os.Getenv("LD_LIBRARY_PATH"), ":"))
	if v := hsaOverride(bin); v != "" {
		cmd.Env = append(cmd.Env, "HSA_OVERRIDE_GFX_VERSION="+v)
	}
	setProcAttrs(cmd)
	out, err := cmd.StderrPipe()
	if err != nil {
		return nil, err
	}
	cmd.Stdout = cmd.Stderr
	p := &proc{model: m, opts: o, cmd: cmd, base: fmt.Sprintf("http://127.0.0.1:%d", port), kvDir: kvDir, fresh: true, done: make(chan struct{}), logDone: make(chan struct{})}
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start llama-server: %w", err)
	}
	e.opt.Log.Info("loading model", "model", m.Name, "pid", cmd.Process.Pid, "args", strings.Join(cmd.Args[1:], " "))
	go p.readLog(out, e.opt.Log)
	go func() {
		err := cmd.Wait()
		p.mu.Lock()
		p.err = err
		p.mu.Unlock()
		close(p.done)
	}()

	// Wait until it answers /health (503 while loading).
	deadline := time.Now().Add(e.opt.LoadTimeout)
	for {
		if !p.running() {
			select { // the last of its output, which says why
			case <-p.logDone:
			case <-time.After(time.Second):
			}
			p.mu.Lock()
			oom := p.oom
			p.mu.Unlock()
			if oom != "" {
				return nil, fmt.Errorf("%w loading %s. Fewer GPU layers or a smaller context would fit (setting “GPU layers” to Auto does this for you). llama-server said: %s", errGPUMemory, m.Name, oom)
			}
			return nil, fmt.Errorf("llama-server stopped while loading %s: %s", m.Name, p.lastLines())
		}
		hctx, cancel := context.WithTimeout(ctx, 2*time.Second)
		req, _ := http.NewRequestWithContext(hctx, http.MethodGet, p.base+"/health", nil)
		res, err := e.http.Do(req)
		cancel()
		if err == nil {
			res.Body.Close()
			if res.StatusCode == http.StatusOK {
				break
			}
		}
		if ctx.Err() != nil || time.Now().After(deadline) {
			stopProcess(cmd)
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			return nil, fmt.Errorf("%s took too long to load", m.Name)
		}
		time.Sleep(150 * time.Millisecond)
	}
	// It can answer /health after an allocation failed while it was
	// reserving memory, and then die on the first request. Treat that as
	// a load that ran out of memory, so the caller falls back.
	p.mu.Lock()
	oom := p.oom
	p.mu.Unlock()
	if oom != "" {
		stopProcess(cmd)
		return nil, fmt.Errorf("%w loading %s. Fewer GPU layers or a smaller context would fit (setting “GPU layers” to Auto does this for you). llama-server said: %s", errGPUMemory, m.Name, oom)
	}
	if len(o.LoRA) > 0 {
		if err := e.setScales(ctx, p, o.LoRA); err != nil {
			stopProcess(cmd)
			return nil, fmt.Errorf("loading the adapters for %s: %w", m.Name, err)
		}
	}
	p.touch(0, e.opt.KeepAlive)
	e.mu.Lock()
	e.procs[m.Name] = p
	e.mu.Unlock()
	p.mu.Lock()
	e.opt.Log.Info("model loaded", "model", m.Name, "ctx", p.ctx, "gpu_layers", fmt.Sprintf("%d/%d", p.layers, p.total), "vram_mib", p.vram>>20, "ram_mib", p.ram>>20)
	p.mu.Unlock()
	return p, nil
}

// hsaOverride is the HSA_OVERRIDE_GFX_VERSION a ROCm build of llama-server
// needs on this GPU, or "". ROCm ships kernels for one chip per family
// (gfx1030, gfx1100), and its siblings (an RX 6700 XT is gfx1031) run
// them fine once told to; Ollama does the same. Only applies when the
// server is a ROCm build and the user hasn't set it.
func hsaOverride(server string) string {
	if os.Getenv("HSA_OVERRIDE_GFX_VERSION") != "" {
		return ""
	}
	if _, err := os.Stat(filepath.Join(filepath.Dir(server), "libggml-hip.so")); err != nil {
		return ""
	}
	nodes, _ := filepath.Glob("/sys/class/kfd/kfd/topology/nodes/*/properties")
	for _, n := range nodes {
		b, err := os.ReadFile(n)
		if err != nil {
			continue
		}
		for _, l := range strings.Split(string(b), "\n") {
			v, ok := strings.CutPrefix(l, "gfx_target_version ")
			if !ok {
				continue
			}
			if o := gfxOverride(v); o != "" {
				return o
			}
		}
	}
	return ""
}

// gfxOverride maps a KFD gfx_target_version (100301 is gfx1031) to the
// override for its family, or "" when it needs none.
func gfxOverride(v string) string {
	n, err := strconv.Atoi(strings.TrimSpace(v))
	if err != nil || n == 0 {
		return ""
	}
	switch major, minor, step := n/10000, n/100%100, n%100; {
	case major == 10 && minor == 3 && step != 0:
		return "10.3.0"
	case major == 11 && minor == 0 && step != 0 && step != 1:
		return "11.0.0"
	}
	return ""
}

// readLog keeps the last lines for error messages and picks out how the
// model was placed: layers on the GPU, buffer sizes, context.
func (p *proc) readLog(r io.Reader, log *slog.Logger) {
	defer close(p.logDone)
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64<<10), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		log.Debug("llama-server", "model", p.model.Name, "out", line)
		p.mu.Lock()
		p.tail = append(p.tail, line)
		if len(p.tail) > 40 {
			p.tail = p.tail[len(p.tail)-40:]
		}
		if m := bufferRe.FindStringSubmatch(line); m != nil {
			mib, _ := strconv.ParseFloat(m[2], 64)
			b := int64(mib * (1 << 20))
			if gpuDevRe.MatchString(m[1]) {
				p.vram += b
			} else {
				p.ram += b
			}
		}
		if p.oom == "" && oomRe.MatchString(line) {
			p.oom = strings.TrimSpace(line)
		}
		if m := offloadRe.FindStringSubmatch(line); m != nil {
			p.layers, _ = strconv.Atoi(m[1])
			p.total, _ = strconv.Atoi(m[2])
		}
		if m := nctxRe.FindStringSubmatch(line); m != nil {
			v := m[1]
			if v == "" {
				v = m[2]
			}
			p.ctx, _ = strconv.Atoi(v)
		}
		p.mu.Unlock()
	}
}

// lastLines is the useful end of the log, for an error message.
func (p *proc) lastLines() string {
	p.mu.Lock()
	defer p.mu.Unlock()
	var keep []string
	for _, l := range p.tail {
		if strings.Contains(l, " E ") || strings.Contains(strings.ToLower(l), "error") || strings.Contains(l, "failed") {
			keep = append(keep, strings.TrimSpace(l))
		}
	}
	if len(keep) == 0 && len(p.tail) > 0 {
		keep = p.tail[max(0, len(p.tail)-3):]
	}
	return strings.Join(keep, " / ")
}

func (e *Engine) stop(p *proc) {
	e.mu.Lock()
	if e.procs[p.model.Name] == p {
		delete(e.procs, p.model.Name)
	}
	e.mu.Unlock()
	if !p.running() {
		return
	}
	e.saveSlot(p)
	stopProcess(p.cmd)
	select {
	case <-p.done:
	case <-time.After(10 * time.Second):
		p.cmd.Process.Kill()
		<-p.done
	}
	e.opt.Log.Info("model unloaded", "model", p.model.Name)
}

// Run unloads idle models until ctx ends, then stops them all.
func (e *Engine) Run(ctx context.Context) {
	tick := time.NewTicker(5 * time.Second)
	defer tick.Stop()
	for {
		select {
		case <-ctx.Done():
			e.StopAll()
			return
		case <-tick.C:
			e.mu.Lock()
			var idle []*proc
			for _, p := range e.procs {
				p.mu.Lock()
				if !p.running() || (!p.pinned && p.busy == 0 && !p.expires.IsZero() && time.Now().After(p.expires)) {
					idle = append(idle, p)
				}
				p.mu.Unlock()
			}
			e.mu.Unlock()
			for _, p := range idle {
				e.stop(p)
			}
		}
	}
}

// StopAll stops every model.
func (e *Engine) StopAll() {
	e.mu.Lock()
	all := make([]*proc, 0, len(e.procs))
	for _, p := range e.procs {
		all = append(all, p)
	}
	e.mu.Unlock()
	for _, p := range all {
		e.stop(p)
	}
}

// ---- the Ollama-shaped API ----

// Version is the llama.cpp build, e.g. "b1-339f324".
func (e *Engine) Version(ctx context.Context) (string, error) {
	bin := e.ServerPath()
	if bin == "" {
		return "", errors.New("no llama-server")
	}
	out, err := exec.CommandContext(ctx, bin, "--version").CombinedOutput()
	if err != nil {
		return "", err
	}
	for _, l := range strings.Split(string(out), "\n") {
		if v, ok := strings.CutPrefix(strings.TrimSpace(l), "version: "); ok {
			return "llama.cpp " + v, nil
		}
	}
	return "llama.cpp", nil
}

// gguf reads (once) the header of a model's files.
func (e *Engine) gguf(m *Model) (*gguf.File, error) {
	e.mu.Lock()
	f := e.meta[m.Name]
	e.mu.Unlock()
	if f != nil {
		return f, nil
	}
	paths := make([]string, len(m.Files))
	for i, p := range m.Files {
		paths[i] = e.store.Blob(p)
	}
	f, err := gguf.ReadSplits(paths)
	if err != nil {
		return nil, err
	}
	e.mu.Lock()
	e.meta[m.Name] = f
	e.mu.Unlock()
	return f, nil
}

func (e *Engine) details(m *Model) ollama.Details {
	d := ollama.Details{Format: "gguf"}
	if f, err := e.gguf(m); err == nil {
		d.Family = f.Arch()
		d.Families = []string{f.Arch()}
		d.ParameterSize = f.SizeLabel()
		d.QuantizationLevel = f.FileType()
	}
	return d
}

// List returns every installed model under each of its names, like
// Ollama listed an alias as its own entry with the same digest.
func (e *Engine) List(ctx context.Context) ([]ollama.Model, error) {
	e.mu.Lock()
	again := e.OllamaDir != "" && time.Since(e.lastImport) > 10*time.Second
	if again {
		e.lastImport = time.Now()
	}
	e.mu.Unlock()
	if again {
		if n, err := e.store.ImportOllama(e.OllamaDir); err == nil && n > 0 {
			e.opt.Log.Info("imported models from Ollama", "count", n)
		}
	}
	var out []ollama.Model
	for _, m := range e.store.List() {
		if m.Draft != "" {
			continue // drafts can't chat, so they're not models to pick
		}
		d := e.details(m)
		for _, n := range m.Names() {
			out = append(out, ollama.Model{Name: n, Model: n, ModifiedAt: m.Modified, Size: m.Size, Digest: m.Digest, Details: d})
		}
	}
	return out, nil
}

// Running lists the loaded models.
func (e *Engine) Running(ctx context.Context) ([]ollama.RunningModel, error) {
	e.mu.Lock()
	procs := make([]*proc, 0, len(e.procs))
	for _, p := range e.procs {
		if p.running() {
			procs = append(procs, p)
		}
	}
	e.mu.Unlock()
	sort.Slice(procs, func(i, j int) bool { return procs[i].model.Name < procs[j].model.Name })
	out := make([]ollama.RunningModel, 0, len(procs))
	for _, p := range procs {
		p.mu.Lock()
		exp := p.expires
		if p.pinned {
			exp = time.Now().Add(100 * 365 * 24 * time.Hour) // "forever", as Ollama reported it
		}
		name := p.asked
		if name == "" {
			name = p.model.Name
		}
		r := ollama.RunningModel{Name: name, Model: name, Size: p.vram + p.ram, SizeVRAM: p.vram, Digest: p.model.Digest, ExpiresAt: exp, ContextLength: p.ctx}
		p.mu.Unlock()
		r.Details = e.details(p.model)
		out = append(out, r)
	}
	return out, nil
}

// Loading is the name of the model being loaded, or "". Running leaves it
// out until it's ready.
func (e *Engine) Loading() string {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.pending == nil || e.pending.ejected {
		return ""
	}
	return e.pending.asked
}

// Show describes a model: GGUF metadata in Ollama's model_info shape,
// capabilities guessed from the chat template, default parameters.
func (e *Engine) Show(ctx context.Context, name string) (*ollama.ShowResponse, error) {
	m, err := e.store.Get(name)
	if err != nil {
		return nil, &ollama.APIError{StatusCode: http.StatusNotFound, Message: err.Error()}
	}
	if m.Adapter != nil {
		return nil, &ollama.APIError{StatusCode: http.StatusNotFound, Message: m.Name + " is a LoRA adapter, not a model"}
	}
	f, err := e.gguf(m)
	if err != nil {
		return nil, err
	}
	s := &ollama.ShowResponse{ModelInfo: f.ModelInfo(), Details: e.details(m), Template: f.ChatTemplate(), ModifiedAt: m.Modified, Parameters: paramsText(m.Params)}
	tpl := f.ChatTemplate()
	if f.IsEmbedding() {
		s.Capabilities = []string{"embedding"}
	} else {
		s.Capabilities = []string{"completion"}
		if strings.Contains(tpl, "tools") {
			s.Capabilities = append(s.Capabilities, "tools")
		}
		if strings.Contains(tpl, "enable_thinking") || strings.Contains(tpl, "<think>") || strings.Contains(tpl, "reasoning") || strings.Contains(tpl, "thinking") {
			s.Capabilities = append(s.Capabilities, "thinking")
		}
		switch {
		case m.Projector != "" && m.options(nil)["vision"] == "off":
			// Turned off in its settings: it can't see images, but the
			// setting to turn it back on stays.
			s.Capabilities = append(s.Capabilities, "vision-off")
		case m.Projector != "":
			s.Capabilities = append(s.Capabilities, "vision")
		}
	}
	return s, nil
}

// paramsText writes default parameters the way Ollama's Modelfile did
// ("temperature 0.6\nstop \"<|im_end|>\""), which ollama.ParseParameters reads.
func paramsText(p map[string]any) string {
	keys := make([]string, 0, len(p))
	for k := range p {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var b strings.Builder
	for _, k := range keys {
		switch v := p[k].(type) {
		case []any:
			for _, x := range v {
				fmt.Fprintf(&b, "%s %q\n", k, fmt.Sprint(x))
			}
		case string:
			fmt.Fprintf(&b, "%s %q\n", k, v)
		default:
			fmt.Fprintf(&b, "%s %v\n", k, v)
		}
	}
	return b.String()
}

// Load loads a model; keepAlive "-1" keeps it until unloaded, "0"
// unloads it, a duration ("5m") sets how long it stays idle.
func (e *Engine) Load(ctx context.Context, model, keepAlive string, options map[string]any) error {
	ka, err := parseKeepAlive(keepAlive)
	if err != nil {
		return err
	}
	if ka == 0 && keepAlive != "" {
		return e.Unload(ctx, model)
	}
	p, _, err := e.ensure(ctx, model, options)
	if err != nil {
		return err
	}
	if keepAlive == "" {
		ka = modelKeepAlive(p.model, options)
	}
	p.touch(ka, e.opt.KeepAlive)
	return nil
}

func parseKeepAlive(v any) (time.Duration, error) {
	switch k := v.(type) {
	case nil:
		return 0, nil
	case int:
		return keepAliveSeconds(float64(k)), nil
	case float64:
		return keepAliveSeconds(k), nil
	case string:
		if k == "" {
			return 0, nil
		}
		if n, err := strconv.ParseFloat(k, 64); err == nil {
			return keepAliveSeconds(n), nil
		}
		d, err := time.ParseDuration(k)
		if err != nil {
			return 0, fmt.Errorf("bad keep_alive %q", k)
		}
		return d, nil
	}
	return 0, fmt.Errorf("bad keep_alive %v", v)
}

// modelKeepAlive is the keep_alive a model's settings (or the request's
// options) ask for when the request itself sets none; 0 if neither does.
func modelKeepAlive(m *Model, options map[string]any) time.Duration {
	ka, err := parseKeepAlive(m.options(options)["keep_alive"])
	if err != nil {
		return 0
	}
	return ka
}

func keepAliveSeconds(n float64) time.Duration {
	if n < 0 {
		return -1
	}
	return time.Duration(n * float64(time.Second))
}

// Unload stops a model.
func (e *Engine) Unload(ctx context.Context, model string) error {
	m, err := e.store.Get(model)
	if err != nil {
		return err
	}
	e.mu.Lock()
	p := e.procs[m.Name]
	if pl := e.pending; pl != nil && pl.model.Name == m.Name {
		pl.ejected = true
		pl.cancel()
	}
	e.mu.Unlock()
	if p != nil {
		e.stop(p)
	}
	return nil
}

// Delete unloads a model and removes it (or just the alias named). A
// name that doesn't exist is fine.
func (e *Engine) Delete(ctx context.Context, model string) error {
	m, err := e.store.Get(model)
	if err != nil {
		// Already gone: deleting a model's full name takes its aliases with
		// it, and the library deletes every name of a model in turn.
		return nil
	}
	if normName(model) == normName(m.Name) {
		e.Unload(ctx, model)
		e.mu.Lock()
		delete(e.meta, m.Name)
		e.mu.Unlock()
	}
	return e.store.Delete(model)
}

func num(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case float32:
		return float64(n), true
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	case string:
		f, err := strconv.ParseFloat(n, 64)
		return f, err == nil
	}
	return 0, false
}

// The rest of the Ollama client's surface, for code that hasn't moved on:
// URL, Host and Port describe where models run; Create was how presets
// were published as Ollama models, which an editor now reaches through
// ollamui's own API instead.

// Settings returns a model's own settings.
func (e *Engine) Settings(name string) (map[string]any, error) {
	m, err := e.store.Get(name)
	if err != nil {
		return nil, err
	}
	return m.Settings, nil
}

// SetSettings replaces a model's own settings; they apply from the next
// request (a change to how it loads restarts it then).
func (e *Engine) SetSettings(name string, s map[string]any) error {
	m, err := e.store.Get(name)
	if err != nil {
		return err
	}
	cp := *m
	cp.Settings = s
	if len(s) == 0 {
		cp.Settings = nil
	}
	return e.store.Put(&cp)
}

// Backend names the GPU library the llama-server build uses, from the
// libraries beside it ("" if it can't tell).
func (e *Engine) Backend() string {
	dir := filepath.Dir(e.ServerPath())
	for _, b := range []struct{ lib, name string }{
		{"libggml-hip.so", "ROCm"}, {"libggml-cuda.so", "CUDA"}, {"libggml-vulkan.so", "Vulkan"}, {"libggml-metal.so", "Metal"},
		{"libggml-metal.dylib", "Metal"}, {"ggml-vulkan.dll", "Vulkan"},
	} {
		if _, err := os.Stat(filepath.Join(dir, b.lib)); err == nil {
			return b.name
		}
	}
	return ""
}

// SavedSettings lists the models that have settings of their own, by
// full name.
func (e *Engine) SavedSettings() map[string]map[string]any {
	out := map[string]map[string]any{}
	for _, m := range e.store.List() {
		if len(m.Settings) > 0 {
			out[m.Name] = m.Settings
		}
	}
	return out
}

// KeepAlive is how long an idle model stays loaded when it doesn't say.
func (e *Engine) KeepAlive() time.Duration { return e.opt.KeepAlive }

// FlashAttn is what flash attention is set to when a model doesn't say.
func (e *Engine) FlashAttn() string { return e.opt.FlashAttn }

// CacheType is the K/V cache type models load with.
func (e *Engine) CacheType() string { return e.opt.CacheType }

// ServerPath is the llama-server binary in use.
func (e *Engine) ServerPath() string {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.opt.Server
}

// SetServer switches to another llama-server binary ("" for none). Models
// already running keep the process they have until they unload; the next
// start uses the new one.
func (e *Engine) SetServer(path string) {
	e.mu.Lock()
	e.opt.Server = path
	e.mu.Unlock()
}

// LoadedNames lists the models the engine has running, so callers can ask
// before replacing the binary under them.
func (e *Engine) LoadedNames() []string {
	e.mu.Lock()
	defer e.mu.Unlock()
	names := make([]string, 0, len(e.procs))
	for n := range e.procs {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

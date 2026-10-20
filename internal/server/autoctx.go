package server

import (
	"context"
	"slices"
	"strings"

	"ollamui/internal/hw"
	"ollamui/internal/ollama"
)

// Context lengths ollamui picks from, smallest first.
var ctxSteps = []int{8192, 16384, 32768, 65536, 131072, 262144}

const (
	// ctxMargin is VRAM left free for Ollama's own buffers, which grow a
	// little with context, and for whatever else starts using the GPU.
	ctxMargin = 400 << 20
	// ctxStickiness lets a model keep the context it's already loaded with
	// even if free VRAM dips a little, so chats don't trigger reloads.
	ctxStickiness = 512 << 20
)

// autoContext is the context length ollamui picked for a model, and why.
type autoContext struct {
	Ctx    int    `json:"ctx"`    // 0 means "leave it to Ollama"
	Reason string `json:"reason"` // short explanation for the UI
}

// chooseContext picks the largest step whose cache fits beside the model's
// weights (base) in budget bytes of VRAM. loaded is the context the model is
// running with now (0 if not loaded); it's kept if it still fits, to avoid
// reloading the model for small swings in free memory.
func chooseContext(base int64, kv hw.KVCache, budget int64, modelMax, loaded int) int {
	fits := func(c int) bool { return base+kv.Bytes(c) <= budget }
	for _, c := range ctxSteps {
		if c == loaded && (modelMax == 0 || c <= modelMax) && base+kv.Bytes(c) <= budget+ctxStickiness {
			if next := nextStep(c); next == 0 || !fits(next) {
				return c
			}
		}
	}
	best := 0
	for _, c := range ctxSteps {
		if modelMax > 0 && c > modelMax {
			break
		}
		if fits(c) {
			best = c
		}
	}
	return best
}

func nextStep(c int) int {
	for _, s := range ctxSteps {
		if s > c {
			return s
		}
	}
	return 0
}

// sameModel compares Ollama model names, treating "x" and "x:latest" alike.
func sameModel(a, b string) bool {
	norm := func(s string) string {
		s = strings.ToLower(s)
		if !strings.Contains(s[strings.LastIndex(s, "/")+1:], ":") {
			s += ":latest"
		}
		return s
	}
	return norm(a) == norm(b)
}

// pickContext works out the largest context that keeps model fully on the
// GPU. It needs the model's weight size, which it measures from Ollama:
// from /api/ps if the model is loaded, from an earlier measurement, or, if
// probe is set, by loading the model briefly. Without a GPU, without layer
// metadata or when the model sets num_ctx itself, it returns Ctx 0.
func (s *Server) pickContext(ctx context.Context, model string, probe bool) autoContext {
	if s.memory.VRAM == 0 {
		return autoContext{Reason: "no GPU detected"}
	}
	show, err := s.ollama.Show(ctx, model)
	if err != nil {
		return autoContext{}
	}
	if _, set := ollama.ParseParameters(show.Parameters)["num_ctx"]; set {
		return autoContext{Reason: "set by the model"}
	}
	kv, ok := hw.KVCacheFromInfo(show.ModelInfo)
	if !ok {
		return autoContext{Reason: "model doesn't describe its layers"}
	}
	kv = kv.Scale(s.kvFactor(ctx))

	running, err := s.ollama.Running(ctx)
	if err != nil {
		return autoContext{}
	}
	find := func(list []ollama.RunningModel) *ollama.RunningModel {
		for i := range list {
			if sameModel(list[i].Name, model) {
				return &list[i]
			}
		}
		return nil
	}
	self := find(running)
	if self != nil && self.SizeVRAM < self.Size && self.ContextLength > 0 && aloneOnGPU(running, self) {
		s.noteOverflow(model, self.ContextLength)
	}
	if self == nil {
		if _, ok := s.ctxBase.Load(strings.ToLower(model)); !ok && probe {
			// Load at Ollama's default context just to measure the weights.
			// The real request right after reloads it with the chosen context.
			if err := s.ollama.Load(ctx, model, "1m", nil); err == nil {
				if running, err = s.ollama.Running(ctx); err == nil {
					self = find(running)
				}
			}
		}
	}

	var base int64
	loadedCtx := 0
	if self != nil && self.SizeVRAM == self.Size {
		base = self.Size - kv.Bytes(self.ContextLength)
		loadedCtx = self.ContextLength
		s.ctxBase.Store(strings.ToLower(model), base)
	} else if v, ok := s.ctxBase.Load(strings.ToLower(model)); ok {
		base = v.(int64)
	} else {
		return autoContext{Reason: "worked out when the model first loads"}
	}

	// VRAM that isn't ours to use: the desktop, browsers, other apps.
	// Ollama unloads its other models to make room, so they don't count.
	// But Ollama reports less than a model really occupies, so "used minus
	// Ollama's models" is only trustworthy when no other model is loaded.
	// Otherwise use the last such reading.
	other := s.memory.VRAM / 10
	if used := s.vramUsed(); used > 0 {
		onlyUs := true
		for _, m := range running {
			used -= m.SizeVRAM
			if self == nil || m.Name != self.Name {
				onlyUs = false
			}
		}
		if onlyUs {
			other = max(used, 0)
			s.otherVRAM.Store(other)
		} else if v := s.otherVRAM.Load(); v > 0 {
			other = v
		} else {
			other = max(used, 0)
		}
	}
	budget := s.memory.VRAM - other - ctxMargin
	modelMax := show.ContextLength()
	capped := false
	if _, ok := s.ctxCap.Load(strings.ToLower(model)); !ok {
		s.capFromBench(ctx, model, modelMax)
	}
	if v, ok := s.ctxCap.Load(strings.ToLower(model)); ok {
		if limit := v.(int); modelMax == 0 || limit < modelMax {
			modelMax, capped = limit, true
		}
	}
	if self != nil && self.SizeVRAM < self.Size {
		loadedCtx = 0 // don't hold on to a context that spilled over
	}
	c := chooseContext(base, kv, budget, modelMax, loadedCtx)
	s.log.Debug("picked context", "model", model, "ctx", c, "base", base, "budget", budget, "other", other, "loaded", loadedCtx)
	if c == 0 {
		return autoContext{Reason: "even 8k context doesn't fit on the GPU"}
	}
	if capped {
		return autoContext{Ctx: c, Reason: "largest that fit on your GPU when tried"}
	}
	return autoContext{Ctx: c, Reason: "largest that fits on your GPU"}
}

// noteOverflow records that model didn't fit on the GPU at ctx, so later
// picks stay below it. The estimate leaves out some of Ollama's buffers,
// which grow with context, so a large context can look like it fits and
// then spill onto the CPU; this is the correction.
func (s *Server) noteOverflow(model string, ctx int) {
	limit := 0
	for _, c := range ctxSteps {
		if c < ctx {
			limit = c
		}
	}
	key := strings.ToLower(model)
	if v, ok := s.ctxCap.Load(key); ok && v.(int) <= limit {
		return
	}
	s.ctxCap.Store(key, limit)
	s.log.Info("context didn't fit on the GPU; picking smaller from now on", "model", model, "ctx", ctx, "limit", limit)
}

// capFromBench starts from a saved speed test, which settled on the
// largest context that stayed on the GPU with the model alone there, so
// what was learned survives a restart.
func (s *Server) capFromBench(ctx context.Context, model string, modelMax int) {
	models, err := s.ollama.List(ctx)
	if err != nil {
		return
	}
	i := slices.IndexFunc(models, func(m ollama.Model) bool { return sameModel(m.Name, model) })
	if i < 0 {
		return
	}
	if r, ok := s.bench.Get(models[i].Digest); ok && r.GPUPct == 100 && r.Ctx > 0 && (modelMax == 0 || r.Ctx < modelMax) {
		s.ctxCap.LoadOrStore(strings.ToLower(model), r.Ctx)
	}
}

// aloneOnGPU reports whether self is the only model using VRAM, so a
// spill onto the CPU is its own context's doing.
func aloneOnGPU(running []ollama.RunningModel, self *ollama.RunningModel) bool {
	for _, m := range running {
		if m.Name != self.Name && m.SizeVRAM > 0 {
			return false
		}
	}
	return true
}

// loadAuto loads model at the context ollamui picks. If it spills onto the
// CPU anyway, it tries the next smaller context, a couple of times at most.
func (s *Server) loadAuto(ctx context.Context, model, keepAlive string) (autoContext, error) {
	var auto autoContext
	for range 3 {
		auto = s.pickContext(ctx, model, true)
		var opts map[string]any
		if auto.Ctx > 0 {
			opts = map[string]any{"num_ctx": auto.Ctx}
		}
		if err := s.ollama.Load(ctx, model, keepAlive, opts); err != nil {
			return auto, err
		}
		if auto.Ctx == 0 {
			return auto, nil
		}
		running, err := s.ollama.Running(ctx)
		if err != nil {
			return auto, nil
		}
		i := slices.IndexFunc(running, func(m ollama.RunningModel) bool { return sameModel(m.Name, model) })
		if i < 0 || running[i].SizeVRAM >= running[i].Size || !aloneOnGPU(running, &running[i]) {
			return auto, nil
		}
		s.noteOverflow(model, running[i].ContextLength)
	}
	return auto, nil
}

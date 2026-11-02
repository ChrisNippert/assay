package server

import (
	"context"
	"strings"

	"assay/internal/hw"
	"assay/internal/remote"
)

// Local teams on one GPU: when the lead and the workers are local models
// that fit in its memory together, both stay loaded (each names the other
// in keep_with, so loading one doesn't unload the other) instead of
// swapping at every step, and the workers' server has a slot per worker,
// so they write side by side. Vision is off for both to save memory.

const (
	togetherLeadCtx   = 32768
	togetherWorkerCtx = 8192 // per worker slot
	togetherMaxSlots  = 4
	// togetherSpare is what both need besides their weights: contexts,
	// compute buffers and some room. On an RX 6700 XT, Qwen3.5 9B (5.97 GB
	// of weights) with its context took 6.45 GB and Gemma 4 E2B (2.62 GB)
	// with four slots 1.78 GB, its per-layer embeddings staying in RAM.
	togetherSpare = 1_000_000_000
)

// together is how a team's local members load to stay on the GPU together.
type together struct {
	lead, worker string
	slots        int
}

// opts are mb's load options.
func (g *together) opts(mb member) map[string]any {
	keep := g.lead + "," + g.worker
	if mb.name == g.lead && g.lead != g.worker {
		return map[string]any{"num_ctx": togetherLeadCtx, "vision": "off", "keep_with": keep}
	}
	ctx := g.slots * togetherWorkerCtx
	if g.lead == g.worker {
		// One server for both: the lead's context, shared with the workers.
		ctx = max(ctx, togetherLeadCtx)
	}
	return map[string]any{"num_ctx": ctx, "num_parallel": g.slots, "vision": "off", "keep_with": keep}
}

// together says how the team's members stay loaded together, or nil when
// one is an API model or they don't fit on the GPU at once.
func (s *Server) together(ctx context.Context, t remote.Team, lead, worker member) *together {
	if lead.api || worker.api {
		return nil
	}
	size := func(name string) int64 {
		if e, ok := s.ollama.(interface{ WeightsSize(string) int64 }); ok {
			return e.WeightsSize(name)
		}
		// Without the engine's sizes: the installed size, projector and all.
		models, _ := s.ollama.List(ctx)
		for _, m := range models {
			if strings.EqualFold(m.Name, name) {
				return m.Size
			}
		}
		return 0
	}
	need := size(worker.name) + togetherSpare
	if lead.name != worker.name {
		need += size(lead.name)
	}
	if size(lead.name) == 0 || size(worker.name) == 0 || need > s.gpuRoom(ctx) {
		return nil
	}
	slots := t.Size
	if t.Strategy == remote.Agent || slots < 1 {
		slots = togetherMaxSlots
	}
	return &together{lead: lead.name, worker: worker.name, slots: min(slots, togetherMaxSlots)}
}

// gpuRoom is the GPU memory Assay's models can have: all of it but what
// other programs use now.
func (s *Server) gpuRoom(ctx context.Context) int64 {
	total := hw.Detect().VRAM
	if total == 0 {
		return 0
	}
	others := hw.VRAMUsed()
	if running, err := s.ollama.Running(ctx); err == nil {
		for _, m := range running {
			others -= m.SizeVRAM
		}
	}
	return total - max(others, 0)
}

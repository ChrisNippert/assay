package engine

import (
	"os"
	"runtime"
	"strconv"

	"ollamui/internal/hw"
)

const (
	defaultFitTarget = 1024 // MiB llama.cpp keeps free per device by default
	draftComputeMiB  = 1024 // room for a draft's context and compute buffers
	gentleUbatch     = 256
	mib              = 1 << 20
)

// liveGPU reads current memory use; tests swap it.
func (e *Engine) liveGPU() hw.Live {
	if e.Live != nil {
		return e.Live()
	}
	return hw.ReadLive()
}

// draftFile is the blob that drafts for this load (the MTP head or the
// draft model), "" if none.
func (e *Engine) draftFile(m *Model, o loadOpts) string {
	switch {
	case o.MTP && m.MTP != "":
		return e.store.Blob(m.MTP)
	case o.Draft != "":
		if d, err := e.store.Get(o.Draft); err == nil && len(d.Files) > 0 {
			return e.store.Blob(d.Files[0])
		}
	}
	return ""
}

// fitTarget returns the --fit-target to pass in MiB, and whether to pass
// one. It only applies when --fit is in effect (no hand-set layers or
// experts): the app-wide VRAM limit, and room for a draft kept on the GPU.
func (e *Engine) fitTarget(m *Model, o loadOpts) (int64, bool) {
	lim := e.Limits.Get()
	if o.GPULayers >= 0 || o.CPUMoE != cpuMoEUnset {
		if lim.MaxVRAMGB > 0 && e.opt.Log != nil {
			e.opt.Log.Info("VRAM limit can't apply: GPU layers or experts are set by hand", "model", m.Name)
		}
		return 0, false
	}
	target, set := int64(defaultFitTarget), false
	if lim.MaxVRAMGB > 0 && !o.Embedding {
		live := e.liveGPU()
		if live.VRAMTotal <= 0 {
			if e.opt.Log != nil {
				e.opt.Log.Info("VRAM limit skipped: GPU memory use is unknown")
			}
		} else {
			var other int64
			e.mu.Lock()
			for name, p := range e.procs {
				if name != m.Name {
					p.mu.Lock()
					other += p.vram
					p.mu.Unlock()
				}
			}
			e.mu.Unlock()
			free := live.VRAMTotal - live.VRAMUsed
			allowed := int64(lim.MaxVRAMGB*float64(1<<30)) - other
			target = max(defaultFitTarget, (free-allowed)/mib)
			set = true
		}
	}
	// llama.cpp's fit can't measure an MTP head or draft model ("fitting
	// without it"), so it fills the GPU with the main model and the draft
	// is squeezed in after: a Gemma 4 26B with its MTP head was left with
	// 360 MiB free and stalled. Leave the draft's room whenever one is used.
	if !o.Embedding {
		if f := e.draftFile(m, o); f != "" {
			if st, err := os.Stat(f); err == nil {
				target += st.Size()/mib + draftComputeMiB
				set = true
			}
		}
	}
	return target, set
}

// gentleArgs make llama-server yield to other apps: it can't cap GPU use,
// but a low priority, a small physical batch and fewer threads leave gaps.
func (e *Engine) gentleArgs(o loadOpts) []string {
	if !e.Limits.Get().Gentle {
		return nil
	}
	a := []string{"--prio", "-1"}
	if !o.Embedding {
		a = append(a, "-ub", strconv.Itoa(gentleUbatch))
	}
	return a
}

// gentleThreads is the thread count gentle mode caps to, when the model
// doesn't set one.
func gentleThreads() int { return max(1, runtime.NumCPU()/2) }

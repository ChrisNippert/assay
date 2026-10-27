// Package hw detects GPU and system memory and estimates whether a model fits.
package hw

import (
	"bufio"
	"os"
	"strconv"
	"strings"
)

// Memory is what's available to run models.
type Memory struct {
	GPU  string `json:"gpu"`  // vendor/name, empty if none found; "4 GPUs (...)" for several
	VRAM int64  `json:"vram"` // bytes across all cards, 0 if no GPU found
	RAM  int64  `json:"ram"`  // bytes
	// Cards lists each GPU found (one entry for a single card).
	Cards []Card `json:"cards,omitempty"`
}

// Detect finds every GPU's memory (AMD via sysfs, NVIDIA via nvidia-smi,
// any vendor via the registry on Windows) and total system RAM. VRAM is the
// total across cards. Missing pieces are left as zero.
func Detect() Memory {
	m := Memory{RAM: totalRAM()}
	cards := detectCards()
	if len(cards) == 0 {
		return m
	}
	names := make([]string, len(cards))
	for i, c := range cards {
		names[i] = c.Name
		m.VRAM += c.Total
	}
	m.GPU, m.Cards = GroupName(names), cards
	return m
}

func detectCards() []Card {
	if fake := FakeGPUs(); fake != nil {
		var out []Card
		for _, f := range fake {
			out = append(out, Card{f.Name, f.Total})
		}
		return out
	}
	if c := amdDetect("/sys/class/drm"); len(c) > 0 {
		return c
	}
	if c := nvidiaDetect(); len(c) > 0 {
		return c
	}
	if name, v := windowsVRAM(); v > 0 {
		return []Card{{name, v}}
	}
	return nil
}

func memTotal(path string) int64 {
	f, err := os.Open(path)
	if err != nil {
		return 0
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		fields := strings.Fields(sc.Text())
		if len(fields) >= 2 && fields[0] == "MemTotal:" {
			kb, _ := strconv.ParseInt(fields[1], 10, 64)
			return kb * 1024
		}
	}
	return 0
}

// amdVRAM totals the amdgpu driver's VRAM over the cards.
func amdVRAM(drmDir string) int64 {
	var sum int64
	for _, c := range amdCards(drmDir) {
		sum += c.total
	}
	return sum
}

// VRAMUsed returns how much of the GPUs' memory is in use right now by
// everything (desktop, browser, loaded models), or 0 if it can't tell.
func VRAMUsed() int64 {
	used, _, _ := sumCards(liveCards())
	return used
}

func readInt(path string) int64 {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0
	}
	v, _ := strconv.ParseInt(strings.TrimSpace(string(b)), 10, 64)
	return v
}

// Fit levels, from best to worst.
const (
	FitGPU     = "gpu"     // runs entirely on the GPU
	FitTight   = "tight"   // on the GPU with little headroom
	FitPartial = "partial" // some layers spill to system RAM: works, but slower
	FitCPU     = "cpu"     // no GPU found; runs from system RAM
	FitNone    = "none"    // doesn't fit in GPU + RAM
)

// Fit is the outcome of an estimate.
type Fit struct {
	Need    int64  `json:"need"`    // estimated bytes
	Level   string `json:"level"`   // one of the Fit* constants
	Label   string `json:"label"`   // short text for the UI
	Percent int    `json:"percent"` // Need as a percentage of VRAM (or RAM without a GPU), capped at 100
}

// Estimate guesses the memory a model needs: the weights, a KV cache for
// ctx tokens and some working buffers. It doesn't know the model's layer
// layout, so the KV part is a rule of thumb scaled by parameter count
// (MoE and hybrid models usually need less).
func Estimate(fileSize, params int64, ctx int, moe bool) int64 {
	return EstimateKV(fileSize, params, ctx, moe, 1)
}

// EstimateKV is Estimate with the cache part scaled by kvFactor, the size
// of the server's K/V cache relative to f16 (see KVTypeFactor), times the
// number of parallel requests. The weights and buffers don't scale.
func EstimateKV(fileSize, params int64, ctx int, moe bool, kvFactor float64) int64 {
	perToken := params / 1e9 * 18_000
	perToken = min(max(perToken, 20_000), 260_000)
	if moe {
		perToken /= 2
	}
	cache := int64(float64(int64(ctx)*perToken) * kvFactor)
	return fileSize + cache + 600_000_000
}

// Classify compares an estimate with the machine's memory.
func Classify(need int64, m Memory) Fit {
	f := Fit{Need: need}
	vram := m.VRAM
	if n := int64(len(m.Cards)); n > 1 {
		// Each extra card needs its own compute buffer.
		vram = max(0, vram-(n-1)*ExtraCardOverhead)
	}
	pct := func(of int64) int {
		if of <= 0 {
			return 100
		}
		return int(min(need*100/of, 100))
	}
	switch {
	case m.VRAM == 0 && need <= m.RAM*8/10:
		f.Level, f.Label, f.Percent = FitCPU, "CPU only", pct(m.RAM)
	case m.VRAM == 0:
		f.Level, f.Label, f.Percent = FitNone, "Too big", 100
	case need <= vram*9/10:
		f.Level, f.Label, f.Percent = FitGPU, "Fits on GPU", pct(vram)
	case need <= vram:
		f.Level, f.Label, f.Percent = FitTight, "Tight", pct(vram)
	case need <= vram+m.RAM*7/10:
		f.Level, f.Label, f.Percent = FitPartial, "Spills to RAM", 100
	default:
		f.Level, f.Label, f.Percent = FitNone, "Too big", 100
	}
	return f
}

// Package hw detects GPU and system memory and estimates whether a model fits.
package hw

import (
	"bufio"
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// Memory is what's available to run models.
type Memory struct {
	GPU  string `json:"gpu"`  // vendor/name, empty if none found
	VRAM int64  `json:"vram"` // bytes, 0 if no GPU found
	RAM  int64  `json:"ram"`  // bytes
}

// Detect finds the largest GPU's memory (AMD via sysfs, NVIDIA via
// nvidia-smi) and total system RAM. Missing pieces are left as zero.
func Detect() Memory {
	m := Memory{RAM: memTotal("/proc/meminfo")}
	if v := amdVRAM("/sys/class/drm"); v > 0 {
		m.GPU, m.VRAM = "AMD GPU", v
	} else if name, v := nvidiaVRAM(); v > 0 {
		m.GPU, m.VRAM = name, v
	}
	return m
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

// amdVRAM reads the amdgpu driver's VRAM size for each card and returns the largest.
func amdVRAM(drmDir string) int64 {
	paths, _ := filepath.Glob(filepath.Join(drmDir, "card*", "device", "mem_info_vram_total"))
	var best int64
	for _, p := range paths {
		b, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		if v, err := strconv.ParseInt(strings.TrimSpace(string(b)), 10, 64); err == nil && v > best {
			best = v
		}
	}
	return best
}

func nvidiaVRAM() (string, int64) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return "", 0
	}
	var name string
	var best int64
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		n, mib, ok := strings.Cut(line, ",")
		if !ok {
			continue
		}
		if v, err := strconv.ParseInt(strings.TrimSpace(mib), 10, 64); err == nil && v*1024*1024 > best {
			name, best = strings.TrimSpace(n), v*1024*1024
		}
	}
	return name, best
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
	perToken := params / 1e9 * 18_000
	perToken = min(max(perToken, 20_000), 260_000)
	if moe {
		perToken /= 2
	}
	return fileSize + int64(ctx)*perToken + 600_000_000
}

// Classify compares an estimate with the machine's memory.
func Classify(need int64, m Memory) Fit {
	f := Fit{Need: need}
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
	case need <= m.VRAM*9/10:
		f.Level, f.Label, f.Percent = FitGPU, "Fits on GPU", pct(m.VRAM)
	case need <= m.VRAM:
		f.Level, f.Label, f.Percent = FitTight, "Tight", pct(m.VRAM)
	case need <= m.VRAM+m.RAM*7/10:
		f.Level, f.Label, f.Percent = FitPartial, "Spills to RAM", 100
	default:
		f.Level, f.Label, f.Percent = FitNone, "Too big", 100
	}
	return f
}

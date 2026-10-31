package hw

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// Card is one GPU.
type Card struct {
	Name  string `json:"name"`
	Total int64  `json:"total"` // bytes of VRAM
}

// LiveCard is a point-in-time reading of one GPU.
type LiveCard struct {
	Name  string `json:"name"`
	Used  int64  `json:"used"`
	Total int64  `json:"total"`
	Busy  int    `json:"busy"` // percent, -1 if unknown
}

// ExtraCardOverhead is the VRAM each GPU beyond the first is assumed to lose
// to its own compute buffer (llama.cpp allocates one per device), which
// Classify subtracts from the total.
const ExtraCardOverhead = 512 << 20

// minRealVRAM separates a real card from an iGPU's small carve-out.
const minRealVRAM = 1 << 30

// FakeEnv names a test-only variable that makes hw report made-up GPUs:
// "name:totalMiB:usedMiB,..." (see parseFakeGPUs).
const FakeEnv = "ASSAY_FAKE_GPUS"

// FakeGPUs returns the cards ASSAY_FAKE_GPUS describes, nil if it is
// unset or malformed.
func FakeGPUs() []LiveCard { return parseFakeGPUs(os.Getenv(FakeEnv)) }

func parseFakeGPUs(s string) []LiveCard {
	if s == "" {
		return nil
	}
	var out []LiveCard
	for _, part := range strings.Split(s, ",") {
		f := strings.Split(part, ":")
		if len(f) != 3 {
			return nil
		}
		total, e1 := strconv.ParseInt(strings.TrimSpace(f[1]), 10, 64)
		used, e2 := strconv.ParseInt(strings.TrimSpace(f[2]), 10, 64)
		if e1 != nil || e2 != nil || total <= 0 {
			return nil
		}
		out = append(out, LiveCard{Name: strings.TrimSpace(f[0]), Used: used << 20, Total: total << 20, Busy: int(used % 100)})
	}
	return out
}

// GroupName is a readable name for a set of cards: the card's own name for
// one, "4 GPUs (3× Quadro P2000, GeForce GTX 1050 Ti)" for several.
func GroupName(names []string) string {
	if len(names) == 1 {
		return names[0]
	}
	var order []string
	count := map[string]int{}
	for _, n := range names {
		n = strings.TrimPrefix(n, "NVIDIA ")
		if count[n] == 0 {
			order = append(order, n)
		}
		count[n]++
	}
	parts := make([]string, len(order))
	for i, n := range order {
		parts[i] = n
		if count[n] > 1 {
			parts[i] = fmt.Sprintf("%d× %s", count[n], n)
		}
	}
	return fmt.Sprintf("%d GPUs (%s)", len(names), strings.Join(parts, ", "))
}

// amdCard is one amdgpu device directory.
type amdCard struct {
	dir   string
	total int64
}

// amdCards lists the amdgpu cards under drmDir. Small carve-outs (an iGPU's
// share of RAM) are left out when a real card exists.
func amdCards(drmDir string) []amdCard {
	paths, _ := filepath.Glob(filepath.Join(drmDir, "card*", "device", "mem_info_vram_total"))
	var all []amdCard
	big := false
	for _, p := range paths {
		if v, ok := readIntOK(p); ok && v > 0 {
			all = append(all, amdCard{filepath.Dir(p), v})
			big = big || v >= minRealVRAM
		}
	}
	if !big {
		return all
	}
	var out []amdCard
	for _, c := range all {
		if c.total >= minRealVRAM {
			out = append(out, c)
		}
	}
	return out
}

// amdName is the card's product name where the driver gives one.
func amdName(dir string) string {
	if b, err := os.ReadFile(filepath.Join(dir, "product_name")); err == nil {
		if n := strings.TrimSpace(string(b)); n != "" {
			return n
		}
	}
	return "AMD GPU"
}

// amdDetect returns every AMD card.
func amdDetect(drmDir string) []Card {
	var out []Card
	for _, c := range amdCards(drmDir) {
		out = append(out, Card{Name: amdName(c.dir), Total: c.total})
	}
	return out
}

// parseNvidiaCards reads `nvidia-smi --query-gpu=name,memory.total` lines.
func parseNvidiaCards(out string) []Card {
	var cards []Card
	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		n, mib, ok := strings.Cut(line, ",")
		if !ok {
			continue
		}
		if v, err := strconv.ParseInt(strings.TrimSpace(mib), 10, 64); err == nil && v > 0 {
			cards = append(cards, Card{Name: strings.TrimSpace(n), Total: v << 20})
		}
	}
	return cards
}

func nvidiaDetect() []Card {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return nil
	}
	return parseNvidiaCards(string(out))
}

// parseNvidiaLive reads `nvidia-smi --query-gpu=[name,]memory.used,memory.total,utilization.gpu`
// lines (MiB and percent); the name column is optional.
func parseNvidiaLive(out string) []LiveCard {
	var cards []LiveCard
	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		f := strings.Split(line, ",")
		if len(f) < 3 {
			continue
		}
		name := strings.TrimSpace(strings.Join(f[:len(f)-3], ","))
		u, e1 := strconv.ParseInt(strings.TrimSpace(f[len(f)-3]), 10, 64)
		t, e2 := strconv.ParseInt(strings.TrimSpace(f[len(f)-2]), 10, 64)
		b, e3 := strconv.Atoi(strings.TrimSpace(f[len(f)-1]))
		if e1 != nil || e2 != nil || t <= 0 {
			continue
		}
		if e3 != nil {
			b = -1
		}
		cards = append(cards, LiveCard{Name: name, Used: u << 20, Total: t << 20, Busy: b})
	}
	return cards
}

// amdLiveCards reads every AMD card's memory and busy percentage.
func amdLiveCards(drmDir string) []LiveCard {
	var out []LiveCard
	for _, c := range amdCards(drmDir) {
		u, ok := readIntOK(filepath.Join(c.dir, "mem_info_vram_used"))
		if !ok {
			continue
		}
		b, bok := readIntOK(filepath.Join(c.dir, "gpu_busy_percent"))
		if !bok {
			b = -1
		}
		out = append(out, LiveCard{Name: amdName(c.dir), Used: u, Total: c.total, Busy: int(b)})
	}
	return out
}

func nvidiaLiveCards() []LiveCard {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "nvidia-smi",
		"--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return nil
	}
	return parseNvidiaLive(string(out))
}

// sumCards totals the cards' memory. Busy is the highest card's: with
// layers split across cards they take turns, so an average would read as a
// mostly idle GPU while one of them is saturated.
func sumCards(cards []LiveCard) (used, total int64, busy int) {
	busy = -1
	for _, c := range cards {
		used += c.Used
		total += c.Total
		busy = max(busy, c.Busy)
	}
	return
}

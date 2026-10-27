package hw

import (
	"bufio"
	"io"
	"os"
	"strconv"
	"strings"
)

// Live is a point-in-time reading of GPU and system memory use. Zero totals
// mean the value is unknown (no GPU found, or not supported on this OS).
type Live struct {
	VRAMUsed  int64 `json:"vramUsed"`
	VRAMTotal int64 `json:"vramTotal"`
	GPUBusy   int   `json:"gpuBusy"` // percent, -1 if unknown; the busiest card's
	RAMUsed   int64 `json:"ramUsed"`
	RAMTotal  int64 `json:"ramTotal"`
	// Cards is each GPU's own reading, only with more than one.
	Cards []LiveCard `json:"cards,omitempty"`
}

// liveCards reads every GPU: AMD via sysfs, else NVIDIA via nvidia-smi.
func liveCards() []LiveCard {
	if fake := FakeGPUs(); fake != nil {
		return fake
	}
	if c := amdLiveCards("/sys/class/drm"); len(c) > 0 {
		return c
	}
	return nvidiaLiveCards()
}

// ReadLive reads current usage: the GPUs' memory summed, GPUBusy the busiest
// card's; RAM from /proc/meminfo (unknown elsewhere).
func ReadLive() Live {
	l := Live{GPUBusy: -1}
	if cards := liveCards(); len(cards) > 0 {
		l.VRAMUsed, l.VRAMTotal, l.GPUBusy = sumCards(cards)
		if len(cards) > 1 {
			l.Cards = cards
		}
	}
	if f, err := os.Open("/proc/meminfo"); err == nil {
		defer f.Close()
		l.RAMUsed, l.RAMTotal = parseMeminfo(f)
	}
	return l
}

// parseMeminfo returns used (total - available) and total bytes.
func parseMeminfo(r io.Reader) (used, total int64) {
	var avail int64
	sc := bufio.NewScanner(r)
	for sc.Scan() {
		f := strings.Fields(sc.Text())
		if len(f) < 2 {
			continue
		}
		kb, _ := strconv.ParseInt(f[1], 10, 64)
		switch f[0] {
		case "MemTotal:":
			total = kb * 1024
		case "MemAvailable:":
			avail = kb * 1024
		}
	}
	if total == 0 {
		return 0, 0
	}
	return total - avail, total
}

func readIntOK(path string) (int64, bool) {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, false
	}
	n, err := strconv.ParseInt(strings.TrimSpace(string(b)), 10, 64)
	return n, err == nil
}

// amdLive sums the cards.
func amdLive(drmDir string) (used, total int64, busy int, ok bool) {
	cards := amdLiveCards(drmDir)
	if len(cards) == 0 {
		return 0, 0, 0, false
	}
	used, total, busy = sumCards(cards)
	return used, total, busy, true
}

// parseNvidiaSMI parses "used, total, util" lines in MiB and percent,
// summing the cards.
func parseNvidiaSMI(out string) (used, total int64, busy int, ok bool) {
	cards := parseNvidiaLive(out)
	if len(cards) == 0 {
		return 0, 0, 0, false
	}
	used, total, busy = sumCards(cards)
	return used, total, busy, true
}

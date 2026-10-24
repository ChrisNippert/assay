package hw

import (
	"bufio"
	"context"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// Live is a point-in-time reading of GPU and system memory use. Zero totals
// mean the value is unknown (no GPU found, or not supported on this OS).
type Live struct {
	VRAMUsed  int64 `json:"vramUsed"`
	VRAMTotal int64 `json:"vramTotal"`
	GPUBusy   int   `json:"gpuBusy"` // percent, -1 if unknown
	RAMUsed   int64 `json:"ramUsed"`
	RAMTotal  int64 `json:"ramTotal"`
}

// ReadLive reads current usage: AMD via sysfs, else NVIDIA via nvidia-smi;
// RAM from /proc/meminfo (unknown elsewhere).
func ReadLive() Live {
	l := Live{GPUBusy: -1}
	if used, total, busy, ok := amdLive("/sys/class/drm"); ok {
		l.VRAMUsed, l.VRAMTotal, l.GPUBusy = used, total, busy
	} else if used, total, busy, ok := nvidiaLive(); ok {
		l.VRAMUsed, l.VRAMTotal, l.GPUBusy = used, total, busy
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

// amdLive reads the card with the most VRAM.
func amdLive(drmDir string) (used, total int64, busy int, ok bool) {
	paths, _ := filepath.Glob(filepath.Join(drmDir, "card*", "device", "mem_info_vram_total"))
	for _, p := range paths {
		t, tok := readIntOK(p)
		if !tok || t <= total {
			continue
		}
		dir := filepath.Dir(p)
		u, uok := readIntOK(filepath.Join(dir, "mem_info_vram_used"))
		if !uok {
			continue
		}
		b, bok := readIntOK(filepath.Join(dir, "gpu_busy_percent"))
		if !bok {
			b = -1
		}
		used, total, busy, ok = u, t, int(b), true
	}
	return
}

// parseNvidiaSMI parses "used, total, util" in MiB and percent.
func parseNvidiaSMI(out string) (used, total int64, busy int, ok bool) {
	line, _, _ := strings.Cut(strings.TrimSpace(out), "\n")
	f := strings.Split(line, ",")
	if len(f) < 3 {
		return 0, 0, 0, false
	}
	u, e1 := strconv.ParseInt(strings.TrimSpace(f[0]), 10, 64)
	t, e2 := strconv.ParseInt(strings.TrimSpace(f[1]), 10, 64)
	b, e3 := strconv.Atoi(strings.TrimSpace(f[2]))
	if e1 != nil || e2 != nil || t <= 0 {
		return 0, 0, 0, false
	}
	if e3 != nil {
		b = -1
	}
	return u << 20, t << 20, b, true
}

func nvidiaLive() (used, total int64, busy int, ok bool) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "nvidia-smi",
		"--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return 0, 0, 0, false
	}
	return parseNvidiaSMI(string(out))
}

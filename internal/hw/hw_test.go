package hw

import (
	"os"
	"path/filepath"
	"testing"
)

func TestAMDVRAM(t *testing.T) {
	dir := t.TempDir()
	for card, v := range map[string]string{"card0": "536870912\n", "card1": "12868124672\n"} {
		p := filepath.Join(dir, card, "device")
		os.MkdirAll(p, 0o755)
		os.WriteFile(filepath.Join(p, "mem_info_vram_total"), []byte(v), 0o644)
	}
	if got := amdVRAM(dir); got != 12868124672 {
		t.Fatalf("got %d", got)
	}
}

func TestMemTotal(t *testing.T) {
	p := filepath.Join(t.TempDir(), "meminfo")
	os.WriteFile(p, []byte("MemTotal:       49231960 kB\nMemFree: 1 kB\n"), 0o644)
	if got := memTotal(p); got != 49231960*1024 {
		t.Fatalf("got %d", got)
	}
}

func TestClassify(t *testing.T) {
	gpu := Memory{VRAM: 12e9, RAM: 48e9}
	cases := []struct {
		need int64
		mem  Memory
		want string
	}{
		{6e9, gpu, FitGPU},
		{11.5e9, gpu, FitTight},
		{30e9, gpu, FitPartial},
		{60e9, gpu, FitNone},
		{10e9, Memory{RAM: 32e9}, FitCPU},
		{40e9, Memory{RAM: 32e9}, FitNone},
	}
	for _, c := range cases {
		if got := Classify(c.need, c.mem).Level; got != c.want {
			t.Errorf("Classify(%d) = %s, want %s", c.need, got, c.want)
		}
	}
}

func TestEstimateGrowsWithContext(t *testing.T) {
	small := Estimate(5e9, 8e9, 4096, false)
	big := Estimate(5e9, 8e9, 32768, false)
	moe := Estimate(5e9, 8e9, 32768, true)
	if !(small < big && moe < big && small > 5e9) {
		t.Fatalf("small=%d big=%d moe=%d", small, big, moe)
	}
}

package hw

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseMeminfo(t *testing.T) {
	used, total := parseMeminfo(strings.NewReader("MemTotal:       1000 kB\nMemFree: 10 kB\nMemAvailable:    400 kB\n"))
	if total != 1000*1024 || used != 600*1024 {
		t.Fatalf("used=%d total=%d", used, total)
	}
	if u, tt := parseMeminfo(strings.NewReader("junk")); u != 0 || tt != 0 {
		t.Fatal("expected unknown")
	}
}

func TestAMDLive(t *testing.T) {
	d := t.TempDir()
	dev := filepath.Join(d, "card0", "device")
	os.MkdirAll(dev, 0o755)
	for n, v := range map[string]string{"mem_info_vram_total": "12000\n", "mem_info_vram_used": "3000\n", "gpu_busy_percent": "42\n"} {
		os.WriteFile(filepath.Join(dev, n), []byte(v), 0o644)
	}
	u, tot, b, ok := amdLive(d)
	if !ok || u != 3000 || tot != 12000 || b != 42 {
		t.Fatalf("%d %d %d %v", u, tot, b, ok)
	}
	if _, _, _, ok := amdLive(t.TempDir()); ok {
		t.Fatal("empty dir should be unknown")
	}
}

func TestParseNvidiaSMI(t *testing.T) {
	u, tot, b, ok := parseNvidiaSMI("1024, 8192, 17\n")
	if !ok || u != 1024<<20 || tot != 8192<<20 || b != 17 {
		t.Fatalf("%d %d %d %v", u, tot, b, ok)
	}
	if _, _, _, ok := parseNvidiaSMI("garbage"); ok {
		t.Fatal("garbage parsed")
	}
}

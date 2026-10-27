package engine

import (
	"strconv"
	"strings"
	"testing"

	"ollamui/internal/hw"
	"ollamui/internal/limits"
)

func fourGPUs() []Device {
	return []Device{
		{ID: "CUDA0", Name: "Quadro P2000", Total: 5120 << 20},
		{ID: "CUDA1", Name: "Quadro P2000", Total: 5120 << 20},
		{ID: "CUDA2", Name: "Quadro P2000", Total: 5120 << 20},
		{ID: "CUDA3", Name: "NVIDIA GeForce GTX 1050 Ti", Total: 4096 << 20},
	}
}

func TestParseDevices(t *testing.T) {
	out := "ggml_cuda_init: found 2 CUDA devices:\nAvailable devices:\n  CUDA0: Quadro P2000 (5120 MiB, 4900 MiB free)\n  Vulkan1: AMD Radeon RX 6700 XT (RADV NAVI22) (12272 MiB, 2442 MiB free)\n"
	devs := parseDevices(out)
	if len(devs) != 2 || devs[0].ID != "CUDA0" || devs[0].Total != 5120<<20 || devs[1].Name != "AMD Radeon RX 6700 XT (RADV NAVI22)" {
		t.Fatalf("%+v", devs)
	}
	if len(parseDevices("Available devices:\n  (none)\n")) != 0 {
		t.Fatal("none parsed as a device")
	}
}

func TestDeviceAndSplitArgs(t *testing.T) {
	e, m := limitEngine(t, limits.Limits{})
	e.ListDevices = fourGPUs

	a := argsFor(e, m, noSpec)
	if flagVal(a, "--device") != "" || flagVal(a, "--split-mode") != "" {
		t.Errorf("defaults passed flags: %v", a)
	}
	// Every device chosen is the default too.
	if a := argsFor(e, m, map[string]any{"gpu_devices": "CUDA0,CUDA1,CUDA2,CUDA3"}); flagVal(a, "--device") != "" {
		t.Errorf("all devices: %v", a)
	}
	a = argsFor(e, m, map[string]any{"gpu_devices": "CUDA0,CUDA2", "split_mode": "row"})
	if flagVal(a, "--device") != "CUDA0,CUDA2" || flagVal(a, "--split-mode") != "row" {
		t.Errorf("subset: %v", a)
	}
	if a := argsFor(e, m, map[string]any{"split_mode": "layer"}); flagVal(a, "--split-mode") != "" {
		t.Errorf("layer is the default: %v", a)
	}
	// A card that is gone is dropped; if none is left, all are used.
	if got := flagVal(argsFor(e, m, map[string]any{"gpu_devices": "CUDA1,CUDA9"}), "--device"); got != "CUDA1" {
		t.Errorf("stale device: %q", got)
	}
	if got := flagVal(argsFor(e, m, map[string]any{"gpu_devices": "CUDA9"}), "--device"); got != "" {
		t.Errorf("only stale devices: %q", got)
	}
	// With one GPU (or none known) the options do nothing.
	e.ListDevices = func() []Device { return fourGPUs()[:1] }
	if got := flagVal(argsFor(e, m, map[string]any{"gpu_devices": "CUDA0"}), "--device"); got != "" {
		t.Errorf("single GPU: %q", got)
	}
}

func TestFitTargetSpreadsOverCards(t *testing.T) {
	e, m := limitEngine(t, limits.Limits{MaxVRAMGB: 8})
	// 4 cards, 40 GiB total, 4 used: 36 free, 8 allowed (plus no others).
	e.Live = func() hw.Live {
		return hw.Live{VRAMTotal: 40 << 30, VRAMUsed: 4 << 30, Cards: make([]hw.LiveCard, 4)}
	}
	e.ListDevices = fourGPUs
	opts := map[string]any{"speedup": "off", "spec_ngram": "off"}
	// (36-8) GiB over 4 cards = 7 GiB each.
	want := 7 * 1024
	if got := flagVal(argsFor(e, m, opts), "--fit-target"); got != strconv.Itoa(want) {
		t.Errorf("4 cards: %s, want %d", got, want)
	}
	// On two of them the same margin is spread over two.
	opts["gpu_devices"] = "CUDA0,CUDA1"
	want = 14 * 1024
	if got := flagVal(argsFor(e, m, opts), "--fit-target"); got != strconv.Itoa(want) {
		t.Errorf("2 cards: %s, want %d", got, want)
	}
	if !strings.Contains(strings.Join(argsFor(e, m, opts), " "), "--device CUDA0,CUDA1") {
		t.Error("no --device")
	}
}

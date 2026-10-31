package engine

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"assay/internal/hw"
)

// Device is a GPU llama-server can run on, as `--list-devices` names it.
type Device struct {
	ID    string `json:"id"`    // what --device takes: CUDA0, Vulkan1, ROCm0
	Name  string `json:"name"`  // e.g. "NVIDIA GeForce GTX 1050 Ti"
	Total int64  `json:"total"` // bytes
}

var deviceLine = regexp.MustCompile(`^\s*(\w+): (.+) \((\d+) MiB, \d+ MiB free\)\s*$`)

// parseDevices reads `llama-server --list-devices` output:
//
//	Available devices:
//	  CUDA0: NVIDIA GeForce GTX 1050 Ti (4096 MiB, 3900 MiB free)
func parseDevices(out string) []Device {
	var devs []Device
	for _, l := range strings.Split(out, "\n") {
		if m := deviceLine.FindStringSubmatch(l); m != nil {
			mib, _ := strconv.ParseInt(m[3], 10, 64)
			devs = append(devs, Device{ID: m[1], Name: m[2], Total: mib << 20})
		}
	}
	return devs
}

// Devices lists the GPUs of the current llama-server. It runs
// `--list-devices` once per binary and remembers the answer; nil when it
// can't tell (then the Tune panel hides the GPU choices). The
// ASSAY_FAKE_GPUS test variable (see package hw) makes up CUDA0.. entries.
func (e *Engine) Devices() []Device {
	if e.ListDevices != nil {
		return e.ListDevices()
	}
	if fake := hw.FakeGPUs(); fake != nil {
		var devs []Device
		for i, f := range fake {
			devs = append(devs, Device{ID: "CUDA" + strconv.Itoa(i), Name: f.Name, Total: f.Total})
		}
		return devs
	}
	bin := e.ServerPath()
	if bin == "" {
		return nil
	}
	e.mu.Lock()
	devs, ok := e.devCache[bin]
	e.mu.Unlock()
	if ok {
		return devs
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, bin, "--list-devices")
	// Its shared libraries sit beside it, as in start.
	cmd.Env = append(os.Environ(), "LD_LIBRARY_PATH="+strings.TrimSuffix(filepath.Dir(bin)+":"+os.Getenv("LD_LIBRARY_PATH"), ":"))
	if v := hsaOverride(bin); v != "" {
		cmd.Env = append(cmd.Env, "HSA_OVERRIDE_GFX_VERSION="+v)
	}
	out, _ := cmd.CombinedOutput()
	devs = parseDevices(string(out))
	e.mu.Lock()
	if e.devCache == nil {
		e.devCache = map[string][]Device{}
	}
	e.devCache[bin] = devs
	e.mu.Unlock()
	return devs
}

// deviceFlag turns the chosen device list into the --device value, "" when
// the default (all GPUs) applies: nothing chosen, a name llama-server
// doesn't know, or every device chosen.
func (e *Engine) deviceFlag(chosen string) string {
	if chosen == "" {
		return "" // without asking llama-server what it has
	}
	all := e.Devices()
	if len(all) < 2 {
		return ""
	}
	var keep []string
	for _, id := range strings.Split(chosen, ",") {
		for _, d := range all {
			if d.ID == id {
				keep = append(keep, id)
				break
			}
		}
	}
	if len(keep) == 0 || len(keep) == len(all) {
		return ""
	}
	return strings.Join(keep, ",")
}

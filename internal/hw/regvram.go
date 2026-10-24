package hw

import (
	"strconv"
	"strings"
)

// parseRegVRAM reads `reg query /s` output: one block per adapter key, with
// DriverDesc and HardwareInformation.qwMemorySize lines.
func parseRegVRAM(out string) (string, int64) {
	var name, bestName string
	var size, best int64
	flush := func() {
		if size > best {
			best, bestName = size, name
		}
		name, size = "", 0
	}
	for _, line := range strings.Split(out, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "HKEY_") {
			flush()
			continue
		}
		f := strings.Fields(line)
		if len(f) < 3 {
			continue
		}
		switch f[0] {
		case "DriverDesc":
			name = strings.Join(f[2:], " ")
		case "HardwareInformation.qwMemorySize":
			if v, err := strconv.ParseInt(strings.TrimPrefix(f[2], "0x"), 16, 64); err == nil {
				size = v
			}
		}
	}
	flush()
	return bestName, best
}

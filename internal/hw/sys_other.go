//go:build !windows

package hw

func totalRAM() int64 { return memTotal("/proc/meminfo") }

func windowsVRAM() (string, int64) { return "", 0 }

package hw

import (
	"context"
	"os/exec"
	"syscall"
	"time"
	"unsafe"
)

// memoryStatusEx is Windows' MEMORYSTATUSEX.
type memoryStatusEx struct {
	Length               uint32
	MemoryLoad           uint32
	TotalPhys            uint64
	AvailPhys            uint64
	TotalPageFile        uint64
	AvailPageFile        uint64
	TotalVirtual         uint64
	AvailVirtual         uint64
	AvailExtendedVirtual uint64
}

var globalMemoryStatusEx = syscall.NewLazyDLL("kernel32.dll").NewProc("GlobalMemoryStatusEx")

func totalRAM() int64 {
	st := memoryStatusEx{}
	st.Length = uint32(unsafe.Sizeof(st))
	if r, _, _ := globalMemoryStatusEx.Call(uintptr(unsafe.Pointer(&st))); r == 0 {
		return 0
	}
	return int64(st.TotalPhys)
}

// displayClass is the registry key of the display adapters; each one's
// driver records its dedicated VRAM as HardwareInformation.qwMemorySize
// (a QWORD, unlike WMI's AdapterRAM, which stops at 4 GB).
const displayClass = `HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}`

// windowsVRAM returns the adapter with the most dedicated memory.
func windowsVRAM() (string, int64) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "reg", "query", displayClass, "/s")
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true}
	out, err := cmd.Output()
	if err != nil {
		return "", 0
	}
	return parseRegVRAM(string(out))
}

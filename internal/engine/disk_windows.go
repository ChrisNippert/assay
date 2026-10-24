package engine

import (
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"unsafe"
)

var getDiskFreeSpaceEx = syscall.NewLazyDLL("kernel32.dll").NewProc("GetDiskFreeSpaceExW")

// diskSpace asks Windows for the space on the volume holding path.
func diskSpace(path string) (free, total int64, err error) {
	p, err := syscall.UTF16PtrFromString(nearest(path))
	if err != nil {
		return 0, 0, err
	}
	var avail, size, all uint64
	r, _, e := getDiskFreeSpaceEx.Call(uintptr(unsafe.Pointer(p)), uintptr(unsafe.Pointer(&avail)), uintptr(unsafe.Pointer(&size)), uintptr(unsafe.Pointer(&all)))
	if r == 0 {
		return 0, 0, e
	}
	return int64(avail), int64(size), nil
}

// DiskFree is the space available to us on the disk holding path (or its
// closest existing parent).
func DiskFree(path string) (int64, error) {
	free, _, err := diskSpace(path)
	return free, err
}

// DiskTotal is the size of the disk holding path.
func DiskTotal(path string) (int64, error) {
	_, total, err := diskSpace(path)
	return total, err
}

// SameDisk reports whether a and b are on one volume (drive letter or
// share), so a rename between them is instant.
func SameDisk(a, b string) bool {
	va, vb := filepath.VolumeName(nearest(a)), filepath.VolumeName(nearest(b))
	return va != "" && strings.EqualFold(va, vb)
}

// linkCount is 1: Windows' FileInfo doesn't carry the link count, so hard
// links into an Ollama install are counted as the store's own files.
func linkCount(info os.FileInfo) uint64 { return 1 }

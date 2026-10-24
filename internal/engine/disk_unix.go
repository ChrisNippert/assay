//go:build unix

package engine

import (
	"os"
	"syscall"
)

// DiskFree is the space available to us on the disk holding path (or its
// closest existing parent).
func DiskFree(path string) (int64, error) {
	var st syscall.Statfs_t
	if err := syscall.Statfs(nearest(path), &st); err != nil {
		return 0, err
	}
	return int64(st.Bavail) * int64(st.Bsize), nil
}

// DiskTotal is the size of the disk holding path.
func DiskTotal(path string) (int64, error) {
	var st syscall.Statfs_t
	if err := syscall.Statfs(nearest(path), &st); err != nil {
		return 0, err
	}
	return int64(st.Blocks) * int64(st.Bsize), nil
}

// SameDisk reports whether a and b (or their closest existing parents) are
// on one filesystem, so a rename between them is instant.
func SameDisk(a, b string) bool {
	var x, y syscall.Stat_t
	if syscall.Stat(nearest(a), &x) != nil || syscall.Stat(nearest(b), &y) != nil {
		return false
	}
	return x.Dev == y.Dev
}

// linkCount is how many names a file has; more than one means a hard link,
// e.g. into an Ollama install.
func linkCount(info os.FileInfo) uint64 {
	if st, ok := info.Sys().(*syscall.Stat_t); ok {
		return uint64(st.Nlink)
	}
	return 1
}

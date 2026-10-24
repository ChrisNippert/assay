//go:build !unix && !windows

package engine

import (
	"errors"
	"os"
)

// DiskFree is the space available on the disk holding path.
func DiskFree(path string) (int64, error) { return 0, errors.New("not supported here") }

// DiskTotal is the size of the disk holding path.
func DiskTotal(path string) (int64, error) { return 0, errors.New("not supported here") }

// SameDisk reports whether a and b are on one filesystem.
func SameDisk(a, b string) bool         { return false }
func linkCount(info os.FileInfo) uint64 { return 1 }

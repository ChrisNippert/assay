package engine

import (
	"fmt"
	"os"

	"assay/internal/hf"
)

// SpareSpace is kept free after a download, so the disk isn't left full.
const SpareSpace = 1 << 30

// need is how much of files is still to be written to dsts (a finished
// file counts nothing, a .partial counts what's left).
func need(files []hf.File, dsts []string) int64 {
	var n int64
	for i, f := range files {
		left := f.Size
		if info, err := os.Stat(dsts[i]); err == nil && (f.Size == 0 || info.Size() == f.Size) {
			left = 0
		} else if info, err := os.Stat(dsts[i] + ".partial"); err == nil {
			left -= info.Size()
		}
		if left > 0 {
			n += left
		}
	}
	return n
}

// reserve checks there is room for n more bytes in the models folder,
// counting other downloads in progress, and holds it until release is
// called. With no way to read free space it lets the download go ahead.
func (e *Engine) reserve(name string, n int64) (release func(), err error) {
	free := e.Free
	if free == nil {
		free = DiskFree
	}
	dir := e.store.Blob("")
	avail, ferr := free(dir)
	e.mu.Lock()
	defer e.mu.Unlock()
	if ferr == nil && n > 0 && n+e.reserved+SpareSpace > avail {
		left := avail - e.reserved
		if left < 0 {
			left = 0
		}
		return nil, fmt.Errorf("not enough disk space for %s: it needs %s plus 1 GB spare, and %s is free in %s. Free up space, or move the models folder in Settings",
			name, sizeText(n), sizeText(left), nearest(dir))
	}
	e.reserved += n
	return func() {
		e.mu.Lock()
		e.reserved -= n
		e.mu.Unlock()
	}, nil
}

func sizeText(n int64) string {
	switch {
	case n >= 1e9:
		return fmt.Sprintf("%.1f GB", float64(n)/1e9)
	case n >= 1e6:
		return fmt.Sprintf("%.0f MB", float64(n)/1e6)
	default:
		return fmt.Sprintf("%d KB", n/1e3)
	}
}

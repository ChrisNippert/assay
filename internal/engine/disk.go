package engine

import (
	"os"
	"path/filepath"
)

// nearest returns path, or its closest ancestor that exists.
func nearest(path string) string {
	for {
		if _, err := os.Stat(path); err == nil {
			return path
		}
		up := filepath.Dir(path)
		if up == path {
			return path
		}
		path = up
	}
}

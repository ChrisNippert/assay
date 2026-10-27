// Package pick opens the machine's own folder/file chooser, for the paths
// ollamui asks the user to type. It runs a dialog tool (zenity, osascript,
// PowerShell, …) on the machine running ollamui, so callers must only offer
// it to a request that came from that machine.
package pick

import (
	"bytes"
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
)

var (
	// ErrCancelled means the user closed the dialog without choosing.
	ErrCancelled = errors.New("cancelled")
	// ErrBusy means a dialog is already open.
	ErrBusy = errors.New("a dialog is already open")
	// ErrNoTool means no dialog tool is installed (or there is no display).
	ErrNoTool = errors.New("no dialog tool available")
)

// Request describes one dialog.
type Request struct {
	Kind   string // "file" or "folder"
	Title  string
	Start  string // where to open: a folder, or a file whose folder is used
	Filter string // file kind only: a glob like "*.gguf"; empty = any file
}

// tool is one way of running a dialog: a program and its arguments.
type tool struct {
	name string
	args []string
}

var mu sync.Mutex

// lookPath finds a program. A var so tests can't open a real dialog.
var lookPath = exec.LookPath

// allowed reports whether a tool name may be used. OLLAMUI_PICK_TOOLS, when
// set (even to ""), is a comma-separated allow list; empty disables native
// dialogs. Test-only: it lets the in-app browser be checked on a desktop.
func allowed(name string) bool {
	v, ok := os.LookupEnv("OLLAMUI_PICK_TOOLS")
	if !ok {
		return true
	}
	for _, n := range strings.Split(v, ",") {
		if strings.TrimSpace(n) == name {
			return true
		}
	}
	return false
}

// find returns the first candidate whose program exists and is allowed.
func find(r Request) (tool, string, bool) {
	if !haveDisplay() {
		return tool{}, "", false
	}
	for _, t := range candidates(r) {
		if !allowed(t.name) {
			continue
		}
		if p, err := lookPath(t.name); err == nil {
			return t, p, true
		}
	}
	return tool{}, "", false
}

// Available reports whether Choose can open a dialog here.
func Available() bool {
	_, _, ok := find(Request{Kind: "folder"})
	return ok
}

// Choose opens the dialog and returns the chosen path. Only one dialog runs
// at a time (ErrBusy). It waits as long as the user takes, but gives up when
// ctx ends.
func Choose(ctx context.Context, r Request) (string, error) {
	if r.Kind != "file" && r.Kind != "folder" {
		return "", errors.New("kind must be file or folder")
	}
	if !mu.TryLock() {
		return "", ErrBusy
	}
	defer mu.Unlock()
	t, path, ok := find(r)
	if !ok {
		return "", ErrNoTool
	}
	cmd := exec.CommandContext(ctx, path, t.args...)
	var out, errb bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &errb
	hideWindow(cmd)
	err := cmd.Run()
	if ctx.Err() != nil {
		return "", ctx.Err()
	}
	got := strings.TrimRight(out.String(), "\r\n")
	if i := strings.IndexByte(got, '\n'); i >= 0 { // several lines: take the first
		got = strings.TrimRight(got[:i], "\r")
	}
	if err != nil {
		var ee *exec.ExitError
		if errors.As(err, &ee) && ee.ExitCode() == 1 && got == "" {
			return "", ErrCancelled
		}
		msg := strings.TrimSpace(errb.String())
		if msg == "" {
			msg = err.Error()
		}
		return "", errors.New(t.name + ": " + msg)
	}
	if got == "" {
		return "", ErrCancelled
	}
	return got, nil
}

// startDir turns Start into an existing folder, walking up from a path that
// isn't there, and falling back to the home folder.
func startDir(start string) string {
	p := strings.TrimSpace(start)
	for p != "" {
		if fi, err := os.Stat(p); err == nil {
			if !fi.IsDir() {
				p = filepath.Dir(p)
			}
			return p
		}
		up := filepath.Dir(p)
		if up == p {
			break
		}
		p = up
	}
	h, _ := os.UserHomeDir()
	return h
}

// ext is the extension a "*.ext" filter names, without the dot.
func ext(filter string) string {
	if strings.HasPrefix(filter, "*.") && !strings.ContainsAny(filter[2:], "*?[/") {
		return filter[2:]
	}
	return ""
}

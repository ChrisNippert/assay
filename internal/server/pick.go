package server

import (
	"errors"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"

	"ollamui/internal/pick"
)

// fromLoopback reports whether the request came from the machine running
// ollamui: only then may it open a dialog on that machine's screen.
func fromLoopback(r *http.Request) bool {
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		host = r.RemoteAddr
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// handlePickInfo says whether the native chooser can be used from here.
func (s *Server) handlePickInfo(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]bool{"native": fromLoopback(r) && pick.Available()})
}

// handlePick opens the machine's own file/folder dialog: POST {kind, title,
// start, filter}. Answers {path}, or {cancelled:true}.
func (s *Server) handlePick(w http.ResponseWriter, r *http.Request) {
	if !fromLoopback(r) {
		writeError(w, http.StatusForbidden, "the system chooser opens only for requests from the machine running ollamui")
		return
	}
	var req struct {
		Kind   string `json:"kind"`
		Title  string `json:"title"`
		Start  string `json:"start"`
		Filter string `json:"filter"`
	}
	if !readJSON(w, r, &req) {
		return
	}
	path, err := pick.Choose(r.Context(), pick.Request{Kind: req.Kind, Title: req.Title, Start: req.Start, Filter: req.Filter})
	switch {
	case err == nil:
		writeJSON(w, http.StatusOK, map[string]any{"path": path})
	case errors.Is(err, pick.ErrCancelled):
		writeJSON(w, http.StatusOK, map[string]any{"cancelled": true})
	case errors.Is(err, pick.ErrBusy):
		writeError(w, http.StatusConflict, err.Error())
	case errors.Is(err, pick.ErrNoTool):
		writeError(w, http.StatusNotImplemented, err.Error())
	case r.Context().Err() != nil:
		// the browser went away; nobody to answer
	default:
		writeError(w, http.StatusInternalServerError, err.Error())
	}
}

// fsEntry is one row of the in-app chooser.
type fsEntry struct {
	Name string `json:"name"`
	Path string `json:"path"`
	Dir  bool   `json:"dir"`
}

// fsRoot is a shortcut the in-app chooser offers.
type fsRoot struct {
	Name string `json:"name"`
	Path string `json:"path"`
}

// handleFS lists a folder on the server for the in-app chooser:
// GET /api/fs?path=&kind=file|folder&filter=*.gguf&hidden=1. Folders come
// first; files only for kind=file (matching the filter). A folder that
// can't be read comes back with "error" set, still 200, so the dialog can
// show it inline.
func (s *Server) handleFS(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	home, _ := os.UserHomeDir()
	p := strings.TrimSpace(q.Get("path"))
	if p == "" {
		p = home
	}
	if p == "" {
		p = string(filepath.Separator)
	}
	p = filepath.Clean(p)
	if !filepath.IsAbs(p) {
		writeError(w, http.StatusBadRequest, "give a full path")
		return
	}
	// A typed path that isn't there (or is a file) shows the nearest folder.
	for {
		fi, err := os.Stat(p)
		if err == nil && fi.IsDir() {
			break
		}
		up := filepath.Dir(p)
		if up == p {
			break
		}
		p = up
	}
	out := map[string]any{"path": p, "home": home, "roots": fsRoots(home), "entries": []fsEntry{}}
	if up := filepath.Dir(p); up != p {
		out["parent"] = up
	}
	des, err := os.ReadDir(p)
	if err != nil {
		out["error"] = strings.TrimPrefix(err.Error(), "open "+p+": ")
		writeJSON(w, http.StatusOK, out)
		return
	}
	wantFiles := q.Get("kind") == "file"
	filter := strings.ToLower(q.Get("filter"))
	hidden := q.Get("hidden") == "1"
	var dirs, files []fsEntry
	for _, de := range des {
		name := de.Name()
		if !hidden && strings.HasPrefix(name, ".") {
			continue
		}
		full := filepath.Join(p, name)
		isDir := de.IsDir()
		if de.Type()&os.ModeSymlink != 0 { // follow links to see what they are
			if fi, err := os.Stat(full); err == nil {
				isDir = fi.IsDir()
			}
		}
		switch {
		case isDir:
			dirs = append(dirs, fsEntry{name, full, true})
		case wantFiles:
			if filter != "" {
				if ok, _ := filepath.Match(filter, strings.ToLower(name)); !ok {
					continue
				}
			}
			files = append(files, fsEntry{name, full, false})
		}
	}
	sortNatural(dirs)
	sortNatural(files)
	out["entries"] = append(dirs, files...)
	writeJSON(w, http.StatusOK, out)
}

// fsRoots are the places worth jumping to: home, and the disks.
func fsRoots(home string) []fsRoot {
	var out []fsRoot
	add := func(name, p string) {
		if fi, err := os.Stat(p); err == nil && fi.IsDir() {
			out = append(out, fsRoot{name, p})
		}
	}
	if runtime.GOOS == "windows" {
		for c := 'A'; c <= 'Z'; c++ {
			add(string(c)+":", string(c)+`:\`)
		}
		return out
	}
	if home != "" {
		add("Home", home)
	}
	add("/", "/")
	if runtime.GOOS == "darwin" {
		add("Volumes", "/Volumes")
	} else {
		add("/mnt", "/mnt")
		add("/media", "/media")
	}
	return out
}

func sortNatural(es []fsEntry) {
	sort.SliceStable(es, func(i, j int) bool { return naturalLess(es[i].Name, es[j].Name) })
}

// naturalLess compares case-insensitively, with digit runs as numbers
// ("model2" before "model10").
func naturalLess(a, b string) bool {
	a, b = strings.ToLower(a), strings.ToLower(b)
	for a != "" && b != "" {
		if isDigit(a[0]) && isDigit(b[0]) {
			na, ra := digits(a)
			nb, rb := digits(b)
			if na != nb {
				if len(na) != len(nb) {
					return len(na) < len(nb)
				}
				return na < nb
			}
			a, b = ra, rb
			continue
		}
		if a[0] != b[0] {
			return a[0] < b[0]
		}
		a, b = a[1:], b[1:]
	}
	return len(a) < len(b)
}

func isDigit(c byte) bool { return c >= '0' && c <= '9' }

// digits splits a leading digit run (without leading zeros) from the rest.
func digits(s string) (num, rest string) {
	i := 0
	for i < len(s) && isDigit(s[i]) {
		i++
	}
	return strings.TrimLeft(s[:i], "0"), s[i:]
}

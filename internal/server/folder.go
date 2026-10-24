package server

import (
	"bufio"
	"context"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"

	"ollamui/internal/engine"
	"ollamui/internal/pulls"
)

// modelFolder is a backend whose model store can move (the engine).
type modelFolder interface {
	Store() *engine.Store
	MoveFolder(ctx context.Context, dst string, opt engine.MoveOptions) error
	FolderPinned() bool
}

// moveJob is the models folder move in progress, or the last one.
type moveJob struct {
	mu     sync.Mutex
	State  string `json:"state"` // "", "running", "done", "failed"
	To     string `json:"to"`
	Kind   string `json:"kind"`
	Done   int64  `json:"done"`
	Total  int64  `json:"total"`
	File   string `json:"file"`
	Error  string `json:"error"`
	cancel context.CancelFunc
}

func (j *moveJob) snapshot() *moveJob {
	j.mu.Lock()
	defer j.mu.Unlock()
	return &moveJob{State: j.State, To: j.To, Kind: j.Kind, Done: j.Done, Total: j.Total, File: j.File, Error: j.Error}
}

// diskMount is a mounted disk the folder picker offers.
type diskMount struct {
	Path    string `json:"path"`    // mount point
	Suggest string `json:"suggest"` // folder to put the models in
	Free    int64  `json:"free"`
	Total   int64  `json:"total"`
}

// mounts lists mounted disks (from /proc/mounts; drive letters on
// Windows) worth offering.
func mounts(cur string) []diskMount {
	if runtime.GOOS == "windows" {
		return driveLetters(cur)
	}
	f, err := os.Open("/proc/mounts")
	if err != nil {
		return nil
	}
	defer f.Close()
	var out []diskMount
	seen := map[string]bool{}
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		fields := strings.Fields(sc.Text())
		if len(fields) < 3 || !strings.HasPrefix(fields[0], "/dev/") || strings.HasPrefix(fields[0], "/dev/loop") {
			continue
		}
		path := strings.ReplaceAll(fields[1], `\040`, " ")
		if seen[fields[0]] || path == "/boot" || strings.HasPrefix(path, "/boot/") || strings.HasPrefix(path, "/efi") || strings.HasPrefix(path, "/var/lib") || strings.HasPrefix(path, "/snap") {
			continue
		}
		seen[fields[0]] = true
		if engine.SameDisk(path, cur) {
			continue // where the models are already
		}
		free, err := engine.DiskFree(path)
		total, _ := engine.DiskTotal(path)
		if err != nil || total == 0 {
			continue
		}
		m := diskMount{Path: path, Free: free, Total: total, Suggest: filepath.Join(path, "ollamui", "models")}
		if path == "/" {
			home, _ := os.UserHomeDir()
			m.Suggest = filepath.Join(home, "ollamui-models") // / itself isn't writable
		}
		out = append(out, m)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Free > out[j].Free })
	return out
}

// driveLetters offers C: to Z: (skipping the one the models are on).
func driveLetters(cur string) []diskMount {
	var out []diskMount
	for c := 'C'; c <= 'Z'; c++ {
		root := string(c) + `:\`
		if _, err := os.Stat(root); err != nil || engine.SameDisk(root, cur) {
			continue
		}
		free, err := engine.DiskFree(root)
		total, _ := engine.DiskTotal(root)
		if err != nil || total == 0 {
			continue
		}
		out = append(out, diskMount{Path: root, Free: free, Total: total, Suggest: filepath.Join(root, "ollamui", "models")})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Free > out[j].Free })
	return out
}

func (s *Server) pullsRunning() bool {
	if s.pulls == nil {
		return false
	}
	for _, j := range s.pulls.List() {
		if j.State == pulls.Running {
			return true
		}
	}
	return false
}

// folderInfo is what the settings page shows about the models folder.
func (s *Server) folderInfo() map[string]any {
	mf, ok := s.ollama.(modelFolder)
	if !ok {
		return nil
	}
	st := mf.Store()
	used, free := st.Usage()
	total, _ := engine.DiskTotal(st.Dir())
	return map[string]any{
		"dir": st.Dir(), "used": used, "free": free, "total": total,
		"pinned": mf.FolderPinned(), "mounts": mounts(st.Dir()),
		"move": s.move.snapshot(), "pulling": s.pullsRunning(),
	}
}

func (s *Server) handleFolder(w http.ResponseWriter, r *http.Request) {
	info := s.folderInfo()
	if info == nil {
		writeError(w, http.StatusNotImplemented, "the models folder needs the llama.cpp engine")
		return
	}
	writeJSON(w, http.StatusOK, info)
}

// handleFolderCheck says what moving to a folder would involve: POST {path}.
func (s *Server) handleFolderCheck(w http.ResponseWriter, r *http.Request) {
	mf, ok := s.ollama.(modelFolder)
	if !ok {
		writeError(w, http.StatusNotImplemented, "the models folder needs the llama.cpp engine")
		return
	}
	var req struct {
		Path string `json:"path"`
	}
	if !readJSON(w, r, &req) {
		return
	}
	p, err := mf.Store().Plan(req.Path)
	if err != nil {
		writeError(w, http.StatusBadRequest, capitalize(err.Error()))
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"from": p.From, "to": p.To, "kind": p.Kind, "sameDisk": p.SameDisk,
		"bytes": p.Bytes, "linked": p.Linked, "linkedCount": p.LinkedCount,
		"free": p.Free, "models": p.Models, "enough": p.SameDisk || p.Kind == engine.PlanSwitch || p.Free >= p.Bytes,
	})
}

func capitalize(s string) string {
	if s == "" {
		return s
	}
	return strings.ToUpper(s[:1]) + s[1:]
}

// handleFolderMove starts moving the models: POST {path}. Progress is
// read from GET /api/models/folder; DELETE cancels.
func (s *Server) handleFolderMove(w http.ResponseWriter, r *http.Request) {
	mf, ok := s.ollama.(modelFolder)
	if !ok {
		writeError(w, http.StatusNotImplemented, "the models folder needs the llama.cpp engine")
		return
	}
	var req struct {
		Path string `json:"path"`
	}
	if !readJSON(w, r, &req) {
		return
	}
	if s.pullsRunning() {
		writeError(w, http.StatusConflict, "A download is running. Wait for it to finish, or cancel it, then move the models.")
		return
	}
	plan, err := mf.Store().Plan(req.Path)
	if err != nil {
		writeError(w, http.StatusBadRequest, capitalize(err.Error()))
		return
	}
	if !plan.SameDisk && plan.Kind == engine.PlanMove && plan.Free < plan.Bytes {
		writeError(w, http.StatusBadRequest, "Not enough space on that disk: the models need "+formatBytes(plan.Bytes)+" and "+formatBytes(plan.Free)+" is free.")
		return
	}
	j := &s.move
	j.mu.Lock()
	if j.State == "running" {
		j.mu.Unlock()
		writeError(w, http.StatusConflict, "A move is already running.")
		return
	}
	ctx, cancel := context.WithCancel(context.Background())
	j.State, j.To, j.Kind, j.Done, j.Total, j.File, j.Error, j.cancel = "running", plan.To, plan.Kind, 0, plan.Bytes, "", "", cancel
	j.mu.Unlock()
	go func() {
		defer cancel()
		err := mf.MoveFolder(ctx, plan.To, engine.MoveOptions{Progress: func(done, total int64, file string) {
			j.mu.Lock()
			j.Done, j.Total, j.File = done, total, file
			j.mu.Unlock()
		}})
		j.mu.Lock()
		defer j.mu.Unlock()
		if err != nil {
			j.State, j.Error = "failed", capitalize(err.Error())
			if ctx.Err() != nil {
				j.Error = "Canceled. The models are where they were."
			}
			s.log.Warn("move models", "to", plan.To, "err", err)
			return
		}
		j.State = "done"
		s.log.Info("models folder changed", "to", plan.To, "kind", plan.Kind)
	}()
	writeJSON(w, http.StatusAccepted, map[string]any{"started": true})
}

func (s *Server) handleFolderCancel(w http.ResponseWriter, r *http.Request) {
	j := &s.move
	j.mu.Lock()
	if j.State == "running" && j.cancel != nil {
		j.cancel()
	}
	j.mu.Unlock()
	w.WriteHeader(http.StatusNoContent)
}

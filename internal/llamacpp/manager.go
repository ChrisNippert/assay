package llamacpp

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"
)

// Where the binary in use comes from.
const (
	SourceFlag    = "flag"    // -llama-server
	SourceCustom  = "custom"  // a llama-server picked in Settings
	SourceOwn     = "own"     // <data>/llama.cpp, the user's own build
	SourceRelease = "release" // installed here from GitHub
	SourcePath    = "path"    // found on PATH
	SourceNone    = "none"
)

// Runner is the engine the manager keeps pointed at the right binary.
type Runner interface {
	ServerPath() string
	SetServer(path string)
	StopAll()
	LoadedNames() []string
}

// Job is an install or update, for the page to show.
type Job struct {
	State       string  `json:"state"` // running, done, failed, canceled
	Tag         string  `json:"tag"`
	Build       string  `json:"build,omitempty"` // ID of the build being installed
	Phase       string  `json:"phase"`           // Downloading, Unpacking, Installing
	Completed   int64   `json:"completed"`
	Total       int64   `json:"total"`
	BytesPerSec float64 `json:"bytesPerSec"`
	Error       string  `json:"error,omitempty"`
	// Finished is when it ended (zero while running).
	Finished time.Time `json:"finished,omitzero"`
}

// Job states.
const (
	Running  = "running"
	Done     = "done"
	Failed   = "failed"
	Canceled = "canceled"
)

// Probe is what a llama-server says about itself.
type Probe struct {
	Version string   `json:"version,omitempty"`
	Devices []string `json:"devices,omitempty"`
	// Listed is whether the binary could list devices at all (older builds
	// can't, and then an empty Devices means nothing).
	Listed bool   `json:"listed"`
	Error  string `json:"error,omitempty"`
}

// Status is everything the settings page needs about llama.cpp.
type Status struct {
	Path      string `json:"path"`
	Source    string `json:"source"`
	Custom    string `json:"custom,omitempty"`    // the llama-server picked in Settings
	OwnPath   string `json:"ownPath,omitempty"`   // the user's own build, if there is one
	Installed string `json:"installed,omitempty"` // tag of the installed release
	// PreferRelease: use the installed release even though there is an own build.
	PreferRelease bool   `json:"preferRelease"`
	Supported     bool   `json:"supported"`
	Label         string `json:"label,omitempty"` // the default build: Vulkan, Metal…
	// InstalledBuild is the installed release's build ID, and InstalledLabel its name.
	InstalledBuild string `json:"installedBuild,omitempty"`
	InstalledLabel string `json:"installedLabel,omitempty"`
	Probe          *Probe `json:"probe,omitempty"` // only for the installed release in use
	Job            *Job   `json:"job,omitempty"`
}

// BuildInfo is a build on offer, for the page.
type BuildInfo struct {
	ID    string `json:"id"`
	Label string `json:"label"`
	Note  string `json:"note"`
	Size  int64  `json:"size"`
	// Files are the downloads' names and URL the first one's address.
	Files []string `json:"files"`
	URL   string   `json:"url"`
}

// Check is what GitHub has, compared with what is installed.
type Check struct {
	Supported bool   `json:"supported"`
	Label     string `json:"label,omitempty"` // of the build that Size and an update refer to
	Tag       string `json:"tag,omitempty"`
	Size      int64  `json:"size,omitempty"`
	Installed string `json:"installed,omitempty"`
	UpToDate  bool   `json:"upToDate"`
	// ReleaseURL is the release's page on GitHub.
	ReleaseURL string `json:"releaseUrl,omitempty"`
	// Builds is every build this release has for this system, default first.
	Builds []BuildInfo `json:"builds,omitempty"`
}

// ReleasePage is where a release tag is described.
const ReleasePage = "https://github.com/ggml-org/llama.cpp/releases/tag/"

// RunningError says models are loaded from the binary about to be replaced.
type RunningError struct{ Models []string }

func (e *RunningError) Error() string {
	return "models are running from the llama.cpp being updated: " + strings.Join(e.Models, ", ")
}

// Manager installs llama.cpp releases and keeps the engine on the right
// binary.
type Manager struct {
	Inst *Installer
	// Flag is the -llama-server path; it beats everything else.
	Flag string
	// Data is ollamui's data folder: <Data>/llama.cpp is the user's own build.
	Data string
	Eng  Runner
	Log  *slog.Logger

	mu      sync.Mutex
	job     *Job
	cancel  context.CancelFunc
	check   *Check
	checkAt time.Time
	cat     *Catalog
	probes  map[string]*Probe
}

// NewManager manages data/llama.cpp-release. Call Resolve to pick the
// binary for the engine's first start.
func NewManager(data, flagPath string, eng Runner, log *slog.Logger) *Manager {
	if log == nil {
		log = slog.Default()
	}
	m := &Manager{Flag: flagPath, Data: data, Eng: eng, Log: log, probes: map[string]*Probe{}}
	m.Inst = &Installer{Dir: filepath.Join(data, "llama.cpp-release")}
	m.Inst.BeforeSwap = m.beforeSwap
	return m
}

func (m *Manager) ownBinary() string {
	name := "llama-server"
	if m.Inst.exe() != name {
		name = m.Inst.exe()
	}
	p := filepath.Join(m.Data, "llama.cpp", name)
	if info, err := os.Stat(p); err == nil && !info.IsDir() {
		return p
	}
	return ""
}

func (m *Manager) prefsPath() string { return filepath.Join(m.Data, "llamacpp.json") }

// prefs are the choices saved in llamacpp.json.
type prefs struct {
	PreferRelease bool `json:"preferRelease"`
	// Custom is a llama-server the user picked; it comes before everything
	// but the -llama-server flag.
	Custom string `json:"custom,omitempty"`
}

func (m *Manager) prefs() prefs {
	var p prefs
	if b, err := os.ReadFile(m.prefsPath()); err == nil {
		json.Unmarshal(b, &p)
	}
	return p
}

func (m *Manager) savePrefs(p prefs) error {
	b, _ := json.Marshal(p)
	if err := os.MkdirAll(m.Data, 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(m.prefsPath(), b, 0o644); err != nil {
		return err
	}
	m.apply()
	return nil
}

func (m *Manager) preferRelease() bool { return m.prefs().PreferRelease }

// SetPreferRelease chooses the installed release over the user's own build
// (or back), and points the engine at it.
func (m *Manager) SetPreferRelease(v bool) error {
	p := m.prefs()
	p.PreferRelease = v
	return m.savePrefs(p)
}

// Custom is the llama-server picked in Settings, or "".
func (m *Manager) Custom() string { return m.prefs().Custom }

// SetCustom uses the llama-server at path (its libraries beside it), or
// stops using one when path is "". It must be a file that runs.
func (m *Manager) SetCustom(path string) error {
	if path != "" {
		abs, err := filepath.Abs(path)
		if err != nil {
			return err
		}
		info, err := os.Stat(abs)
		if err != nil {
			return fmt.Errorf("no file at %s", abs)
		}
		if info.IsDir() {
			// A folder: the llama-server in it.
			abs = filepath.Join(abs, m.Inst.exe())
			if info, err = os.Stat(abs); err != nil || info.IsDir() {
				return fmt.Errorf("no %s in that folder", m.Inst.exe())
			}
		}
		path = abs
	}
	p := m.prefs()
	p.Custom = path
	return m.savePrefs(p)
}

// Resolve finds the llama-server to use: the flag, then the user's own
// build in <data>/llama.cpp, then the installed release, then PATH. (A
// saved preference puts the installed release before the user's build.)
func (m *Manager) Resolve() (path, source string) {
	if p := m.Flag; p != "" {
		if info, err := os.Stat(p); err == nil && !info.IsDir() {
			return p, SourceFlag
		}
	}
	if p := m.Custom(); p != "" {
		if info, err := os.Stat(p); err == nil && !info.IsDir() {
			return p, SourceCustom
		}
	}
	own, rel := m.ownBinary(), m.Inst.Binary()
	if rel != "" && m.preferRelease() {
		return rel, SourceRelease
	}
	if own != "" {
		return own, SourceOwn
	}
	if rel != "" {
		return rel, SourceRelease
	}
	if p, err := exec.LookPath("llama-server"); err == nil {
		return p, SourcePath
	}
	return "", SourceNone
}

// apply points the engine at whatever Resolve finds now.
func (m *Manager) apply() {
	if m.Eng == nil {
		return
	}
	p, _ := m.Resolve()
	if p != m.Eng.ServerPath() {
		m.Eng.SetServer(p)
	}
}

// usingRelease says whether the engine's binary is inside the install.
func (m *Manager) usingRelease() bool {
	if m.Eng == nil {
		return false
	}
	p := m.Eng.ServerPath()
	return p != "" && filepath.Dir(p) == m.Inst.Dir
}

// beforeSwap stops the models running from the install being replaced.
func (m *Manager) beforeSwap() error {
	if m.usingRelease() && m.Eng != nil {
		m.Eng.StopAll()
	}
	return nil
}

// Status describes the current setup. It runs the binary in use once when
// that is the installed release (to see its GPU devices).
func (m *Manager) Status(ctx context.Context) Status {
	path, source := m.Resolve()
	goos, goarch := m.Inst.platform()
	label := Label(goos, goarch)
	st := Status{Path: path, Source: source, Custom: m.Custom(), OwnPath: m.ownBinary(), Installed: m.Inst.Installed(), PreferRelease: m.preferRelease(), Supported: label != "", Label: label}
	st.InstalledBuild = m.Inst.InstalledBuild()
	st.InstalledLabel = buildLabel(st.InstalledBuild, label)
	if m.Eng != nil {
		st.Path = m.Eng.ServerPath()
		if st.Path != path {
			// The engine hasn't switched (can't happen after apply); trust it.
			st.Source = SourcePath
		}
	}
	if source == SourceRelease && st.Path == path {
		st.Probe = m.probe(ctx, path)
	}
	m.mu.Lock()
	if m.job != nil {
		j := *m.job
		st.Job = &j
	}
	m.mu.Unlock()
	return st
}

var (
	versionRe = regexp.MustCompile(`(?m)^version:\s*(.+)$`)
	deviceRe  = regexp.MustCompile(`(?m)^\s+(\S+): (.+)$`)
)

// probe asks the binary for its version and devices, remembering the answer.
func (m *Manager) probe(ctx context.Context, bin string) *Probe {
	m.mu.Lock()
	info, _ := os.Stat(bin)
	key := bin
	if info != nil {
		key += "@" + info.ModTime().String()
	}
	if p, ok := m.probes[key]; ok {
		m.mu.Unlock()
		return p
	}
	m.mu.Unlock()
	p := ProbeBinary(ctx, bin)
	m.mu.Lock()
	m.probes[key] = p
	m.mu.Unlock()
	return p
}

// ProbeBinary runs llama-server --version and --list-devices. The shared
// libraries sit beside the binary.
func ProbeBinary(ctx context.Context, bin string) *Probe {
	ctx, cancel := context.WithTimeout(ctx, 20*time.Second)
	defer cancel()
	run := func(arg string) (string, error) {
		cmd := exec.CommandContext(ctx, bin, arg)
		cmd.Env = append(os.Environ(), "LD_LIBRARY_PATH="+strings.TrimSuffix(filepath.Dir(bin)+":"+os.Getenv("LD_LIBRARY_PATH"), ":"))
		var out bytes.Buffer
		cmd.Stdout, cmd.Stderr = &out, &out
		err := cmd.Run()
		return out.String(), err
	}
	p := &Probe{}
	out, err := run("--version")
	if err != nil {
		msg := strings.TrimSpace(out)
		if i := strings.LastIndex(msg, "\n"); i >= 0 {
			msg = msg[i+1:]
		}
		if msg == "" {
			msg = err.Error()
		}
		p.Error = msg
		return p
	}
	if mm := versionRe.FindStringSubmatch(out); mm != nil {
		p.Version = strings.TrimSpace(mm[1])
	}
	if out, err = run("--list-devices"); err == nil {
		p.Listed = true
		if _, rest, ok := strings.Cut(out, "Available devices:"); ok {
			for _, mm := range deviceRe.FindAllStringSubmatch(rest, -1) {
				p.Devices = append(p.Devices, strings.TrimSpace(mm[1]+": "+mm[2]))
			}
		}
	}
	return p
}

// buildLabel names a build ID; def is for installs from before builds had IDs.
func buildLabel(id, def string) string {
	switch id {
	case BuildCPU:
		return "CPU"
	case BuildVulkan:
		return "Vulkan"
	case BuildROCm:
		return "ROCm (HIP)"
	case BuildCUDA:
		return "CUDA"
	case BuildMetal:
		return "Metal"
	}
	return def
}

// current is the build an update or "Check" refers to: the installed one,
// else the default.
func (c *Catalog) current(installed string) Build {
	for _, b := range c.Builds {
		if b.ID == installed {
			return b
		}
	}
	return c.Builds[0]
}

func (c *Catalog) check(installed, installedBuild string) *Check {
	b := c.current(installedBuild)
	out := &Check{Supported: true, Label: b.Label, Tag: c.Tag, Size: b.Size(), Installed: installed, ReleaseURL: ReleasePage + c.Tag}
	out.UpToDate = installed != "" && TagNumber(installed) >= TagNumber(c.Tag)
	for _, b := range c.Builds {
		bi := BuildInfo{ID: b.ID, Label: b.Label, Note: b.Note, Size: b.Size(), URL: b.Asset.URL, Files: []string{b.Asset.Name}}
		for _, e := range b.Extra {
			bi.Files = append(bi.Files, e.Name)
		}
		out.Builds = append(out.Builds, bi)
	}
	return out
}

// Probe asks bin for its version and devices (remembered per file).
func (m *Manager) Probe(ctx context.Context, bin string) *Probe { return m.probe(ctx, bin) }

// Check looks for the newest release (at most every ten minutes unless
// fresh) and compares it with the install.
func (m *Manager) Check(ctx context.Context, fresh bool) (*Check, error) {
	m.mu.Lock()
	cat, at := m.cat, m.checkAt
	m.mu.Unlock()
	if fresh || cat == nil || time.Since(at) >= 10*time.Minute {
		var err error
		cat, err = m.Inst.Catalog(ctx)
		if errors.Is(err, ErrUnsupported) {
			return &Check{Supported: false}, nil
		}
		if err != nil {
			return nil, err
		}
		m.mu.Lock()
		m.cat, m.checkAt = cat, time.Now()
		m.mu.Unlock()
	}
	return cat.check(m.Inst.Installed(), m.Inst.InstalledBuild()), nil
}

// Install starts installing (or updating to) the newest release of build
// ("" keeps the installed build, else the default) in the background. If
// models are running from the install it would replace, it fails with a
// *RunningError unless force is set, in which case they are stopped just
// before the swap.
func (m *Manager) Install(force bool, build string) (*Job, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.job != nil && m.job.State == Running {
		j := *m.job
		return &j, nil
	}
	if !force && m.usingRelease() && m.Inst.Binary() != "" {
		if names := m.Eng.LoadedNames(); len(names) > 0 {
			return nil, &RunningError{Models: names}
		}
	}
	if build == "" {
		build = m.Inst.InstalledBuild()
	}
	ctx, cancel := context.WithCancel(context.Background())
	m.cancel = cancel
	m.job = &Job{State: Running, Phase: "Looking for the latest release", Build: build}
	go m.run(ctx, build)
	j := *m.job
	return &j, nil
}

// Cancel stops an install under way.
func (m *Manager) Cancel() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.job == nil || m.job.State != Running || m.cancel == nil {
		return false
	}
	m.cancel()
	return true
}

func (m *Manager) run(ctx context.Context, build string) {
	var lastAt time.Time
	var lastDone int64
	update := func(f func(j *Job)) {
		m.mu.Lock()
		f(m.job)
		m.mu.Unlock()
	}
	r, err := m.Inst.Latest(ctx, build)
	if err == nil {
		update(func(j *Job) { j.Tag, j.Build, j.Phase = r.Tag, r.Build, "Downloading" })
		err = m.Inst.Install(ctx, r, func(p Progress) {
			update(func(j *Job) {
				j.Phase, j.Completed, j.Total = p.Phase, p.Done, p.Total
				if p.Phase != "Downloading" {
					j.BytesPerSec = 0
					return
				}
				if now := time.Now(); lastAt.IsZero() {
					lastAt, lastDone = now, p.Done
				} else if dt := now.Sub(lastAt); dt >= time.Second {
					j.BytesPerSec = float64(p.Done-lastDone) / dt.Seconds()
					lastAt, lastDone = now, p.Done
				}
			})
		})
	}
	if err == nil {
		m.apply()
		if bin := m.Inst.Binary(); bin != "" {
			m.probe(context.Background(), bin) // warm the cache: the page asks next
		}
	}
	update(func(j *Job) {
		j.BytesPerSec = 0
		j.Finished = time.Now()
		switch {
		case errors.Is(err, context.Canceled):
			j.State, j.Phase = Canceled, "Canceled"
		case err != nil:
			j.State, j.Phase, j.Error = Failed, "Failed", err.Error()
			m.Log.Warn("llama.cpp install failed", "err", err)
		default:
			j.State, j.Phase = Done, "Installed"
			j.Completed = j.Total
			m.Log.Info("llama.cpp installed", "tag", j.Tag, "dir", m.Inst.Dir)
		}
	})
}

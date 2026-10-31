// Package service starts the app when the computer (or the user's session)
// does: a systemd user unit on Linux, a LaunchAgent on macOS, a Run key in
// the registry on Windows. Every outside effect goes through Runner and
// Home, so tests never touch the real service managers.
package service

import (
	"context"
	"encoding/xml"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"sync"
)

// Name is what the service is called on disk, in the registry and in
// launchd. It is the only place the app's name appears here.
const Name = "ollamui"

// envPassthrough are the variables worth keeping when they were set at
// install time: the service manager starts the app with almost no
// environment, and these change which GPU or llama-server it finds.
var envPassthrough = []string{
	"PATH",
	"HSA_OVERRIDE_GFX_VERSION",
	"HIP_VISIBLE_DEVICES",
	"ROCR_VISIBLE_DEVICES",
	"CUDA_VISIBLE_DEVICES",
	"GGML_VK_VISIBLE_DEVICES",
	"OLLAMA_MODELS",
	"OLLAMUI_LLAMA_SERVER",
	"OLLAMUI_VOICE",
}

// Status is what Settings and `service status` show.
type Status struct {
	Platform string `json:"platform"`
	// Supported is false where there is no service manager to use.
	Supported bool `json:"supported"`
	Enabled   bool `json:"enabled"`
	// Running is whether the service manager has it running now (always
	// false on Windows, which has none: the Run key starts it at sign-in).
	Running bool `json:"running"`
	// Command is what it runs; Where is the file or key that says so.
	Command string `json:"command,omitempty"`
	Where   string `json:"where,omitempty"`
}

// Service installs, removes and reports on the start-on-boot entry.
type Service interface {
	// Install writes the entry to run the app with args. startNow also
	// starts it (not wanted from the web page: that instance holds the port).
	// The notes are things the person should know.
	Install(ctx context.Context, args []string, startNow bool) (notes []string, err error)
	// Uninstall removes it; stop also stops the running service.
	Uninstall(ctx context.Context, stop bool) error
	Status(ctx context.Context) Status
}

// Runner runs a command and returns its combined output.
type Runner func(ctx context.Context, name string, args ...string) ([]byte, error)

// ExecRunner runs the real thing.
func ExecRunner(ctx context.Context, name string, args ...string) ([]byte, error) {
	return exec.CommandContext(ctx, name, args...).CombinedOutput()
}

// Manager is the Service for the current platform.
type Manager struct {
	GOOS string
	// Exe is the binary to start (absolute, symlinks resolved).
	Exe string
	// Home is the user's home directory; UID is the user's id (macOS).
	Home string
	UID  int
	Env  func(string) string
	Run  Runner
}

// New is a Manager for this machine, run by the real tools.
func New() (*Manager, error) {
	exe, err := Executable()
	if err != nil {
		return nil, err
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, err
	}
	return &Manager{GOOS: runtime.GOOS, Exe: exe, Home: home, UID: os.Getuid(), Env: os.Getenv, Run: ExecRunner}, nil
}

// Executable is the running binary's absolute path, symlinks resolved.
func Executable() (string, error) {
	exe, err := os.Executable()
	if err != nil {
		return "", err
	}
	if r, err := filepath.EvalSymlinks(exe); err == nil {
		exe = r
	}
	return exe, nil
}

const (
	winRunKey = `HKCU\Software\Microsoft\Windows\CurrentVersion\Run`
	label     = "local." + Name // the LaunchAgent's label
)

func (m *Manager) unitPath() string {
	return filepath.Join(m.Home, ".config", "systemd", "user", Name+".service")
}

func (m *Manager) plistPath() string {
	return filepath.Join(m.Home, "Library", "LaunchAgents", label+".plist")
}

func (m *Manager) domain() string { return fmt.Sprintf("gui/%d", m.UID) }

// env is the pass-through variables that are set, in a stable order.
func (m *Manager) env() [][2]string {
	var out [][2]string
	for _, k := range envPassthrough {
		if v := m.Env(k); v != "" {
			out = append(out, [2]string{k, v})
		}
	}
	return out
}

// Install implements Service.
func (m *Manager) Install(ctx context.Context, args []string, startNow bool) ([]string, error) {
	switch m.GOOS {
	case "linux":
		return m.installLinux(ctx, args, startNow)
	case "darwin":
		return m.installDarwin(ctx, args, startNow)
	case "windows":
		return m.installWindows(ctx, args)
	}
	return nil, fmt.Errorf("starting on boot isn't supported on %s", m.GOOS)
}

// Uninstall implements Service.
func (m *Manager) Uninstall(ctx context.Context, stop bool) error {
	switch m.GOOS {
	case "linux":
		if _, err := os.Stat(m.unitPath()); err != nil {
			return errors.New("not installed")
		}
		args := []string{"--user", "disable"}
		if stop {
			args = append(args, "--now")
		}
		if out, err := m.Run(ctx, "systemctl", append(args, Name+".service")...); err != nil {
			return cmdErr("systemctl", out, err)
		}
		if err := os.Remove(m.unitPath()); err != nil {
			return err
		}
		m.Run(ctx, "systemctl", "--user", "daemon-reload")
		return nil
	case "darwin":
		if _, err := os.Stat(m.plistPath()); err != nil {
			return errors.New("not installed")
		}
		if err := os.Remove(m.plistPath()); err != nil {
			return err
		}
		if stop {
			m.Run(ctx, "launchctl", "bootout", m.domain()+"/"+label) // not loaded is fine
		}
		return nil
	case "windows":
		if out, err := m.Run(ctx, "reg", "delete", winRunKey, "/v", Name, "/f"); err != nil {
			return cmdErr("reg", out, err)
		}
		return nil
	}
	return fmt.Errorf("starting on boot isn't supported on %s", m.GOOS)
}

// Status implements Service.
func (m *Manager) Status(ctx context.Context) Status {
	st := Status{Platform: m.GOOS, Supported: m.GOOS == "linux" || m.GOOS == "darwin" || m.GOOS == "windows"}
	switch m.GOOS {
	case "linux":
		st.Where = m.unitPath()
		b, err := os.ReadFile(st.Where)
		if err != nil {
			return st
		}
		st.Command = unitExecStart(string(b))
		_, err = m.Run(ctx, "systemctl", "--user", "is-enabled", Name+".service")
		st.Enabled = err == nil
		_, err = m.Run(ctx, "systemctl", "--user", "is-active", Name+".service")
		st.Running = err == nil
	case "darwin":
		st.Where = m.plistPath()
		b, err := os.ReadFile(st.Where)
		if err != nil {
			return st
		}
		st.Command = plistCommand(string(b))
		st.Enabled = true
		_, err = m.Run(ctx, "launchctl", "print", m.domain()+"/"+label)
		st.Running = err == nil
	case "windows":
		st.Where = winRunKey
		out, err := m.Run(ctx, "reg", "query", winRunKey, "/v", Name)
		if err != nil {
			return st
		}
		st.Command = regValue(string(out))
		st.Enabled = st.Command != ""
	}
	return st
}

func cmdErr(name string, out []byte, err error) error {
	if msg := strings.TrimSpace(string(out)); msg != "" {
		return fmt.Errorf("%s: %s", name, msg)
	}
	return fmt.Errorf("%s: %w", name, err)
}

// ---- Linux ----

func (m *Manager) installLinux(ctx context.Context, args []string, startNow bool) ([]string, error) {
	path := m.unitPath()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	if err := os.WriteFile(path, []byte(m.unitFile(args)), 0o644); err != nil {
		return nil, err
	}
	if out, err := m.Run(ctx, "systemctl", "--user", "daemon-reload"); err != nil {
		return nil, cmdErr("systemctl", out, err)
	}
	enable := []string{"--user", "enable"}
	if startNow {
		enable = append(enable, "--now")
	}
	if out, err := m.Run(ctx, "systemctl", append(enable, Name+".service")...); err != nil {
		return nil, cmdErr("systemctl", out, err)
	}
	notes := []string{"Wrote " + path + "."}
	user := m.Env("USER")
	if out, _ := m.Run(ctx, "loginctl", "show-user", user, "-p", "Linger"); !strings.Contains(string(out), "Linger=yes") {
		if user == "" {
			user = "$USER"
		}
		notes = append(notes,
			"It starts when you log in. To start at boot, before anyone logs in, run:",
			"  loginctl enable-linger "+user)
	}
	return notes, nil
}

func (m *Manager) unitFile(args []string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "[Unit]\nDescription=%s\n\n[Service]\n", Name)
	words := make([]string, 0, len(args)+1)
	for _, w := range append([]string{m.Exe}, args...) {
		words = append(words, systemdWord(w, false))
	}
	fmt.Fprintf(&b, "ExecStart=%s\n", strings.Join(words, " "))
	b.WriteString("Restart=on-failure\nRestartSec=5\n")
	for _, kv := range m.env() {
		fmt.Fprintf(&b, "Environment=%s\n", systemdWord(kv[0]+"="+kv[1], true))
	}
	b.WriteString("\n[Install]\nWantedBy=default.target\n")
	return b.String()
}

// systemdWord quotes w when it holds anything systemd would split or
// expand; % and $ are doubled either way.
func systemdWord(w string, always bool) string {
	w = strings.NewReplacer("%", "%%", "$", "$$").Replace(w)
	if !always && w != "" && !strings.ContainsAny(w, " \t\n\"'\\;") {
		return w
	}
	return `"` + strings.NewReplacer(`\`, `\\`, `"`, `\"`, "\n", `\n`).Replace(w) + `"`
}

func unitExecStart(unit string) string {
	for _, l := range strings.Split(unit, "\n") {
		if v, ok := strings.CutPrefix(l, "ExecStart="); ok {
			return v
		}
	}
	return ""
}

// ---- macOS ----

func (m *Manager) installDarwin(ctx context.Context, args []string, startNow bool) ([]string, error) {
	path := m.plistPath()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	if err := os.WriteFile(path, []byte(m.plist(args)), 0o644); err != nil {
		return nil, err
	}
	notes := []string{"Wrote " + path + "."}
	if startNow {
		m.Run(ctx, "launchctl", "bootout", m.domain()+"/"+label) // replace an older one that is loaded
		if out, err := m.Run(ctx, "launchctl", "bootstrap", m.domain(), path); err != nil {
			return nil, cmdErr("launchctl", out, err)
		}
	}
	return append(notes, "It starts whenever you log in."), nil
}

func (m *Manager) plist(args []string) string {
	var b strings.Builder
	b.WriteString(`<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>Label</key>
	<string>` + label + `</string>
	<key>ProgramArguments</key>
	<array>
`)
	for _, a := range append([]string{m.Exe}, args...) {
		b.WriteString("\t\t<string>" + xmlEscape(a) + "</string>\n")
	}
	b.WriteString("\t</array>\n\t<key>RunAtLoad</key>\n\t<true/>\n\t<key>KeepAlive</key>\n\t<true/>\n\t<key>ThrottleInterval</key>\n\t<integer>5</integer>\n")
	if env := m.env(); len(env) > 0 {
		b.WriteString("\t<key>EnvironmentVariables</key>\n\t<dict>\n")
		for _, kv := range env {
			b.WriteString("\t\t<key>" + xmlEscape(kv[0]) + "</key>\n\t\t<string>" + xmlEscape(kv[1]) + "</string>\n")
		}
		b.WriteString("\t</dict>\n")
	}
	b.WriteString("</dict>\n</plist>\n")
	return b.String()
}

func xmlEscape(s string) string {
	var b strings.Builder
	xml.EscapeText(&b, []byte(s))
	return b.String()
}

var (
	plistArgs   = regexp.MustCompile(`(?s)<key>ProgramArguments</key>\s*<array>(.*?)</array>`)
	plistString = regexp.MustCompile(`<string>(.*?)</string>`)
	xmlUnescape = strings.NewReplacer("&lt;", "<", "&gt;", ">", "&quot;", `"`, "&apos;", "'", "&#39;", "'", "&#34;", `"`, "&amp;", "&")
)

// plistCommand is the ProgramArguments as one display line.
func plistCommand(p string) string {
	m := plistArgs.FindStringSubmatch(p)
	if m == nil {
		return ""
	}
	var words []string
	for _, s := range plistString.FindAllStringSubmatch(m[1], -1) {
		words = append(words, xmlUnescape.Replace(s[1]))
	}
	return shellJoin(words)
}

// shellJoin is words as one line, quoted where they hold spaces or quotes.
func shellJoin(words []string) string {
	q := make([]string, len(words))
	for i, w := range words {
		q[i] = w
		if w == "" || strings.ContainsAny(w, " \t\"'") {
			q[i] = `"` + strings.ReplaceAll(w, `"`, `\"`) + `"`
		}
	}
	return strings.Join(q, " ")
}

// ---- Windows ----

// winCommand is the Run key's value. -open=false comes last so it beats
// anything in args: signing in shouldn't throw a browser tab at the person.
func (m *Manager) winCommand(args []string) string {
	words := append([]string{m.Exe}, args...)
	return shellJoin(append(words, "-open=false"))
}

func (m *Manager) installWindows(ctx context.Context, args []string) ([]string, error) {
	out, err := m.Run(ctx, "reg", "add", winRunKey, "/v", Name, "/t", "REG_SZ", "/d", m.winCommand(args), "/f")
	if err != nil {
		return nil, cmdErr("reg", out, err)
	}
	return []string{"Added " + Name + " to " + winRunKey + ".", "It starts the next time you sign in."}, nil
}

// regValue pulls the data out of `reg query` output:
//
//	name    REG_SZ    "C:\...\app.exe" -open=false
func regValue(out string) string {
	for _, l := range strings.Split(out, "\n") {
		if _, v, ok := strings.Cut(l, "REG_SZ"); ok {
			return strings.TrimSpace(v)
		}
	}
	return ""
}

// ---- dry run ----

// Fake keeps the entry in memory, for trying the Settings page without
// installing anything (OLLAMUI_SERVICE_DRYRUN=1).
type Fake struct {
	exe     string
	mu      sync.Mutex
	on      bool
	command string
}

// NewFake is a Service that changes nothing outside itself.
func NewFake(exe string) *Fake { return &Fake{exe: exe} }

func (f *Fake) Install(_ context.Context, args []string, _ bool) ([]string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.on, f.command = true, shellJoin(append([]string{f.exe}, args...))
	return []string{"Dry run: nothing was installed."}, nil
}

func (f *Fake) Uninstall(context.Context, bool) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if !f.on {
		return errors.New("not installed")
	}
	f.on = false
	return nil
}

func (f *Fake) Status(context.Context) Status {
	f.mu.Lock()
	defer f.mu.Unlock()
	st := Status{Platform: "dry run", Supported: true, Enabled: f.on, Where: "nowhere: dry run"}
	if f.on {
		st.Command = f.command
	}
	return st
}

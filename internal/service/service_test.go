package service

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// rec is a Runner that records commands and answers from a table.
type rec struct {
	calls []string
	fail  map[string]bool // command prefix → fail
	out   map[string]string
}

func (r *rec) run(_ context.Context, name string, args ...string) ([]byte, error) {
	line := name + " " + strings.Join(args, " ")
	r.calls = append(r.calls, line)
	for p, o := range r.out {
		if strings.HasPrefix(line, p) {
			return []byte(o), nil
		}
	}
	for p := range r.fail {
		if strings.HasPrefix(line, p) {
			return []byte("boom"), errors.New("exit status 1")
		}
	}
	return nil, nil
}

func newMgr(t *testing.T, goos string, env map[string]string) (*Manager, *rec) {
	r := &rec{}
	return &Manager{
		GOOS: goos, Exe: "/opt/my apps/" + Name, Home: t.TempDir(), UID: 501,
		Env: func(k string) string { return env[k] }, Run: r.run,
	}, r
}

func TestLinuxInstall(t *testing.T) {
	m, r := newMgr(t, "linux", map[string]string{"HSA_OVERRIDE_GFX_VERSION": "10.3.0", "USER": "sam", "PATH": "/usr/bin"})
	notes, err := m.Install(context.Background(), []string{"-listen", "0.0.0.0:8484", "-data", "/my data"}, true)
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(filepath.Join(m.Home, ".config/systemd/user/"+Name+".service"))
	if err != nil {
		t.Fatal(err)
	}
	unit := string(b)
	for _, want := range []string{
		`ExecStart="/opt/my apps/` + Name + `" -listen 0.0.0.0:8484 -data "/my data"`,
		"Restart=on-failure",
		`Environment="HSA_OVERRIDE_GFX_VERSION=10.3.0"`,
		"WantedBy=default.target",
	} {
		if !strings.Contains(unit, want) {
			t.Errorf("unit lacks %q:\n%s", want, unit)
		}
	}
	if strings.Contains(unit, "HIP_VISIBLE_DEVICES") {
		t.Errorf("unset variable written:\n%s", unit)
	}
	if got := strings.Join(r.calls[:2], "|"); got != "systemctl --user daemon-reload|systemctl --user enable --now "+Name+".service" {
		t.Errorf("calls: %v", r.calls)
	}
	if !strings.Contains(strings.Join(notes, "\n"), "loginctl enable-linger sam") {
		t.Errorf("no linger hint: %v", notes)
	}
	// With linger already on there is nothing to say.
	r.out = map[string]string{"loginctl": "Linger=yes\n"}
	notes, _ = m.Install(context.Background(), nil, false)
	if strings.Contains(strings.Join(notes, "\n"), "linger") {
		t.Errorf("linger hint with linger on: %v", notes)
	}
	if last := r.calls[len(r.calls)-2]; strings.Contains(last, "--now") {
		t.Errorf("startNow=false still started it: %s", last)
	}
	for _, c := range r.calls {
		if strings.Contains(c, "enable-linger") && !strings.HasPrefix(c, "loginctl show") {
			t.Errorf("must not run enable-linger: %s", c)
		}
	}
}

func TestLinuxStatusAndUninstall(t *testing.T) {
	m, r := newMgr(t, "linux", nil)
	ctx := context.Background()
	if st := m.Status(ctx); st.Enabled || st.Command != "" || !st.Supported {
		t.Fatalf("before install: %+v", st)
	}
	if err := m.Uninstall(ctx, true); err == nil {
		t.Fatal("uninstalling nothing should fail")
	}
	if _, err := m.Install(ctx, []string{"-open=false"}, true); err != nil {
		t.Fatal(err)
	}
	st := m.Status(ctx)
	if !st.Enabled || !st.Running || !strings.HasSuffix(st.Command, "-open=false") {
		t.Fatalf("after install: %+v", st)
	}
	r.fail = map[string]bool{"systemctl --user is-active": true}
	if st := m.Status(ctx); st.Running || !st.Enabled {
		t.Fatalf("stopped: %+v", st)
	}
	if err := m.Uninstall(ctx, true); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(strings.Join(r.calls, "|"), "disable --now "+Name+".service") {
		t.Errorf("calls: %v", r.calls)
	}
	if _, err := os.Stat(m.unitPath()); err == nil {
		t.Error("unit file left behind")
	}
}

func TestLinuxInstallFailure(t *testing.T) {
	m, r := newMgr(t, "linux", nil)
	r.fail = map[string]bool{"systemctl --user enable": true}
	if _, err := m.Install(context.Background(), nil, true); err == nil || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("err: %v", err)
	}
}

func TestSystemdQuoting(t *testing.T) {
	for in, want := range map[string]string{
		"plain":    "plain",
		"a b":      `"a b"`,
		`say "hi"`: `"say \"hi\""`,
		"50%":      "50%%",
		"$HOME":    "$$HOME",
		"":         `""`,
	} {
		if got := systemdWord(in, false); got != want {
			t.Errorf("systemdWord(%q) = %s, want %s", in, got, want)
		}
	}
}

func TestDarwin(t *testing.T) {
	m, r := newMgr(t, "darwin", map[string]string{"HSA_OVERRIDE_GFX_VERSION": "11.0.0"})
	ctx := context.Background()
	if _, err := m.Install(ctx, []string{"-listen", "0.0.0.0:8484", "-voice", "a&b"}, true); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(filepath.Join(m.Home, "Library/LaunchAgents/local."+Name+".plist"))
	if err != nil {
		t.Fatal(err)
	}
	p := string(b)
	for _, want := range []string{
		"<string>local." + Name + "</string>", "<string>/opt/my apps/" + Name + "</string>",
		"<string>0.0.0.0:8484</string>", "<string>a&amp;b</string>",
		"<key>KeepAlive</key>", "<key>HSA_OVERRIDE_GFX_VERSION</key>",
	} {
		if !strings.Contains(p, want) {
			t.Errorf("plist lacks %q:\n%s", want, p)
		}
	}
	if !strings.Contains(strings.Join(r.calls, "|"), "launchctl bootstrap gui/501 "+m.plistPath()) {
		t.Errorf("calls: %v", r.calls)
	}
	st := m.Status(ctx)
	if !st.Enabled || !st.Running || !strings.Contains(st.Command, `"/opt/my apps/`+Name+`" -listen 0.0.0.0:8484 -voice a&b`) {
		t.Fatalf("status: %+v", st)
	}
	if err := m.Uninstall(ctx, true); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(strings.Join(r.calls, "|"), "launchctl bootout gui/501/local."+Name) {
		t.Errorf("calls: %v", r.calls)
	}
	if st := m.Status(ctx); st.Enabled {
		t.Fatalf("still enabled: %+v", st)
	}
}

func TestWindows(t *testing.T) {
	m, r := newMgr(t, "windows", nil)
	m.Exe = `C:\Program Files\App\` + Name + ".exe"
	ctx := context.Background()
	if _, err := m.Install(ctx, []string{"-listen", "0.0.0.0:8484"}, true); err != nil {
		t.Fatal(err)
	}
	want := `reg add HKCU\Software\Microsoft\Windows\CurrentVersion\Run /v ` + Name + ` /t REG_SZ /d "C:\Program Files\App\` + Name + `.exe" -listen 0.0.0.0:8484 -open=false /f`
	if r.calls[0] != want {
		t.Errorf("got  %s\nwant %s", r.calls[0], want)
	}
	r.out = map[string]string{"reg query": "\r\nHKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Run\r\n    " + Name + "    REG_SZ    \"C:\\x\\a.exe\" -open=false\r\n\r\n"}
	st := m.Status(ctx)
	if !st.Enabled || st.Command != `"C:\x\a.exe" -open=false` {
		t.Fatalf("status: %+v", st)
	}
	r.out = nil
	r.fail = map[string]bool{"reg query": true}
	if st := m.Status(ctx); st.Enabled {
		t.Fatalf("missing value reads as enabled: %+v", st)
	}
	r.fail = nil
	if err := m.Uninstall(ctx, true); err != nil {
		t.Fatal(err)
	}
	if got := r.calls[len(r.calls)-1]; got != `reg delete HKCU\Software\Microsoft\Windows\CurrentVersion\Run /v `+Name+` /f` {
		t.Errorf("delete: %s", got)
	}
}

func TestFake(t *testing.T) {
	f := NewFake("/x/app")
	ctx := context.Background()
	if f.Status(ctx).Enabled {
		t.Fatal("starts off")
	}
	f.Install(ctx, []string{"-listen", ":1"}, false)
	if st := f.Status(ctx); !st.Enabled || st.Command != "/x/app -listen :1" {
		t.Fatalf("%+v", st)
	}
	if err := f.Uninstall(ctx, false); err != nil || f.Status(ctx).Enabled {
		t.Fatal("uninstall")
	}
}

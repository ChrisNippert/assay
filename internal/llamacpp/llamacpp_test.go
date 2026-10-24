package llamacpp

import (
	"archive/tar"
	"archive/zip"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func fakeTarGz(t *testing.T, top string, extra func(*tar.Writer)) []byte {
	t.Helper()
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gz)
	add := func(name, body string, mode int64) {
		tw.WriteHeader(&tar.Header{Name: name, Mode: mode, Size: int64(len(body)), Typeflag: tar.TypeReg})
		tw.Write([]byte(body))
	}
	if top != "" {
		tw.WriteHeader(&tar.Header{Name: top + "/", Typeflag: tar.TypeDir, Mode: 0o755})
		top += "/"
	}
	add(top+"llama-server", "#!/bin/sh\necho version: 1\n", 0o755)
	add(top+"libggml-vulkan.so", "lib", 0o755)
	tw.WriteHeader(&tar.Header{Name: top + "libggml.so", Typeflag: tar.TypeSymlink, Linkname: "libggml-vulkan.so"})
	if extra != nil {
		extra(tw)
	}
	tw.Close()
	gz.Close()
	return buf.Bytes()
}

func fakeZip(t *testing.T) []byte {
	t.Helper()
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	w, _ := zw.Create("llama-server.exe")
	w.Write([]byte("MZ"))
	w, _ = zw.Create("ggml-vulkan.dll")
	w.Write([]byte("dll"))
	zw.Close()
	return buf.Bytes()
}

// fakeGitHub serves a release list and the files it points to.
type fakeGitHub struct {
	*httptest.Server
	files   map[string][]byte
	hits    int
	status  int // non-zero: answer the list with this
	headers map[string]string
}

func newFakeGitHub(t *testing.T) *fakeGitHub {
	g := &fakeGitHub{files: map[string][]byte{}}
	mux := http.NewServeMux()
	mux.HandleFunc("/releases", func(w http.ResponseWriter, r *http.Request) {
		g.hits++
		for k, v := range g.headers {
			w.Header().Set(k, v)
		}
		if g.status != 0 {
			w.WriteHeader(g.status)
			return
		}
		asset := func(name string) map[string]any {
			a := map[string]any{"name": name, "browser_download_url": g.URL + "/dl/" + name, "size": len(g.files[name])}
			if b, ok := g.files[name]; ok {
				sum := sha256.Sum256(b)
				a["digest"] = "sha256:" + hex.EncodeToString(sum[:])
			}
			return a
		}
		json.NewEncoder(w).Encode([]any{
			// Newest has no Linux build yet (CI still running): skipped.
			map[string]any{"tag_name": "b200", "assets": []any{asset("llama-b200-bin-win-vulkan-x64.zip")}},
			map[string]any{"tag_name": "b150", "draft": true, "assets": []any{asset("llama-b150-bin-ubuntu-vulkan-x64.tar.gz")}},
			map[string]any{"tag_name": "b100", "prerelease": true, "assets": []any{
				asset("llama-b100-bin-ubuntu-x64.tar.gz"),
				asset("llama-b100-bin-ubuntu-vulkan-x64.tar.gz"),
				asset("llama-b100-bin-macos-arm64.tar.gz"),
				asset("llama-b100-bin-win-vulkan-x64.zip"),
			}},
		})
	})
	mux.HandleFunc("/dl/", func(w http.ResponseWriter, r *http.Request) {
		b, ok := g.files[strings.TrimPrefix(r.URL.Path, "/dl/")]
		if !ok {
			http.NotFound(w, r)
			return
		}
		w.Write(b)
	})
	g.Server = httptest.NewServer(mux)
	t.Cleanup(g.Close)
	return g
}

func linuxInstaller(g *fakeGitHub, dir string) *Installer {
	return &Installer{Dir: dir, API: g.URL + "/releases", HTTP: g.Client(), GOOS: "linux", GOARCH: "amd64"}
}

func TestPickAsset(t *testing.T) {
	tag := "b100"
	names := []string{
		"llama-b100-bin-ubuntu-x64.tar.gz", "llama-b100-bin-ubuntu-vulkan-x64.tar.gz", "llama-b100-bin-ubuntu-vulkan-arm64.tar.gz",
		"llama-b100-bin-macos-arm64.tar.gz", "llama-b100-bin-macos-x64.tar.gz",
		"llama-b100-bin-win-vulkan-x64.zip", "llama-b100-bin-win-cpu-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip",
	}
	var assets []Asset
	for _, n := range names {
		assets = append(assets, Asset{Name: n})
	}
	for _, c := range []struct{ os, arch, want string }{
		{"linux", "amd64", "llama-b100-bin-ubuntu-vulkan-x64.tar.gz"},
		{"linux", "arm64", "llama-b100-bin-ubuntu-vulkan-arm64.tar.gz"},
		{"darwin", "arm64", "llama-b100-bin-macos-arm64.tar.gz"},
		{"windows", "amd64", "llama-b100-bin-win-vulkan-x64.zip"},
	} {
		a, err := PickAsset(assets, tag, c.os, c.arch)
		if err != nil || a.Name != c.want {
			t.Errorf("%s/%s: got %q, %v; want %q", c.os, c.arch, a.Name, err, c.want)
		}
	}
	for _, c := range [][2]string{{"freebsd", "amd64"}, {"linux", "riscv64"}, {"windows", "386"}} {
		if _, err := PickAsset(assets, tag, c[0], c[1]); !errors.Is(err, ErrUnsupported) {
			t.Errorf("%v: want ErrUnsupported, got %v", c, err)
		}
	}
	if _, err := PickAsset(assets, "b999", "linux", "amd64"); !errors.Is(err, ErrUnsupported) {
		t.Errorf("wrong tag: got %v", err)
	}
}

func TestLatest(t *testing.T) {
	g := newFakeGitHub(t)
	g.files["llama-b100-bin-ubuntu-vulkan-x64.tar.gz"] = []byte("x")
	rel, err := linuxInstaller(g, t.TempDir()).Latest(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if rel.Tag != "b100" || rel.Asset.Name != "llama-b100-bin-ubuntu-vulkan-x64.tar.gz" || rel.Asset.Digest == "" {
		t.Errorf("got %+v", rel)
	}
	in := linuxInstaller(g, t.TempDir())
	in.GOOS, in.GOARCH = "plan9", "mips"
	if _, err := in.Latest(context.Background()); !errors.Is(err, ErrUnsupported) {
		t.Errorf("unsupported system: %v", err)
	}
}

func TestLatestRateLimit(t *testing.T) {
	g := newFakeGitHub(t)
	g.status = http.StatusForbidden
	g.headers = map[string]string{"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": fmt.Sprint(time.Now().Add(time.Hour).Unix())}
	_, err := linuxInstaller(g, t.TempDir()).Latest(context.Background())
	var rl *RateLimitError
	if !errors.As(err, &rl) || rl.Reset.IsZero() || !strings.Contains(err.Error(), "rate limit") {
		t.Errorf("got %v", err)
	}
	g.status, g.headers = http.StatusForbidden, nil // a plain 403 is not a rate limit
	if _, err = linuxInstaller(g, t.TempDir()).Latest(context.Background()); errors.As(err, &rl) {
		t.Errorf("plain 403 taken for a rate limit")
	}
	g.status = http.StatusTooManyRequests
	if _, err = linuxInstaller(g, t.TempDir()).Latest(context.Background()); !errors.As(err, &rl) {
		t.Errorf("429: got %v", err)
	}
}

func TestInstallSwapLeavesOwnBuildAlone(t *testing.T) {
	g := newFakeGitHub(t)
	g.files["llama-b100-bin-ubuntu-vulkan-x64.tar.gz"] = fakeTarGz(t, "llama-b100", nil)
	data := t.TempDir()
	own := filepath.Join(data, "llama.cpp")
	os.MkdirAll(own, 0o755)
	os.WriteFile(filepath.Join(own, "llama-server"), []byte("mine"), 0o755)
	os.WriteFile(filepath.Join(own, "libggml-hip.so"), []byte("rocm"), 0o755)

	in := linuxInstaller(g, filepath.Join(data, "llama.cpp-release"))
	ctx := context.Background()
	rel, err := in.Latest(ctx)
	if err != nil {
		t.Fatal(err)
	}
	var phases []string
	if err := in.Install(ctx, rel, func(p Progress) {
		if len(phases) == 0 || phases[len(phases)-1] != p.Phase {
			phases = append(phases, p.Phase)
		}
	}); err != nil {
		t.Fatal(err)
	}
	if strings.Join(phases, ",") != "Downloading,Unpacking,Installing" {
		t.Errorf("phases %v", phases)
	}
	if in.Binary() != filepath.Join(in.Dir, "llama-server") || in.Installed() != "b100" {
		t.Errorf("binary %q installed %q", in.Binary(), in.Installed())
	}
	if l, err := os.Readlink(filepath.Join(in.Dir, "libggml.so")); err != nil || l != "libggml-vulkan.so" {
		t.Errorf("symlink: %q %v", l, err)
	}
	if info, _ := os.Stat(in.Binary()); info.Mode()&0o111 == 0 {
		t.Errorf("llama-server isn't executable: %v", info.Mode())
	}

	// Update: new content replaces the old one completely, nothing left over.
	os.WriteFile(filepath.Join(in.Dir, "stale-file"), []byte("old"), 0o644)
	g.files["llama-b100-bin-ubuntu-vulkan-x64.tar.gz"] = fakeTarGz(t, "", nil) // no top folder this time
	rel, _ = in.Latest(ctx)
	if err := in.Install(ctx, rel, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(in.Dir, "stale-file")); !os.IsNotExist(err) {
		t.Errorf("old files survived the swap")
	}
	for _, leftover := range []string{in.Dir + ".new", in.Dir + ".old"} {
		if _, err := os.Stat(leftover); !os.IsNotExist(err) {
			t.Errorf("%s left behind", leftover)
		}
	}
	if b, _ := os.ReadFile(filepath.Join(own, "llama-server")); string(b) != "mine" {
		t.Errorf("own build changed: %q", b)
	}
	if b, _ := os.ReadFile(filepath.Join(own, "libggml-hip.so")); string(b) != "rocm" {
		t.Errorf("own build's libraries changed")
	}
	entries, _ := os.ReadDir(data)
	for _, e := range entries {
		if e.Name() != "llama.cpp" && e.Name() != "llama.cpp-release" {
			t.Errorf("stray entry %q in the data folder", e.Name())
		}
	}
}

func TestInstallRefusesTheUsersFolder(t *testing.T) {
	in := &Installer{Dir: filepath.Join(t.TempDir(), "llama.cpp")}
	if err := in.Install(context.Background(), &Release{}, nil); err == nil {
		t.Error("installed into llama.cpp")
	}
}

func TestFailedInstallKeepsTheOldOne(t *testing.T) {
	g := newFakeGitHub(t)
	name := "llama-b100-bin-ubuntu-vulkan-x64.tar.gz"
	g.files[name] = fakeTarGz(t, "llama-b100", nil)
	in := linuxInstaller(g, filepath.Join(t.TempDir(), "llama.cpp-release"))
	ctx := context.Background()
	rel, _ := in.Latest(ctx)
	if err := in.Install(ctx, rel, nil); err != nil {
		t.Fatal(err)
	}

	bad := map[string][]byte{
		"corrupt":   []byte("not a gzip"),
		"no server": fakeTarGz(t, "", func(tw *tar.Writer) {}), // replaced below
		"traversal": fakeTarGz(t, "x", func(tw *tar.Writer) {
			tw.WriteHeader(&tar.Header{Name: "../../evil", Mode: 0o644, Size: 1, Typeflag: tar.TypeReg})
			tw.Write([]byte("e"))
		}),
		"link out": fakeTarGz(t, "x", func(tw *tar.Writer) {
			tw.WriteHeader(&tar.Header{Name: "x/l", Typeflag: tar.TypeSymlink, Linkname: "../../etc/passwd"})
		}),
		"absolute": fakeTarGz(t, "x", func(tw *tar.Writer) {
			tw.WriteHeader(&tar.Header{Name: "/tmp/evil", Mode: 0o644, Size: 1, Typeflag: tar.TypeReg})
			tw.Write([]byte("e"))
		}),
	}
	var noServer bytes.Buffer
	gz := gzip.NewWriter(&noServer)
	tw := tar.NewWriter(gz)
	tw.WriteHeader(&tar.Header{Name: "readme", Mode: 0o644, Size: 1, Typeflag: tar.TypeReg})
	tw.Write([]byte("r"))
	tw.Close()
	gz.Close()
	bad["no server"] = noServer.Bytes()
	for what, b := range bad {
		g.files[name] = b
		rel2 := *rel
		rel2.Asset.Digest = "" // the fake's digest was for the good file
		if err := in.Install(ctx, &rel2, nil); err == nil {
			t.Errorf("%s: install succeeded", what)
		}
		if in.Installed() != "b100" {
			t.Errorf("%s: old install lost (%q)", what, in.Installed())
		}
		if _, err := os.Stat(in.Dir + ".new"); !os.IsNotExist(err) {
			t.Errorf("%s: staging left behind", what)
		}
	}
	// A wrong checksum is refused too.
	g.files[name] = fakeTarGz(t, "llama-b100", func(tw *tar.Writer) {
		tw.WriteHeader(&tar.Header{Name: "llama-b100/other", Mode: 0o644, Size: 1, Typeflag: tar.TypeReg})
		tw.Write([]byte("o"))
	})
	rel3 := *rel
	rel3.Tag = "b101"
	if err := in.Install(ctx, &rel3, nil); err == nil || !strings.Contains(err.Error(), "checksum") {
		t.Errorf("checksum: %v", err)
	}
	if in.Installed() != "b100" {
		t.Errorf("old install lost after checksum failure")
	}
	// And so is a download that doesn't exist.
	rel4 := *rel
	rel4.Asset.URL = g.URL + "/dl/missing"
	if err := in.Install(ctx, &rel4, nil); err == nil {
		t.Error("missing download accepted")
	}
}

func TestInstallWindowsZip(t *testing.T) {
	g := newFakeGitHub(t)
	g.files["llama-b200-bin-win-vulkan-x64.zip"] = fakeZip(t)
	in := linuxInstaller(g, filepath.Join(t.TempDir(), "llama.cpp-release"))
	in.GOOS = "windows"
	rel, err := in.Latest(context.Background())
	if err != nil || rel.Tag != "b200" {
		t.Fatalf("%+v %v", rel, err)
	}
	if err := in.Install(context.Background(), rel, nil); err != nil {
		t.Fatal(err)
	}
	if filepath.Base(in.Binary()) != "llama-server.exe" || in.Installed() != "b200" {
		t.Errorf("binary %q installed %q", in.Binary(), in.Installed())
	}
}

func TestCancelledInstall(t *testing.T) {
	g := newFakeGitHub(t)
	g.files["llama-b100-bin-ubuntu-vulkan-x64.tar.gz"] = fakeTarGz(t, "llama-b100", nil)
	in := linuxInstaller(g, filepath.Join(t.TempDir(), "llama.cpp-release"))
	rel, _ := in.Latest(context.Background())
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := in.Install(ctx, rel, nil); !errors.Is(err, context.Canceled) {
		t.Errorf("got %v", err)
	}
	if in.Binary() != "" {
		t.Error("installed anyway")
	}
}

type fakeRunner struct {
	path    string
	loaded  []string
	stopped int
}

func (f *fakeRunner) ServerPath() string    { return f.path }
func (f *fakeRunner) SetServer(p string)    { f.path = p }
func (f *fakeRunner) StopAll()              { f.stopped++; f.loaded = nil }
func (f *fakeRunner) LoadedNames() []string { return f.loaded }

func newTestManager(t *testing.T, g *fakeGitHub, flag string) (*Manager, *fakeRunner, string) {
	data := t.TempDir()
	t.Setenv("PATH", t.TempDir()) // no llama-server on PATH
	fr := &fakeRunner{}
	m := NewManager(data, flag, fr, nil)
	m.Inst = linuxInstaller(g, filepath.Join(data, "llama.cpp-release"))
	m.Inst.BeforeSwap = m.beforeSwap
	return m, fr, data
}

func waitJob(t *testing.T, m *Manager) *Job {
	t.Helper()
	for range 400 {
		m.mu.Lock()
		j := *m.job
		m.mu.Unlock()
		if j.State != Running {
			return &j
		}
		time.Sleep(25 * time.Millisecond)
	}
	t.Fatal("install didn't finish")
	return nil
}

func TestResolveOrder(t *testing.T) {
	g := newFakeGitHub(t)
	flagBin := filepath.Join(t.TempDir(), "my-llama-server")
	os.WriteFile(flagBin, []byte("x"), 0o755)
	m, _, data := newTestManager(t, g, "")
	if p, src := m.Resolve(); p != "" || src != SourceNone {
		t.Fatalf("empty: %q %q", p, src)
	}
	// Installed release only.
	os.MkdirAll(m.Inst.Dir, 0o755)
	os.WriteFile(filepath.Join(m.Inst.Dir, "llama-server"), []byte("x"), 0o755)
	if p, src := m.Resolve(); src != SourceRelease || p != filepath.Join(m.Inst.Dir, "llama-server") {
		t.Fatalf("release: %q %q", p, src)
	}
	// The user's own build wins over the release...
	os.MkdirAll(filepath.Join(data, "llama.cpp"), 0o755)
	own := filepath.Join(data, "llama.cpp", "llama-server")
	os.WriteFile(own, []byte("x"), 0o755)
	if p, src := m.Resolve(); src != SourceOwn || p != own {
		t.Fatalf("own: %q %q", p, src)
	}
	// ...unless the user chose the release.
	if err := m.SetPreferRelease(true); err != nil {
		t.Fatal(err)
	}
	if _, src := m.Resolve(); src != SourceRelease {
		t.Fatalf("preferred release: %q", src)
	}
	if m.Eng.ServerPath() != filepath.Join(m.Inst.Dir, "llama-server") {
		t.Errorf("engine not switched: %q", m.Eng.ServerPath())
	}
	m.SetPreferRelease(false)
	if _, src := m.Resolve(); src != SourceOwn {
		t.Fatalf("back to own: %q", src)
	}
	// The flag beats everything; a flag naming nothing is ignored.
	m.Flag = flagBin
	if p, src := m.Resolve(); src != SourceFlag || p != flagBin {
		t.Fatalf("flag: %q %q", p, src)
	}
	m.Flag = filepath.Join(data, "nope")
	if _, src := m.Resolve(); src != SourceOwn {
		t.Fatalf("missing flag path: %q", src)
	}
}

func TestManagerInstallSwitchesTheEngine(t *testing.T) {
	g := newFakeGitHub(t)
	g.files["llama-b100-bin-ubuntu-vulkan-x64.tar.gz"] = fakeTarGz(t, "llama-b100", nil)
	m, fr, data := newTestManager(t, g, "")

	c, err := m.Check(context.Background(), true)
	if err != nil || !c.Supported || c.Tag != "b100" || c.UpToDate || c.Size == 0 || c.Label != "Vulkan" {
		t.Fatalf("check: %+v %v", c, err)
	}
	if _, err := m.Check(context.Background(), false); err != nil || g.hits != 1 {
		t.Errorf("cached check hit GitHub again (%d hits, %v)", g.hits, err)
	}

	if _, err := m.Install(false); err != nil {
		t.Fatal(err)
	}
	j := waitJob(t, m)
	if j.State != Done || j.Tag != "b100" {
		t.Fatalf("job %+v", j)
	}
	if want := filepath.Join(data, "llama.cpp-release", "llama-server"); fr.path != want {
		t.Errorf("engine path %q, want %q", fr.path, want)
	}
	st := m.Status(context.Background())
	if st.Source != SourceRelease || st.Installed != "b100" || !st.Supported || st.Probe == nil {
		t.Errorf("status %+v", st)
	}
	if c, _ := m.Check(context.Background(), false); !c.UpToDate {
		t.Errorf("not up to date after installing the newest: %+v", c)
	}
}

func TestUpdateWithRunningModels(t *testing.T) {
	g := newFakeGitHub(t)
	g.files["llama-b100-bin-ubuntu-vulkan-x64.tar.gz"] = fakeTarGz(t, "llama-b100", nil)
	m, fr, _ := newTestManager(t, g, "")
	m.Install(false)
	waitJob(t, m)

	fr.loaded = []string{"qwen3:8b"}
	_, err := m.Install(false)
	var re *RunningError
	if !errors.As(err, &re) || re.Models[0] != "qwen3:8b" {
		t.Fatalf("want RunningError, got %v", err)
	}
	if fr.stopped != 0 {
		t.Error("stopped models without being told to")
	}
	if _, err := m.Install(true); err != nil {
		t.Fatal(err)
	}
	if j := waitJob(t, m); j.State != Done {
		t.Fatalf("job %+v", j)
	}
	if fr.stopped != 1 {
		t.Errorf("models stopped %d times", fr.stopped)
	}
}

func TestRunningModelsOnOwnBuildDontBlockInstall(t *testing.T) {
	g := newFakeGitHub(t)
	g.files["llama-b100-bin-ubuntu-vulkan-x64.tar.gz"] = fakeTarGz(t, "llama-b100", nil)
	m, fr, data := newTestManager(t, g, "")
	os.MkdirAll(filepath.Join(data, "llama.cpp"), 0o755)
	own := filepath.Join(data, "llama.cpp", "llama-server")
	os.WriteFile(own, []byte("mine"), 0o755)
	m.apply()
	fr.loaded = []string{"gemma"}
	if _, err := m.Install(false); err != nil {
		t.Fatal(err)
	}
	if j := waitJob(t, m); j.State != Done {
		t.Fatalf("job %+v", j)
	}
	if fr.stopped != 0 || fr.path != own {
		t.Errorf("own build disturbed: stopped %d, path %q", fr.stopped, fr.path)
	}
	if b, _ := os.ReadFile(own); string(b) != "mine" {
		t.Error("own build overwritten")
	}
	if st := m.Status(context.Background()); st.Source != SourceOwn || st.Installed != "b100" || st.OwnPath != own {
		t.Errorf("status %+v", st)
	}
}

func TestManagerInstallFailure(t *testing.T) {
	g := newFakeGitHub(t)
	g.status = http.StatusForbidden
	g.headers = map[string]string{"X-RateLimit-Remaining": "0"}
	m, _, _ := newTestManager(t, g, "")
	m.Install(false)
	j := waitJob(t, m)
	if j.State != Failed || !strings.Contains(j.Error, "rate limit") {
		t.Errorf("job %+v", j)
	}
	if _, err := m.Check(context.Background(), true); err == nil {
		t.Error("check hid the rate limit")
	}
}

func TestUnsupportedSystem(t *testing.T) {
	g := newFakeGitHub(t)
	m, _, _ := newTestManager(t, g, "")
	m.Inst.GOOS, m.Inst.GOARCH = "freebsd", "amd64"
	c, err := m.Check(context.Background(), true)
	if err != nil || c.Supported {
		t.Errorf("check: %+v %v", c, err)
	}
	if st := m.Status(context.Background()); st.Supported {
		t.Error("status says supported")
	}
	m.Install(false)
	if j := waitJob(t, m); j.State != Failed {
		t.Errorf("job %+v", j)
	}
}

func TestProbeBinary(t *testing.T) {
	dir := t.TempDir()
	script := `#!/bin/sh
case "$1" in
--version) echo "version: 0.7.0-dev (build 100, commit abc)"; echo "built with GNU" ;;
--list-devices) echo "Available devices:"; echo "  Vulkan0: Fake GPU (8192 MiB, 8000 MiB free)" ;;
esac
`
	bin := filepath.Join(dir, "llama-server")
	os.WriteFile(bin, []byte(script), 0o755)
	p := ProbeBinary(context.Background(), bin)
	if p.Error != "" || p.Version != "0.7.0-dev (build 100, commit abc)" || !p.Listed || len(p.Devices) != 1 || p.Devices[0] != "Vulkan0: Fake GPU (8192 MiB, 8000 MiB free)" {
		t.Errorf("%+v", p)
	}
	os.WriteFile(bin, []byte("#!/bin/sh\ncase \"$1\" in --version) echo 'version: 1';; --list-devices) echo 'Available devices:';; esac\n"), 0o755)
	if p := ProbeBinary(context.Background(), bin); !p.Listed || len(p.Devices) != 0 {
		t.Errorf("no devices: %+v", p)
	}
	os.WriteFile(bin, []byte("#!/bin/sh\necho 'error while loading shared libraries: libvulkan.so.1: cannot open shared object file' >&2\nexit 127\n"), 0o755)
	if p := ProbeBinary(context.Background(), bin); !strings.Contains(p.Error, "libvulkan.so.1") {
		t.Errorf("load error: %+v", p)
	}
}

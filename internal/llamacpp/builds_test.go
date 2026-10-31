package llamacpp

import (
	"archive/tar"
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// b12019Names are the file names of a real release (trimmed to what matters).
var b12019Names = []string{
	"cudart-llama-b12019-bin-ubuntu-cuda-12.8-x64.tar.gz",
	"cudart-llama-b12019-bin-ubuntu-cuda-13.4-arm64.tar.gz",
	"cudart-llama-b12019-bin-ubuntu-cuda-13.4-x64.tar.gz",
	"cudart-llama-bin-win-cuda-12.4-x64.zip",
	"cudart-llama-bin-win-cuda-13.4-arm64.zip",
	"cudart-llama-bin-win-cuda-13.4-x64.zip",
	"llama-b12019-bin-android-arm64.tar.gz",
	"llama-b12019-bin-macos-arm64.tar.gz",
	"llama-b12019-bin-macos-x64.tar.gz",
	"llama-b12019-bin-ubuntu-arm64.tar.gz",
	"llama-b12019-bin-ubuntu-cuda-12.8-x64.tar.gz",
	"llama-b12019-bin-ubuntu-cuda-13.4-arm64.tar.gz",
	"llama-b12019-bin-ubuntu-cuda-13.4-x64.tar.gz",
	"llama-b12019-bin-ubuntu-openvino-2026.4.1-x64.tar.gz",
	"llama-b12019-bin-ubuntu-rocm-10.0-x64.tar.gz",
	"llama-b12019-bin-ubuntu-sycl-fp16-x64.tar.gz",
	"llama-b12019-bin-ubuntu-vulkan-arm64.tar.gz",
	"llama-b12019-bin-ubuntu-vulkan-x64.tar.gz",
	"llama-b12019-bin-ubuntu-x64.tar.gz",
	"llama-b12019-bin-win-cpu-arm64.zip",
	"llama-b12019-bin-win-cpu-x64.zip",
	"llama-b12019-bin-win-cuda-12.4-x64.zip",
	"llama-b12019-bin-win-cuda-13.4-arm64.zip",
	"llama-b12019-bin-win-cuda-13.4-x64.zip",
	"llama-b12019-bin-win-rocm-10.0-x64.zip",
	"llama-b12019-bin-win-vulkan-arm64.zip",
	"llama-b12019-bin-win-vulkan-x64.zip",
}

func fixtureAssets(names ...string) []Asset {
	if len(names) == 0 {
		names = b12019Names
	}
	var out []Asset
	for _, n := range names {
		out = append(out, Asset{Name: n, Size: 1000})
	}
	return out
}

func buildIDs(bs []Build) []string {
	var ids []string
	for _, b := range bs {
		ids = append(ids, b.ID)
	}
	return ids
}

func TestBuildsPerSystem(t *testing.T) {
	for _, c := range []struct {
		os, arch string
		ids      []string
		cuda     string // the CUDA build's main file, "" if none
		extra    string // and its runtime
	}{
		{"linux", "amd64", []string{"vulkan", "cpu", "rocm", "cuda"}, "llama-b12019-bin-ubuntu-cuda-12.8-x64.tar.gz", "cudart-llama-b12019-bin-ubuntu-cuda-12.8-x64.tar.gz"},
		{"linux", "arm64", []string{"vulkan", "cpu", "cuda"}, "llama-b12019-bin-ubuntu-cuda-13.4-arm64.tar.gz", "cudart-llama-b12019-bin-ubuntu-cuda-13.4-arm64.tar.gz"},
		{"windows", "amd64", []string{"vulkan", "cpu", "rocm", "cuda"}, "llama-b12019-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip"},
		{"windows", "arm64", []string{"vulkan", "cpu", "cuda"}, "llama-b12019-bin-win-cuda-13.4-arm64.zip", "cudart-llama-bin-win-cuda-13.4-arm64.zip"},
		{"darwin", "arm64", []string{"metal"}, "", ""},
		{"darwin", "amd64", []string{"cpu"}, "", ""},
	} {
		bs := Builds(fixtureAssets(), "b12019", c.os, c.arch)
		if got := buildIDs(bs); !reflect.DeepEqual(got, c.ids) {
			t.Errorf("%s/%s: builds %v, want %v", c.os, c.arch, got, c.ids)
			continue
		}
		for _, b := range bs {
			if b.ID != BuildCUDA {
				if len(b.Extra) != 0 {
					t.Errorf("%s/%s %s: unexpected extra files", c.os, c.arch, b.ID)
				}
				continue
			}
			if b.Asset.Name != c.cuda || len(b.Extra) != 1 || b.Extra[0].Name != c.extra {
				t.Errorf("%s/%s cuda: %s + %v, want %s + %s", c.os, c.arch, b.Asset.Name, b.Extra, c.cuda, c.extra)
			}
			if b.Size() != 2000 {
				t.Errorf("size counts the runtime too: %d", b.Size())
			}
		}
	}
	// The default is the Vulkan build, or the only one.
	if a, err := PickAsset(fixtureAssets(), "b12019", "linux", "amd64"); err != nil || !strings.Contains(a.Name, "vulkan") {
		t.Errorf("default build: %v %v", a.Name, err)
	}
	if _, err := PickBuild(fixtureAssets(), "b12019", "darwin", "arm64", "cuda"); err == nil {
		t.Error("a Mac has no CUDA build")
	}
}

func TestBuildsSkipsCUDAWithoutRuntimeAndPicksVersions(t *testing.T) {
	// Without its cudart file CUDA isn't offered.
	bs := Builds(fixtureAssets("llama-b1-bin-win-cuda-12.4-x64.zip", "llama-b1-bin-win-cpu-x64.zip"), "b1", "windows", "amd64")
	if got := buildIDs(bs); !reflect.DeepEqual(got, []string{"cpu"}) {
		t.Errorf("got %v", got)
	}
	// CUDA takes the lowest version (older drivers), ROCm the highest.
	bs = Builds(fixtureAssets(
		"llama-b1-bin-ubuntu-cuda-13.4-x64.tar.gz", "cudart-llama-b1-bin-ubuntu-cuda-13.4-x64.tar.gz",
		"llama-b1-bin-ubuntu-cuda-12.8-x64.tar.gz", "cudart-llama-b1-bin-ubuntu-cuda-12.8-x64.tar.gz",
		"llama-b1-bin-ubuntu-rocm-7.2-x64.tar.gz", "llama-b1-bin-ubuntu-rocm-10.0-x64.tar.gz",
		"llama-b1-bin-ubuntu-rocmfoo-x64.tar.gz",
	), "b1", "linux", "amd64")
	for _, b := range bs {
		switch b.ID {
		case BuildCUDA:
			if !strings.Contains(b.Asset.Name, "12.8") {
				t.Errorf("cuda: %s", b.Asset.Name)
			}
		case BuildROCm:
			if !strings.Contains(b.Asset.Name, "10.0") {
				t.Errorf("rocm: %s", b.Asset.Name)
			}
		}
	}
	if len(bs) != 2 {
		t.Errorf("got %v", buildIDs(bs))
	}
}

func TestRecommend(t *testing.T) {
	all := Builds(fixtureAssets(), "b12019", "linux", "amd64")
	noGPU := Builds(fixtureAssets("llama-b1-bin-ubuntu-vulkan-x64.tar.gz", "llama-b1-bin-ubuntu-x64.tar.gz"), "b1", "linux", "amd64")
	mac := Builds(fixtureAssets(), "b12019", "darwin", "arm64")
	for _, c := range []struct {
		name   string
		builds []Build
		gpus   []string
		want   string
	}{
		{"amd", all, []string{"AMD Radeon RX 6700 XT"}, "rocm"},
		{"amd without rocm", noGPU, []string{"AMD Radeon RX 6700 XT"}, "vulkan"},
		{"nvidia", all, []string{"NVIDIA GeForce RTX 4070"}, "cuda"},
		{"quadro", all, []string{"Quadro P2000"}, "cuda"},
		{"nvidia without cuda", noGPU, []string{"NVIDIA GeForce RTX 4070"}, "vulkan"},
		{"intel", all, []string{"Intel(R) Arc(TM) A770"}, "vulkan"},
		{"none", all, nil, "cpu"},
		{"mixed vendors", all, []string{"NVIDIA GeForce RTX 4070", "AMD Radeon RX 6700 XT"}, "vulkan"},
		{"mac", mac, []string{"Apple M2"}, "metal"},
		{"mac without gpu info", mac, nil, "metal"},
		{"nothing", nil, nil, ""},
	} {
		if got := Recommend(c.builds, c.gpus); got != c.want {
			t.Errorf("%s: got %q, want %q", c.name, got, c.want)
		}
	}
}

func TestInstallBuildWithRuntime(t *testing.T) {
	g := newFakeGitHub(t)
	main := fakeTarGz(t, "llama-b100", nil)
	g.files["llama-b100-bin-ubuntu-cuda-12.8-x64.tar.gz"] = main
	g.files["cudart-llama-b100-bin-ubuntu-cuda-12.8-x64.tar.gz"] = fakeTarGz(t, "", func(tw *tar.Writer) {
		body := "runtime"
		tw.WriteHeader(&tar.Header{Name: "cudart-lib.so", Mode: 0o755, Size: int64(len(body)), Typeflag: tar.TypeReg})
		tw.Write([]byte(body))
	})
	g.extra = []string{"llama-b100-bin-ubuntu-cuda-12.8-x64.tar.gz", "cudart-llama-b100-bin-ubuntu-cuda-12.8-x64.tar.gz"}
	cudart := int64(len(g.files["cudart-llama-b100-bin-ubuntu-cuda-12.8-x64.tar.gz"]))
	in := linuxInstaller(g, filepath.Join(t.TempDir(), "llama.cpp-release"))
	ctx := context.Background()

	cat, err := in.Catalog(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if cat.Tag != "b100" || !reflect.DeepEqual(buildIDs(cat.Builds), []string{"vulkan", "cpu", "cuda"}) {
		t.Fatalf("catalog %s %v", cat.Tag, buildIDs(cat.Builds))
	}
	rel, err := in.Latest(ctx, BuildCUDA)
	if err != nil {
		t.Fatal(err)
	}
	if rel.Build != BuildCUDA || len(rel.Extra) != 1 {
		t.Fatalf("release %+v", rel)
	}
	var last Progress
	if err := in.Install(ctx, rel, func(p Progress) {
		if p.Phase == "Downloading" {
			last = p
		}
	}); err != nil {
		t.Fatal(err)
	}
	if in.InstalledBuild() != BuildCUDA || in.Installed() != "b100" {
		t.Errorf("installed %q %q", in.Installed(), in.InstalledBuild())
	}
	// The runtime's files sit beside llama-server; the build's own win.
	for _, f := range []string{"llama-server", "libggml-vulkan.so", "cudart-lib.so"} {
		if _, err := os.Stat(filepath.Join(in.Dir, f)); err != nil {
			t.Errorf("missing %s: %v", f, err)
		}
	}
	if b, _ := os.ReadFile(filepath.Join(in.Dir, "llama-server")); !strings.Contains(string(b), "version: 1") {
		t.Errorf("llama-server was overwritten: %q", b)
	}
	if last.Total != int64(len(main))+cudart {
		t.Errorf("progress total %d counts both downloads", last.Total)
	}
	if _, err := in.Latest(ctx, "metal"); err == nil {
		t.Error("an unknown build should fail")
	}
}

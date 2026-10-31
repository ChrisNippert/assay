// Package llamacpp installs and updates llama.cpp's ready-made llama-server
// from its GitHub releases, into a folder of ollamui's own.
package llamacpp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"runtime"
	"strconv"
	"strings"
	"time"
)

// DefaultAPI lists llama.cpp's releases, newest first.
const DefaultAPI = "https://api.github.com/repos/ggml-org/llama.cpp/releases"

// ErrUnsupported means llama.cpp publishes no ready-made build for this
// system.
var ErrUnsupported = errors.New("no ready-made llama.cpp build for this system")

// Asset is one downloadable file of a release.
type Asset struct {
	Name   string `json:"name"`
	URL    string `json:"browser_download_url"`
	Size   int64  `json:"size"`
	Digest string `json:"digest"` // "sha256:…" when GitHub provides it
}

// Release is a llama.cpp release with the build chosen for this system.
type Release struct {
	Tag   string // e.g. "b11849"
	Build string // build ID (BuildVulkan…)
	Asset Asset
	// Extra are further files the build needs unpacked beside it (the CUDA
	// runtime libraries).
	Extra []Asset
}

// RateLimitError is GitHub refusing anonymous requests for a while.
type RateLimitError struct{ Reset time.Time }

func (e *RateLimitError) Error() string {
	if e.Reset.IsZero() {
		return "GitHub's rate limit for anonymous requests is reached; try again in a while"
	}
	return "GitHub's rate limit for anonymous requests is reached; try again after " + e.Reset.Local().Format("15:04")
}

// Build IDs: which GPU library a llama-server was compiled for.
const (
	BuildCPU    = "cpu"
	BuildVulkan = "vulkan"
	BuildROCm   = "rocm"
	BuildCUDA   = "cuda"
	BuildMetal  = "metal"
)

// Build is one of the ready-made llama-servers a release has for this system.
type Build struct {
	ID    string `json:"id"`
	Label string `json:"label"`
	Note  string `json:"note"` // who it is for, in plain words
	Asset Asset  `json:"asset"`
	// Extra files are unpacked beside it (the CUDA runtime libraries).
	Extra []Asset `json:"extra,omitempty"`
}

// Size is the whole download.
func (b Build) Size() int64 {
	n := b.Asset.Size
	for _, e := range b.Extra {
		n += e.Size
	}
	return n
}

// spec says what a build's file is called: "llama-<tag>-bin-" + prefix +
// version + suffix + ext. Builds with a version in the name (CUDA 12.8,
// ROCm 10.0) take the lowest one the release offers if low is set, else the
// highest: CUDA 12 runs on older drivers than 13.
type spec struct {
	id, label, note string
	prefix, suffix  string
	versioned, low  bool
	cudart          bool // has a separate cudart-… file to unpack beside it
}

const (
	noteCPU    = "Runs on the processor only. Works everywhere, but is slow for anything big."
	noteVulkan = "Any GPU with a Vulkan driver (AMD, NVIDIA, Intel). Needs the system's Vulkan loader."
	noteROCm   = "AMD GPUs. Usually faster than Vulkan on small models. Needs AMD's ROCm libraries installed."
	noteCUDA   = "NVIDIA GPUs, the fastest choice there. Includes the CUDA runtime it needs, which makes the download large."
	noteMetal  = "Apple silicon's GPU."
)

// specs lists the builds llama.cpp publishes for goos/goarch, the default
// (the one that works on most GPUs) first.
func specs(goos, goarch string) []spec {
	vulkan := func(p string) spec { return spec{id: BuildVulkan, label: "Vulkan", note: noteVulkan, prefix: p} }
	cpu := func(p string) spec { return spec{id: BuildCPU, label: "CPU", note: noteCPU, prefix: p} }
	rocm := func(os string) spec {
		return spec{id: BuildROCm, label: "ROCm (HIP)", note: noteROCm, prefix: os + "-rocm-", suffix: "-x64", versioned: true}
	}
	cuda := func(os, arch string) spec {
		return spec{id: BuildCUDA, label: "CUDA", note: noteCUDA, prefix: os + "-cuda-", suffix: "-" + arch, versioned: true, low: true, cudart: true}
	}
	switch goos + "/" + goarch {
	case "linux/amd64":
		return []spec{vulkan("ubuntu-vulkan-x64"), cpu("ubuntu-x64"), rocm("ubuntu"), cuda("ubuntu", "x64")}
	case "linux/arm64":
		return []spec{vulkan("ubuntu-vulkan-arm64"), cpu("ubuntu-arm64"), cuda("ubuntu", "arm64")}
	case "darwin/arm64":
		return []spec{{id: BuildMetal, label: "Metal", note: noteMetal, prefix: "macos-arm64"}}
	case "darwin/amd64":
		return []spec{cpu("macos-x64")}
	case "windows/amd64":
		return []spec{vulkan("win-vulkan-x64"), cpu("win-cpu-x64"), rocm("win"), cuda("win", "x64")}
	case "windows/arm64":
		return []spec{vulkan("win-vulkan-arm64"), cpu("win-cpu-arm64"), cuda("win", "arm64")}
	}
	return nil
}

func archiveExt(goos string) string {
	if goos == "windows" {
		return ".zip"
	}
	return ".tar.gz"
}

// versionOf is the "12.8" in name between head and tail, or "" if name
// isn't head + a dotted number + tail.
func versionOf(name, head, tail string) string {
	if !strings.HasPrefix(name, head) || !strings.HasSuffix(name, tail) || len(name) <= len(head)+len(tail) {
		return ""
	}
	v := name[len(head) : len(name)-len(tail)]
	for _, r := range v {
		if (r < '0' || r > '9') && r != '.' {
			return ""
		}
	}
	return v
}

// versionLess compares dotted numbers ("12.8" < "13.4").
func versionLess(a, b string) bool {
	pa, pb := strings.Split(a, "."), strings.Split(b, ".")
	for i := 0; i < len(pa) && i < len(pb); i++ {
		x, _ := strconv.Atoi(pa[i])
		y, _ := strconv.Atoi(pb[i])
		if x != y {
			return x < y
		}
	}
	return len(pa) < len(pb)
}

// find locates sp among a release's files.
func (sp spec) find(assets []Asset, tag, goos string) (Build, bool) {
	ext := archiveExt(goos)
	head := "llama-" + tag + "-bin-" + sp.prefix
	b := Build{ID: sp.id, Label: sp.label, Note: sp.note}
	if !sp.versioned {
		for _, a := range assets {
			if a.Name == head+sp.suffix+ext {
				b.Asset = a
				return b, true
			}
		}
		return b, false
	}
	var ver string
	for _, a := range assets {
		v := versionOf(a.Name, head, sp.suffix+ext)
		if v == "" {
			continue
		}
		if ver == "" || sp.low == versionLess(v, ver) {
			ver, b.Asset = v, a
		}
	}
	if ver == "" {
		return b, false
	}
	if sp.cudart {
		// "cudart-llama-<tag>-bin-…" on Linux, "cudart-llama-bin-…" on Windows.
		tail := sp.prefix + ver + sp.suffix + ext
		for _, a := range assets {
			if a.Name == "cudart-llama-"+tag+"-bin-"+tail || a.Name == "cudart-llama-bin-"+tail {
				b.Extra = append(b.Extra, a)
				break
			}
		}
		if b.Extra == nil {
			return b, false // useless without its runtime
		}
	}
	return b, true
}

// Builds lists the builds a release has for goos/goarch, the default first.
func Builds(assets []Asset, tag, goos, goarch string) []Build {
	var out []Build
	for _, sp := range specs(goos, goarch) {
		if b, ok := sp.find(assets, tag, goos); ok {
			out = append(out, b)
		}
	}
	return out
}

// Supported says whether llama.cpp publishes any ready-made build for
// goos/goarch.
func Supported(goos, goarch string) bool { return len(specs(goos, goarch)) > 0 }

// Label names the GPU library of the default build for goos/goarch ("" if
// none).
func Label(goos, goarch string) string {
	if sp := specs(goos, goarch); len(sp) > 0 {
		return sp[0].label
	}
	return ""
}

// PickBuild finds one build (id "" is the default) among a release's files.
func PickBuild(assets []Asset, tag, goos, goarch, id string) (Build, error) {
	bs := Builds(assets, tag, goos, goarch)
	if len(bs) == 0 {
		return Build{}, ErrUnsupported
	}
	if id == "" {
		return bs[0], nil
	}
	for _, b := range bs {
		if b.ID == id {
			return b, nil
		}
	}
	return Build{}, ErrUnsupported
}

// PickAsset finds the default build for goos/goarch among a release's files.
func PickAsset(assets []Asset, tag, goos, goarch string) (Asset, error) {
	b, err := PickBuild(assets, tag, goos, goarch, "")
	return b.Asset, err
}

// Recommend picks the build to offer first for these GPU names: AMD → ROCm,
// NVIDIA → CUDA, either falling back to Vulkan; other GPUs → Vulkan; no GPU
// → CPU. Where only one build exists (a Mac) it is that one.
func Recommend(builds []Build, gpus []string) string {
	has := func(id string) bool {
		for _, b := range builds {
			if b.ID == id {
				return true
			}
		}
		return false
	}
	first := func(ids ...string) string {
		for _, id := range ids {
			if has(id) {
				return id
			}
		}
		if len(builds) > 0 {
			return builds[0].ID
		}
		return ""
	}
	if len(builds) == 1 {
		return builds[0].ID
	}
	if len(gpus) == 0 {
		return first(BuildCPU, BuildVulkan)
	}
	var amd, nvidia bool
	for _, g := range gpus {
		l := strings.ToLower(g)
		switch {
		case containsAny(l, "nvidia", "geforce", "quadro", "tesla", "rtx", "gtx"):
			nvidia = true
		case containsAny(l, "amd", "radeon", "ati "), strings.HasPrefix(l, "rx "):
			amd = true
		}
	}
	switch {
	case nvidia && !amd:
		return first(BuildCUDA, BuildVulkan)
	case amd && !nvidia:
		return first(BuildROCm, BuildVulkan)
	}
	return first(BuildVulkan, BuildCPU)
}

func containsAny(s string, subs ...string) bool {
	for _, x := range subs {
		if strings.Contains(s, x) {
			return true
		}
	}
	return false
}

// TagNumber is the build number of a tag like "b11849" (0 if it has none).
func TagNumber(tag string) int {
	n, _ := strconv.Atoi(strings.TrimPrefix(tag, "b"))
	return n
}

type githubRelease struct {
	Tag    string  `json:"tag_name"`
	Draft  bool    `json:"draft"`
	Assets []Asset `json:"assets"`
}

// releases reads GitHub's release list. It is the list, not
// /releases/latest: llama.cpp marks its continuous builds as pre-releases,
// so "latest" points at an old tag.
func (in *Installer) releases(ctx context.Context) ([]githubRelease, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, in.api()+"?per_page=10", nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/vnd.github+json")
	req.Header.Set("User-Agent", "ollamui")
	resp, err := in.client().Do(req)
	if err != nil {
		return nil, fmt.Errorf("can't reach GitHub: %w", err)
	}
	defer resp.Body.Close()
	switch {
	case resp.StatusCode == http.StatusOK:
	case resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode == http.StatusForbidden && resp.Header.Get("X-RateLimit-Remaining") == "0":
		return nil, &RateLimitError{Reset: resetTime(resp.Header)}
	default:
		return nil, fmt.Errorf("GitHub answered %s", resp.Status)
	}
	var releases []githubRelease
	if err := json.NewDecoder(io.LimitReader(resp.Body, 32<<20)).Decode(&releases); err != nil {
		return nil, fmt.Errorf("unreadable release list: %w", err)
	}
	return releases, nil
}

// Catalog is the newest release and the builds it has for this system.
type Catalog struct {
	Tag    string
	Builds []Build
}

// Catalog finds the newest release that has any build for this system.
func (in *Installer) Catalog(ctx context.Context) (*Catalog, error) {
	goos, goarch := in.platform()
	if !Supported(goos, goarch) {
		return nil, ErrUnsupported
	}
	rels, err := in.releases(ctx)
	if err != nil {
		return nil, err
	}
	for _, r := range rels {
		if r.Draft {
			continue
		}
		if bs := Builds(r.Assets, r.Tag, goos, goarch); len(bs) > 0 {
			return &Catalog{Tag: r.Tag, Builds: bs}, nil
		}
	}
	return nil, errors.New("none of the latest llama.cpp releases has a build for this system")
}

// Latest finds the newest release that has the build id ("" is the
// default).
func (in *Installer) Latest(ctx context.Context, id string) (*Release, error) {
	goos, goarch := in.platform()
	if !Supported(goos, goarch) {
		return nil, ErrUnsupported
	}
	rels, err := in.releases(ctx)
	if err != nil {
		return nil, err
	}
	for _, r := range rels {
		if r.Draft {
			continue
		}
		if b, err := PickBuild(r.Assets, r.Tag, goos, goarch, id); err == nil {
			return &Release{Tag: r.Tag, Build: b.ID, Asset: b.Asset, Extra: b.Extra}, nil
		}
	}
	return nil, errors.New("none of the latest llama.cpp releases has this build for this system")
}

func resetTime(h http.Header) time.Time {
	if n, err := strconv.ParseInt(h.Get("X-RateLimit-Reset"), 10, 64); err == nil {
		return time.Unix(n, 0)
	}
	if n, err := strconv.Atoi(h.Get("Retry-After")); err == nil {
		return time.Now().Add(time.Duration(n) * time.Second)
	}
	return time.Time{}
}

func (in *Installer) platform() (string, string) {
	goos, goarch := in.GOOS, in.GOARCH
	if goos == "" {
		goos = runtime.GOOS
	}
	if goarch == "" {
		goarch = runtime.GOARCH
	}
	return goos, goarch
}

func (in *Installer) api() string {
	if in.API != "" {
		return in.API
	}
	return DefaultAPI
}

func (in *Installer) client() *http.Client {
	if in.HTTP != nil {
		return in.HTTP
	}
	return http.DefaultClient
}

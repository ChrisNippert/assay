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

// Release is a llama.cpp release with the build for this system.
type Release struct {
	Tag   string // e.g. "b11849"
	Asset Asset
}

// RateLimitError is GitHub refusing anonymous requests for a while.
type RateLimitError struct{ Reset time.Time }

func (e *RateLimitError) Error() string {
	if e.Reset.IsZero() {
		return "GitHub's rate limit for anonymous requests is reached; try again in a while"
	}
	return "GitHub's rate limit for anonymous requests is reached; try again after " + e.Reset.Local().Format("15:04")
}

// variant says which of the release files fits goos/goarch: the part of
// the file name after "-bin-", its extension, and what the build is.
func variant(goos, goarch string) (part, ext, label string, ok bool) {
	switch goos + "/" + goarch {
	case "linux/amd64":
		return "ubuntu-vulkan-x64", ".tar.gz", "Vulkan", true
	case "linux/arm64":
		return "ubuntu-vulkan-arm64", ".tar.gz", "Vulkan", true
	case "darwin/arm64":
		return "macos-arm64", ".tar.gz", "Metal", true
	case "darwin/amd64":
		return "macos-x64", ".tar.gz", "CPU", true
	case "windows/amd64":
		return "win-vulkan-x64", ".zip", "Vulkan", true
	case "windows/arm64":
		return "win-vulkan-arm64", ".zip", "Vulkan", true
	}
	return "", "", "", false
}

// Label names the GPU library of the build for goos/goarch ("" if none).
func Label(goos, goarch string) string {
	_, _, l, _ := variant(goos, goarch)
	return l
}

// PickAsset finds the build for goos/goarch among a release's files.
func PickAsset(assets []Asset, tag, goos, goarch string) (Asset, error) {
	part, ext, _, ok := variant(goos, goarch)
	if !ok {
		return Asset{}, ErrUnsupported
	}
	want := "llama-" + tag + "-bin-" + part + ext
	for _, a := range assets {
		if a.Name == want {
			return a, nil
		}
	}
	return Asset{}, ErrUnsupported
}

// TagNumber is the build number of a tag like "b11849" (0 if it has none).
func TagNumber(tag string) int {
	n, _ := strconv.Atoi(strings.TrimPrefix(tag, "b"))
	return n
}

// Latest finds the newest release that has a build for this system. It
// reads the release list, not /releases/latest: llama.cpp marks its
// continuous builds as pre-releases, so "latest" points at an old tag.
func (in *Installer) Latest(ctx context.Context) (*Release, error) {
	goos, goarch := in.platform()
	if _, _, _, ok := variant(goos, goarch); !ok {
		return nil, ErrUnsupported
	}
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
	var releases []struct {
		Tag    string  `json:"tag_name"`
		Draft  bool    `json:"draft"`
		Assets []Asset `json:"assets"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 32<<20)).Decode(&releases); err != nil {
		return nil, fmt.Errorf("unreadable release list: %w", err)
	}
	for _, r := range releases {
		if r.Draft {
			continue
		}
		if a, err := PickAsset(r.Assets, r.Tag, goos, goarch); err == nil {
			return &Release{Tag: r.Tag, Asset: a}, nil
		}
	}
	return nil, errors.New("none of the latest llama.cpp releases has a build for this system")
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

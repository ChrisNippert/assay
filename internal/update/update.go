// Package update tells the person when a newer release exists. It only
// looks (GitHub's "latest release" endpoint, at most once a day, cached);
// installing is left to them.
package update

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Repo is "owner/name" on GitHub. Empty (until the project has a home)
// hides the whole feature and makes no network calls.
const Repo = ""

const (
	apiBase  = "https://api.github.com"
	interval = 24 * time.Hour
)

// state is the settings file.
type state struct {
	// Off turns checking off (the zero value checks).
	Off       bool      `json:"off,omitempty"`
	CheckedAt time.Time `json:"checkedAt,omitzero"`
	Latest    string    `json:"latest,omitempty"`
	URL       string    `json:"url,omitempty"`
}

// Checker looks for a newer release.
type Checker struct {
	Repo    string // "owner/name"; empty disables
	Current string // this build's version
	// Base and Client are for tests.
	Base   string
	Client *http.Client
	Now    func() time.Time

	path string // settings file; "" keeps them in memory
	mu   sync.Mutex
	st   state
}

// Info is what the settings page shows.
type Info struct {
	// Enabled is whether the feature exists at all (a repo is set).
	Enabled   bool   `json:"enabled"`
	Check     bool   `json:"check"`
	Current   string `json:"current"`
	Latest    string `json:"latest,omitempty"`
	URL       string `json:"url,omitempty"`
	Available bool   `json:"available"`
}

// Open reads path; a missing or damaged file means the defaults.
func Open(path, repo, current string) *Checker {
	c := &Checker{Repo: repo, Current: current, Base: apiBase, Client: &http.Client{Timeout: 5 * time.Second}, Now: time.Now, path: path}
	if path != "" {
		if b, err := os.ReadFile(path); err == nil {
			json.Unmarshal(b, &c.st)
		}
	}
	return c
}

func (c *Checker) save() {
	if c.path == "" {
		return
	}
	b, _ := json.MarshalIndent(c.st, "", "  ")
	if os.MkdirAll(filepath.Dir(c.path), 0o755) != nil {
		return
	}
	if os.WriteFile(c.path+".tmp", append(b, '\n'), 0o644) == nil {
		os.Rename(c.path+".tmp", c.path)
	}
}

// SetCheck turns the daily check on or off.
func (c *Checker) SetCheck(on bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.st.Off = !on
	c.save()
}

// Get is the current picture, asking GitHub first when the answer is more
// than a day old (a failed ask counts too, so an offline machine doesn't
// retry on every page view).
func (c *Checker) Get(ctx context.Context) Info {
	c.mu.Lock()
	defer c.mu.Unlock()
	info := Info{Enabled: c.Repo != "", Check: !c.st.Off, Current: c.Current}
	if !info.Enabled {
		return info
	}
	if info.Check && c.Now().Sub(c.st.CheckedAt) >= interval {
		c.st.CheckedAt = c.Now()
		if tag, url, err := c.fetch(ctx); err == nil {
			c.st.Latest, c.st.URL = tag, url
		}
		c.save()
	}
	if info.Check && c.st.Latest != "" && newer(c.st.Latest, c.Current) {
		info.Available, info.Latest, info.URL = true, c.st.Latest, c.st.URL
	}
	return info
}

func (c *Checker) fetch(ctx context.Context) (tag, url string, err error) {
	req, err := http.NewRequestWithContext(ctx, "GET", c.Base+"/repos/"+c.Repo+"/releases/latest", nil)
	if err != nil {
		return "", "", err
	}
	req.Header.Set("Accept", "application/vnd.github+json")
	resp, err := c.Client.Do(req)
	if err != nil {
		return "", "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", "", fmt.Errorf("github: %s", resp.Status)
	}
	var rel struct {
		Tag string `json:"tag_name"`
		URL string `json:"html_url"`
	}
	if err := json.NewDecoder(http.MaxBytesReader(nil, resp.Body, 1<<20)).Decode(&rel); err != nil {
		return "", "", err
	}
	if rel.Tag == "" {
		return "", "", errors.New("github: no tag")
	}
	return rel.Tag, rel.URL, nil
}

// newer reports whether tag is a later version than current. Builds that
// aren't a release ("dev", or git-describe output after a tag) never count
// as older.
func newer(tag, current string) bool {
	t, ok := parse(tag)
	if !ok {
		return false
	}
	c, ok := parse(current)
	if !ok {
		return false
	}
	for i := range t {
		if t[i] != c[i] {
			return t[i] > c[i]
		}
	}
	return false
}

// parse reads "v1.2.3" (or "1.2", "1.2.3-rc1") as three numbers.
func parse(v string) (n [3]int, ok bool) {
	v = strings.TrimPrefix(strings.TrimSpace(v), "v")
	if i := strings.IndexAny(v, "-+"); i >= 0 {
		// A suffix after a tag ("-3-gabc123-dirty") means a build past it,
		// which isn't comparable; "-rc1" would be, but is rare enough to skip.
		return n, false
	}
	parts := strings.Split(v, ".")
	if len(parts) == 0 || len(parts) > 3 {
		return n, false
	}
	for i, p := range parts {
		x, err := strconv.Atoi(p)
		if err != nil || x < 0 {
			return n, false
		}
		n[i] = x
	}
	return n, true
}

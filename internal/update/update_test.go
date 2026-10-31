package update

import (
	"context"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"
)

func TestNewer(t *testing.T) {
	for _, c := range []struct {
		tag, cur string
		want     bool
	}{
		{"v1.2.0", "v1.1.9", true},
		{"v1.2.0", "1.2.0", false},
		{"v1.10.0", "v1.9.0", true},
		{"v1.0.0", "dev", false},
		{"v1.0.0", "v0.9.0-3-gabc-dirty", false},
		{"nightly", "v1.0.0", false},
		{"v2", "v1.9.9", true},
	} {
		if got := newer(c.tag, c.cur); got != c.want {
			t.Errorf("newer(%q, %q) = %v", c.tag, c.cur, got)
		}
	}
}

func fakeGitHub(t *testing.T, calls *atomic.Int32, status int) *httptest.Server {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		if r.URL.Path != "/repos/o/n/releases/latest" {
			t.Errorf("path %s", r.URL.Path)
		}
		w.WriteHeader(status)
		w.Write([]byte(`{"tag_name":"v1.3.0","html_url":"https://example.test/rel"}`))
	}))
	t.Cleanup(srv.Close)
	return srv
}

func TestNoRepoNoNetwork(t *testing.T) {
	c := Open("", "", "v1.0.0")
	c.Client = &http.Client{Transport: failTransport{t}}
	if info := c.Get(context.Background()); info.Enabled || info.Available {
		t.Fatalf("%+v", info)
	}
}

type failTransport struct{ t *testing.T }

func (f failTransport) RoundTrip(*http.Request) (*http.Response, error) {
	f.t.Error("network call with no repo")
	return nil, http.ErrNotSupported
}

func TestCheckOncePerDay(t *testing.T) {
	var calls atomic.Int32
	srv := fakeGitHub(t, &calls, 200)
	path := filepath.Join(t.TempDir(), "update.json")
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	c := Open(path, "o/n", "v1.2.0")
	c.Base, c.Now = srv.URL, func() time.Time { return now }
	ctx := context.Background()
	info := c.Get(ctx)
	if !info.Available || info.Latest != "v1.3.0" || info.URL != "https://example.test/rel" || !info.Check {
		t.Fatalf("%+v", info)
	}
	c.Get(ctx)
	now = now.Add(23 * time.Hour)
	c.Get(ctx)
	if calls.Load() != 1 {
		t.Fatalf("%d calls within a day", calls.Load())
	}
	// A new process reads the cache instead of asking.
	c2 := Open(path, "o/n", "v1.2.0")
	c2.Base, c2.Now = srv.URL, func() time.Time { return now }
	if !c2.Get(ctx).Available || calls.Load() != 1 {
		t.Fatalf("cache not used: %d calls", calls.Load())
	}
	now = now.Add(2 * time.Hour)
	c.Get(ctx)
	if calls.Load() != 2 {
		t.Fatalf("%d calls after a day", calls.Load())
	}
}

func TestSwitchOffAndFailure(t *testing.T) {
	var calls atomic.Int32
	srv := fakeGitHub(t, &calls, 500)
	c := Open("", "o/n", "v1.0.0")
	c.Base = srv.URL
	ctx := context.Background()
	if info := c.Get(ctx); info.Available {
		t.Fatalf("%+v", info)
	}
	c.Get(ctx)
	if calls.Load() != 1 {
		t.Fatalf("a failed check retried at once: %d", calls.Load())
	}
	c2 := Open("", "o/n", "v1.0.0")
	c2.Base = srv.URL
	c2.SetCheck(false)
	if info := c2.Get(ctx); info.Check || calls.Load() != 1 {
		t.Fatalf("checked while off: %+v", info)
	}
}

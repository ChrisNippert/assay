package server

import (
	"encoding/json"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// doFrom is do with a chosen client address.
func doFrom(s *Server, addr, method, path, body string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	req.Host = "localhost:8484"
	req.RemoteAddr = addr
	if body != "" {
		req.Header.Set("Content-Type", "application/json")
	}
	rec := httptest.NewRecorder()
	s.ServeHTTP(rec, req)
	return rec
}

func TestPickRefusesRemote(t *testing.T) {
	s, _, _ := newTestServer(t)
	t.Setenv("OLLAMUI_PICK_TOOLS", "") // never a real dialog from a test
	rec := doFrom(s, "192.168.1.20:5555", "POST", "/api/pick", `{"kind":"folder"}`)
	if rec.Code != 403 {
		t.Fatalf("remote POST: %d %s", rec.Code, rec.Body)
	}
	rec = doFrom(s, "192.168.1.20:5555", "GET", "/api/pick", "")
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), `"native":false`) {
		t.Fatalf("remote GET: %d %s", rec.Code, rec.Body)
	}
	// Even from loopback, with no tool allowed, native is off and POST fails.
	rec = doFrom(s, "127.0.0.1:5555", "GET", "/api/pick", "")
	if !strings.Contains(rec.Body.String(), `"native":false`) {
		t.Fatalf("loopback without a tool: %s", rec.Body)
	}
	rec = doFrom(s, "[::1]:5555", "POST", "/api/pick", `{"kind":"folder"}`)
	if rec.Code != 501 {
		t.Fatalf("loopback without a tool POST: %d %s", rec.Code, rec.Body)
	}
}

func TestPickNativeFromLoopback(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("the fake zenity is a shell script")
	}
	s, _, _ := newTestServer(t)
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "zenity"), []byte("#!/bin/sh\necho /picked/here\n"), 0o755)
	t.Setenv("PATH", dir+":/usr/bin:/bin")
	t.Setenv("DISPLAY", ":fake")
	t.Setenv("OLLAMUI_PICK_TOOLS", "zenity")
	rec := doFrom(s, "127.0.0.1:1", "GET", "/api/pick", "")
	if !strings.Contains(rec.Body.String(), `"native":true`) {
		t.Fatalf("GET: %s", rec.Body)
	}
	rec = doFrom(s, "127.0.0.1:1", "POST", "/api/pick", `{"kind":"folder","title":"x"}`)
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), `"path":"/picked/here"`) {
		t.Fatalf("POST: %d %s", rec.Code, rec.Body)
	}
}

type fsResp struct {
	Path    string
	Parent  string
	Error   string
	Entries []fsEntry
}

func listFS(t *testing.T, s *Server, query string) fsResp {
	t.Helper()
	rec := do(s, "GET", "/api/fs?"+query, "")
	if rec.Code != 200 {
		t.Fatalf("%s: %d %s", query, rec.Code, rec.Body)
	}
	var out fsResp
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	return out
}

func entryNames(es []fsEntry) string {
	var n []string
	for _, e := range es {
		if e.Dir {
			n = append(n, e.Name+"/")
		} else {
			n = append(n, e.Name)
		}
	}
	return strings.Join(n, " ")
}

func TestFSListing(t *testing.T) {
	s, _, _ := newTestServer(t)
	d := t.TempDir()
	for _, f := range []string{"b.gguf", "A.GGUF", "notes.txt", ".hidden.gguf"} {
		os.WriteFile(filepath.Join(d, f), nil, 0o644)
	}
	for _, f := range []string{"m10", "m2", "Zed", ".git"} {
		os.Mkdir(filepath.Join(d, f), 0o755)
	}
	got := listFS(t, s, "kind=file&filter=*.gguf&path="+d)
	if got.Path != d || got.Parent != filepath.Dir(d) {
		t.Fatalf("path %q parent %q", got.Path, got.Parent)
	}
	if want := "m2/ m10/ Zed/ A.GGUF b.gguf"; entryNames(got.Entries) != want {
		t.Fatalf("got %q want %q", entryNames(got.Entries), want)
	}
	if got := listFS(t, s, "kind=folder&path="+d); entryNames(got.Entries) != "m2/ m10/ Zed/" {
		t.Fatalf("folder kind: %q", entryNames(got.Entries))
	}
	if got := listFS(t, s, "kind=file&hidden=1&path="+d); !strings.Contains(entryNames(got.Entries), ".git/") || !strings.Contains(entryNames(got.Entries), ".hidden.gguf") {
		t.Fatalf("hidden: %q", entryNames(got.Entries))
	}
	// a path that isn't there shows the nearest folder that is
	if got := listFS(t, s, "path="+filepath.Join(d, "m2", "nope", "deeper")); got.Path != filepath.Join(d, "m2") {
		t.Fatalf("missing path -> %q", got.Path)
	}
	// a file shows its folder
	if got := listFS(t, s, "path="+filepath.Join(d, "notes.txt")); got.Path != d {
		t.Fatalf("file path -> %q", got.Path)
	}
	// no path: home
	if home, _ := os.UserHomeDir(); home != "" && listFS(t, s, "").Path != home {
		t.Fatal("empty path should open home")
	}
	if rec := do(s, "GET", "/api/fs?path=relative/x", ""); rec.Code != 400 {
		t.Fatalf("relative path: %d", rec.Code)
	}
}

func TestFSUnreadable(t *testing.T) {
	if os.Geteuid() == 0 || runtime.GOOS == "windows" {
		t.Skip("needs a folder this user can't read")
	}
	s, _, _ := newTestServer(t)
	d := filepath.Join(t.TempDir(), "locked")
	os.Mkdir(d, 0o000)
	t.Cleanup(func() { os.Chmod(d, 0o755) })
	got := listFS(t, s, "path="+d)
	if got.Path != d || got.Error == "" {
		t.Fatalf("want an inline error, got %+v", got)
	}
}

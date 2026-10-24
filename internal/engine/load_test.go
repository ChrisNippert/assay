package engine

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

// slowServer is a llama-server that never becomes ready.
func slowServer(t *testing.T) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("uses a shell script as llama-server")
	}
	bin := filepath.Join(t.TempDir(), "llama-server")
	if err := os.WriteFile(bin, []byte("#!/bin/sh\nexec sleep 30\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	return bin
}

func TestLoadOutlivesRequestAndEjectStopsIt(t *testing.T) {
	store, err := OpenStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	m := &Model{Name: "hf.co/o/r:Q4", Files: []string{"x.gguf"}}
	if err := store.Put(m); err != nil {
		t.Fatal(err)
	}
	os.MkdirAll(filepath.Dir(store.Blob("x.gguf")), 0o755)
	os.WriteFile(store.Blob("x.gguf"), []byte("not really a model"), 0o644)
	e := New(store, Options{})
	e.SetServer(slowServer(t))

	ctx, cancel := context.WithCancel(context.Background())
	errc := make(chan error, 1)
	go func() {
		_, _, err := e.ensure(ctx, m.Name, nil)
		errc <- err
	}()
	waitFor(t, func() bool { return e.Loading() == m.Name })

	// The page that asked went away: the load carries on.
	cancel()
	select {
	case err := <-errc:
		t.Fatalf("load stopped with its request: %v", err)
	case <-time.After(500 * time.Millisecond):
	}
	if e.Loading() != m.Name {
		t.Fatalf("Loading() = %q, want %q", e.Loading(), m.Name)
	}

	// Ejecting it does stop it.
	if err := e.Unload(context.Background(), m.Name); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-errc:
		if !errors.Is(err, errEjected) {
			t.Fatalf("err = %v, want ejected", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("eject didn't stop the load")
	}
	if e.Loading() != "" {
		t.Fatalf("still loading %q", e.Loading())
	}
}

func waitFor(t *testing.T, ok func() bool) {
	t.Helper()
	for i := 0; i < 100; i++ {
		if ok() {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatal("timed out")
}

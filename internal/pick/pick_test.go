package pick

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// fakeTool puts a shell script named zenity alone on PATH, so a test never
// opens a dialog on the desktop running it.
func fakeTool(t *testing.T, script string) string {
	t.Helper()
	if runtime.GOOS == "windows" || runtime.GOOS == "darwin" {
		t.Skip("the fake zenity is a Linux shell script")
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "zenity"), []byte("#!/bin/sh\n"+script+"\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir+":/usr/bin:/bin")
	t.Setenv("DISPLAY", ":fake")
	t.Setenv("ASSAY_PICK_TOOLS", "zenity")
	return dir
}

func TestChoose(t *testing.T) {
	fakeTool(t, `echo /chosen/path`)
	if !Available() {
		t.Fatal("fake zenity not found")
	}
	got, err := Choose(context.Background(), Request{Kind: "folder", Title: "Pick it", Start: t.TempDir()})
	if err != nil || got != "/chosen/path" {
		t.Fatalf("got %q, %v", got, err)
	}
}

func TestChooseArgs(t *testing.T) {
	dir := fakeTool(t, `echo "$@" > "$(dirname "$0")/args"; echo /x`)
	if _, err := Choose(context.Background(), Request{Kind: "file", Title: "Adapter", Start: dir, Filter: "*.gguf"}); err != nil {
		t.Fatal(err)
	}
	b, _ := os.ReadFile(filepath.Join(dir, "args"))
	for _, want := range []string{"--file-selection", "--title=Adapter", "--filename=" + dir + "/", "*.gguf"} {
		if !strings.Contains(string(b), want) {
			t.Errorf("args %q lack %q", b, want)
		}
	}
	if strings.Contains(string(b), "--directory") {
		t.Errorf("file dialog asked for a directory: %q", b)
	}
}

func TestChooseCancelled(t *testing.T) {
	fakeTool(t, `exit 1`)
	if _, err := Choose(context.Background(), Request{Kind: "file"}); !errors.Is(err, ErrCancelled) {
		t.Fatalf("exit 1: %v", err)
	}
	fakeTool(t, `echo`)
	if _, err := Choose(context.Background(), Request{Kind: "file"}); !errors.Is(err, ErrCancelled) {
		t.Fatalf("empty output: %v", err)
	}
}

func TestChooseNoTool(t *testing.T) {
	fakeTool(t, `echo /x`)
	t.Setenv("ASSAY_PICK_TOOLS", "")
	if Available() {
		t.Fatal("ASSAY_PICK_TOOLS= should disable native dialogs")
	}
	if _, err := Choose(context.Background(), Request{Kind: "file"}); !errors.Is(err, ErrNoTool) {
		t.Fatal(err)
	}
}

func TestChooseBusyAndContext(t *testing.T) {
	dir := fakeTool(t, `touch "$(dirname "$0")/started"; exec sleep 30`)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { _, err := Choose(ctx, Request{Kind: "folder"}); done <- err }()
	deadline := time.Now().Add(5 * time.Second)
	for _, err := os.Stat(filepath.Join(dir, "started")); err != nil; _, err = os.Stat(filepath.Join(dir, "started")) {
		if time.Now().After(deadline) {
			t.Fatal("dialog never started")
		}
		time.Sleep(10 * time.Millisecond)
	}
	for {
		_, err := Choose(context.Background(), Request{Kind: "folder"})
		if errors.Is(err, ErrBusy) {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("second dialog never reported busy: %v", err)
		}
		time.Sleep(10 * time.Millisecond)
	}
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("got %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("dialog kept running after the context ended")
	}
}

package voice

import (
	"bytes"
	"context"
	"errors"
	"log/slog"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"time"
)

// Launcher starts the bundled speech service on demand and stops it when
// voice mode hasn't been used for a while, so nothing runs in the
// background unless you're talking. A service someone else started on the
// same URL is used as is and never stopped.
type Launcher struct {
	client *Client
	script []byte
	log    *slog.Logger
	// Idle is how long the service may go unused before it's stopped.
	Idle time.Duration

	mu       sync.Mutex
	cmd      *exec.Cmd
	done     chan struct{} // closed when cmd exits
	err      error         // why the last start failed
	lastUsed time.Time
}

// ErrStarting means the service is on its way up; try again shortly.
var ErrStarting = errors.New("the speech service is starting")

// NewLauncher manages the service behind client, running script with uv.
func NewLauncher(client *Client, script []byte, log *slog.Logger) *Launcher {
	return &Launcher{client: client, script: script, log: log, Idle: 10 * time.Minute}
}

// Ready reports whether the service answers. With start set it starts the
// service if it isn't running, returning ErrStarting until it's up.
func (l *Launcher) Ready(ctx context.Context, start bool) error {
	l.touch()
	if l.client.Available(ctx) == nil {
		if l.Managed() {
			l.markInstalled()
		}
		return nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.cmd != nil {
		return ErrStarting
	}
	if !start {
		if l.err != nil {
			return l.err
		}
		return errors.New("the speech service isn't running")
	}
	l.err = l.startLocked()
	if l.err != nil {
		return l.err
	}
	return ErrStarting
}

// Wait starts the service if needed and waits until it answers.
func (l *Launcher) Wait(ctx context.Context) error {
	for {
		err := l.Ready(ctx, true)
		if !errors.Is(err, ErrStarting) {
			return err
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(500 * time.Millisecond):
		}
	}
}

// Managed reports whether ollamui started the running service.
func (l *Launcher) Managed() bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.cmd != nil
}

func (l *Launcher) touch() {
	l.mu.Lock()
	l.lastUsed = time.Now()
	l.mu.Unlock()
}

// CanStart reports why the bundled service couldn't be started, if so.
func (l *Launcher) CanStart() error {
	if _, err := findUV(); err != nil {
		return errors.New("voice mode is optional and needs uv to run its speech service (https://docs.astral.sh/uv/), or start one yourself and pass -voice URL")
	}
	u, _ := url.Parse(l.client.URL())
	if h := u.Hostname(); h != "127.0.0.1" && h != "localhost" {
		return errors.New("the speech service at " + l.client.URL() + " isn't answering")
	}
	return nil
}

func (l *Launcher) startLocked() error {
	if err := l.CanStart(); err != nil {
		return err
	}
	uv, _ := findUV()
	u, _ := url.Parse(l.client.URL())
	path, err := l.writeScript()
	if err != nil {
		return err
	}
	cmd := exec.Command(uv, "run", "--script", path, "--listen", u.Host)
	cmd.Stdout, cmd.Stderr = logWriter{l.log}, logWriter{l.log}
	setProcAttrs(cmd)
	if err := cmd.Start(); err != nil {
		return err
	}
	l.log.Info("started the speech service", "pid", cmd.Process.Pid, "url", l.client.URL())
	l.cmd, l.done = cmd, make(chan struct{})
	go func(cmd *exec.Cmd, done chan struct{}) {
		err := cmd.Wait()
		l.mu.Lock()
		if l.cmd == cmd {
			l.cmd = nil
		}
		l.mu.Unlock()
		close(done)
		l.log.Info("speech service stopped", "err", err)
	}(cmd, l.done)
	return nil
}

// writeScript puts the embedded script where uv can run it. uv keeps one
// environment per script content, so the first start installs packages
// (a minute or so) and later ones take a few seconds.
func (l *Launcher) writeScript() (string, error) {
	dir, err := os.UserCacheDir()
	if err != nil {
		return "", err
	}
	path := filepath.Join(dir, "ollamui", "voice_server.py")
	if old, err := os.ReadFile(path); err == nil && bytes.Equal(old, l.script) {
		return path, nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return "", err
	}
	return path, os.WriteFile(path, l.script, 0o644)
}

// Run stops the service after Idle without use, and when ctx ends.
func (l *Launcher) Run(ctx context.Context) {
	tick := time.NewTicker(min(30*time.Second, max(l.Idle/4, time.Second)))
	defer tick.Stop()
	for {
		select {
		case <-ctx.Done():
			l.Stop()
			return
		case <-tick.C:
			l.mu.Lock()
			idle := l.cmd != nil && l.Idle > 0 && time.Since(l.lastUsed) > l.Idle
			l.mu.Unlock()
			if idle {
				l.log.Info("stopping the idle speech service")
				l.Stop()
			}
		}
	}
}

// Stop stops a service ollamui started.
func (l *Launcher) Stop() {
	l.mu.Lock()
	cmd, done := l.cmd, l.done
	l.mu.Unlock()
	if cmd == nil {
		return
	}
	stopProcess(cmd)
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		cmd.Process.Kill()
	}
}

func findUV() (string, error) {
	if p, err := exec.LookPath("uv"); err == nil {
		return p, nil
	}
	if home, err := os.UserHomeDir(); err == nil {
		for _, p := range []string{".local/bin/uv", ".cargo/bin/uv"} {
			if _, err := os.Stat(filepath.Join(home, p)); err == nil {
				return filepath.Join(home, p), nil
			}
		}
	}
	return "", exec.ErrNotFound
}

// logWriter sends the service's output to ollamui's log, a line at a time.
type logWriter struct{ log *slog.Logger }

func (w logWriter) Write(p []byte) (int, error) {
	for _, line := range bytes.Split(bytes.TrimRight(p, "\n"), []byte("\n")) {
		if len(line) > 0 {
			w.log.Debug("speech service", "out", string(line))
		}
	}
	return len(p), nil
}

//go:build unix

package voice

import (
	"os/exec"
	"syscall"
)

// The service runs in its own process group, so stopping it also stops the
// Python process uv starts.
func setProcAttrs(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

func stopProcess(cmd *exec.Cmd) {
	syscall.Kill(-cmd.Process.Pid, syscall.SIGTERM)
}

//go:build linux

package engine

import (
	"os/exec"
	"syscall"
)

// Each server runs in its own process group, so stopping it stops any
// helper it started, and Ctrl-C in Assay's terminal doesn't reach it
// before Assay has unloaded it cleanly. If Assay dies, the kernel
// stops the server too (Pdeathsig), so no model is left holding VRAM.
func setProcAttrs(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true, Pdeathsig: syscall.SIGTERM}
}

func stopProcess(cmd *exec.Cmd) {
	syscall.Kill(-cmd.Process.Pid, syscall.SIGTERM)
}

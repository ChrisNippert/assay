//go:build !unix

package engine

import "os/exec"

func setProcAttrs(*exec.Cmd) {}

func stopProcess(cmd *exec.Cmd) { cmd.Process.Kill() }

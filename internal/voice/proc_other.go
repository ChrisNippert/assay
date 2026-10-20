//go:build !unix

package voice

import "os/exec"

func setProcAttrs(*exec.Cmd) {}

func stopProcess(cmd *exec.Cmd) { cmd.Process.Kill() }

//go:build !unix

package voice

import (
	"os"
	"os/exec"
)

func setProcAttrs(*exec.Cmd) {}

func stopProcess(cmd *exec.Cmd) { cmd.Process.Kill() }

func fileID(os.FileInfo) (uint64, bool) { return 0, false }

//go:build unix

package voice

import (
	"os"
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

// fileID identifies a file across hard links, so uv's linked packages are
// counted once.
func fileID(info os.FileInfo) (uint64, bool) {
	st, ok := info.Sys().(*syscall.Stat_t)
	if !ok {
		return 0, false
	}
	return uint64(st.Dev)<<32 ^ st.Ino, true
}

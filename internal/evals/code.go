package evals

import (
	"bytes"
	"context"
	"crypto/rand"
	_ "embed"
	"encoding/hex"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"time"
)

var codeBlockRe = regexp.MustCompile("(?s)```[ \\t]*(python3?|py)?[ \\t]*\\n(.*?)```")

// ExtractCode returns the reply's last Python (or unlabelled) code block.
func ExtractCode(reply string) (string, bool) {
	reply = StripThinking(reply)
	m := codeBlockRe.FindAllStringSubmatch(reply, -1)
	for i := len(m) - 1; i >= 0; i-- {
		if code := strings.TrimSpace(m[i][2]); code != "" {
			return code + "\n", true
		}
	}
	return "", false
}

//go:embed harness.py
var harness string

// CodeTimeout is how long one question's code and tests may run.
var CodeTimeout = 15 * time.Second

// maxOutput caps what is kept of the code's output.
const maxOutput = 4000

// RunCode runs the model's code, then the tests, with Python in a fresh
// temporary folder. Limits: CodeTimeout, 1 GB of memory, 10 MB per file
// written, no network when Linux allows an unprivileged network namespace.
// It is not a full sandbox: the code runs as you.
func RunCode(ctx context.Context, code, tests string) (passed bool, output string, err error) {
	py, err := pythonPath()
	if err != nil {
		return false, "", err
	}
	dir, err := os.MkdirTemp("", "assay-eval-")
	if err != nil {
		return false, "", err
	}
	defer os.RemoveAll(dir)
	for name, body := range map[string]string{"harness.py": harness, "solution.py": code, "tests.py": tests} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o600); err != nil {
			return false, "", err
		}
	}
	ctx, cancel := context.WithTimeout(ctx, CodeTimeout)
	defer cancel()
	var nonce [12]byte
	rand.Read(nonce[:])
	token := hex.EncodeToString(nonce[:])
	args := []string{py, "-I", "-B", "harness.py", token}
	if noNetwork() {
		args = append([]string{"unshare", "--user", "--net", "--map-root-user"}, args...)
	}
	cmd := exec.CommandContext(ctx, args[0], args[1:]...)
	cmd.Dir = dir
	cmd.Env = []string{"PATH=" + os.Getenv("PATH"), "HOME=" + dir, "TMPDIR=" + dir, "PYTHONHASHSEED=0", "LANG=C.UTF-8"}
	if runtime.GOOS == "windows" {
		cmd.Env = append(cmd.Env, "SYSTEMROOT="+os.Getenv("SYSTEMROOT"))
	}
	var out bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &out
	cmd.WaitDelay = 2 * time.Second
	runErr := cmd.Run()
	output = out.String()
	finished := strings.Contains(output, token)
	output = strings.ReplaceAll(output, token, "")
	if len(output) > maxOutput {
		output = output[:maxOutput/2] + "\n[…]\n" + output[len(output)-maxOutput/2:]
	}
	if ctx.Err() == context.DeadlineExceeded {
		return false, strings.TrimSpace(output + "\nStopped after " + CodeTimeout.String() + "."), nil
	}
	if ctx.Err() != nil {
		return false, output, ctx.Err()
	}
	return runErr == nil && finished, strings.TrimSpace(output), nil
}

var (
	pyOnce sync.Once
	pyPath string
	pyErr  error
	nsOnce sync.Once
	nsOK   bool
)

// pythonPath finds Python 3.
func pythonPath() (string, error) {
	pyOnce.Do(func() {
		for _, name := range []string{"python3", "python", "py"} {
			p, err := exec.LookPath(name)
			if err != nil {
				continue
			}
			out, err := exec.Command(p, "-c", "import sys; print(sys.version_info[0])").Output()
			if err == nil && strings.TrimSpace(string(out)) == "3" {
				pyPath = p
				return
			}
		}
		pyErr = errors.New("code questions need Python 3, and none was found on PATH")
	})
	return pyPath, pyErr
}

// HavePython says code questions can be run here.
func HavePython() bool { _, err := pythonPath(); return err == nil }

// noNetwork says the code can be run without network access.
func noNetwork() bool {
	nsOnce.Do(func() {
		if runtime.GOOS != "linux" {
			return
		}
		nsOK = exec.Command("unshare", "--user", "--net", "--map-root-user", "true").Run() == nil
	})
	return nsOK
}

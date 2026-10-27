//go:build darwin

package pick

import (
	"os/exec"
	"strings"
)

func haveDisplay() bool { return true }

func hideWindow(*exec.Cmd) {}

func candidates(r Request) []tool {
	q := func(s string) string { return `"` + strings.NewReplacer(`\`, `\\`, `"`, `\"`).Replace(s) + `"` }
	title := r.Title
	if title == "" {
		title = "Choose"
	}
	what := "choose folder"
	if r.Kind == "file" {
		what = "choose file"
	}
	script := "POSIX path of (" + what + " with prompt " + q(title) + " default location (POSIX file " + q(startDir(r.Start)) + ")"
	if e := ext(r.Filter); r.Kind == "file" && e != "" {
		script += " of type {" + q(e) + "}"
	}
	script += ")"
	return []tool{{"osascript", []string{"-e", script}}}
}

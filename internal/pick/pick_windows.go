//go:build windows

package pick

import (
	"os/exec"
	"strings"
	"syscall"
)

func haveDisplay() bool { return true }

// hideWindow keeps PowerShell's own console from flashing up.
func hideWindow(c *exec.Cmd) {
	c.SysProcAttr = &syscall.SysProcAttr{HideWindow: true}
}

func candidates(r Request) []tool {
	q := func(s string) string { return "'" + strings.ReplaceAll(s, "'", "''") + "'" }
	title := r.Title
	if title == "" {
		title = "Choose"
	}
	script := "Add-Type -AssemblyName System.Windows.Forms; "
	if r.Kind == "file" {
		script += "$d = New-Object System.Windows.Forms.OpenFileDialog; $d.InitialDirectory = " + q(startDir(r.Start)) + "; "
		if e := ext(r.Filter); e != "" {
			script += "$d.Filter = " + q(strings.ToUpper(e)+" files|"+r.Filter+"|All files|*.*") + "; "
		}
		script += "$d.Title = " + q(title) + "; if ($d.ShowDialog() -eq 'OK') { $d.FileName }"
	} else {
		script += "$d = New-Object System.Windows.Forms.FolderBrowserDialog; $d.SelectedPath = " + q(startDir(r.Start)) + "; $d.Description = " + q(title) + "; "
		script += "if ($d.ShowDialog() -eq 'OK') { $d.SelectedPath }"
	}
	return []tool{{"powershell", []string{"-NoProfile", "-STA", "-WindowStyle", "Hidden", "-Command", script}}}
}

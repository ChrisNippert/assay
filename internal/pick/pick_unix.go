//go:build !windows && !darwin

package pick

import (
	"os"
	"os/exec"
	"strings"
)

func haveDisplay() bool {
	return os.Getenv("DISPLAY") != "" || os.Getenv("WAYLAND_DISPLAY") != ""
}

func hideWindow(*exec.Cmd) {}

func candidates(r Request) []tool {
	dir := startDir(r.Start)
	if !strings.HasSuffix(dir, "/") {
		dir += "/"
	}
	title := r.Title
	if title == "" {
		title = "Choose"
	}
	folder := r.Kind == "folder"
	filter := r.Filter

	z := []string{"--file-selection", "--title=" + title, "--filename=" + dir}
	y := []string{"--file", "--title=" + title, "--filename=" + dir, "--center"}
	if folder {
		z = append(z, "--directory")
		y = append(y, "--directory")
	} else if filter != "" {
		z = append(z, "--file-filter="+filter+" | "+filter)
		y = append(y, "--file-filter="+filter+" | "+filter)
	}
	k := []string{"--title", title}
	if folder {
		k = append(k, "--getexistingdirectory", dir)
	} else {
		k = append(k, "--getopenfilename", dir)
		if filter != "" {
			k = append(k, filter)
		}
	}
	return []tool{{"zenity", z}, {"kdialog", k}, {"yad", y}}
}

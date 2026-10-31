package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"assay/internal/service"
	"assay/internal/update"
)

// newService is the start-on-boot manager: the real one, or an in-memory
// stand-in when ASSAY_SERVICE_DRYRUN=1 so the UI can be tried safely.
func newService() (service.Service, error) {
	if os.Getenv("ASSAY_SERVICE_DRYRUN") == "1" {
		exe, _ := service.Executable()
		return service.NewFake(exe), nil
	}
	return service.New()
}

// newUpdates reads the update-check settings from the config folder.
func newUpdates(configDir string) *update.Checker {
	return update.Open(filepath.Join(configDir, "update.json"), update.Repo, version)
}

// subcommand runs `version` and `service ...` when args start with one,
// reporting whether it did. Anything else is left to the flags.
func subcommand(args []string) (bool, error) {
	if len(args) == 0 {
		return false, nil
	}
	switch args[0] {
	case "version":
		fmt.Println(service.Name, version)
		return true, nil
	case "service":
		return true, serviceCmd(args[1:])
	}
	return false, nil
}

const serviceUsage = `usage: ` + service.Name + ` service install [flags to run with]
       ` + service.Name + ` service uninstall
       ` + service.Name + ` service status

install makes it start when you log in (or at boot, see the notes it
prints), running with the flags you give, e.g.
  ` + service.Name + ` service install -listen 0.0.0.0:8484`

func serviceCmd(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("%s", serviceUsage)
	}
	svc, err := newService()
	if err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	switch args[0] {
	case "install":
		notes, err := svc.Install(ctx, args[1:], true)
		if err != nil {
			return err
		}
		fmt.Println(strings.Join(notes, "\n"))
		printStatus(svc.Status(ctx))
	case "uninstall":
		if err := svc.Uninstall(ctx, true); err != nil {
			return err
		}
		fmt.Println("Removed. It won't start on its own any more.")
	case "status":
		printStatus(svc.Status(ctx))
	default:
		return fmt.Errorf("unknown service command %q\n%s", args[0], serviceUsage)
	}
	return nil
}

func printStatus(st service.Status) {
	if !st.Supported {
		fmt.Println("Starting on boot isn't supported on this system.")
		return
	}
	state := "not installed"
	switch {
	case st.Enabled && st.Running:
		state = "enabled, running"
	case st.Enabled && st.Platform == "windows":
		state = "enabled (starts when you sign in)"
	case st.Enabled:
		state = "enabled, not running"
	}
	fmt.Println("Start on boot:", state)
	if st.Command != "" {
		fmt.Println("Runs:", st.Command)
	}
	if st.Where != "" {
		fmt.Println("Set in:", st.Where)
	}
}

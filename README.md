# assay

A web UI to find, download, tune and chat with AI models that run on your own
machine, on top of llama.cpp's `llama-server`. One Go binary, plain files, no
database.

## Install

Download the archive for your system from the Releases page, unpack it and
check it against `SHA256SUMS`. Or build it: `go build ./cmd/assay`
(Go, no cgo).

## Run

    ./assay

then open http://localhost:8484. The first time, open Settings and press
"Install llama.cpp" (or put your own `llama-server` on PATH), then download a
model from Discover. `assay -help` lists the options; `assay --version`
prints the version.

## Start on boot

    assay service install -listen 0.0.0.0:8484   # flags are optional
    assay service status
    assay service uninstall

This sets up a systemd user unit (Linux), a LaunchAgent (macOS) or a Run key
(Windows). On Linux, starting before you log in also needs
`loginctl enable-linger $USER`. Settings → This machine has the same switch.

See ARCHITECTURE.md for how it all fits together.

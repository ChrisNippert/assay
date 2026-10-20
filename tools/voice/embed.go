// Package voiceserver embeds the speech service script, so ollamui can
// start it on demand without a separate install.
package voiceserver

import _ "embed"

// Script is voice_server.py, run with `uv run --script`.
//
//go:embed voice_server.py
var Script []byte

// Package web holds the UI's templates and static files, embedded into the binary.
package web

import "embed"

//go:embed templates static
var FS embed.FS

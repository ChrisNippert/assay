package ollama

import (
	"errors"
	"fmt"
	"net/http"
)

// APIError is an error with the HTTP status it maps to, from the engine or
// a llama-server.
type APIError struct {
	StatusCode int
	Message    string
}

func (e *APIError) Error() string {
	// Errors that arrive mid-stream come with a 200 status, which says nothing useful.
	if e.StatusCode == http.StatusOK {
		return e.Message
	}
	return fmt.Sprintf("%s (HTTP %d)", e.Message, e.StatusCode)
}

// IsNotFound reports whether err says the model doesn't exist.
func IsNotFound(err error) bool {
	var apiErr *APIError
	return errors.As(err, &apiErr) && apiErr.StatusCode == http.StatusNotFound
}

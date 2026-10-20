// Package ollama is a small typed client for the Ollama HTTP API.
package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
)

// DefaultURL is where Ollama listens unless OLLAMA_HOST says otherwise.
const DefaultURL = "http://127.0.0.1:11434"

// Client talks to one Ollama server.
type Client struct {
	base *url.URL
	http *http.Client
}

// New returns a client for the Ollama server at baseURL.
func New(baseURL string) (*Client, error) {
	u, err := url.Parse(baseURL)
	if err != nil {
		return nil, fmt.Errorf("parse ollama url: %w", err)
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return nil, fmt.Errorf("ollama url must be http or https: %q", baseURL)
	}
	// No client-wide timeout: chat and pull streams can run for a long time.
	// Callers bound non-streaming calls with their context.
	return &Client{base: u, http: &http.Client{}}, nil
}

// URL returns the server address the client talks to.
func (c *Client) URL() string { return c.base.String() }

// ResolveHost turns an OLLAMA_HOST style value ("0.0.0.0", ":11434",
// "host:port", "https://host") into a URL a client can connect to.
func ResolveHost(v string) string {
	v = strings.TrimSpace(v)
	if v == "" {
		return DefaultURL
	}
	scheme := "http"
	if i := strings.Index(v, "://"); i >= 0 {
		scheme, v = v[:i], v[i+3:]
	}
	host, port, err := net.SplitHostPort(v)
	if err != nil {
		host, port = v, "11434"
	}
	if host == "" || host == "0.0.0.0" || host == "::" {
		host = "127.0.0.1"
	}
	return scheme + "://" + net.JoinHostPort(host, port)
}

// APIError is an error response from Ollama.
type APIError struct {
	StatusCode int
	Message    string
}

func (e *APIError) Error() string {
	// Errors that arrive mid-stream come with a 200 status, which says nothing useful.
	if e.StatusCode == http.StatusOK {
		return "ollama: " + e.Message
	}
	return fmt.Sprintf("ollama: %s (HTTP %d)", e.Message, e.StatusCode)
}

// IsNotFound reports whether err is Ollama saying the model doesn't exist.
func IsNotFound(err error) bool {
	var apiErr *APIError
	return errors.As(err, &apiErr) && apiErr.StatusCode == http.StatusNotFound
}

func (c *Client) do(ctx context.Context, method, path string, body any) (*http.Response, error) {
	var rdr io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		rdr = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.base.JoinPath(path).String(), rdr)
	if err != nil {
		return nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 300 {
		defer resp.Body.Close()
		return nil, readError(resp)
	}
	return resp, nil
}

func readError(resp *http.Response) error {
	b, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
	msg := strings.TrimSpace(string(b))
	var e struct {
		Error string `json:"error"`
	}
	if json.Unmarshal(b, &e) == nil && e.Error != "" {
		msg = e.Error
	}
	if msg == "" {
		msg = resp.Status
	}
	return &APIError{StatusCode: resp.StatusCode, Message: msg}
}

// call makes a non-streaming request and decodes the response into out
// (if out is non-nil).
func (c *Client) call(ctx context.Context, method, path string, in, out any) error {
	resp, err := c.do(ctx, method, path, in)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if out == nil {
		_, err = io.Copy(io.Discard, resp.Body)
		return err
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

// stream makes a request whose response is NDJSON and calls fn for each
// object. An {"error": ...} object in the stream becomes an *APIError.
// If fn returns an error, the stream is abandoned and that error returned.
func stream[T any](ctx context.Context, c *Client, path string, in any, fn func(T) error) error {
	resp, err := c.do(ctx, http.MethodPost, path, in)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	dec := json.NewDecoder(resp.Body)
	for {
		var raw json.RawMessage
		if err := dec.Decode(&raw); err != nil {
			if errors.Is(err, io.EOF) {
				return nil
			}
			return err
		}
		var e struct {
			Error string `json:"error"`
		}
		if json.Unmarshal(raw, &e) == nil && e.Error != "" {
			return &APIError{StatusCode: resp.StatusCode, Message: e.Error}
		}
		var v T
		if err := json.Unmarshal(raw, &v); err != nil {
			return fmt.Errorf("decode stream: %w", err)
		}
		if err := fn(v); err != nil {
			return err
		}
	}
}

func ptr[T any](v T) *T { return &v }

// Version returns the Ollama server version.
func (c *Client) Version(ctx context.Context) (string, error) {
	var out struct {
		Version string `json:"version"`
	}
	err := c.call(ctx, http.MethodGet, "/api/version", nil, &out)
	return out.Version, err
}

// List returns the models installed locally.
func (c *Client) List(ctx context.Context) ([]Model, error) {
	var out struct {
		Models []Model `json:"models"`
	}
	err := c.call(ctx, http.MethodGet, "/api/tags", nil, &out)
	return out.Models, err
}

// Running returns the models currently loaded in memory.
func (c *Client) Running(ctx context.Context) ([]RunningModel, error) {
	var out struct {
		Models []RunningModel `json:"models"`
	}
	err := c.call(ctx, http.MethodGet, "/api/ps", nil, &out)
	return out.Models, err
}

// Show returns details about one model.
func (c *Client) Show(ctx context.Context, model string) (*ShowResponse, error) {
	var out ShowResponse
	if err := c.call(ctx, http.MethodPost, "/api/show", map[string]string{"model": model}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// Delete removes a model from disk.
func (c *Client) Delete(ctx context.Context, model string) error {
	return c.call(ctx, http.MethodDelete, "/api/delete", map[string]string{"model": model}, nil)
}

// Copy creates dst as another name for src. No weights are duplicated.
func (c *Client) Copy(ctx context.Context, src, dst string) error {
	return c.call(ctx, http.MethodPost, "/api/copy", map[string]string{"source": src, "destination": dst}, nil)
}

// Load loads a model into memory and keeps it there for keepAlive
// (a duration like "30m", or "-1" for until ejected; empty means Ollama's default).
func (c *Client) Load(ctx context.Context, model, keepAlive string) error {
	req := map[string]any{"model": model, "stream": false}
	if keepAlive != "" {
		req["keep_alive"] = keepAliveValue(keepAlive)
	}
	return c.call(ctx, http.MethodPost, "/api/generate", req, nil)
}

// Unload ejects a model from memory.
func (c *Client) Unload(ctx context.Context, model string) error {
	req := map[string]any{"model": model, "stream": false, "keep_alive": 0}
	return c.call(ctx, http.MethodPost, "/api/generate", req, nil)
}

// keepAliveValue sends bare numbers as numbers (seconds, or -1 for
// forever) and anything else as a duration string.
func keepAliveValue(s string) any {
	var n json.Number = json.Number(s)
	if _, err := n.Int64(); err == nil {
		return n
	}
	return s
}

// Pull downloads a model, calling fn with each progress update.
// Names like "hf.co/unsloth/Qwen3-8B-GGUF:Q4_K_M" pull from Hugging Face.
func (c *Client) Pull(ctx context.Context, model string, fn func(PullProgress) error) error {
	req := map[string]any{"model": model, "stream": true}
	return stream(ctx, c, "/api/pull", req, fn)
}

// Create creates or replaces a model, calling fn with each status update.
func (c *Client) Create(ctx context.Context, req CreateRequest, fn func(Status) error) error {
	req.Stream = ptr(true)
	return stream(ctx, c, "/api/create", req, fn)
}

// Chat sends a conversation and streams the reply to fn.
func (c *Client) Chat(ctx context.Context, req ChatRequest, fn func(ChatResponse) error) error {
	req.Stream = ptr(true)
	return stream(ctx, c, "/api/chat", req, fn)
}

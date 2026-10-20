// Package hf searches Hugging Face for GGUF models and lists their quants.
package hf

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"
)

// DefaultBaseURL is the public Hugging Face hub.
const DefaultBaseURL = "https://huggingface.co"

// ErrNotFound means the repo or file doesn't exist (or is private).
var ErrNotFound = errors.New("not found on Hugging Face")

// Client talks to the Hugging Face hub API.
type Client struct {
	base  string
	token string
	http  *http.Client
}

// New returns a client. token is optional; it's needed for gated or private repos.
func New(baseURL, token string) *Client {
	return &Client{
		base:  strings.TrimRight(baseURL, "/"),
		token: token,
		http:  &http.Client{Timeout: 20 * time.Second},
	}
}

var repoPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$`)

// ValidRepo reports whether id looks like "owner/name". Anything else is
// rejected before it's put into a URL.
func ValidRepo(id string) bool { return repoPattern.MatchString(id) && !strings.Contains(id, "..") }

// GGUF is the metadata Hugging Face extracts from a repo's GGUF files.
type GGUF struct {
	Total         int64  `json:"total"` // parameter count
	Architecture  string `json:"architecture"`
	ContextLength int    `json:"context_length"`
}

// Repo is a model repository.
type Repo struct {
	ID           string    `json:"id"`
	Downloads    int       `json:"downloads"`
	Likes        int       `json:"likes"`
	LastModified time.Time `json:"lastModified"`
	Tags         []string  `json:"tags"`
	PipelineTag  string    `json:"pipeline_tag"`
	GGUF         *GGUF     `json:"gguf"`
	Gated        any       `json:"gated"` // false, "auto" or "manual"
}

// Author is the part of the ID before the slash.
func (r Repo) Author() string { a, _, _ := strings.Cut(r.ID, "/"); return a }

// Name is the part of the ID after the slash.
func (r Repo) Name() string { _, n, _ := strings.Cut(r.ID, "/"); return n }

// Params is the parameter count, or 0 if unknown.
func (r Repo) Params() int64 {
	if r.GGUF == nil {
		return 0
	}
	return r.GGUF.Total
}

// Arch is the GGUF architecture name, e.g. "qwen3" or "qwen3moe".
func (r Repo) Arch() string {
	if r.GGUF == nil {
		return ""
	}
	return r.GGUF.Architecture
}

// ContextLength is the model's trained context length, or 0 if unknown.
func (r Repo) ContextLength() int {
	if r.GGUF == nil {
		return 0
	}
	return r.GGUF.ContextLength
}

// Vision reports whether the repo is tagged as taking images.
func (r Repo) Vision() bool {
	if r.PipelineTag == "image-text-to-text" {
		return true
	}
	for _, t := range r.Tags {
		if t == "image-text-to-text" || t == "vision" {
			return true
		}
	}
	return false
}

// IsGated reports whether downloading requires accepting terms on the website.
func (r Repo) IsGated() bool {
	b, ok := r.Gated.(bool)
	return r.Gated != nil && !(ok && !b)
}

var moeArchs = map[string]bool{
	"mixtral": true, "deepseek2": true, "deepseek3": true, "gptoss": true, "gpt-oss": true,
	"llama4": true, "olmoe": true, "dbrx": true, "jamba": true, "arctic": true, "dots1": true,
}

var activeParamsPattern = regexp.MustCompile(`(?i)-A\d+(\.\d+)?B\b`)

// IsMoE guesses whether a model is a mixture of experts from its
// architecture name, or from an "A3B"-style active-parameter suffix.
func IsMoE(arch, repoID string) bool {
	a := strings.ToLower(arch)
	return strings.Contains(a, "moe") || moeArchs[a] || activeParamsPattern.MatchString(repoID)
}

var expandFields = []string{"gguf", "downloads", "likes", "lastModified", "tags", "pipeline_tag", "gated"}

// SearchParams narrows a search.
type SearchParams struct {
	Query  string
	Author string
	Sort   string // downloads, likes, lastModified, trendingScore
	Limit  int
}

// Search finds repos that contain GGUF files.
func (c *Client) Search(ctx context.Context, p SearchParams) ([]Repo, error) {
	q := url.Values{}
	q.Set("filter", "gguf")
	if p.Query != "" {
		q.Set("search", p.Query)
	}
	if p.Author != "" {
		q.Set("author", p.Author)
	}
	switch p.Sort {
	case "downloads", "likes", "lastModified", "trendingScore":
	default:
		p.Sort = "downloads"
	}
	q.Set("sort", p.Sort)
	q.Set("direction", "-1")
	if p.Limit <= 0 || p.Limit > 200 {
		p.Limit = 100
	}
	q.Set("limit", fmt.Sprint(p.Limit))
	for _, f := range expandFields {
		q.Add("expand[]", f)
	}
	var out []Repo
	if err := c.getJSON(ctx, "/api/models?"+q.Encode(), &out); err != nil {
		return nil, err
	}
	return out, nil
}

// Info returns one repo's details.
func (c *Client) Info(ctx context.Context, repo string) (*Repo, error) {
	if !ValidRepo(repo) {
		return nil, ErrNotFound
	}
	q := url.Values{}
	for _, f := range expandFields {
		q.Add("expand[]", f)
	}
	var out Repo
	if err := c.getJSON(ctx, "/api/models/"+repo+"?"+q.Encode(), &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// File is one entry in a repo.
type File struct {
	Type string `json:"type"` // "file" or "directory"
	Path string `json:"path"`
	Size int64  `json:"size"`
}

// Files lists every file in a repo's main branch.
func (c *Client) Files(ctx context.Context, repo string) ([]File, error) {
	if !ValidRepo(repo) {
		return nil, ErrNotFound
	}
	var out []File
	if err := c.getJSON(ctx, "/api/models/"+repo+"/tree/main?recursive=true", &out); err != nil {
		return nil, err
	}
	return out, nil
}

// RecommendedParams reads the "params" file some uploaders (notably Unsloth)
// ship with Ollama-style recommended settings. It returns nil if there is none.
func (c *Client) RecommendedParams(ctx context.Context, repo string) (map[string]any, error) {
	if !ValidRepo(repo) {
		return nil, ErrNotFound
	}
	var out map[string]any
	err := c.getJSON(ctx, "/"+repo+"/resolve/main/params", &out)
	if errors.Is(err, ErrNotFound) {
		return nil, nil
	}
	return out, err
}

func (c *Client) getJSON(ctx context.Context, path string, out any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.base+path, nil)
	if err != nil {
		return err
	}
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return ctx.Err() // the caller gave up; don't dress it up
		}
		var netErr net.Error
		if errors.As(err, &netErr) && netErr.Timeout() {
			return errors.New("Hugging Face didn't respond in time. Check your connection and try again")
		}
		return errors.New("couldn't reach Hugging Face. Check your connection and try again")
	}
	defer resp.Body.Close()
	switch {
	case resp.StatusCode == http.StatusNotFound:
		return ErrNotFound
	case resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden:
		return fmt.Errorf("hugging face: access denied (gated or private repo; set HF_TOKEN)")
	case resp.StatusCode >= 300:
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<10))
		return fmt.Errorf("hugging face: %s: %s", resp.Status, strings.TrimSpace(string(b)))
	}
	return json.NewDecoder(io.LimitReader(resp.Body, 32<<20)).Decode(out)
}

package engine

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"

	"ollamui/internal/ollama"
)

// IsEmbedding reports whether an installed model only embeds.
func (e *Engine) IsEmbedding(name string) bool {
	m, err := e.store.Get(name)
	if err != nil {
		return false
	}
	f, err := e.gguf(m)
	return err == nil && f.IsEmbedding()
}

// Proxy sends an OpenAI-style request body to the model's llama-server,
// loading the model first if needed, and returns the raw response (a
// stream when the body asks for one). path is "/v1/chat/completions" or
// "/v1/embeddings". options are Ollama-named settings that apply over the
// model's own (a preset's); for chat, the sampling ones fill in only what
// the body leaves out. The caller closes the response body, which also
// marks the model idle again.
func (e *Engine) Proxy(ctx context.Context, model string, options map[string]any, path string, body []byte) (*http.Response, error) {
	var req map[string]any
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, &ollama.APIError{StatusCode: http.StatusBadRequest, Message: "invalid JSON: " + err.Error()}
	}
	p, _, err := e.ensure(ctx, model, options)
	if errors.Is(err, ErrNotFound) {
		return nil, &ollama.APIError{StatusCode: http.StatusNotFound, Message: "model " + model + " not found"}
	}
	if err != nil {
		return nil, err
	}
	switch path {
	case "/v1/chat/completions":
		if p.opts.Embedding {
			return nil, &ollama.APIError{StatusCode: http.StatusBadRequest, Message: model + " is an embedding model, not a chat model"}
		}
		opts := p.model.options(options)
		for k, v := range opts {
			if f, ok := sampling[k]; ok {
				if _, set := req[f]; !set {
					req[f] = v
				}
			}
		}
		_, a := req["max_tokens"]
		_, b := req["max_completion_tokens"]
		if n, ok := num(opts["num_predict"]); ok && n > 0 && !a && !b {
			req["max_tokens"] = int(n)
		}
		// A reply format from the options (a preset's) applies only when the
		// body asks for no format of its own.
		_, rf := req["response_format"]
		_, gr := req["grammar"]
		if !rf && !gr {
			for k, v := range replyFields(opts) {
				req[k] = v
			}
		}
	case "/v1/embeddings":
		if !p.opts.Embedding {
			return nil, &ollama.APIError{StatusCode: http.StatusBadRequest, Message: model + " isn't an embedding model"}
		}
	default:
		return nil, fmt.Errorf("proxy: unsupported path %s", path)
	}
	req["model"] = p.model.Name
	out, _ := json.Marshal(req)

	ka := modelKeepAlive(p.model, options)
	p.touch(ka, e.opt.KeepAlive)
	p.mu.Lock()
	p.busy++
	p.mu.Unlock()
	release := func() {
		p.mu.Lock()
		p.busy--
		p.mu.Unlock()
		p.touch(ka, e.opt.KeepAlive)
	}
	hreq, _ := http.NewRequestWithContext(ctx, http.MethodPost, p.base+path, bytes.NewReader(out))
	hreq.Header.Set("Content-Type", "application/json")
	res, err := e.http.Do(hreq)
	if err != nil {
		release()
		if !p.running() {
			return nil, fmt.Errorf("llama-server stopped: %s", p.lastLines())
		}
		return nil, err
	}
	res.Body = &releaseBody{ReadCloser: res.Body, release: release}
	return res, nil
}

// releaseBody runs release once when the body is closed.
type releaseBody struct {
	io.ReadCloser
	release func()
	closed  bool
}

func (b *releaseBody) Close() error {
	err := b.ReadCloser.Close()
	if !b.closed {
		b.closed = true
		b.release()
	}
	return err
}

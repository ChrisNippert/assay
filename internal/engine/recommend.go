package engine

import (
	"context"
	"maps"
	"slices"
	"sync"
	"time"

	"assay/internal/gguf"
	"assay/internal/hf"
)

// Recommended sampling settings, from three places: the GGUF itself
// (general.sampling.*, written by recent llama.cpp conversions and by
// uploaders such as Unsloth), the model card's prose ("temperature=1.0,
// top_p=0.95, top_k=20"), and the Ollama-style params file some uploaders
// ship. Using one copies its settings into the model's Params, the layer
// under its own settings and each chat's.

// RecProfile is one set of recommended settings.
type RecProfile struct {
	Label   string         `json:"label"`
	Source  string         `json:"source"` // "file", "card" or "params"
	Options map[string]any `json:"options"`
}

// ggufSampling maps llama.cpp's general.sampling.* keys to option names.
var ggufSampling = map[string]string{
	"general.sampling.temp":           "temperature",
	"general.sampling.top_p":          "top_p",
	"general.sampling.top_k":          "top_k",
	"general.sampling.min_p":          "min_p",
	"general.sampling.penalty_repeat": "repeat_penalty",
	"general.sampling.penalty_last_n": "repeat_last_n",
}

// samplingFromGGUF reads the file's recommended settings. Float32 values
// come back as 0.949999988; they are rounded to what was meant.
func samplingFromGGUF(f *gguf.File) map[string]any {
	out := map[string]any{}
	for k, opt := range ggufSampling {
		v, ok := f.Meta[k].(float64)
		if !ok {
			continue
		}
		if opt == "top_k" || opt == "repeat_last_n" {
			out[opt] = int(v)
		} else {
			out[opt] = roundTo(v, 4)
		}
	}
	return out
}

func roundTo(v float64, digits int) float64 {
	p := 1.0
	for range digits {
		p *= 10
	}
	if v < 0 {
		return -float64(int64(-v*p+0.5)) / p
	}
	return float64(int64(v*p+0.5)) / p
}

// CardReader is the part of hf.Client recommendations need beyond HF.
type CardReader interface {
	Card(ctx context.Context, repo string) (string, error)
}

var cardCache = struct {
	sync.Mutex
	m map[string]cachedCard
}{m: map[string]cachedCard{}}

type cachedCard struct {
	text string
	at   time.Time
}

func (e *Engine) card(ctx context.Context, repo string) string {
	cr, ok := e.HF.(CardReader)
	if !ok || repo == "" {
		return ""
	}
	cardCache.Lock()
	c, hit := cardCache.m[repo]
	cardCache.Unlock()
	if hit && time.Since(c.at) < time.Hour {
		return c.text
	}
	text, err := cr.Card(ctx, repo)
	if err != nil {
		return ""
	}
	cardCache.Lock()
	cardCache.m[repo] = cachedCard{text, time.Now()}
	cardCache.Unlock()
	return text
}

// Recommended lists the recommended settings for a model, the file's
// first. Asking Hugging Face is skipped when ctx says offline (it just
// fails quickly).
func (e *Engine) Recommended(ctx context.Context, name string) ([]RecProfile, error) {
	m, err := e.store.Get(name)
	if err != nil {
		return nil, err
	}
	var out []RecProfile
	add := func(p RecProfile) {
		if len(p.Options) == 0 {
			return
		}
		for _, q := range out {
			if maps.Equal(q.Options, p.Options) {
				return
			}
		}
		out = append(out, p)
	}
	if len(m.Files) > 0 {
		if f, err := gguf.Read(e.store.Blob(m.Files[0])); err == nil {
			add(RecProfile{Label: "Set in the model file", Source: "file", Options: samplingFromGGUF(f)})
		}
	}
	if m.Repo != "" && e.HF != nil {
		cctx, cancel := context.WithTimeout(ctx, 8*time.Second)
		defer cancel()
		for _, p := range hf.CardSettings(e.card(cctx, m.Repo)) {
			add(RecProfile{Label: p.Label, Source: "card", Options: p.Options})
		}
		if p, err := e.HF.RecommendedParams(cctx, m.Repo); err == nil {
			add(RecProfile{Label: "The uploader's params file", Source: "params", Options: onlySampling(p)})
		}
	}
	return out, nil
}

// onlySampling keeps the sampling settings of a params map (not stop
// words and the like).
func onlySampling(p map[string]any) map[string]any {
	out := map[string]any{}
	for k, v := range p {
		if slices.Contains(hf.SamplingKeys, k) || k == "repeat_last_n" {
			out[k] = v
		}
	}
	return out
}

// UseRecommended makes opts the recommended sampling of the model and of
// the other downloads of the same repo (other quants of the same model):
// the sampling in their Params is replaced, other Params (stop words) kept.
func (e *Engine) UseRecommended(name string, opts map[string]any) error {
	m, err := e.store.Get(name)
	if err != nil {
		return err
	}
	for _, x := range e.store.List() {
		if x != m && (m.Repo == "" || x.Repo != m.Repo) {
			continue
		}
		cp := *x
		cp.Params = map[string]any{}
		for k, v := range x.Params {
			if _, s := onlySampling(map[string]any{k: v})[k]; !s {
				cp.Params[k] = v
			}
		}
		maps.Copy(cp.Params, onlySampling(opts))
		if len(cp.Params) == 0 {
			cp.Params = nil
		}
		if err := e.store.Put(&cp); err != nil {
			return err
		}
	}
	return nil
}

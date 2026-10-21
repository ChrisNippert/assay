package server

import (
	"context"
	"slices"
	"strings"
	"sync"
	"time"
)

// embedSuggestion is a download offered on the Knowledge settings.
type embedSuggestion struct {
	Name      string
	Size      string
	Desc      string
	Installed string // the installed name ("" if not installed)
}

var embedSuggestions = []embedSuggestion{
	{Name: "qwen3-embedding:0.6b", Size: "~640 MB", Desc: "Strong retrieval for its size, multilingual, long inputs"},
	{Name: "embeddinggemma", Size: "~620 MB", Desc: "Google's small multilingual embedding model"},
	{Name: "nomic-embed-text", Size: "~275 MB", Desc: "Light and fast, English-focused"},
	{Name: "mxbai-embed-large", Size: "~670 MB", Desc: "Strong English retrieval"},
	{Name: "snowflake-arctic-embed2", Size: "~1.2 GB", Desc: "Multilingual, high quality"},
	{Name: "bge-m3", Size: "~1.2 GB", Desc: "Multilingual, long inputs"},
}

type embedModel struct {
	Name     string
	Size     int64
	Selected bool
}

// knowledgePage is what the Knowledge section renders.
type knowledgePage struct {
	Current     string
	TopK        int
	TopKs       []int
	Models      []embedModel // installed embedding models
	Suggestions []embedSuggestion
	HasFiles    bool   // some project has files, so changing the model re-indexes
	Notice      string // result of a just-finished download
	NoticeBad   bool
}

func sameEmbed(a, b string) bool {
	return a == b || strings.TrimSuffix(a, ":latest") == strings.TrimSuffix(b, ":latest")
}

// embeddingCapable reports, for each model name, whether Ollama says it can embed.
func (s *Server) embeddingCapable(ctx context.Context, names []string) map[string]bool {
	ctx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	var mu sync.Mutex
	var wg sync.WaitGroup
	out := map[string]bool{}
	for _, n := range names {
		wg.Add(1)
		go func() {
			defer wg.Done()
			sh, err := s.ollama.Show(ctx, n)
			if err != nil {
				return
			}
			mu.Lock()
			out[n] = slices.Contains(sh.Capabilities, "embedding")
			mu.Unlock()
		}()
	}
	wg.Wait()
	return out
}

func (s *Server) knowledgePage(ctx context.Context, pulled string) knowledgePage {
	st := s.ragSt.Get()
	k := knowledgePage{Current: st.EmbedModel, TopK: st.TopK, TopKs: []int{3, 4, 6, 8, 10, 12}}
	if !slices.Contains(k.TopKs, k.TopK) {
		k.TopKs = append(k.TopKs, k.TopK)
		slices.Sort(k.TopKs)
	}
	models, _ := s.ollama.List(ctx)
	var names []string
	size := map[string]int64{}
	for _, m := range models {
		names = append(names, m.Name)
		size[m.Name] = m.Size
	}
	caps := s.embeddingCapable(ctx, names)
	for _, n := range names {
		if caps[n] {
			k.Models = append(k.Models, embedModel{n, size[n], sameEmbed(n, st.EmbedModel)})
		}
	}
	k.Suggestions = slices.Clone(embedSuggestions)
	for i, sg := range k.Suggestions {
		for _, m := range k.Models {
			if sameEmbed(m.Name, sg.Name) {
				k.Suggestions[i].Installed = m.Name
			}
		}
	}
	if s.projects != nil {
		if ps, err := s.projects.List(); err == nil {
			for _, p := range ps {
				if len(p.Files) > 0 {
					k.HasFiles = true
				}
			}
		}
	}
	if pulled != "" {
		for _, n := range names {
			if !sameEmbed(n, pulled) {
				continue
			}
			if caps[n] {
				k.Notice = n + " is installed. Choose it above to use it."
			} else {
				k.Notice, k.NoticeBad = n+" isn't an embedding model, so it can't index files. Choose another one.", true
			}
			return k
		}
		k.Notice, k.NoticeBad = "Couldn't find "+pulled+" after the download.", true
	}
	return k
}

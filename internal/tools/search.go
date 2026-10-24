package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"html"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"
)

const searchResults = 5

// Hit is one web search result.
type Hit struct {
	Title, URL, Snippet string
}

// Searcher runs web searches. The SearXNG address comes from the settings;
// DDGURL is html.duckduckgo.com unless a test points it elsewhere.
type Searcher struct {
	HTTP   *http.Client
	DDGURL string
}

// NewSearcher returns a Searcher for the real services. Search backends
// are chosen by the user, so (unlike fetch_url) a local SearXNG is fine.
func NewSearcher() *Searcher {
	return &Searcher{HTTP: &http.Client{Timeout: 15 * time.Second}, DDGURL: "https://html.duckduckgo.com/html/"}
}

func searxBase(raw string) (*url.URL, error) {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		return nil, errors.New("the SearXNG address should look like http://localhost:8888")
	}
	u.Path = strings.TrimSuffix(strings.TrimSuffix(u.Path, "/"), "/search")
	return u, nil
}

// Search asks the backend for results.
func (s *Searcher) Search(ctx context.Context, set Settings, query string) ([]Hit, error) {
	switch set.Search {
	case SearchSearXNG:
		return s.searx(ctx, set.SearxURL, query)
	case SearchDuckDuckGo:
		return s.ddg(ctx, query)
	}
	return nil, errors.New("web search is off")
}

func (s *Searcher) get(ctx context.Context, u string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; ollamui; +local assistant)")
	res, err := s.HTTP.Do(req)
	if err != nil {
		return nil, unwrapURLError(err)
	}
	defer res.Body.Close()
	b, _ := io.ReadAll(io.LimitReader(res.Body, 2<<20))
	if res.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("the search service answered %s", res.Status)
	}
	return b, nil
}

func (s *Searcher) searx(ctx context.Context, base, query string) ([]Hit, error) {
	u, err := searxBase(base)
	if err != nil {
		return nil, err
	}
	u = u.JoinPath("search")
	u.RawQuery = url.Values{"q": {query}, "format": {"json"}}.Encode()
	b, err := s.get(ctx, u.String())
	if err != nil {
		return nil, err
	}
	var out struct {
		Results []struct {
			Title, URL, Content string
		} `json:"results"`
	}
	if err := json.Unmarshal(b, &out); err != nil {
		return nil, errors.New("SearXNG didn't answer with JSON (is the json format enabled in its settings?)")
	}
	var hits []Hit
	for _, r := range out.Results {
		if r.URL == "" {
			continue
		}
		hits = append(hits, Hit{Title: r.Title, URL: r.URL, Snippet: r.Content})
		if len(hits) == searchResults {
			break
		}
	}
	return hits, nil
}

var (
	reDDGLink    = regexp.MustCompile(`(?is)<a\b[^>]*class="[^"]*\bresult__a\b[^"]*"[^>]*>`)
	reDDGSnippet = regexp.MustCompile(`(?is)<a\b[^>]*class="[^"]*\bresult__snippet\b[^"]*"[^>]*>(.*?)</a>`)
	reHref       = regexp.MustCompile(`(?is)\bhref="([^"]*)"`)
)

// ddg scrapes DuckDuckGo's HTML page, which has no API and may change.
func (s *Searcher) ddg(ctx context.Context, query string) ([]Hit, error) {
	b, err := s.get(ctx, s.DDGURL+"?"+url.Values{"q": {query}}.Encode())
	if err != nil {
		return nil, err
	}
	return parseDDG(string(b)), nil
}

func parseDDG(page string) []Hit {
	var hits []Hit
	links := reDDGLink.FindAllStringIndex(page, -1)
	for i, loc := range links {
		end := len(page)
		if i+1 < len(links) {
			end = links[i+1][0]
		}
		tag := page[loc[0]:loc[1]]
		m := reHref.FindStringSubmatch(tag)
		if m == nil {
			continue
		}
		target := ddgTarget(html.UnescapeString(m[1]))
		if target == "" {
			continue
		}
		// The title is the link's text; the snippet is the next snippet link.
		rest := page[loc[1]:end]
		title, _, _ := strings.Cut(rest, "</a>")
		h := Hit{Title: plainText(title), URL: target}
		if sm := reDDGSnippet.FindStringSubmatch(rest); sm != nil {
			h.Snippet = plainText(sm[1])
		}
		hits = append(hits, h)
		if len(hits) == searchResults {
			break
		}
	}
	return hits
}

// ddgTarget unwraps DuckDuckGo's redirect (//duckduckgo.com/l/?uddg=...).
// Ads (y.js) are dropped.
func ddgTarget(href string) string {
	if strings.HasPrefix(href, "//") {
		href = "https:" + href
	}
	u, err := url.Parse(href)
	if err != nil {
		return ""
	}
	if strings.HasSuffix(u.Host, "duckduckgo.com") {
		if strings.HasPrefix(u.Path, "/y.js") {
			return ""
		}
		t := u.Query().Get("uddg")
		if t == "" {
			return ""
		}
		return t
	}
	if u.Scheme == "http" || u.Scheme == "https" {
		return href
	}
	return ""
}

func plainText(s string) string {
	return strings.Join(strings.Fields(html.UnescapeString(reTag.ReplaceAllString(s, ""))), " ")
}

func webSearchTool(s *Searcher, set Settings) Tool {
	return Tool{
		Name:        "web_search",
		Description: "Search the web and get the top results (title, address, short snippet). Use it for current events, facts you are unsure of, or anything recent. Then use fetch_url on a result to read the page. Keep the query short, like a search-engine query.",
		Parameters:  objSchema("query", "What to search for, e.g. \"rust 1.80 release notes\""),
		Run: func(ctx context.Context, args json.RawMessage) Result {
			q, err := strArg(args, "query")
			if err != nil {
				return fail("Searched the web", err.Error())
			}
			hits, err := s.Search(ctx, set, q)
			label := "Searched the web for “" + short(q, 60) + "”"
			if err != nil {
				return fail(label, err.Error())
			}
			if len(hits) == 0 {
				return Result{Text: "No results.", Label: label + " · no results"}
			}
			var b strings.Builder
			for i, h := range hits {
				fmt.Fprintf(&b, "%d. %s\n   %s\n", i+1, h.Title, h.URL)
				if h.Snippet != "" {
					b.WriteString("   " + short(h.Snippet, 300) + "\n")
				}
			}
			return Result{Text: strings.TrimSpace(b.String()), Label: label + " · " + plural(len(hits), "result", "results")}
		},
	}
}

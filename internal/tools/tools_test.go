package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/netip"
	"net/url"
	"strings"
	"testing"
	"time"

	"assay/internal/chats"
)

func TestEval(t *testing.T) {
	ok := map[string]string{
		"2^10":                    "1024",
		"2**3":                    "8",
		"-2^2":                    "-4",
		"2^3^2":                   "512",
		"(1+2)*3 - 4/2":           "7",
		"10 % 4":                  "2",
		"sqrt(144) + abs(-3)":     "15",
		"1.5e3 + 1":               "1501",
		"max(1, 5, 3) * min(4,2)": "10",
		"pow(2, 0.5)":             "1.41421356237",
		"pi * 2":                  "6.28318530718",
		"round(2.5) + floor(1.9)": "4",
		"0.1 + 0.2":               "0.3",
		"2 * -3":                  "-6",
		"1_000 * 3":               "3000",
		"log(1000) + ln(e)":       "4",
	}
	for in, want := range ok {
		v, err := Eval(in)
		if err != nil {
			t.Errorf("%q: %v", in, err)
			continue
		}
		if got := FormatNumber(v); got != want {
			t.Errorf("%q = %s, want %s", in, got, want)
		}
	}
	for _, bad := range []string{"", "1/0", "2 +", "(1", "1)", "foo(1)", "x", "1 2", "sqrt(1,2)", "9^9^9^9", "system('ls')", "__import__('os')", strings.Repeat("(", 200) + "1" + strings.Repeat(")", 200), "5 % 0"} {
		if v, err := Eval(bad); err == nil {
			t.Errorf("%q = %v, want an error", bad, v)
		}
	}
}

func TestCalculatorTool(t *testing.T) {
	set := Build(Deps{})
	r := Run(context.Background(), set, "calculator", json.RawMessage(`{"expression":"2^10"}`))
	if r.Err || r.Text != "1024" || r.Label != "Calculated 2^10 = 1024" {
		t.Errorf("%+v", r)
	}
	r = Run(context.Background(), set, "calculator", json.RawMessage(`{"expression":"1/0"}`))
	if !r.Err || !strings.Contains(r.Text, "division by zero") {
		t.Errorf("%+v", r)
	}
	r = Run(context.Background(), set, "calculator", json.RawMessage(`"2+2"`))
	if !r.Err {
		t.Errorf("a non-object argument should fail: %+v", r)
	}
	r = Run(context.Background(), set, "nope", nil)
	if !r.Err || !strings.Contains(r.Text, "calculator") {
		t.Errorf("unknown tool should list the real ones: %+v", r)
	}
}

func TestBlocked(t *testing.T) {
	for _, s := range []string{"127.0.0.1", "127.1.2.3", "::1", "10.0.0.5", "172.16.0.1", "172.31.255.255", "192.168.1.1", "169.254.169.254", "100.64.0.1", "100.100.100.100", "100.127.255.255", "0.0.0.0", "::", "fe80::1", "fc00::1", "fd12::1", "::ffff:127.0.0.1", "::ffff:10.1.1.1", "224.0.0.1", "ff02::1", "198.18.0.1"} {
		if !Blocked(netip.MustParseAddr(s)) {
			t.Errorf("%s should be blocked", s)
		}
	}
	for _, s := range []string{"93.184.216.34", "8.8.8.8", "100.63.255.255", "100.128.0.1", "172.32.0.1", "2606:2800:220:1:248:1893:25c8:1946"} {
		if Blocked(netip.MustParseAddr(s)) {
			t.Errorf("%s should be allowed", s)
		}
	}
}

func allowPorts(ports ...string) func(netip.AddrPort) bool {
	return func(ap netip.AddrPort) bool {
		for _, p := range ports {
			if fmt.Sprint(ap.Port()) == p {
				return true
			}
		}
		return false
	}
}

func portOf(ts *httptest.Server) string { u, _ := url.Parse(ts.URL); return u.Port() }

func TestFetchBlocksLocalAddresses(t *testing.T) {
	hit := false
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { hit = true }))
	defer ts.Close()
	f := NewFetcher() // no test exemption
	for _, u := range []string{ts.URL, "http://localhost:" + portOf(ts), "http://[::1]:" + portOf(ts) + "/", "http://0.0.0.0:" + portOf(ts), "http://169.254.169.254/latest/meta-data", "http://100.64.1.1/"} {
		if _, err := f.Fetch(context.Background(), u); err == nil {
			t.Errorf("%s: fetched", u)
		}
	}
	if hit {
		t.Error("the local server was reached")
	}
	for _, u := range []string{"file:///etc/passwd", "ftp://example.com/x", "gopher://x/"} {
		if _, err := f.Fetch(context.Background(), u); err == nil || !strings.Contains(err.Error(), "http") {
			t.Errorf("%s: %v", u, err)
		}
	}
}

func TestFetchRedirectToLoopbackIsBlocked(t *testing.T) {
	secret := false
	inner := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		secret = true
		fmt.Fprint(w, "secret")
	}))
	defer inner.Close()
	// The "public" server is let through; it sends the fetch to 127.0.0.1.
	outer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, inner.URL+"/admin", http.StatusFound)
	}))
	defer outer.Close()
	f := newFetcher(allowPorts(portOf(outer)))
	_, err := f.Fetch(context.Background(), outer.URL)
	if err == nil || !strings.Contains(err.Error(), "private network") {
		t.Errorf("err = %v", err)
	}
	if secret {
		t.Error("the redirect target was reached")
	}
}

func TestFetchRedirectLoopAndNonHTTP(t *testing.T) {
	var ts *httptest.Server
	ts = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/ftp" {
			http.Redirect(w, r, "ftp://example.com/", http.StatusFound)
			return
		}
		http.Redirect(w, r, "/loop", http.StatusFound)
	}))
	defer ts.Close()
	f := newFetcher(allowPorts(portOf(ts)))
	if _, err := f.Fetch(context.Background(), ts.URL+"/loop"); err == nil || !strings.Contains(err.Error(), "redirects") {
		t.Errorf("loop: %v", err)
	}
	if _, err := f.Fetch(context.Background(), ts.URL+"/ftp"); err == nil {
		t.Error("redirect to ftp was followed")
	}
}

const page = `<!doctype html><html><head><title>Example &amp; Co</title>
<style>body{color:red}</style><script>var x = "<p>nope</p>";</script></head>
<body><nav><a href="/">Home</a> <a href="/about">About</a></nav>
<header>Site banner</header>
<main><h1>Welcome</h1>
<p>This is the <b>first</b> paragraph, with a <a href="/docs/start">link to the docs</a> in it.</p>
<h2>Things</h2><ul><li>one</li><li>two &lt;2&gt;</li></ul>
<p>` + "Lorem ipsum dolor sit amet. " + `</p>
<form><input name=q><button>Go</button></form></main>
<footer>Copyright</footer></body></html>`

func TestFetchHTMLToText(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/json":
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"a":1}`)
		case "/bin":
			w.Header().Set("Content-Type", "image/png")
			fmt.Fprint(w, "\x89PNG")
		case "/404":
			http.NotFound(w, r)
		case "/big":
			w.Header().Set("Content-Type", "text/plain")
			fmt.Fprint(w, strings.Repeat("word ", 5000))
		default:
			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			// Padding so <main> is long enough to be preferred over the page.
			fmt.Fprint(w, strings.Replace(page, "Lorem ipsum", strings.Repeat("padding ", 60)+"Lorem ipsum", 1))
		}
	}))
	defer ts.Close()
	f := newFetcher(allowPorts(portOf(ts)))
	set := Build(Deps{Fetcher: f})
	run := func(path string) Result {
		b, _ := json.Marshal(map[string]string{"url": ts.URL + path})
		return Run(context.Background(), set, "fetch_url", b)
	}

	r := run("/page")
	if r.Err {
		t.Fatalf("%+v", r)
	}
	for _, want := range []string{"Title: Example & Co", "# Welcome", "## Things", "- one", "- two <2>", "link to the docs (" + ts.URL + "/docs/start)", "first paragraph"} {
		if !strings.Contains(r.Text, want) {
			t.Errorf("missing %q in:\n%s", want, r.Text)
		}
	}
	for _, bad := range []string{"nope", "Site banner", "Copyright", "color:red", "Home", "<p>", "Go"} {
		if strings.Contains(r.Text, bad) {
			t.Errorf("%q should be stripped from:\n%s", bad, r.Text)
		}
	}
	if !strings.HasPrefix(r.Label, "Read 127.0.0.1:") || !strings.HasSuffix(r.Label, "/page") {
		t.Errorf("label %q", r.Label)
	}

	if r := run("/json"); r.Err || !strings.Contains(r.Text, `{"a":1}`) {
		t.Errorf("json: %+v", r)
	}
	if r := run("/bin"); !r.Err || !strings.Contains(r.Text, "image/png") {
		t.Errorf("binary: %+v", r)
	}
	if r := run("/404"); !r.Err || !strings.Contains(r.Text, "404") {
		t.Errorf("404: %+v", r)
	}
	r = run("/big")
	if r.Err || !strings.Contains(r.Text, "[Cut off: showing the first") || len(r.Text) > fetchMaxChars+200 {
		t.Errorf("big: err=%v len=%d tail=%q", r.Err, len(r.Text), r.Text[max(0, len(r.Text)-100):])
	}
}

func TestFetchSizeCap(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		w.Write([]byte(strings.Repeat("x", fetchMaxBytes+5000)))
	}))
	defer ts.Close()
	p, err := newFetcher(allowPorts(portOf(ts))).Fetch(context.Background(), ts.URL)
	if err != nil || len(p.Text) != fetchMaxBytes {
		t.Errorf("len=%d err=%v", len(p.Text), err)
	}
}

func TestSearXNG(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/search" || r.URL.Query().Get("format") != "json" {
			http.NotFound(w, r)
			return
		}
		type res struct{ Title, URL, Content string }
		out := map[string]any{"results": []res{}}
		var rs []res
		for i := 1; i <= 8; i++ {
			rs = append(rs, res{fmt.Sprintf("Result %d for %s", i, r.URL.Query().Get("q")), fmt.Sprintf("https://example.org/%d", i), "A snippet."})
		}
		out["results"] = rs
		json.NewEncoder(w).Encode(out)
	}))
	defer ts.Close()
	set := Settings{Search: SearchSearXNG, SearxURL: ts.URL + "/"}
	tools := Build(Deps{Settings: set})
	r := Run(context.Background(), tools, "web_search", json.RawMessage(`{"query":"go generics"}`))
	if r.Err || !strings.Contains(r.Text, "1. Result 1 for go generics\n   https://example.org/1\n   A snippet.") || strings.Contains(r.Text, "6. ") {
		t.Errorf("%+v", r)
	}
	if r.Label != "Searched the web for “go generics” · 5 results" {
		t.Errorf("label %q", r.Label)
	}
	// Not JSON: say how to fix it.
	bad := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { fmt.Fprint(w, "<html>") }))
	defer bad.Close()
	r = Run(context.Background(), Build(Deps{Settings: Settings{Search: SearchSearXNG, SearxURL: bad.URL}}), "web_search", json.RawMessage(`{"query":"x"}`))
	if !r.Err || !strings.Contains(r.Text, "JSON") {
		t.Errorf("%+v", r)
	}
}

const ddgPage = `<div class="results">
<div class="result results_links"><h2 class="result__title"><a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fgo.dev%2Fdoc%2Ftutorial%2Fgenerics&amp;rut=abc">Tutorial: Getting started with <b>generics</b></a></h2>
<a class="result__snippet" href="//duckduckgo.com/l/?uddg=x">How to use <b>generics</b> in Go &amp; why.</a></div>
<div class="result result--ad"><a rel="nofollow" class="result__a" href="https://duckduckgo.com/y.js?ad_provider=bing">An ad</a><a class="result__snippet" href="x">Buy now</a></div>
<div class="result"><a rel="nofollow" class="result__a" href="https://example.com/direct">Direct link</a></div>
</div>`

func TestDuckDuckGo(t *testing.T) {
	hits := parseDDG(ddgPage)
	if len(hits) != 2 {
		t.Fatalf("hits = %+v", hits)
	}
	if hits[0].URL != "https://go.dev/doc/tutorial/generics" || hits[0].Title != "Tutorial: Getting started with generics" || hits[0].Snippet != "How to use generics in Go & why." {
		t.Errorf("%+v", hits[0])
	}
	if hits[1].URL != "https://example.com/direct" || hits[1].Snippet != "" {
		t.Errorf("%+v", hits[1])
	}

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("q") != "go generics" {
			http.Error(w, "no", 400)
			return
		}
		fmt.Fprint(w, ddgPage)
	}))
	defer ts.Close()
	s := NewSearcher()
	s.DDGURL = ts.URL + "/html/"
	r := Run(context.Background(), Build(Deps{Settings: Settings{Search: SearchDuckDuckGo}, Searcher: s}), "web_search", json.RawMessage(`{"query":"go generics"}`))
	if r.Err || r.Label != "Searched the web for “go generics” · 2 results" {
		t.Errorf("%+v", r)
	}
}

func TestWebSearchOffByDefault(t *testing.T) {
	for _, set := range []Settings{{}, {Search: SearchOff}, {Search: SearchSearXNG}, {Search: SearchDuckDuckGo, Off: []string{"web_search"}}} {
		for _, tl := range Build(Deps{Settings: set}) {
			if tl.Name == "web_search" {
				t.Errorf("%+v offers web_search", set)
			}
		}
	}
	names := func(set Settings) string {
		var n []string
		for _, tl := range Build(Deps{Settings: set, Chats: fakeChats{}, Project: func(context.Context, string) ([]Passage, error) { return nil, nil }}) {
			n = append(n, tl.Name)
		}
		return strings.Join(n, ",")
	}
	if got := names(Settings{Search: SearchDuckDuckGo}); got != "web_search,fetch_url,calculator,search_chats,search_project,current_time" {
		t.Error(got)
	}
	if got := names(Settings{Off: []string{"fetch_url", "calculator"}}); got != "search_chats,search_project,current_time" {
		t.Error(got)
	}
}

type fakeChats map[string]*chats.Chat

func (f fakeChats) List() ([]chats.Summary, error) {
	var out []chats.Summary
	for id, c := range f {
		out = append(out, chats.Summary{ID: id, Title: c.Title})
	}
	return out, nil
}
func (f fakeChats) Get(id string) (*chats.Chat, error) { return f[id], nil }

func TestSearchChats(t *testing.T) {
	mk := func(id, title string, msgs ...string) *chats.Chat {
		c := &chats.Chat{ID: id, Title: title, Updated: time.Date(2026, 10, 1, 0, 0, 0, 0, time.UTC)}
		for i, m := range msgs {
			role := "user"
			if i%2 == 1 {
				role = "assistant"
			}
			c.Messages = append(c.Messages, chats.Message{Role: role, Content: m})
		}
		return c
	}
	st := fakeChats{
		"20261001-00000001": mk("20261001-00000001", "Bread help", "How do I feed a sourdough starter?", "Feed it equal weights of flour and water every day."),
		"20261001-00000002": mk("20261001-00000002", "Bike", "My tyre pressure keeps dropping"),
		"20261001-00000003": mk("20261001-00000003", "Sourdough rye", "Rye flour needs a wetter dough and more starter than wheat."),
		"20261001-00000004": mk("20261001-00000004", "Current", "sourdough starter flour"),
	}
	hits, err := SearchChats(st, "sourdough starter", "20261001-00000004")
	if err != nil || len(hits) != 2 {
		t.Fatalf("%+v %v", hits, err)
	}
	// Both match both words; the title hit ranks the rye chat first.
	if hits[0].ID != "20261001-00000003" || hits[1].ID != "20261001-00000001" || !strings.Contains(hits[1].Snippet, "You: How do I feed a sourdough starter?") || !strings.Contains(hits[1].Snippet, "Assistant: Feed it equal weights") {
		t.Errorf("%+v", hits)
	}
	r := Run(context.Background(), Build(Deps{Chats: st, ExceptChat: "20261001-00000004"}), "search_chats", json.RawMessage(`{"query":"tyre"}`))
	if r.Err || !strings.Contains(r.Text, "Bike (chat id 20261001-00000002, 2026-10-01)") || r.Label != "Searched your chats for “tyre” · 1 chat" {
		t.Errorf("%+v", r)
	}
	r = Run(context.Background(), Build(Deps{Chats: st}), "search_chats", json.RawMessage(`{"query":"zebra"}`))
	if r.Err || !strings.Contains(r.Label, "no results") {
		t.Errorf("%+v", r)
	}
	if r := Run(context.Background(), Build(Deps{Chats: st}), "search_chats", json.RawMessage(`{"query":"!!"}`)); !r.Err {
		t.Errorf("%+v", r)
	}
}

func TestSearchProjectAndTime(t *testing.T) {
	set := Build(Deps{
		Project: func(_ context.Context, q string) ([]Passage, error) {
			return []Passage{{File: "notes.md", Heading: "Setup", Text: "Run make for " + q}, {File: "a.pdf", Page: 3, Text: "page text"}}, nil
		},
		Now: func() time.Time { return time.Date(2026, 10, 24, 14, 5, 0, 0, time.UTC) },
	})
	r := Run(context.Background(), set, "search_project", json.RawMessage(`{"query":"build"}`))
	if r.Err || !strings.Contains(r.Text, "[1] notes.md (Setup)\nRun make for build") || !strings.Contains(r.Text, "[2] a.pdf, page 3") || !strings.HasSuffix(r.Label, "· 2 passages") {
		t.Errorf("%+v", r)
	}
	r = Run(context.Background(), set, "current_time", nil)
	if !strings.HasPrefix(r.Text, "Saturday 24 October 2026, 14:05") || r.Label != "Checked the time · 14:05" {
		t.Errorf("%+v", r)
	}
}

func TestStore(t *testing.T) {
	path := t.TempDir() + "/tools.json"
	st, err := OpenStore(path)
	if err != nil {
		t.Fatal(err)
	}
	if st.Get().Search != SearchOff {
		t.Error("search should start off")
	}
	if err := st.Set(Settings{Search: "bing"}); err == nil {
		t.Error("unknown backend accepted")
	}
	if err := st.Set(Settings{Search: SearchSearXNG, SearxURL: "ftp://x"}); err == nil {
		t.Error("bad SearXNG URL accepted")
	}
	if err := st.Set(Settings{Off: []string{"hack"}}); err == nil {
		t.Error("unknown tool accepted")
	}
	want := Settings{Search: SearchSearXNG, SearxURL: "http://localhost:8888", Off: []string{"calculator"}}
	if err := st.Set(want); err != nil {
		t.Fatal(err)
	}
	st2, _ := OpenStore(path)
	if got := st2.Get(); got.Search != want.Search || got.SearxURL != want.SearxURL || len(got.Off) != 1 {
		t.Errorf("%+v", got)
	}
}

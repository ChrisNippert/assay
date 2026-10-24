package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"html"
	"io"
	"mime"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"regexp"
	"strings"
	"syscall"
	"time"
)

const (
	fetchMaxBytes = 2 << 20 // largest page read
	fetchMaxChars = 8000    // what goes back to the model
	fetchTimeout  = 20 * time.Second
)

// Fetcher downloads pages for fetch_url. It refuses to connect to
// addresses on this machine or its networks, so a page (or a model that
// was told to) can't reach services behind the user's firewall.
type Fetcher struct {
	client *http.Client
}

var cgnat = netip.MustParsePrefix("100.64.0.0/10") // carrier-grade NAT, also Tailscale

// Blocked reports whether a fetched page may not be read from ip:
// loopback, private, link-local, CGNAT/Tailscale, multicast and the
// unspecified address.
func Blocked(ip netip.Addr) bool {
	ip = ip.Unmap()
	return !ip.IsValid() || ip.IsLoopback() || ip.IsPrivate() || ip.IsLinkLocalUnicast() ||
		ip.IsLinkLocalMulticast() || ip.IsInterfaceLocalMulticast() || ip.IsMulticast() ||
		ip.IsUnspecified() || cgnat.Contains(ip) ||
		netip.MustParsePrefix("192.0.0.0/24").Contains(ip) || netip.MustParsePrefix("198.18.0.0/15").Contains(ip)
}

// NewFetcher returns a Fetcher with the address guard on.
func NewFetcher() *Fetcher { return newFetcher(nil) }

// newFetcher builds a Fetcher; allow, if set, can let an address through
// that Blocked would refuse (tests talk to servers on 127.0.0.1).
func newFetcher(allow func(netip.AddrPort) bool) *Fetcher {
	// The check runs on the address about to be connected to, after DNS, for
	// the first request and every redirect alike. Checking the URL's host
	// beforehand would be open to DNS rebinding.
	dialer := &net.Dialer{
		Timeout: 10 * time.Second,
		Control: func(network, address string, _ syscall.RawConn) error {
			ap, err := netip.ParseAddrPort(address)
			if err != nil {
				return fmt.Errorf("bad address %q", address)
			}
			if allow != nil && allow(ap) {
				return nil
			}
			if Blocked(ap.Addr()) {
				return errors.New("that address is on this computer or a private network, so it can't be fetched")
			}
			return nil
		},
	}
	tr := &http.Transport{
		Proxy:                 nil, // a proxy would do the connecting, past the guard
		DialContext:           dialer.DialContext,
		ForceAttemptHTTP2:     true,
		TLSHandshakeTimeout:   10 * time.Second,
		ResponseHeaderTimeout: fetchTimeout,
		MaxIdleConns:          4,
		IdleConnTimeout:       30 * time.Second,
	}
	return &Fetcher{client: &http.Client{
		Transport: tr,
		Timeout:   fetchTimeout,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= 5 {
				return errors.New("too many redirects")
			}
			return checkURL(req.URL)
		},
	}}
}

func checkURL(u *url.URL) error {
	if u.Scheme != "http" && u.Scheme != "https" {
		return fmt.Errorf("only http and https addresses can be fetched, not %q", u.Scheme)
	}
	if u.Hostname() == "" {
		return errors.New("the address has no host")
	}
	return nil
}

// Page is a fetched page, as text.
type Page struct {
	URL   string // after redirects
	Title string
	Text  string
}

// Fetch downloads rawURL and returns its readable text.
func (f *Fetcher) Fetch(ctx context.Context, rawURL string) (*Page, error) {
	u, err := url.Parse(rawURL)
	if err != nil || u.Host == "" {
		// "example.com/page" without a scheme is a common slip.
		if u2, err2 := url.Parse("https://" + rawURL); err2 == nil && u2.Host != "" && !strings.Contains(rawURL, "://") {
			u, err = u2, nil
		}
	}
	if err != nil {
		return nil, errors.New("that isn't a valid web address")
	}
	if err := checkURL(u); err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(ctx, fetchTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; ollamui; +local assistant)")
	req.Header.Set("Accept", "text/html,application/xhtml+xml,text/plain,application/json;q=0.9,*/*;q=0.5")
	res, err := f.client.Do(req)
	if err != nil {
		return nil, unwrapURLError(err)
	}
	defer res.Body.Close()
	if res.StatusCode >= 400 {
		return nil, fmt.Errorf("the server answered %s", res.Status)
	}
	ct, _, _ := mime.ParseMediaType(res.Header.Get("Content-Type"))
	body, err := io.ReadAll(io.LimitReader(res.Body, fetchMaxBytes))
	if err != nil {
		return nil, unwrapURLError(err)
	}
	text := strings.ToValidUTF8(string(body), "")
	page := &Page{URL: res.Request.URL.String()}
	switch {
	case ct == "text/html" || ct == "application/xhtml+xml" || ct == "" && strings.Contains(text[:min(len(text), 512)], "<html"):
		page.Title, page.Text = HTMLToText(text, res.Request.URL)
	case strings.HasPrefix(ct, "text/") || ct == "application/json" || strings.HasSuffix(ct, "+json") || strings.HasSuffix(ct, "xml"):
		page.Text = strings.TrimSpace(text)
	default:
		return nil, fmt.Errorf("it is %s, which can't be read as text", ct)
	}
	return page, nil
}

func unwrapURLError(err error) error {
	var ue *url.Error
	if errors.As(err, &ue) {
		err = ue.Err
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return errors.New("the server took too long to answer")
	}
	return err
}

func fetchTool(f *Fetcher) Tool {
	return Tool{
		Name:        "fetch_url",
		Description: "Download a web page (an http or https address) and read its text. Use it to read a page the user gave you, or a page found with web_search. Long pages are cut off after about 8000 characters. It can't open local or private network addresses.",
		Parameters:  objSchema("url", "The full address, e.g. https://example.com/article"),
		Run: func(ctx context.Context, args json.RawMessage) Result {
			raw, err := strArg(args, "url")
			if err != nil {
				return fail("Read a page", err.Error())
			}
			page, err := f.Fetch(ctx, raw)
			if err != nil {
				return fail("Couldn’t read "+short(displayURL(raw), 60), err.Error())
			}
			var b strings.Builder
			b.WriteString("URL: " + page.URL + "\n")
			if page.Title != "" {
				b.WriteString("Title: " + page.Title + "\n")
			}
			b.WriteString("\n")
			if page.Text == "" {
				b.WriteString("(The page has no readable text. It may need JavaScript.)")
			} else {
				b.WriteString(page.Text)
			}
			return Result{Text: Clip(b.String(), fetchMaxChars), Label: "Read " + short(displayURL(page.URL), 60)}
		},
	}
}

// displayURL drops the scheme and a trailing slash: "example.com/page".
func displayURL(raw string) string {
	s := raw
	if i := strings.Index(s, "://"); i >= 0 {
		s = s[i+3:]
	}
	s = strings.TrimPrefix(s, "www.")
	return strings.TrimSuffix(s, "/")
}

// --- HTML to text ---

var (
	reDropBlocks = func() []*regexp.Regexp {
		var out []*regexp.Regexp
		for _, t := range []string{"script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg", "iframe", "template", "select", "button"} {
			out = append(out, regexp.MustCompile(`(?is)<`+t+`\b[^>]*>.*?</`+t+`\s*>`))
		}
		return out
	}()
	reComment  = regexp.MustCompile(`(?s)<!--.*?-->`)
	reTitle    = regexp.MustCompile(`(?is)<title[^>]*>(.*?)</title>`)
	reMain     = regexp.MustCompile(`(?is)<(main|article)\b[^>]*>(.*)</(main|article)\s*>`)
	reHeading  = regexp.MustCompile(`(?is)<h([1-6])\b[^>]*>`)
	reHeadEnd  = regexp.MustCompile(`(?i)</h[1-6]\s*>`)
	reItem     = regexp.MustCompile(`(?i)<li\b[^>]*>`)
	reBlock    = regexp.MustCompile(`(?i)</?(p|div|section|ul|ol|table|tr|blockquote|pre|figure|dl|dt|dd|details|summary|article|main)\b[^>]*>`)
	reBreak    = regexp.MustCompile(`(?i)<br\s*/?>`)
	reLink     = regexp.MustCompile(`(?is)<a\b[^>]*?href\s*=\s*("([^"]*)"|'([^']*)'|([^\s>]+))[^>]*>(.*?)</a\s*>`)
	reTag      = regexp.MustCompile(`(?s)<[^>]*>`)
	reSpaces   = regexp.MustCompile(`[ \t\f\v\x{a0}]+`)
	reBlankRun = regexp.MustCompile(`\n{3,}`)
)

// HTMLToText turns a page into readable text: scripts, styles and page
// furniture (nav, header, footer, forms) are dropped, headings become
// "# lines", list items "- lines", and links "text (address)". It returns
// the page title too. base resolves relative links.
func HTMLToText(s string, base *url.URL) (title, text string) {
	if m := reTitle.FindStringSubmatch(s); m != nil {
		title = strings.TrimSpace(strings.Join(strings.Fields(html.UnescapeString(reTag.ReplaceAllString(m[1], ""))), " "))
	}
	s = reComment.ReplaceAllString(s, "")
	for _, re := range reDropBlocks {
		s = re.ReplaceAllString(s, "")
	}
	// A page with a <main> or <article> keeps just that.
	if m := reMain.FindStringSubmatch(s); m != nil && len(m[2]) > 400 {
		s = m[2]
	}
	s = reLink.ReplaceAllStringFunc(s, func(a string) string {
		m := reLink.FindStringSubmatch(a)
		href := html.UnescapeString(m[2] + m[3] + m[4])
		label := strings.Join(strings.Fields(html.UnescapeString(reTag.ReplaceAllString(m[5], ""))), " ")
		if label == "" || strings.HasPrefix(href, "#") || strings.HasPrefix(strings.ToLower(href), "javascript:") || strings.HasPrefix(strings.ToLower(href), "mailto:") {
			return m[5]
		}
		if base != nil {
			if u, err := base.Parse(href); err == nil {
				href = u.String()
			}
		}
		return label + " (" + href + ")"
	})
	s = reHeading.ReplaceAllStringFunc(s, func(h string) string {
		return "\n\n" + strings.Repeat("#", int(reHeading.FindStringSubmatch(h)[1][0]-'0')) + " "
	})
	s = reHeadEnd.ReplaceAllString(s, "\n\n")
	s = reItem.ReplaceAllString(s, "\n- ")
	s = reBlock.ReplaceAllString(s, "\n\n")
	s = reBreak.ReplaceAllString(s, "\n")
	s = reTag.ReplaceAllString(s, "")
	s = html.UnescapeString(s)
	lines := strings.Split(s, "\n")
	for i, l := range lines {
		lines[i] = strings.TrimSpace(reSpaces.ReplaceAllString(l, " "))
	}
	s = reBlankRun.ReplaceAllString(strings.Join(lines, "\n"), "\n\n")
	return title, strings.TrimSpace(s)
}

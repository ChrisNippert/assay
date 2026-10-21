// Package rag turns project files into plain text and cuts that text into
// chunks sized for embedding.
package rag

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/xml"
	"errors"
	"fmt"
	"html"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"unicode/utf8"
)

// MaxFileSize is the largest file Extract accepts (20 MiB).
const MaxFileSize = 20 << 20

// maxXMLSize caps how much of a zipped XML part we will inflate.
const maxXMLSize = 8 * MaxFileSize

// ErrUnsupported is returned for file types Extract can't read.
var ErrUnsupported = errors.New("unsupported file type")

// ErrTooLarge is returned when a file is over MaxFileSize.
var ErrTooLarge = errors.New("file is too large (limit 20 MiB)")

// textExts are the extensions read as plain text.
var textExts = setOf(
	".txt", ".md", ".markdown", ".rst", ".org", ".csv", ".tsv", ".json", ".jsonl",
	".yaml", ".yml", ".toml", ".xml", ".ini", ".log", ".tex",
	".go", ".py", ".js", ".ts", ".tsx", ".jsx", ".rs", ".c", ".h", ".cpp", ".hpp",
	".java", ".kt", ".swift", ".rb", ".php", ".sh", ".zsh", ".bash", ".lua", ".sql",
	".css", ".scss", ".vue", ".svelte", ".zig", ".nim", ".hs", ".ml", ".ex", ".exs",
	".clj", ".dart", ".r", ".jl", ".m", ".cs", ".fs", ".scala", ".pl",
)

// otherExts are the formats that need their own reader.
var otherExts = setOf(".html", ".htm", ".pdf", ".docx", ".odt")

func setOf(xs ...string) map[string]bool {
	m := make(map[string]bool, len(xs))
	for _, x := range xs {
		m[x] = true
	}
	return m
}

func ext(name string) string { return strings.ToLower(filepath.Ext(name)) }

// Supported reports whether Extract can read a file with this name (by extension).
func Supported(name string) bool {
	e := ext(name)
	return textExts[e] || otherExts[e]
}

// Extract returns the plain text of a file. PDF pages are separated by form
// feeds, which Split uses for page numbers.
func Extract(ctx context.Context, name string, data []byte) (string, error) {
	if len(data) > MaxFileSize {
		return "", ErrTooLarge
	}
	e := ext(name)
	switch {
	case textExts[e]:
		return decodeText(data)
	case e == ".html" || e == ".htm":
		s, err := decodeText(data)
		if err != nil {
			return "", err
		}
		return htmlText(s), nil
	case e == ".pdf":
		return pdfText(ctx, data)
	case e == ".docx":
		return zipXML(data, "word/document.xml", docxText)
	case e == ".odt":
		return zipXML(data, "content.xml", odtText)
	}
	return "", ErrUnsupported
}

// decodeText checks that data looks like UTF-8 text and tidies line endings.
func decodeText(data []byte) (string, error) {
	data = bytes.TrimPrefix(data, []byte("\xef\xbb\xbf"))
	if !utf8.Valid(data) || bytes.IndexByte(data, 0) >= 0 {
		return "", ErrUnsupported
	}
	s := string(data)
	if strings.Contains(s, "\r") {
		s = strings.ReplaceAll(s, "\r\n", "\n")
		s = strings.ReplaceAll(s, "\r", "\n")
	}
	return s, nil
}

var (
	reHTMLDrop  = regexp.MustCompile(`(?is)<script\b.*?</script\s*>|<style\b.*?</style\s*>|<!--.*?-->`)
	reHTMLPara  = regexp.MustCompile(`(?i)</?(?:p|div|h[1-6]|ul|ol|table|section|article|header|footer|nav|pre|blockquote|dl|form|main|aside|figure|title|body|html|head)\b[^>]*>`)
	reHTMLLine  = regexp.MustCompile(`(?i)<(?:br|hr|li|tr|dt|dd)\b[^>]*>`)
	reHTMLTag   = regexp.MustCompile(`<[a-zA-Z/!?][^>]*>`)
	reHSpace    = regexp.MustCompile(`[ \t\f\v]+`)
	reBlankRuns = regexp.MustCompile(`\n{3,}`)
)

// htmlText strips tags, scripts and styles, keeping paragraph breaks.
func htmlText(s string) string {
	s = reHTMLDrop.ReplaceAllString(s, "")
	s = reHTMLPara.ReplaceAllString(s, "\n\n")
	s = reHTMLLine.ReplaceAllString(s, "\n")
	s = reHTMLTag.ReplaceAllString(s, "")
	s = html.UnescapeString(s)
	return tidy(s)
}

// tidy collapses runs of spaces, trims lines and limits blank lines to one.
func tidy(s string) string {
	s = strings.ReplaceAll(s, " ", " ")
	lines := strings.Split(s, "\n")
	for i, l := range lines {
		lines[i] = strings.TrimSpace(reHSpace.ReplaceAllString(l, " "))
	}
	s = strings.Join(lines, "\n")
	s = reBlankRuns.ReplaceAllString(s, "\n\n")
	return strings.TrimSpace(s)
}

// pdfText runs pdftotext on the data.
func pdfText(ctx context.Context, data []byte) (string, error) {
	path, err := exec.LookPath("pdftotext")
	if err != nil {
		return "", errors.New("reading PDFs needs pdftotext (poppler-utils), which is not installed")
	}
	f, err := os.CreateTemp("", "ollamui-*.pdf")
	if err != nil {
		return "", err
	}
	defer os.Remove(f.Name())
	_, werr := f.Write(data)
	if cerr := f.Close(); werr == nil {
		werr = cerr
	}
	if werr != nil {
		return "", werr
	}
	var out, errOut bytes.Buffer
	cmd := exec.CommandContext(ctx, path, "-enc", "UTF-8", f.Name(), "-")
	cmd.Stdout, cmd.Stderr = &out, &errOut
	if err := cmd.Run(); err != nil {
		if ctx.Err() != nil {
			return "", ctx.Err()
		}
		msg := strings.TrimSpace(errOut.String())
		if msg == "" {
			msg = err.Error()
		}
		return "", fmt.Errorf("pdftotext failed: %s", msg)
	}
	s := strings.ToValidUTF8(out.String(), "")
	if strings.TrimSpace(strings.ReplaceAll(s, "\f", "")) == "" {
		return "", errors.New("no text found in the PDF (it may be scanned images)")
	}
	return s, nil
}

// zipXML opens a zip file, reads one part and hands it to parse.
func zipXML(data []byte, part string, parse func(io.Reader) (string, error)) (string, error) {
	zr, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return "", ErrUnsupported
	}
	for _, f := range zr.File {
		if f.Name != part {
			continue
		}
		rc, err := f.Open()
		if err != nil {
			return "", err
		}
		defer rc.Close()
		s, err := parse(io.LimitReader(rc, maxXMLSize))
		if err != nil {
			return "", err
		}
		return strings.TrimSpace(s), nil
	}
	return "", ErrUnsupported
}

func newDecoder(r io.Reader) *xml.Decoder {
	d := xml.NewDecoder(r)
	d.Strict = false
	return d
}

// docxText reads word/document.xml: w:p paragraphs, w:t text, w:tab, w:br.
func docxText(r io.Reader) (string, error) {
	var b strings.Builder
	d := newDecoder(r)
	inT := false
	for {
		tok, err := d.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", err
		}
		switch t := tok.(type) {
		case xml.StartElement:
			switch t.Name.Local {
			case "t":
				inT = true
			case "tab":
				b.WriteByte('\t')
			case "br", "cr":
				b.WriteByte('\n')
			}
		case xml.EndElement:
			switch t.Name.Local {
			case "t":
				inT = false
			case "p":
				b.WriteString("\n\n")
			}
		case xml.CharData:
			if inT {
				b.Write(t)
			}
		}
	}
	return b.String(), nil
}

// odtText reads content.xml: text:p and text:h paragraphs plus spaces and tabs.
func odtText(r io.Reader) (string, error) {
	var b strings.Builder
	d := newDecoder(r)
	depth := 0 // nesting inside a paragraph or heading
	for {
		tok, err := d.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", err
		}
		switch t := tok.(type) {
		case xml.StartElement:
			switch t.Name.Local {
			case "p", "h":
				depth++
			case "tab":
				b.WriteByte('\t')
			case "line-break":
				b.WriteByte('\n')
			case "s":
				n := 1
				for _, a := range t.Attr {
					if a.Name.Local == "c" {
						if v := atoiSmall(a.Value); v > 0 {
							n = v
						}
					}
				}
				b.WriteString(strings.Repeat(" ", n))
			}
		case xml.EndElement:
			if t.Name.Local == "p" || t.Name.Local == "h" {
				if depth > 0 {
					depth--
				}
				if depth == 0 {
					b.WriteString("\n\n")
				}
			}
		case xml.CharData:
			if depth > 0 {
				b.Write(t)
			}
		}
	}
	return b.String(), nil
}

// atoiSmall parses a small decimal number, returning 0 on anything odd.
func atoiSmall(s string) int {
	n := 0
	for _, c := range s {
		if c < '0' || c > '9' || n > 1000 {
			return 0
		}
		n = n*10 + int(c-'0')
	}
	return n
}

// Accept lists the supported extensions for an <input type="file" accept>.
func Accept() string {
	var exts []string
	for e := range textExts {
		exts = append(exts, e)
	}
	for e := range otherExts {
		exts = append(exts, e)
	}
	sort.Strings(exts)
	return strings.Join(exts, ",")
}

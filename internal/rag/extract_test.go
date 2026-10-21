package rag

import (
	"archive/zip"
	"bytes"
	"context"
	"errors"
	"fmt"
	"os/exec"
	"strings"
	"testing"
)

func makeZip(t *testing.T, name, body string) []byte {
	t.Helper()
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	w, err := zw.Create(name)
	if err != nil {
		t.Fatal(err)
	}
	w.Write([]byte(body))
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// miniPDF builds a one-page PDF with a correct xref table.
func miniPDF(text string) []byte {
	stream := fmt.Sprintf("BT /F1 24 Tf 72 700 Td (%s) Tj ET", text)
	objs := []string{
		"<< /Type /Catalog /Pages 2 0 R >>",
		"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
		"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
		fmt.Sprintf("<< /Length %d >>\nstream\n%s\nendstream", len(stream), stream),
		"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
	}
	var b bytes.Buffer
	b.WriteString("%PDF-1.4\n")
	var offs []int
	for i, o := range objs {
		offs = append(offs, b.Len())
		fmt.Fprintf(&b, "%d 0 obj\n%s\nendobj\n", i+1, o)
	}
	xref := b.Len()
	fmt.Fprintf(&b, "xref\n0 %d\n0000000000 65535 f \n", len(objs)+1)
	for _, o := range offs {
		fmt.Fprintf(&b, "%010d 00000 n \n", o)
	}
	fmt.Fprintf(&b, "trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n", len(objs)+1, xref)
	return b.Bytes()
}

func TestSupported(t *testing.T) {
	for _, n := range []string{"a.txt", "A.MD", "x.go", "x.py", "x.tsx", "d.pdf", "d.docx", "d.odt", "p.html", "p.htm", "z.yaml", "q.sql", "s.svelte", "s.pl"} {
		if !Supported(n) {
			t.Errorf("Supported(%q) = false", n)
		}
	}
	for _, n := range []string{"a.png", "a.exe", "noext", "a.zip", "a.doc"} {
		if Supported(n) {
			t.Errorf("Supported(%q) = true", n)
		}
	}
}

func TestExtractText(t *testing.T) {
	tests := []struct {
		name, file string
		in         []byte
		want       string
		err        error
	}{
		{"plain", "a.txt", []byte("hello\nworld"), "hello\nworld", nil},
		{"markdown", "a.md", []byte("# T\n\nbody"), "# T\n\nbody", nil},
		{"markdown long", "a.markdown", []byte("x"), "x", nil},
		{"rst", "a.rst", []byte("x"), "x", nil},
		{"org", "a.org", []byte("x"), "x", nil},
		{"csv", "a.csv", []byte("a,b\n1,2"), "a,b\n1,2", nil},
		{"tsv", "a.tsv", []byte("a\tb"), "a\tb", nil},
		{"json", "a.json", []byte(`{"a":1}`), `{"a":1}`, nil},
		{"jsonl", "a.jsonl", []byte("{}\n{}"), "{}\n{}", nil},
		{"yaml", "a.yaml", []byte("a: 1"), "a: 1", nil},
		{"yml", "a.yml", []byte("a: 1"), "a: 1", nil},
		{"toml", "a.toml", []byte("a = 1"), "a = 1", nil},
		{"xml", "a.xml", []byte("<a/>"), "<a/>", nil},
		{"ini", "a.ini", []byte("[a]"), "[a]", nil},
		{"log", "a.log", []byte("line"), "line", nil},
		{"tex", "a.tex", []byte("\\a"), "\\a", nil},
		{"go", "a.go", []byte("package a"), "package a", nil},
		{"python", "a.py", []byte("print(1)"), "print(1)", nil},
		{"js", "a.js", []byte("let a"), "let a", nil},
		{"upper ext", "A.TXT", []byte("x"), "x", nil},
		{"bom", "a.txt", []byte("\xef\xbb\xbfhi"), "hi", nil},
		{"crlf", "a.txt", []byte("a\r\nb\r\nc"), "a\nb\nc", nil},
		{"utf8", "a.txt", []byte("héllo ✓"), "héllo ✓", nil},
		{"binary invalid utf8", "a.txt", []byte{0xff, 0xfe, 0x00, 0x01}, "", ErrUnsupported},
		{"binary nul", "a.txt", []byte("ab\x00cd"), "", ErrUnsupported},
		{"unknown ext", "a.png", []byte("x"), "", ErrUnsupported},
		{"bad docx", "a.docx", []byte("not a zip"), "", ErrUnsupported},
		{"empty", "a.txt", nil, "", nil},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := Extract(context.Background(), tc.file, tc.in)
			if !errors.Is(err, tc.err) {
				t.Fatalf("err = %v, want %v", err, tc.err)
			}
			if got != tc.want {
				t.Errorf("got %q, want %q", got, tc.want)
			}
		})
	}
}

func TestExtractTooLarge(t *testing.T) {
	_, err := Extract(context.Background(), "a.txt", make([]byte, MaxFileSize+1))
	if !errors.Is(err, ErrTooLarge) {
		t.Fatalf("err = %v", err)
	}
}

func TestExtractHTML(t *testing.T) {
	in := `<!doctype html><html><head><title>Doc</title><style>p{color:red}</style>
<script>var x = "<p>nope</p>";</script></head><body>
<!-- hidden comment -->
<h1>Heading</h1><p>First &amp; <b>bold</b> &lt;para&gt;.</p><p>Second&nbsp;para</p>
<ul><li>one</li><li>two</li></ul>a < b
</body></html>`
	got, err := Extract(context.Background(), "p.html", []byte(in))
	if err != nil {
		t.Fatal(err)
	}
	for _, bad := range []string{"color:red", "nope", "hidden comment", "<b>", "&amp;"} {
		if strings.Contains(got, bad) {
			t.Errorf("output still contains %q:\n%s", bad, got)
		}
	}
	for _, want := range []string{"Heading\n\nFirst & bold <para>.\n\nSecond para", "one\ntwo", "a < b"} {
		if !strings.Contains(got, want) {
			t.Errorf("output missing %q:\n%s", want, got)
		}
	}
	if _, err := Extract(context.Background(), "p.htm", []byte("<p>x</p>")); err != nil {
		t.Error(err)
	}
}

func TestExtractDocx(t *testing.T) {
	doc := `<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
<w:p><w:r><w:t>Hello</w:t></w:r><w:r><w:t xml:space="preserve"> world</w:t></w:r></w:p>
<w:p><w:r><w:t>A</w:t><w:tab/><w:t>B</w:t><w:br/><w:t>C</w:t></w:r></w:p>
</w:body></w:document>`
	got, err := Extract(context.Background(), "a.docx", makeZip(t, "word/document.xml", doc))
	if err != nil {
		t.Fatal(err)
	}
	if want := "Hello world\n\nA\tB\nC"; got != want {
		t.Errorf("got %q, want %q", got, want)
	}
	if _, err := Extract(context.Background(), "a.docx", makeZip(t, "other.xml", "<a/>")); !errors.Is(err, ErrUnsupported) {
		t.Errorf("missing part: err = %v", err)
	}
}

func TestExtractODT(t *testing.T) {
	doc := `<?xml version="1.0"?><office:document-content xmlns:office="o" xmlns:text="t"><office:automatic-styles>
</office:automatic-styles><office:body><office:text>
<text:h text:outline-level="1">Title</text:h>
<text:p>One <text:span>two</text:span><text:s text:c="2"/>three<text:tab/>four<text:line-break/>five</text:p>
<text:p>Next</text:p>
</office:text></office:body></office:document-content>`
	got, err := Extract(context.Background(), "a.odt", makeZip(t, "content.xml", doc))
	if err != nil {
		t.Fatal(err)
	}
	if want := "Title\n\nOne two  three\tfour\nfive\n\nNext"; got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestExtractPDF(t *testing.T) {
	if _, err := exec.LookPath("pdftotext"); err != nil {
		t.Skip("pdftotext not installed")
	}
	got, err := Extract(context.Background(), "a.pdf", miniPDF("Hello PDF"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(got, "Hello PDF") {
		t.Errorf("got %q", got)
	}
	if !strings.Contains(got, "\f") {
		t.Errorf("expected form feed page separator in %q", got)
	}
	if _, err := Extract(context.Background(), "a.pdf", []byte("nope")); err == nil {
		t.Error("garbage PDF should fail")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := Extract(ctx, "a.pdf", miniPDF("Hello")); err == nil {
		t.Error("cancelled context should fail")
	}
}

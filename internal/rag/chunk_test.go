package rag

import (
	"strings"
	"testing"
	"time"
)

func para(n int, word string) string {
	return strings.TrimSpace(strings.Repeat(word+" ", n))
}

func TestSplitEmpty(t *testing.T) {
	for _, in := range []string{"", "   \n\n\t", "\f\f"} {
		if got := Split(in, 100, 10); got != nil {
			t.Errorf("Split(%q) = %v, want nil", in, got)
		}
	}
}

func TestSplitSmall(t *testing.T) {
	got := Split("  hello world \n", 100, 10)
	if len(got) != 1 || got[0].Text != "hello world" || got[0].Page != 0 || got[0].Heading != "" {
		t.Errorf("got %+v", got)
	}
}

func TestSplitSizesAndOverlap(t *testing.T) {
	var sb strings.Builder
	for i := 0; i < 200; i++ {
		sb.WriteString("Sentence number ")
		sb.WriteString(strings.Repeat("x", i%7))
		sb.WriteString(" ends here. ")
		if i%10 == 9 {
			sb.WriteString("\n\n")
		}
	}
	text := sb.String()
	const size, overlap = 300, 60
	chunks := Split(text, size, overlap)
	if len(chunks) < 5 {
		t.Fatalf("only %d chunks", len(chunks))
	}
	for i, c := range chunks {
		if len(c.Text) > size || len(c.Text) == 0 || c.Text != strings.TrimSpace(c.Text) {
			t.Errorf("chunk %d bad: len %d %q", i, len(c.Text), c.Text)
		}
		if !strings.Contains(text, c.Text) {
			t.Errorf("chunk %d is not a substring of the input", i)
		}
		if i > 0 {
			// The tail of the previous chunk shows up at the head of this one.
			prev := chunks[i-1].Text
			w := strings.Fields(c.Text)[0]
			if !strings.Contains(prev[len(prev)/2:], w) {
				t.Errorf("chunk %d does not overlap its predecessor (first word %q)", i, w)
			}
		}
	}
	// Nothing lost: last chunk reaches the end.
	if !strings.HasSuffix(strings.TrimSpace(text), chunks[len(chunks)-1].Text) {
		t.Error("last chunk does not end the text")
	}
}

func TestSplitNoOverlap(t *testing.T) {
	text := para(500, "word")
	chunks := Split(text, 100, 0)
	var total int
	for _, c := range chunks {
		total += len(strings.Fields(c.Text))
		if len(c.Text) > 100 {
			t.Errorf("chunk too long: %d", len(c.Text))
		}
	}
	if total != 500 {
		t.Errorf("words = %d, want 500", total)
	}
}

func TestSplitNeverCutsWords(t *testing.T) {
	text := para(300, "abcdefghij")
	for _, c := range Split(text, 95, 20) {
		for _, w := range strings.Fields(c.Text) {
			if w != "abcdefghij" {
				t.Fatalf("word was cut: %q", w)
			}
		}
	}
}

func TestSplitLongWord(t *testing.T) {
	text := "start " + strings.Repeat("é", 400) + " end"
	chunks := Split(text, 100, 10)
	if len(chunks) < 4 {
		t.Fatalf("got %d chunks", len(chunks))
	}
	for _, c := range chunks {
		if len(c.Text) > 100 {
			t.Errorf("len %d > 100", len(c.Text))
		}
		if !strings.Contains(text, c.Text) {
			t.Error("chunk broke a multi-byte rune")
		}
	}
}

func TestSplitPrefersParagraphs(t *testing.T) {
	a := para(30, "alpha") // 179 chars
	b := para(30, "beta")
	text := a + "\n\n" + b
	chunks := Split(text, 250, 0)
	if len(chunks) != 2 || chunks[0].Text != a || chunks[1].Text != b {
		t.Errorf("got %+v", chunks)
	}
}

func TestSplitHeadings(t *testing.T) {
	text := "# Top\n\n" + para(40, "one") + "\n\n## Sub\n\n" + para(40, "two") +
		"\n\n```\n# not a heading\n```\n\n#nospace\n\n" + para(40, "three")
	chunks := Split(text, 200, 30)
	seen := map[string]bool{}
	for _, c := range chunks {
		seen[c.Heading] = true
		switch {
		case strings.HasPrefix(c.Text, "one"):
			if c.Heading != "Top" {
				t.Errorf("one chunk heading %q", c.Heading)
			}
		case strings.HasPrefix(c.Text, "two") && c.Heading != "Sub":
			t.Errorf("two chunk heading %q", c.Heading)
		}
		if c.Heading == "not a heading" || c.Heading == "nospace" {
			t.Errorf("bad heading %q", c.Heading)
		}
	}
	if !seen["Top"] || !seen["Sub"] {
		t.Errorf("headings seen: %v", seen)
	}
	// A heading starts a new chunk when it is in range.
	small := Split("# A\n\n"+para(20, "aa")+"\n\n# B\n\n"+para(20, "bb"), 120, 0)
	for _, c := range small {
		if strings.Contains(c.Text, "aa") && strings.Contains(c.Text, "# B") {
			t.Errorf("heading B not used as a break: %q", c.Text)
		}
	}
}

func TestSplitHeadingHashes(t *testing.T) {
	cs := Split("###### Deep ##\n\ntext\n\n####### seven\n\nmore", 1000, 0)
	if len(cs) != 1 {
		t.Fatalf("got %d chunks", len(cs))
	}
	if cs[0].Heading != "Deep" {
		t.Errorf("heading = %q", cs[0].Heading)
	}
}

func TestSplitCodeFenceWhole(t *testing.T) {
	code := "```go\n" + strings.Repeat("x := 1\n\nfoo()\n", 8) + "```"
	text := para(25, "intro") + "\n\n" + code + "\n\n" + para(25, "outro")
	chunks := Split(text, 220, 30)
	found := false
	for _, c := range chunks {
		if strings.Contains(c.Text, "```go") {
			found = true
			if !strings.Contains(c.Text, code) {
				t.Errorf("fence was split: %q", c.Text)
			}
		}
	}
	if !found {
		t.Error("no chunk has the code block")
	}
}

func TestSplitHugeFenceStillSplits(t *testing.T) {
	code := "```\n" + strings.Repeat("line of code here\n", 100) + "```"
	chunks := Split(code, 200, 20)
	if len(chunks) < 8 {
		t.Fatalf("got %d chunks", len(chunks))
	}
	for _, c := range chunks {
		if len(c.Text) > 200 {
			t.Errorf("len %d", len(c.Text))
		}
	}
}

func TestSplitPages(t *testing.T) {
	p1 := para(30, "first")
	p2 := para(20, "second")
	p3 := para(20, "third")
	text := p1 + "\n\f" + p2 + "\n\f" + p3 + "\n\f"
	chunks := Split(text, 200, 0)
	pages := map[string]int{}
	for _, c := range chunks {
		if strings.ContainsRune(c.Text, '\f') {
			t.Errorf("form feed left in %q", c.Text)
		}
		pages[strings.Fields(c.Text)[0]] = c.Page
	}
	if pages["first"] != 1 || pages["second"] != 2 || pages["third"] != 3 {
		t.Errorf("pages = %v", pages)
	}
	if got := Split("no pages here", 100, 0); got[0].Page != 0 {
		t.Errorf("unpaged page = %d", got[0].Page)
	}
}

func TestSplitDefaultsAndBadArgs(t *testing.T) {
	text := para(2000, "word")
	if len(Split(text, 0, -5)) < 2 {
		t.Error("size 0 should fall back to a default")
	}
	// Overlap larger than size must still terminate.
	if len(Split(text, 50, 500)) == 0 {
		t.Error("no chunks")
	}
}

func TestSplitLargeIsFast(t *testing.T) {
	line := "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.\n"
	var sb strings.Builder
	for i := 0; sb.Len() < MaxFileSize; i++ {
		if i%50 == 0 {
			sb.WriteString("\n## Section\n\n")
		}
		sb.WriteString(line)
	}
	start := time.Now()
	chunks := Split(sb.String(), 1500, 200)
	if d := time.Since(start); d > 5*time.Second {
		t.Errorf("took %v", d)
	}
	if len(chunks) < 10000 {
		t.Errorf("only %d chunks", len(chunks))
	}
}

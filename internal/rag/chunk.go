package rag

import (
	"sort"
	"strings"
	"unicode/utf8"
)

// Chunk is a piece of a document sized for embedding.
type Chunk struct {
	Text    string // the chunk, trimmed
	Page    int    // 1-based PDF page the chunk starts on, 0 when not paged
	Heading string // nearest markdown heading above it, if any ("" otherwise)
}

type heading struct {
	off  int
	text string
}

type span struct{ start, end int }

// Split cuts text into chunks of about size bytes with overlap bytes carried
// over between neighbours. It prefers to cut at headings, then blank lines,
// line ends, sentence ends and spaces, and keeps fenced code blocks whole when
// they fit.
func Split(text string, size, overlap int) []Chunk {
	if size <= 0 {
		size = 1500
	}
	if overlap < 0 {
		overlap = 0
	}
	if overlap > size/2 {
		overlap = size / 2
	}

	// Strip form feeds, remembering where each one was.
	var breaks []int
	if strings.Contains(text, "\f") {
		var b strings.Builder
		b.Grow(len(text))
		for {
			i := strings.IndexByte(text, '\f')
			if i < 0 {
				b.WriteString(text)
				break
			}
			b.WriteString(text[:i])
			text = text[i+1:]
			breaks = append(breaks, b.Len())
		}
		text = b.String()
	}
	text = strings.TrimRight(text, " \t\r\n")
	if strings.TrimSpace(text) == "" {
		return nil
	}
	paged := breaks != nil
	heads, fences := scanStructure(text)

	var out []Chunk
	start := 0
	for {
		for start < len(text) && isSpace(text[start]) {
			start++
		}
		if start >= len(text) {
			break
		}
		cut := pickCut(text, start, size, heads, fences)
		if t := strings.TrimSpace(text[start:cut]); t != "" {
			c := Chunk{Text: t}
			if paged {
				c.Page = 1 + sort.Search(len(breaks), func(i int) bool { return breaks[i] > start })
			}
			if i := sort.Search(len(heads), func(i int) bool { return heads[i].off > start }); i > 0 {
				c.Heading = heads[i-1].text
			}
			out = append(out, c)
		}
		if cut >= len(text) {
			break
		}
		next := cut - overlap
		if next <= start {
			next = cut
		}
		// Snap forward so we never begin in the middle of a word.
		if next < cut && !isSpace(text[next]) && !isSpace(text[next-1]) {
			for next < cut && !isSpace(text[next]) {
				next++
			}
		}
		start = next
	}
	return out
}

func isSpace(b byte) bool { return b == ' ' || b == '\n' || b == '\t' || b == '\r' }

// scanStructure finds markdown headings and fenced code blocks in one pass.
func scanStructure(text string) ([]heading, []span) {
	var heads []heading
	var fences []span
	inFence := false
	fenceStart := 0
	for off := 0; off < len(text); {
		end := strings.IndexByte(text[off:], '\n')
		next := len(text)
		line := text[off:]
		if end >= 0 {
			line = text[off : off+end]
			next = off + end + 1
		}
		trimmed := strings.TrimLeft(line, " ")
		switch {
		case len(line)-len(trimmed) <= 3 && strings.HasPrefix(trimmed, "```"):
			if inFence {
				fences = append(fences, span{fenceStart, next})
			} else {
				fenceStart = off
			}
			inFence = !inFence
		case !inFence && len(line)-len(trimmed) <= 3:
			n := 0
			for n < len(trimmed) && trimmed[n] == '#' {
				n++
			}
			if n >= 1 && n <= 6 && len(trimmed) > n && trimmed[n] == ' ' {
				if t := strings.TrimSpace(strings.TrimRight(trimmed[n:], "# ")); t != "" {
					heads = append(heads, heading{off, t})
				}
			}
		}
		off = next
	}
	if inFence {
		fences = append(fences, span{fenceStart, len(text)})
	}
	return heads, fences
}

// pickCut chooses where the chunk beginning at start ends. The result is
// always greater than start.
func pickCut(text string, start, size int, heads []heading, fences []span) int {
	end := start + size
	if end >= len(text) {
		return len(text)
	}
	for end > start && !utf8.RuneStart(text[end]) {
		end--
	}
	lo := start + size/2
	if lo <= start {
		lo = start + 1
	}
	cut := bestBreak(text, start, lo, end, heads)

	// Don't cut inside a code fence that began after this chunk did.
	i := sort.Search(len(fences), func(i int) bool { return fences[i].end > cut })
	if i < len(fences) && fences[i].start < cut && fences[i].start > start {
		cut = fences[i].start
	}
	return cut
}

// bestBreak looks in text[lo:end] for the best place to cut, falling back to
// a space anywhere after start and finally to a hard cut at end.
func bestBreak(text string, start, lo, end int, heads []heading) int {
	// Headings: cut just before the last one in the window.
	i := sort.Search(len(heads), func(i int) bool { return heads[i].off > end })
	if i > 0 && heads[i-1].off >= lo {
		return heads[i-1].off
	}
	win := text[lo:end]
	if text[end-1] == '\n' && end-2 >= lo && text[end-2] == '\n' {
		return end
	}
	if k := strings.LastIndex(win, "\n\n"); k >= 0 {
		return lo + k + 2
	}
	if text[end] == '\n' {
		return end
	}
	if k := strings.LastIndexByte(win, '\n'); k >= 0 {
		return lo + k + 1
	}
	if k := strings.LastIndex(win, ". "); k >= 0 {
		return lo + k + 2
	}
	if isSpace(text[end]) {
		return end
	}
	if k := strings.LastIndexByte(text[start+1:end], ' '); k >= 0 {
		return start + 1 + k + 1
	}
	return end
}

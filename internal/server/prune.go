package server

import (
	"strings"

	"assay/internal/ollama"
)

const (
	// pruneAt is the share of the context a prompt must fill before old
	// duplicated code is pruned. Below it prompts stay identical from turn to
	// turn, which keeps the engine's prefix cache valid.
	pruneAt = 0.6
	// pruneMinChars is the size under which a block isn't worth replacing.
	pruneMinChars = 120
	pruneNote     = "[earlier version of this code, replaced later in the chat]"
)

// fenceBlock is a fenced code block in a message: lines[start:end] with the
// fence lines included.
type fenceBlock struct {
	start, end int
	key        string
	size       int
}

// fenceBlocks finds the fenced code blocks of lines. A block's key is its
// attached file name when an "Attached file: NAME" line precedes it, else its
// language and first non-empty line (usually a filename comment or a
// signature).
func fenceBlocks(lines []string) []fenceBlock {
	var out []fenceBlock
	for i := 0; i < len(lines); i++ {
		t := strings.TrimSpace(lines[i])
		n := len(t) - len(strings.TrimLeft(t, "`"))
		if n < 3 {
			continue
		}
		lang := strings.TrimSpace(t[n:])
		end := -1
		for j := i + 1; j < len(lines); j++ {
			c := strings.TrimSpace(lines[j])
			if len(c) >= n && strings.Trim(c, "`") == "" {
				end = j
				break
			}
		}
		if end < 0 {
			break
		}
		key, size := "", 0
		if i > 0 && strings.HasPrefix(lines[i-1], "Attached file: ") {
			key = "file:" + strings.TrimSpace(strings.TrimPrefix(lines[i-1], "Attached file: "))
		}
		first := ""
		for _, l := range lines[i+1 : end] {
			size += len(l) + 1
			if first == "" {
				first = strings.TrimSpace(l)
			}
		}
		if key == "" && first != "" {
			key = "code:" + strings.ToLower(lang) + "\n" + first
		}
		out = append(out, fenceBlock{start: i, end: end, key: key, size: size})
		i = end
	}
	return out
}

// pruneMessages returns msgs with code that a later message supersedes cut
// down to a short note: a fenced block (or attached file) whose key matches a
// block in a later message. The last keep messages and system messages are
// left alone, and msgs itself is not changed.
func pruneMessages(msgs []ollama.Message, keep int) []ollama.Message {
	// later holds the keys seen in messages after the one being looked at.
	later := map[string]bool{}
	out := make([]ollama.Message, len(msgs))
	copy(out, msgs)
	for i := len(msgs) - 1; i >= 0; i-- {
		m := msgs[i]
		if m.Role == "system" || !strings.Contains(m.Content, "```") {
			continue
		}
		lines := strings.Split(m.Content, "\n")
		blocks := fenceBlocks(lines)
		if i < len(msgs)-keep {
			var res []string
			pos := 0
			for _, b := range blocks {
				if b.key == "" || b.size < pruneMinChars || !later[b.key] {
					continue
				}
				res = append(res, lines[pos:b.start]...)
				res = append(res, pruneNote)
				pos = b.end + 1
			}
			if pos > 0 {
				res = append(res, lines[pos:]...)
				out[i].Content = strings.Join(res, "\n")
			}
		}
		for _, b := range blocks {
			if b.key != "" {
				later[b.key] = true
			}
		}
	}
	return out
}

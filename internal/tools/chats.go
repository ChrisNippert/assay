package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"unicode"
)

const chatResults = 5

func searchChatsTool(st ChatStore, except string) Tool {
	return Tool{
		Name:        "search_chats",
		Description: "Search the user's saved chats (titles and messages) for words. Use it when the user refers to something discussed before (\"what did we decide about...\", \"that recipe from last week\"). Returns the best matching chats with an excerpt. Use a few distinctive keywords, not a sentence.",
		Parameters:  objSchema("query", "Keywords to look for, e.g. \"sourdough starter\""),
		Run: func(ctx context.Context, args json.RawMessage) Result {
			q, err := strArg(args, "query")
			if err != nil {
				return fail("Searched your chats", err.Error())
			}
			label := "Searched your chats for “" + short(q, 60) + "”"
			hits, err := SearchChats(st, q, except)
			if err != nil {
				return fail(label, err.Error())
			}
			if len(hits) == 0 {
				return Result{Text: "No saved chat matches.", Label: label + " · no results"}
			}
			var b strings.Builder
			for i, h := range hits {
				fmt.Fprintf(&b, "%d. %s (chat id %s, %s)\n   %s\n", i+1, h.Title, h.ID, h.Updated, h.Snippet)
			}
			return Result{Text: strings.TrimSpace(b.String()), Label: label + " · " + plural(len(hits), "chat", "chats")}
		},
	}
}

// ChatHit is one chat that matched a search.
type ChatHit struct {
	ID, Title, Updated, Snippet string
	score                       int
}

// words splits a query into lowercase search words.
func words(q string) []string {
	return strings.FieldsFunc(strings.ToLower(q), func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsDigit(r) })
}

// SearchChats scores each chat by how many query words its title and
// messages contain (title hits count triple) and returns the best few. A
// chat matching more distinct words ranks above one repeating a word.
func SearchChats(st ChatStore, query, except string) ([]ChatHit, error) {
	ws := words(query)
	if len(ws) == 0 {
		return nil, fmt.Errorf("the query has no words to look for")
	}
	list, err := st.List()
	if err != nil {
		return nil, err
	}
	var hits []ChatHit
	for _, sum := range list {
		if sum.ID == except {
			continue
		}
		c, err := st.Get(sum.ID)
		if err != nil {
			continue
		}
		title := strings.ToLower(c.Title)
		score, distinct := 0, 0
		seen := make([]bool, len(ws))
		var snippets []string
		for i, w := range ws {
			if n := strings.Count(title, w); n > 0 {
				score += 3 * n
				seen[i] = true
			}
		}
		answered := -1 // the reply already quoted as the answer to a matching question
		for mi, m := range c.Messages {
			if m.Role != "user" && m.Role != "assistant" {
				continue
			}
			low := strings.ToLower(m.Content)
			first := -1
			for i, w := range ws {
				if at := strings.Index(low, w); at >= 0 {
					score += min(strings.Count(low, w), 3)
					seen[i] = true
					if first < 0 || at < first {
						first = at
					}
				}
			}
			if first >= 0 && len(snippets) < 4 && mi != answered {
				who := "You: "
				if m.Role == "assistant" {
					who = "Assistant: "
				}
				snippets = append(snippets, who+excerpt(m.Content, first, 400))
				// A question that matched: its answer is what's wanted.
				if m.Role == "user" && mi+1 < len(c.Messages) && c.Messages[mi+1].Role == "assistant" {
					snippets = append(snippets, "Assistant: "+excerpt(c.Messages[mi+1].Content, 0, 400))
					answered = mi + 1
				}
			}
		}
		for _, s := range seen {
			if s {
				distinct++
			}
		}
		if score == 0 {
			continue
		}
		snippet := "(matched in the title)"
		if len(snippets) > 0 {
			snippet = strings.Join(snippets, "\n   ")
		}
		hits = append(hits, ChatHit{ID: c.ID, Title: c.Title, Updated: c.Updated.Format("2006-01-02"), Snippet: snippet, score: distinct*1000 + score})
	}
	sort.SliceStable(hits, func(i, j int) bool { return hits[i].score > hits[j].score })
	if len(hits) > chatResults {
		hits = hits[:chatResults]
	}
	return hits, nil
}

// excerpt returns about n runes of text around byte offset at.
func excerpt(text string, at, n int) string {
	r := []rune(text)
	// at is a byte offset into the lowercased text; map it to a rune index
	// of the original (lowercasing can change byte lengths, so clamp).
	ri := len([]rune(text[:min(at, len(text))]))
	start := max(0, ri-n/3)
	end := min(len(r), start+n)
	s := strings.Join(strings.Fields(string(r[start:end])), " ")
	if start > 0 {
		s = "…" + s
	}
	if end < len(r) {
		s += "…"
	}
	return s
}

func searchProjectTool(search func(ctx context.Context, query string) ([]Passage, error)) Tool {
	return Tool{
		Name:        "search_project",
		Description: "Search the files of this chat's project for passages relevant to a question. Use it when the answer may be in the user's project documents. Returns numbered excerpts with their file names; cite the file when you use one.",
		Parameters:  objSchema("query", "What to look for, as a short question or keywords"),
		Run: func(ctx context.Context, args json.RawMessage) Result {
			q, err := strArg(args, "query")
			if err != nil {
				return fail("Searched the project", err.Error())
			}
			label := "Searched the project’s files for “" + short(q, 60) + "”"
			ps, err := search(ctx, q)
			if err != nil {
				return fail(label, err.Error())
			}
			if len(ps) == 0 {
				return Result{Text: "No passages match.", Label: label + " · no results"}
			}
			var b strings.Builder
			for i, p := range ps {
				fmt.Fprintf(&b, "[%d] %s", i+1, p.File)
				if p.Page > 0 {
					fmt.Fprintf(&b, ", page %d", p.Page)
				}
				if p.Heading != "" {
					fmt.Fprintf(&b, " (%s)", p.Heading)
				}
				b.WriteString("\n" + p.Text + "\n\n")
			}
			return Result{Text: Clip(strings.TrimSpace(b.String()), 8000), Label: label + " · " + plural(len(ps), "passage", "passages")}
		},
	}
}

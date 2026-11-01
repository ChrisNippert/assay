package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"slices"
	"strings"

	"assay/internal/chats"
	"assay/internal/tools"
)

// editCodeTool lets the model change a code block from earlier in the chat
// by sending the exact text to find and what replaces it, instead of
// reprinting the whole block. Saved messages are never changed: the updated
// block is the tool's result, which the reply keeps as a new call.
//
// The tool is built for one reply, and the edits it makes during that reply
// aren't in c.Messages yet, so it keeps them itself: each edit applies on
// top of the ones before it in the same reply.
func editCodeTool(c *chats.Chat, file fileText) tools.Tool {
	var thisReply []chats.ToolCall
	return tools.Tool{
		Name:        "edit_code",
		Description: editCodeHelp,
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"find":    map[string]any{"type": "string", "description": "A short piece of the existing code, copied exactly (same spaces and indentation). Enough to be unique in its block, usually one line."},
				"replace": map[string]any{"type": "string", "description": "What that piece becomes. Empty deletes it."},
			},
			"required": []string{"find", "replace"},
		},
		Run: func(ctx context.Context, args json.RawMessage) tools.Result {
			var a struct{ Find, Replace *string }
			if err := json.Unmarshal(args, &a); err != nil || a.Find == nil || a.Replace == nil {
				return editFail("find and replace are both required")
			}
			msgs := append(slices.Clone(c.Messages), chats.Message{Role: "assistant", Calls: thisReply})
			e, err := applyEdit(msgs, file, *a.Find, *a.Replace)
			if err != nil {
				return editFail(err.Error())
			}
			label, what := "Edited a code block", "Edited"
			if e.name != "" {
				label, what = "Edited "+e.name, "Edited "+e.name+"."
			}
			text := what + " The updated block:\n" + e.block
			if !e.whole {
				text = fmt.Sprintf("%s Lines %d–%d of %d now read:\n%s\nThe user sees the whole updated block.", what, e.from, e.to, e.of, e.excerpt)
			}
			thisReply = append(thisReply, chats.ToolCall{Name: "edit_code", Result: text, Block: e.block, Label: label})
			return tools.Result{Text: text, Full: e.block, Label: label}
		},
	}
}

// editCodeHelp is the tool's description. Small models misread a terse
// one ("the latest block that contains it" read as "only the last
// block"), so it says plainly which block is edited and shows an example.
const editCodeHelp = `Change code that was already shown in this chat (by you or the user) without writing the whole block again.

How it works:
- find: copy a short piece of the existing code exactly, usually one line, with the same indentation.
- replace: what that piece becomes (empty to delete it).
- Any earlier code block can be edited, not only the last one: the tool looks through the code blocks from newest to oldest and edits the first one that contains find. After an edit, the edited version counts as the newest block, so further edits apply to it.
- find must appear only once in that block; if it appears more often, copy more of the line or the line around it.
- Send only the lines that change, never the whole block. Make one call per change (several calls for several changes).
- The user sees the updated block in full, so don't print it again before or after; just say in a sentence what you changed.
- Attached files count as code blocks too. For a long block the result shows only the lines around your change; the rest is unchanged.

Example: an earlier block contains
    System.out.println("Hello, World!");
To make it end with a question mark, call edit_code with
    find:    System.out.println("Hello, World!");
    replace: System.out.println("Hello, World?");`

func editFail(msg string) tools.Result {
	return tools.Result{Text: "Error: " + msg, Label: "Edit code · failed", Err: true}
}

// codeBlock is a fenced block: its fence lines and the text between them,
// and the attached file it is, if it is one.
type codeBlock struct{ open, close, body, name string }

func blocksIn(text string) []codeBlock {
	if !strings.Contains(text, "```") {
		return nil
	}
	lines := strings.Split(text, "\n")
	var out []codeBlock
	for _, b := range fenceBlocks(lines) {
		out = append(out, codeBlock{
			open:  lines[b.start],
			close: lines[b.end],
			body:  strings.Join(lines[b.start+1:b.end], "\n"),
		})
	}
	return out
}

// fileText reads an attached document's text for the edit tool.
type fileText func(a chats.Attachment) (string, bool)

// editCandidates lists the code blocks the chat shows, oldest first: a
// message's attached files (as the model saw them, before its text), the
// blocks earlier edits made, then the blocks in the text. Within a reply
// the edits come before its own text, which is where they happened.
func editCandidates(msgs []chats.Message, file fileText) []codeBlock {
	var out []codeBlock
	for _, m := range msgs {
		for _, a := range m.Attachments {
			if a.Kind == "image" || file == nil {
				continue
			}
			if text, ok := file(a); ok {
				f := fence(text)
				i, j := strings.IndexByte(f, '\n'), strings.LastIndexByte(f, '\n')
				out = append(out, codeBlock{open: f[:i] + strings.TrimPrefix(filepath.Ext(a.Name), "."), close: f[j+1:], body: f[i+1 : j], name: a.Name})
			}
		}
		for _, c := range m.Calls {
			if c.Name == "edit_code" && !c.Error {
				src := c.Block
				if src == "" {
					src = c.Result // chats from before Block was kept
				}
				name, _ := strings.CutPrefix(c.Label, "Edited ")
				if name == "a code block" || name == c.Label {
					name = ""
				}
				for _, b := range blocksIn(src) {
					b.name = name
					out = append(out, b)
				}
			}
		}
		out = append(out, blocksIn(m.Content)...)
	}
	return out
}

// edited is what one edit made.
type edited struct {
	block   string // the whole updated block, fenced
	name    string // the attached file it is, if any
	excerpt string // the changed lines and a few around them, fenced
	whole   bool   // the block is short enough that excerpt is all of it
	from    int    // excerpt's first line (1-based), and the block's length
	to, of  int
}

// excerptLines is how long a block may be for the model to get all of it
// back after an edit; longer ones send only the lines around the change, so
// a big file doesn't fill the context edit after edit.
const excerptLines = 60

// applyEdit finds the latest code block that contains find and replaces its
// one occurrence of find. An error means find is empty, is in no block, or
// occurs more than once in the block that holds it.
func applyEdit(msgs []chats.Message, file fileText, find, replace string) (edited, error) {
	if find == "" {
		return edited{}, errors.New("find is empty: copy the exact text to change")
	}
	cands := editCandidates(msgs, file)
	if t := strings.TrimSpace(find); t != find && t != "" && !strings.Contains(t, "\n") && !slices.ContainsFunc(cands, func(b codeBlock) bool { return strings.Contains(b.body, find) }) {
		// One line with the wrong indentation: match the line's text and
		// keep the block's own indentation.
		find, replace = t, strings.TrimSpace(replace)
	}
	for i := len(cands) - 1; i >= 0; i-- {
		b := cands[i]
		n := strings.Count(b.body, find)
		if n == 0 {
			continue
		}
		if n > 1 {
			return edited{}, fmt.Errorf("find occurs %d times in the newest block that contains it: copy more of the line, or the line around it, so it occurs once", n)
		}
		at := strings.Index(b.body, find)
		body := b.body[:at] + replace + b.body[at+len(find):]
		e := edited{block: b.open + "\n" + body + "\n" + b.close, name: b.name}
		lines := strings.Split(body, "\n")
		first := strings.Count(body[:at], "\n")
		last := first + strings.Count(replace, "\n")
		lo, hi := max(0, first-3), min(len(lines)-1, last+3)
		e.whole = len(lines) <= excerptLines
		if e.whole {
			lo, hi = 0, len(lines)-1
		}
		e.from, e.to, e.of = lo+1, hi+1, len(lines)
		e.excerpt = b.open + "\n" + strings.Join(lines[lo:hi+1], "\n") + "\n" + b.close
		return e, nil
	}
	return edited{}, errors.New("no code block in this chat contains find: copy the exact text from the block")
}

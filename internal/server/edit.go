package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"

	"assay/internal/chats"
	"assay/internal/tools"
)

// editCodeTool lets the model change a code block from earlier in the chat
// by sending the exact text to find and what replaces it, instead of
// reprinting the whole block. Saved messages are never changed: the updated
// block is the tool's result, which the reply keeps as a new call.
func editCodeTool(c *chats.Chat) tools.Tool {
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
			block, err := applyEdit(c.Messages, *a.Find, *a.Replace)
			if err != nil {
				return editFail(err.Error())
			}
			return tools.Result{Text: "Edited. The updated block:\n" + block, Label: "Edited a code block"}
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

Example: an earlier block contains
    System.out.println("Hello, World!");
To make it end with a question mark, call edit_code with
    find:    System.out.println("Hello, World!");
    replace: System.out.println("Hello, World?");`

func editFail(msg string) tools.Result {
	return tools.Result{Text: "Error: " + msg, Label: "Edit code · failed", Err: true}
}

// codeBlock is a fenced block: its fence lines and the text between them.
type codeBlock struct{ open, close, body string }

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

// editCandidates lists the code blocks the chat shows, oldest first. Within a
// reply the edits come before its own text, which is where they happened.
func editCandidates(msgs []chats.Message) []codeBlock {
	var out []codeBlock
	for _, m := range msgs {
		for _, c := range m.Calls {
			if c.Name == "edit_code" && !c.Error {
				out = append(out, blocksIn(c.Result)...)
			}
		}
		out = append(out, blocksIn(m.Content)...)
	}
	return out
}

// applyEdit finds the latest code block that contains find and replaces its
// one occurrence of find. It returns the whole updated block. An error means
// find is empty, is in no block, or occurs more than once in the block that
// holds it.
func applyEdit(msgs []chats.Message, find, replace string) (string, error) {
	if find == "" {
		return "", errors.New("find is empty: copy the exact text to change")
	}
	cands := editCandidates(msgs)
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
			return "", fmt.Errorf("find occurs %d times in the newest block that contains it: copy more of the line, or the line around it, so it occurs once", n)
		}
		return b.open + "\n" + strings.Replace(b.body, find, replace, 1) + "\n" + b.close, nil
	}
	return "", errors.New("no code block in this chat contains find: copy the exact text from the block")
}

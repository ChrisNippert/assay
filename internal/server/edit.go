package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
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
		Description: "Change a code block from earlier in this chat without reprinting it. find is exact text copied from the block; it must occur exactly once in the latest code block that contains it. replace is what it becomes (empty deletes it). The updated block is shown to the user in full.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"find":    map[string]any{"type": "string", "description": "Exact text from the block, including indentation"},
				"replace": map[string]any{"type": "string", "description": "The text that replaces it"},
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
	for i := len(cands) - 1; i >= 0; i-- {
		b := cands[i]
		n := strings.Count(b.body, find)
		if n == 0 {
			continue
		}
		if n > 1 {
			return "", fmt.Errorf("find occurs %d times in the latest block that contains it: quote more of the code so it occurs once", n)
		}
		return b.open + "\n" + strings.Replace(b.body, find, replace, 1) + "\n" + b.close, nil
	}
	return "", errors.New("no code block in this chat contains find: copy the exact text from the block")
}

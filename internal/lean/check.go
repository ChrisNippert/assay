package lean

import (
	"bytes"
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

var execLookPath = exec.LookPath

// CheckTimeout bounds one proof check.
var CheckTimeout = 60 * time.Second

// Result is the outcome of checking one proof.
type Result struct {
	OK     bool
	Output string // Lean's messages, or why the proof was refused
}

// forbidden are ways around the checker: unfinished proofs, new axioms,
// and steps Lean doesn't check itself.
var forbidden = regexp.MustCompile(`\b(sorry|admit|axiom|native_decide|implemented_by|extern|unsafe|ofReduceBool|debug\.skipKernelTC|Lean\.Elab|elab|macro_rules|run_cmd|run_tac)\b|@\[\s*csimp|set_option\s+(debug|maxRecDepth\s+\d{6,})`)

// allowedAxioms are the standard ones; anything else (sorryAx,
// Lean.ofReduceBool, a new axiom) fails the proof.
var allowedAxioms = map[string]bool{"propext": true, "Classical.choice": true, "Quot.sound": true}

// Check proves `theorem goal : statement` with the model's code. The file
// is the preamble (definitions the question gives), the model's code
// (which must define a theorem named goal, with any helper lemmas before
// it), then a check that goal proves exactly the statement and uses only
// the standard axioms.
func Check(ctx context.Context, bin, preamble, statement, code string) (Result, error) {
	if m := forbidden.FindString(code); m != "" {
		return Result{Output: "Refused: the proof uses `" + m + "`, which isn't allowed."}, nil
	}
	dir, err := os.MkdirTemp("", "assay-lean-")
	if err != nil {
		return Result{}, err
	}
	defer os.RemoveAll(dir)
	var src strings.Builder
	if p := strings.TrimSpace(preamble); p != "" {
		src.WriteString(p + "\n\n")
	}
	src.WriteString("namespace AssayProof\n\n")
	src.WriteString(strings.TrimSpace(code))
	src.WriteString("\n\nend AssayProof\n\n")
	src.WriteString("theorem assay_check : " + strings.TrimSpace(statement) + " := AssayProof.goal\n")
	src.WriteString("#print axioms assay_check\n")
	file := filepath.Join(dir, "Proof.lean")
	if err := os.WriteFile(file, []byte(src.String()), 0o600); err != nil {
		return Result{}, err
	}
	ctx, cancel := context.WithTimeout(ctx, CheckTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, bin, file)
	cmd.Dir = dir
	var out bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &out
	cmd.WaitDelay = 2 * time.Second
	runErr := cmd.Run()
	text := strings.ReplaceAll(out.String(), file, "Proof.lean")
	if len(text) > 4000 {
		text = text[:2000] + "\n[…]\n" + text[len(text)-2000:]
	}
	if ctx.Err() == context.DeadlineExceeded {
		return Result{Output: strings.TrimSpace(text + "\nStopped after " + CheckTimeout.String() + ".")}, nil
	}
	if ctx.Err() != nil {
		return Result{}, ctx.Err()
	}
	if runErr != nil || strings.Contains(text, "error:") {
		return Result{Output: strings.TrimSpace(text)}, nil
	}
	if bad := badAxioms(text); bad != "" {
		return Result{Output: strings.TrimSpace(text + "\nRefused: depends on " + bad + ".")}, nil
	}
	return Result{OK: true, Output: strings.TrimSpace(text)}, nil
}

var axiomsRe = regexp.MustCompile(`depends on axioms: \[([^\]]*)\]`)

// badAxioms names any axiom outside the standard three in #print axioms'
// output, or says the output is missing.
func badAxioms(out string) string {
	if strings.Contains(out, "does not depend on any axioms") {
		return ""
	}
	m := axiomsRe.FindStringSubmatch(out)
	if m == nil {
		return "unknown axioms (no #print axioms output)"
	}
	for _, a := range strings.Split(m[1], ",") {
		if a = strings.TrimSpace(a); a != "" && !allowedAxioms[a] {
			return a
		}
	}
	return ""
}

// ExtractCode returns the reply's last ```lean code block.
func ExtractCode(reply string) (string, bool) {
	m := leanBlockRe.FindAllStringSubmatch(reply, -1)
	for i := len(m) - 1; i >= 0; i-- {
		if c := strings.TrimSpace(m[i][1]); c != "" {
			return c, true
		}
	}
	return "", false
}

var leanBlockRe = regexp.MustCompile("(?s)```[ \\t]*(?:lean4?)[ \\t]*\\n(.*?)```")

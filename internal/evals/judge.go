package evals

import (
	"regexp"
	"strings"
)

// JudgeSystem tells the judge model how to grade.
const JudgeSystem = `You grade answers to hard problems, usually proofs. You are given the problem, a reference solution and a candidate's answer. The candidate passes only if their answer is complete and correct: every step follows, nothing important is assumed without justification, and it proves exactly the statement asked (a different route from the reference is fine). Small slips in wording or arithmetic that don't affect the argument are fine. Fail answers with a gap in the logic, circular reasoning, a proof of only special cases, a wrong claim, or no proof at all. If the statement is false and the reference shows that, pass only an answer that shows it is false. Be strict but fair. Explain briefly, then end with a final line that is exactly "Verdict: PASS" or "Verdict: FAIL".`

// JudgeMessages builds the judge's prompt for one answer.
func JudgeMessages(q Question, reply string) (system, user string) {
	var b strings.Builder
	b.WriteString("## Problem\n\n")
	b.WriteString(strings.TrimSpace(q.Prompt))
	b.WriteString("\n\n## Reference solution\n\n")
	b.WriteString(strings.TrimSpace(q.Answer))
	b.WriteString("\n\n## Candidate's answer\n\n")
	if r := strings.TrimSpace(StripThinking(reply)); r != "" {
		b.WriteString(r)
	} else {
		b.WriteString("(empty)")
	}
	b.WriteString("\n\nGrade the candidate's answer.")
	return JudgeSystem, b.String()
}

var verdictRe = regexp.MustCompile(`(?i)verdict\s*[:：]?\s*\**\s*(pass|fail)`)

// ParseVerdict reads the judge's last verdict; ok is false if it gave none.
func ParseVerdict(judge string) (pass, ok bool) {
	m := verdictRe.FindAllStringSubmatch(StripThinking(judge), -1)
	if m == nil {
		return false, false
	}
	return strings.EqualFold(m[len(m)-1][1], "pass"), true
}

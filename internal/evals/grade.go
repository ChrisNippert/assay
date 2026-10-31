package evals

import (
	"math"
	"regexp"
	"strconv"
	"strings"
	"unicode"
)

// Messages builds what the model is asked: the set's system message, if
// any, and the question with its choices and how to give the answer.
func (s *Set) Messages(q Question) (system, user string) {
	var b strings.Builder
	b.WriteString(strings.TrimSpace(q.Prompt))
	switch s.CheckOf(q) {
	case CheckChoice:
		b.WriteString("\n\n")
		for i, c := range q.Choices {
			b.WriteString(string(rune('A'+i)) + ". " + c + "\n")
		}
		b.WriteString("\nEnd your reply with a final line of the form \"Answer: X\", where X is the letter of your choice.")
	case CheckNumber:
		b.WriteString("\n\nEnd your reply with a final line of the form \"Answer: N\", where N is just the number.")
	case CheckExact:
		b.WriteString("\n\nEnd your reply with a final line of the form \"Answer: …\" giving only the answer.")
	}
	return s.System, b.String()
}

// Grade checks a reply. got is the answer it read from the reply (for the
// UI); ok whether it's right.
func (s *Set) Grade(q Question, reply string) (got string, ok bool) {
	reply = StripThinking(reply)
	check := s.CheckOf(q)
	switch check {
	case CheckRegex, CheckContains:
		match := false
		for _, a := range append([]string{q.Answer}, q.Accept...) {
			if check == CheckRegex {
				re, err := regexp.Compile(a)
				match = err == nil && re.MatchString(reply)
			} else {
				match = strings.Contains(strings.ToLower(reply), strings.ToLower(a))
			}
			if match {
				break
			}
		}
		return "", match != q.Not
	}
	got = extract(reply, check, len(q.Choices))
	for _, a := range append([]string{q.Answer}, q.Accept...) {
		switch check {
		case CheckNumber:
			want, _ := parseNumber(a)
			n, okN := parseNumber(got)
			tol := q.Tolerance
			if tol == 0 {
				tol = math.Max(1e-9, math.Abs(want)*1e-4)
			}
			if okN && math.Abs(n-want) <= tol {
				return got, true
			}
		case CheckChoice:
			if letterIndex(got) >= 0 && letterIndex(got) == letterIndex(a) {
				return got, true
			}
		default:
			if normalize(got) == normalize(a) {
				return got, true
			}
		}
	}
	return got, false
}

var thinkRe = regexp.MustCompile(`(?s)<think>.*?</think>`)

// StripThinking drops thinking a model wrote into its reply.
func StripThinking(s string) string {
	s = thinkRe.ReplaceAllString(s, "")
	if i := strings.Index(s, "</think>"); i >= 0 {
		s = s[i+len("</think>"):]
	}
	return strings.TrimSpace(s)
}

var (
	answerRe = regexp.MustCompile(`(?im)^[\s>*#_-]*(?:final\s+)?answer\s*(?:is)?\s*[:：]\s*(.+?)\s*$`)
	fracRe   = regexp.MustCompile(`\\d?frac\{([^{}]*)\}\{([^{}]*)\}`)
	boxedRe  = regexp.MustCompile(`\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}`)
	numRe    = regexp.MustCompile(`-?\d[\d,]*(?:\.\d+)?(?:\s*/\s*\d+)?|-?\.\d+`)
	letterRe = regexp.MustCompile(`\b([A-Z])\b`)
)

// extract reads the final answer out of a reply: the last "Answer: …"
// line, else the last \boxed{…}, else a best guess from the end.
func extract(reply, check string, choices int) string {
	got := ""
	if m := answerRe.FindAllStringSubmatch(reply, -1); m != nil {
		got = m[len(m)-1][1]
	} else if m := boxedRe.FindAllStringSubmatch(reply, -1); m != nil {
		got = m[len(m)-1][1]
	}
	if b := boxedRe.FindStringSubmatch(got); b != nil {
		got = b[1]
	}
	got = strings.Trim(strings.TrimSpace(got), "*_`$ ")
	switch check {
	case CheckNumber:
		src := got
		if src == "" {
			src = reply
		}
		if m := numRe.FindAllString(fracRe.ReplaceAllString(src, "$1/$2"), -1); m != nil {
			if got != "" {
				return m[0]
			}
			return m[len(m)-1]
		}
	case CheckChoice:
		src := got
		if src == "" {
			src = lastLine(reply)
		} else if t := strings.TrimLeft(src, "( "); len(t) >= 1 && t[0] >= 'A' && t[0] <= 'Z' && (len(t) == 1 || !unicode.IsLetter(rune(t[1]))) {
			return t[:1] // "C", "(C) …", "C. …"
		}
		// Otherwise the last lone capital that names a choice ("I pick C").
		m := letterRe.FindAllStringSubmatch(src, -1)
		for i := len(m) - 1; i >= 0; i-- {
			if choices == 0 || int(m[i][1][0]-'A') < choices {
				return m[i][1]
			}
		}
	default:
		if got == "" {
			got = lastLine(reply)
		}
	}
	return got
}

func lastLine(s string) string {
	lines := strings.Split(strings.TrimSpace(s), "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		if l := strings.TrimSpace(lines[i]); l != "" {
			return strings.Trim(l, "*_` ")
		}
	}
	return ""
}

// parseNumber reads "1,234", "$12.50", "75%", "3/4" or "-2".
func parseNumber(s string) (float64, bool) {
	s = strings.TrimSpace(s)
	s = strings.NewReplacer(",", "", "$", "", "%", "", "€", "", "£", "", " ", "").Replace(s)
	if a, b, ok := strings.Cut(s, "/"); ok {
		x, err1 := strconv.ParseFloat(a, 64)
		y, err2 := strconv.ParseFloat(b, 64)
		if err1 != nil || err2 != nil || y == 0 {
			return 0, false
		}
		return x / y, true
	}
	f, err := strconv.ParseFloat(s, 64)
	return f, err == nil
}

func letterIndex(s string) int {
	s = strings.ToUpper(strings.Trim(strings.TrimSpace(s), "().*"))
	if len(s) != 1 || s[0] < 'A' || s[0] > 'Z' {
		return -1
	}
	return int(s[0] - 'A')
}

// normalize ignores case, spacing, quotes and end punctuation.
func normalize(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	s = strings.Trim(s, ".!?\"'“”‘’`*_ ")
	return strings.Join(strings.Fields(s), " ")
}

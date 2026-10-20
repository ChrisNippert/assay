package voice

import (
	"regexp"
	"strings"
)

// SpeechText rewrites text so a text-to-speech engine reads it the way a
// person would. Engines built on espeak (Kokoro, Piper) read the dot in
// "1.67" as the end of a sentence ("one. sixty-seven"), so decimals, money,
// percentages, multipliers, ranges and a few abbreviations are spelled out.
func SpeechText(s string) string {
	for _, r := range speechRules {
		s = r.re.ReplaceAllStringFunc(s, r.fn)
	}
	return strings.Join(strings.Fields(s), " ")
}

type speechRule struct {
	re *regexp.Regexp
	fn func(string) string
}

func rule(pattern string, fn func(m []string) string) speechRule {
	re := regexp.MustCompile(pattern)
	return speechRule{re, func(s string) string { return fn(re.FindStringSubmatch(s)) }}
}

func literal(pattern, repl string) speechRule {
	return rule(pattern, func([]string) string { return repl })
}

var speechRules = []speechRule{
	// Thousands separators: 1,250,000 → 1250000 (engines read the digits fine).
	rule(`\d{1,3}(?:,\d{3})+\b`, func(m []string) string { return strings.ReplaceAll(m[0], ",", "") }),
	// Money: $1.5 billion → 1.5 billion dollars; $20 → 20 dollars.
	rule(`\$(\d+(?:\.\d+)?)(\s*(?:thousand|million|billion|trillion))?`, func(m []string) string {
		return m[1] + m[2] + " dollars"
	}),
	rule(`(\d)\s?%`, func(m []string) string { return m[1] + " percent" }),
	rule(`(\d)\s?°\s?C\b`, func(m []string) string { return m[1] + " degrees Celsius" }),
	rule(`(\d)\s?°\s?F\b`, func(m []string) string { return m[1] + " degrees Fahrenheit" }),
	rule(`(\d)\s?°`, func(m []string) string { return m[1] + " degrees" }),
	// 2x faster → 2 times faster.
	rule(`(\d)[x×](\W|$)`, func(m []string) string { return m[1] + " times" + m[2] }),
	// Ranges: 100–150, 100-150 → 100 to 150.
	rule(`(\d)\s?[–—-]\s?(\d)`, func(m []string) string { return m[1] + " to " + m[2] }),
	// Decimals: 1.67 → 1 point 6 7, read digit by digit as people do.
	rule(`(\d)\.(\d+)`, func(m []string) string {
		return m[1] + " point " + strings.Join(strings.Split(m[2], ""), " ")
	}),
	rule(`~\s?(\d)`, func(m []string) string { return "about " + m[1] }),
	literal(`\be\.g\.,?`, "for example,"),
	literal(`\bi\.e\.,?`, "that is,"),
	literal(`\betc\.`, "et cetera"),
	literal(`\bvs\.?\s`, "versus "),
	literal(`\bapprox\.`, "approximately"),
	literal(`\s&\s`, " and "),
	literal(`(?:->|→)`, " to "),
}

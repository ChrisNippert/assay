package server

import "testing"

func TestFirstSentence(t *testing.T) {
	for in, want := range map[string]string{
		"Let me think about that. Then more.":  "Let me think about that.",
		"I'm checking 3.5 versus 4 now!\nMore": "I'm checking 3.5 versus 4 now!",
		"No ending here":                       "No ending here",
		"Ok. Let me work through the numbers.": "Ok. Let me work through the numbers.",
	} {
		if got := firstSentence(in); got != want {
			t.Errorf("firstSentence(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestTail(t *testing.T) {
	if got := tail("short", 10); got != "short" {
		t.Errorf("tail kept %q", got)
	}
	if got := tail("one two three four five", 12); got != "…four five" {
		t.Errorf("tail = %q", got)
	}
}

func TestGoodLine(t *testing.T) {
	q := "How many primes are there between 100 and 150?"
	said := []string{"I'm finding which numbers between 100 and 150 are prime."}
	for line, want := range map[string]bool{
		"I'm checking each odd number for divisors.":                   true,
		"I'm finding which numbers between 100 and 150 are prime.":     false, // repeat
		"I'm finding which numbers between 100 and 150 are primes.":    false, // near-repeat
		"I've located 12 primes so far.":                               false, // made-up number
		"I'm going through them from 101":                              false, // cut off
		"I'm working through the range from 100 to 150 one at a time.": true,
		"I have found that there are exactly ten primes.":              false,
		"I am ready to tell you the primes I found.":                   false,
		"I'm counting the twelve cases now.":                           false,
	} {
		if got := goodLine(line, q, said); got != want {
			t.Errorf("goodLine(%q) = %v, want %v", line, got, want)
		}
	}
}

func TestVoiceLabel(t *testing.T) {
	if got, ok := voiceLabel("bf_emma"); !ok || got != "Emma · British, female" {
		t.Errorf("bf_emma: %q %v", got, ok)
	}
	if got, ok := voiceLabel("am_michael"); !ok || got != "Michael · American, male" {
		t.Errorf("am_michael: %q %v", got, ok)
	}
	if _, ok := voiceLabel("jf_alpha"); ok {
		t.Error("Japanese voice offered")
	}
}

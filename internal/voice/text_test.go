package voice

import "testing"

func TestSpeechText(t *testing.T) {
	for in, want := range map[string]string{
		"The economy is worth 1.67 trillion dollars.": "The economy is worth 1 point 6 7 trillion dollars.",
		"It grew 3.5% to $1.2 billion.":               "It grew 3 point 5 percent to 1 point 2 billion dollars.",
		"About 1,250,000 people, 2.0x faster.":        "About 1250000 people, 2 point 0 times faster.",
		"Primes between 100–150, e.g. 101.":           "Primes between 100 to 150, for example, 101.",
		"It's ~20°C vs. 68°F.":                        "It's about 20 degrees Celsius versus 68 degrees Fahrenheit.",
		"Version 3 is fine. Then 4.":                  "Version 3 is fine. Then 4.",
		"Salt & pepper, A -> B.":                      "Salt and pepper, A to B.",
	} {
		if got := SpeechText(in); got != want {
			t.Errorf("SpeechText(%q)\n got %q\nwant %q", in, got, want)
		}
	}
}

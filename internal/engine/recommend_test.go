package engine

import (
	"context"
	"fmt"
	"testing"
)

type cardHF struct{ *fakeHF }

func (cardHF) Card(context.Context, string) (string, error) {
	return "Recommended:\n- Thinking: temperature=0.6, top_p=0.95, top_k=20\n- Non-thinking: temperature=0.7, top_p=0.8, top_k=20\n", nil
}
func (cardHF) RecommendedParams(context.Context, string) (map[string]any, error) {
	return map[string]any{"stop": []any{"<end>"}}, nil
}

func TestRecommended(t *testing.T) {
	st, err := OpenStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	e := &Engine{store: st, HF: cardHF{&fakeHF{}}}
	for _, m := range []*Model{
		{Name: "hf.co/u/x-GGUF:Q4_K_M", Repo: "u/x-GGUF", Files: []string{"missing.gguf"}, Params: map[string]any{"stop": []any{"<end>"}, "temperature": 0.2}},
		{Name: "hf.co/u/x-GGUF:Q8_0", Repo: "u/x-GGUF", Files: []string{"missing.gguf"}},
		{Name: "hf.co/v/y-GGUF:Q4_K_M", Repo: "v/y-GGUF", Files: []string{"missing.gguf"}},
	} {
		if err := st.Put(m); err != nil {
			t.Fatal(err)
		}
	}
	ps, err := e.Recommended(context.Background(), "hf.co/u/x-GGUF:Q4_K_M")
	if err != nil || len(ps) != 2 || ps[0].Label != "Thinking" || ps[0].Source != "card" || ps[1].Options["temperature"] != 0.7 {
		t.Fatalf("%+v %v", ps, err)
	}
	if err := e.UseRecommended("hf.co/u/x-GGUF:Q4_K_M", ps[1].Options); err != nil {
		t.Fatal(err)
	}
	a, _ := st.Get("hf.co/u/x-GGUF:Q4_K_M")
	b, _ := st.Get("hf.co/u/x-GGUF:Q8_0")
	c, _ := st.Get("hf.co/v/y-GGUF:Q4_K_M")
	if a.Params["temperature"] != 0.7 || a.Params["stop"] == nil || fmt.Sprint(a.Params["top_k"]) != "20" {
		t.Errorf("model: %v", a.Params)
	}
	if b.Params["top_p"] != 0.8 {
		t.Errorf("other quant of the same repo: %v", b.Params)
	}
	if c.Params != nil {
		t.Errorf("another repo changed: %v", c.Params)
	}
}

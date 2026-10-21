package index

import (
	"testing"
)

func TestPutSearchPersist(t *testing.T) {
	dir := t.TempDir()
	ix, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	err = ix.Put("emb", "a.md", []Entry{{File: "a.md", Text: "cats"}, {File: "a.md", Text: "dogs"}}, [][]float32{{1, 0, 0}, {0, 1, 0}})
	if err != nil {
		t.Fatal(err)
	}
	if err := ix.Put("emb", "b.md", []Entry{{File: "b.md", Text: "birds"}}, [][]float32{{0, 0, 2}}); err != nil {
		t.Fatal(err)
	}
	if err := ix.Put("other", "c.md", []Entry{{File: "c.md"}}, [][]float32{{1, 1, 1}}); err == nil {
		t.Error("mixing embedding models should fail")
	}

	ix, err = Open(dir) // reload from disk
	if err != nil {
		t.Fatal(err)
	}
	hits := ix.Search([]float32{0.1, 0.9, 0}, 2)
	if len(hits) != 2 || hits[0].Text != "dogs" || hits[1].Text != "cats" {
		t.Fatalf("hits = %+v", hits)
	}
	if hits[0].Score < 0.9 || hits[0].Score > 1.0001 {
		t.Errorf("cosine score = %v", hits[0].Score)
	}

	// Re-putting a file replaces its chunks; removing drops them.
	if err := ix.Put("emb", "a.md", []Entry{{File: "a.md", Text: "fish"}}, [][]float32{{0, 1, 0}}); err != nil {
		t.Fatal(err)
	}
	if ix.Len() != 2 || ix.Search([]float32{0, 1, 0}, 1)[0].Text != "fish" {
		t.Errorf("after replace: len %d", ix.Len())
	}
	if err := ix.Remove("a.md"); err != nil {
		t.Fatal(err)
	}
	if err := ix.Remove("b.md"); err != nil {
		t.Fatal(err)
	}
	if ix.Len() != 0 || ix.Model() != "" {
		t.Errorf("empty index keeps model %q", ix.Model())
	}
	if ix, err = Open(dir); err != nil || ix.Len() != 0 {
		t.Fatalf("reopen empty: %v", err)
	}
	if err := ix.Put("other", "c.md", []Entry{{File: "c.md"}}, [][]float32{{1, 1}}); err != nil {
		t.Errorf("an emptied index takes a new model: %v", err)
	}
}

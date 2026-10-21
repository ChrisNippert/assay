package projects

import (
	"errors"
	"testing"
)

func TestStore(t *testing.T) {
	s, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Create("   "); err == nil {
		t.Error("blank name accepted")
	}
	b, _ := s.Create("Taxes 2026")
	a, _ := s.Create("  ollamui\n dev ")
	if a.Name != "ollamui dev" || !ValidID(a.ID) {
		t.Errorf("created %+v", a)
	}
	list, _ := s.List()
	if len(list) != 2 || list[0].ID != a.ID || list[1].ID != b.ID {
		t.Errorf("list should be by name: %+v", list)
	}
	p, err := s.Update(a.ID, func(p *Project) error {
		p.Files = append(p.Files, File{Name: "notes.md", Status: FileReady})
		return nil
	})
	if err != nil || p.File("notes.md") == nil || p.Ready() != 1 {
		t.Fatalf("update: %v %+v", err, p)
	}
	if got, _ := s.Get(a.ID); got.File("notes.md") == nil {
		t.Error("update not saved")
	}
	if ix, err := s.Index(a.ID); err != nil || ix.Len() != 0 {
		t.Fatalf("index: %v", err)
	}
	if err := s.Delete(a.ID); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Get(a.ID); !errors.Is(err, ErrNotFound) {
		t.Errorf("deleted project still there: %v", err)
	}
	for _, bad := range []string{"", "../x", "p123", "P0123456789"} {
		if _, err := s.Get(bad); !errors.Is(err, ErrNotFound) {
			t.Errorf("Get(%q) = %v", bad, err)
		}
	}
}

func TestCleanFileName(t *testing.T) {
	for in, want := range map[string]string{
		"notes.md":              "notes.md",
		"../../etc/passwd":      "passwd",
		`C:\Users\me\tax.pdf`:   "tax.pdf",
		".hidden":               "hidden",
		"a/b/My Report (1).pdf": "My Report (1).pdf",
		"weird<>|name?.txt":     "weird_name_.txt",
		"..":                    "",
	} {
		if got := CleanFileName(in); got != want {
			t.Errorf("CleanFileName(%q) = %q, want %q", in, got, want)
		}
	}
}

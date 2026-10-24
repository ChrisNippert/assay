package engine

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// moveFixture is a store with a plain blob, a split model's blob in a
// folder, a hard link (like Ollama imports) and a symlink to elsewhere.
func moveFixture(t *testing.T) (s *Store, outside string) {
	t.Helper()
	base := t.TempDir()
	s, err := OpenStore(filepath.Join(base, "models"))
	if err != nil {
		t.Fatal(err)
	}
	outside = filepath.Join(base, "ollama-blob")
	write := func(p, data string) {
		os.MkdirAll(filepath.Dir(p), 0o755)
		if err := os.WriteFile(p, []byte(data), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write(s.Blob("a.gguf"), "weights-a")
	write(s.Blob("split/b-00001.gguf"), "weights-b")
	write(outside, "outside-weights")
	if err := os.Link(outside, s.Blob("linked.gguf")); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, s.Blob("sym.gguf")); err != nil {
		t.Fatal(err)
	}
	for _, m := range []*Model{
		{Name: "a:1", Files: []string{"a.gguf"}, Source: "hf"},
		{Name: "b:1", Files: []string{"split/b-00001.gguf"}, Source: "hf"},
		{Name: "l:1", Files: []string{"linked.gguf"}, Source: "ollama"},
		{Name: "s:1", Files: []string{"sym.gguf"}, Source: "ollama"},
	} {
		if err := s.Put(m); err != nil {
			t.Fatal(err)
		}
	}
	return s, outside
}

func checkStore(t *testing.T, s *Store, dir string) {
	t.Helper()
	if s.Dir() != dir {
		t.Fatalf("dir = %s, want %s", s.Dir(), dir)
	}
	if len(s.List()) != 4 {
		t.Fatalf("models = %d", len(s.List()))
	}
	for name, want := range map[string]string{"a:1": "weights-a", "b:1": "weights-b", "l:1": "outside-weights", "s:1": "outside-weights"} {
		m, err := s.Get(name)
		if err != nil {
			t.Fatal(err)
		}
		if b, err := os.ReadFile(s.Blob(m.Files[0])); err != nil || string(b) != want {
			t.Errorf("%s: %q %v", name, b, err)
		}
	}
}

func TestMoveRename(t *testing.T) {
	s, outside := moveFixture(t)
	old := s.Dir()
	dst := filepath.Join(t.TempDir(), "new", "models")
	plan, err := s.Plan(dst)
	if err != nil || plan.Kind != PlanMove || !plan.SameDisk || plan.Bytes != 0 {
		t.Fatalf("plan = %+v %v", plan, err)
	}
	if err := s.MoveTo(context.Background(), dst, MoveOptions{}); err != nil {
		t.Fatal(err)
	}
	checkStore(t, s, dst)
	if _, err := os.Stat(old); !os.IsNotExist(err) {
		t.Error("old folder left behind")
	}
	// A rename keeps the hard link.
	a, _ := os.Stat(outside)
	b, _ := os.Stat(s.Blob("linked.gguf"))
	if !os.SameFile(a, b) {
		t.Error("hard link lost")
	}
}

func TestMoveCopy(t *testing.T) {
	s, outside := moveFixture(t)
	old := s.Dir()
	dst := filepath.Join(t.TempDir(), "disk2")
	var last, total int64
	opt := MoveOptions{ForceCopy: true, Progress: func(done, tot int64, _ string) { last, total = done, tot }}
	if err := s.MoveTo(context.Background(), dst, opt); err != nil {
		t.Fatal(err)
	}
	checkStore(t, s, dst)
	// a, b and the hard link are copied (plus manifests); the symlink is not.
	want := int64(len("weights-a") + len("weights-b") + len("outside-weights"))
	if last != total || total < want {
		t.Errorf("progress %d/%d, want %d", last, total, want)
	}
	if _, err := os.Stat(old); !os.IsNotExist(err) {
		t.Error("old folder left behind")
	}
	a, _ := os.Stat(outside)
	b, _ := os.Stat(s.Blob("linked.gguf"))
	if os.SameFile(a, b) {
		t.Error("hard link should be a real copy now")
	}
	if fi, _ := os.Lstat(s.Blob("sym.gguf")); fi.Mode()&os.ModeSymlink == 0 {
		t.Error("symlink should stay a symlink")
	}
	if _, err := os.Stat(outside); err != nil {
		t.Error("the shared file outside the store must survive")
	}
	// Deleting a model removes the copy only.
	if err := s.Delete("l:1"); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(outside); err != nil {
		t.Error("delete touched the outside file")
	}
}

func TestMoveCancelLeavesSource(t *testing.T) {
	s, _ := moveFixture(t)
	old := s.Dir()
	dst := filepath.Join(t.TempDir(), "disk2")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := s.MoveTo(ctx, dst, MoveOptions{ForceCopy: true}); err == nil {
		t.Fatal("canceled move succeeded")
	}
	checkStore(t, s, old)
	var left []string
	filepath.WalkDir(dst, func(p string, d os.DirEntry, _ error) error {
		if d != nil && !d.IsDir() {
			left = append(left, p)
		}
		return nil
	})
	if len(left) > 0 {
		t.Errorf("copies left behind: %v", left)
	}
	if err := s.Put(&Model{Name: "c:1", Files: []string{"a.gguf"}}); err != nil {
		t.Errorf("store stuck in moving state: %v", err)
	}
}

func TestPlanRefusals(t *testing.T) {
	s, _ := moveFixture(t)
	other := t.TempDir()
	file := filepath.Join(other, "photo.jpg")
	os.WriteFile(file, []byte("x"), 0o644)
	for dst, want := range map[string]string{
		s.Dir():                         "already",
		filepath.Join(s.Dir(), "blobs"): "inside",
		filepath.Dir(s.Dir()):           "contains",
		other:                           "isn't empty",
		file:                            "file",
		"relative/path":                 "full path",
		"":                              "enter",
		filepath.Join(file, "sub"):      "file",
	} {
		if _, err := s.Plan(dst); err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("Plan(%q) = %v, want %q", dst, err, want)
		}
	}
	// A symlink into the store is still inside it.
	link := filepath.Join(other, "link")
	os.Symlink(s.Dir(), link)
	if _, err := s.Plan(filepath.Join(link, "x")); err == nil {
		t.Error("symlinked path into the store accepted")
	}
}

func TestSwitchToExistingStore(t *testing.T) {
	a, _ := moveFixture(t)
	other, err := OpenStore(filepath.Join(t.TempDir(), "other"))
	if err != nil {
		t.Fatal(err)
	}
	other.Put(&Model{Name: "z:1", Files: []string{"z.gguf"}})
	plan, err := a.Plan(other.Dir())
	if err != nil || plan.Kind != PlanSwitch || plan.Models != 1 {
		t.Fatalf("plan = %+v %v", plan, err)
	}
	oldDir := a.Dir()
	if err := a.MoveTo(context.Background(), other.Dir(), MoveOptions{}); err != nil {
		t.Fatal(err)
	}
	if _, err := a.Get("z:1"); err != nil || len(a.List()) != 1 {
		t.Errorf("after switch: %v %v", a.List(), err)
	}
	if _, err := os.Stat(filepath.Join(oldDir, "blobs", "a.gguf")); err != nil {
		t.Error("a switch must leave the old files alone")
	}
}

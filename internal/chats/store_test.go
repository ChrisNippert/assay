package chats

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestSaveGetList(t *testing.T) {
	s, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	a := New("qwen3:8b")
	a.Title = "first"
	a.Messages = append(a.Messages, Message{Role: "user", Content: "hi"})
	if err := s.Save(a); err != nil {
		t.Fatal(err)
	}
	time.Sleep(10 * time.Millisecond)
	b := New("gemma3:12b")
	b.Title = "second"
	if err := s.Save(b); err != nil {
		t.Fatal(err)
	}

	got, err := s.Get(a.ID)
	if err != nil {
		t.Fatal(err)
	}
	if got.Title != "first" || len(got.Messages) != 1 || got.Messages[0].Content != "hi" {
		t.Fatalf("got %+v", got)
	}

	list, err := s.List()
	if err != nil {
		t.Fatal(err)
	}
	if len(list) != 2 || list[0].ID != b.ID {
		t.Fatalf("list = %+v, want newest (%s) first", list, b.ID)
	}

	if err := s.Delete(a.ID); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Get(a.ID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("after delete: %v", err)
	}
}

func TestRejectsBadIDs(t *testing.T) {
	dir := t.TempDir()
	s, _ := Open(dir)
	os.WriteFile(filepath.Join(filepath.Dir(dir), "secret.json"), []byte(`{}`), 0o600)
	for _, id := range []string{"../secret", "", "20260101-zzzzzzzz", "20260101-abcd1234/../x"} {
		if _, err := s.Get(id); !errors.Is(err, ErrNotFound) {
			t.Errorf("Get(%q) = %v, want ErrNotFound", id, err)
		}
	}
	if err := s.Save(&Chat{ID: "../evil"}); err == nil {
		t.Error("Save accepted a bad id")
	}
}

func TestListSkipsCorruptFiles(t *testing.T) {
	dir := t.TempDir()
	s, _ := Open(dir)
	good := New("m")
	s.Save(good)
	os.WriteFile(filepath.Join(dir, "20260101-deadbeef.json"), []byte("{not json"), 0o600)
	list, err := s.List()
	if err != nil {
		t.Fatal(err)
	}
	if len(list) != 1 || list[0].ID != good.ID {
		t.Fatalf("list = %+v", list)
	}
}

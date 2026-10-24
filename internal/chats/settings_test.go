package chats

import (
	"path/filepath"
	"testing"
)

func TestChatSettingsRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "chats.json")
	st, err := OpenSettings(path)
	if err != nil {
		t.Fatal(err)
	}
	if !st.Get().CompressAuto {
		t.Fatal("compression should default to on")
	}
	if err := st.Set(Settings{CompressAuto: false}); err != nil {
		t.Fatal(err)
	}
	again, err := OpenSettings(path)
	if err != nil {
		t.Fatal(err)
	}
	if again.Get().CompressAuto {
		t.Error("the saved choice didn't come back")
	}
}

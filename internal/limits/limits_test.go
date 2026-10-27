package limits

import (
	"path/filepath"
	"testing"
)

func TestRoundTrip(t *testing.T) {
	p := filepath.Join(t.TempDir(), "sub", "limits.json")
	st, err := Open(p)
	if err != nil {
		t.Fatal(err)
	}
	if st.Get() != (Limits{}) {
		t.Fatal("default should be no limits")
	}
	if err := st.Set(Limits{MaxVRAMGB: 3, Gentle: true}); err != nil {
		t.Fatal(err)
	}
	st2, err := Open(p)
	if err != nil {
		t.Fatal(err)
	}
	if got := st2.Get(); got.MaxVRAMGB != 3 || !got.Gentle {
		t.Fatalf("got %+v", got)
	}
	if st.Set(Limits{MaxVRAMGB: -1}) == nil || st.Set(Limits{MaxVRAMGB: 0.1}) == nil {
		t.Fatal("bad limits accepted")
	}
	var nilStore *Store
	if nilStore.Get() != (Limits{}) {
		t.Fatal("nil store")
	}
}

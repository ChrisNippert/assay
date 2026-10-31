package server

import (
	"path/filepath"
	"strings"
	"testing"

	"assay/internal/limits"
)

func TestLimitsEndpoint(t *testing.T) {
	st, err := limits.Open(filepath.Join(t.TempDir(), "limits.json"))
	if err != nil {
		t.Fatal(err)
	}
	s, err := New(Options{Limits: st})
	if err != nil {
		t.Fatal(err)
	}
	if rec := do(s, "GET", "/api/limits", ""); rec.Code != 200 || !strings.Contains(rec.Body.String(), `"maxVramGb":0`) {
		t.Fatalf("get: %d %s", rec.Code, rec.Body)
	}
	if rec := do(s, "PUT", "/api/limits", `{"maxVramGb":3,"gentle":true}`); rec.Code != 200 {
		t.Fatalf("put: %d %s", rec.Code, rec.Body)
	}
	if got := st.Get(); got.MaxVRAMGB != 3 || !got.Gentle {
		t.Errorf("saved %+v", got)
	}
	if rec := do(s, "PUT", "/api/limits", `{"maxVramGb":-2}`); rec.Code != 400 {
		t.Errorf("negative: %d", rec.Code)
	}
}

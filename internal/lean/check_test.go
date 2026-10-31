package lean

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// testBinary finds Lean: $ASSAY_LEAN, else Assay's install, else PATH.
func testBinary(t *testing.T) string {
	if p := os.Getenv("ASSAY_LEAN"); p != "" {
		return p
	}
	home, _ := os.UserHomeDir()
	if b := (&Installer{Dir: filepath.Join(home, ".local/share/assay/lean")}).Binary(); b != "" {
		return b
	}
	t.Skip("Lean isn't installed")
	return ""
}

func TestCheck(t *testing.T) {
	bin := testBinary(t)
	ctx := context.Background()
	pre := "def double (n : Nat) : Nat := n + n"
	stmt := "∀ n : Nat, double n = 2 * n"
	cases := []struct {
		name, code string
		ok         bool
		why        string
	}{
		{"right", "theorem goal : ∀ n : Nat, double n = 2 * n := by\n  intro n; unfold double; omega", true, ""},
		{"helper", "theorem two (n : Nat) : n + n = 2 * n := by omega\ntheorem goal : ∀ n : Nat, double n = 2 * n := fun n => two n", true, ""},
		{"sorry", "theorem goal : ∀ n : Nat, double n = 2 * n := by sorry", false, "sorry"},
		{"axiom", "axiom cheat : ∀ n : Nat, double n = 2 * n\ntheorem goal : ∀ n : Nat, double n = 2 * n := cheat", false, "axiom"},
		{"native", "theorem goal : ∀ n : Nat, double n = 2 * n := by native_decide", false, "native_decide"},
		{"weaker", "theorem goal : double 0 = 2 * 0 := rfl", false, "mismatch"},
		{"missing", "theorem other : ∀ n : Nat, double n = 2 * n := by intro n; unfold double; omega", false, "goal"},
		{"wrong", "theorem goal : ∀ n : Nat, double n = 2 * n := by intro n; rfl", false, "error"},
	}
	for _, c := range cases {
		r, err := Check(ctx, bin, pre, stmt, c.code)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if r.OK != c.ok {
			t.Errorf("%s: ok=%v, want %v\n%s", c.name, r.OK, c.ok, r.Output)
		}
		if !c.ok && r.Output == "" {
			t.Errorf("%s: no reason given", c.name)
		}
	}
}

func TestBadAxioms(t *testing.T) {
	if b := badAxioms("'x' depends on axioms: [propext, Classical.choice, Quot.sound]"); b != "" {
		t.Error(b)
	}
	if b := badAxioms("'x' depends on axioms: [propext, sorryAx]"); b != "sorryAx" {
		t.Error(b)
	}
	if b := badAxioms("'x' does not depend on any axioms"); b != "" {
		t.Error(b)
	}
	if b := badAxioms("nothing"); !strings.Contains(b, "unknown") {
		t.Error(b)
	}
}

func TestExtractCode(t *testing.T) {
	r := "Idea first.\n```lean\ntheorem goal : 1 = 1 := rfl\n```\nOr better:\n```lean4\ntheorem goal : 1 = 1 := by decide\n```"
	if c, ok := ExtractCode(r); !ok || c != "theorem goal : 1 = 1 := by decide" {
		t.Errorf("%q", c)
	}
	if _, ok := ExtractCode("```python\nx\n```"); ok {
		t.Error("took a python block")
	}
}

func TestAssetName(t *testing.T) {
	for _, c := range [][4]string{{"v4.35.0", "linux", "amd64", "lean-4.35.0-linux.zip"}, {"v4.35.0", "linux", "arm64", "lean-4.35.0-linux_aarch64.zip"}, {"v4.35.0", "windows", "amd64", "lean-4.35.0-windows.zip"}, {"v4.35.0", "darwin", "arm64", "lean-4.35.0-darwin_aarch64.zip"}} {
		if got := assetName(c[0], c[1], c[2]); got != c[3] {
			t.Errorf("%v: %s", c, got)
		}
	}
}

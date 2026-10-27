package hw

import (
	"os"
	"path/filepath"
	"testing"
)

const smiTotals = "Quadro P2000, 5120\nQuadro P2000, 5120\nQuadro P2000, 5120\nNVIDIA GeForce GTX 1050 Ti, 4096\n"

func TestParseNvidiaCards(t *testing.T) {
	cards := parseNvidiaCards(smiTotals)
	if len(cards) != 4 || cards[0].Total != 5120<<20 || cards[3].Name != "NVIDIA GeForce GTX 1050 Ti" {
		t.Fatalf("%+v", cards)
	}
	if len(parseNvidiaCards("garbage")) != 0 {
		t.Fatal("garbage parsed")
	}
}

func TestGroupName(t *testing.T) {
	if got := GroupName([]string{"NVIDIA GeForce RTX 3090"}); got != "NVIDIA GeForce RTX 3090" {
		t.Errorf("single: %q", got)
	}
	var names []string
	for _, c := range parseNvidiaCards(smiTotals) {
		names = append(names, c.Name)
	}
	if got, want := GroupName(names), "4 GPUs (3× Quadro P2000, GeForce GTX 1050 Ti)"; got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

// fakeDRM makes card dirs holding total, used and (if set) busy files.
func fakeDRM(t *testing.T, cards map[string][3]string) string {
	t.Helper()
	dir := t.TempDir()
	for card, v := range cards {
		p := filepath.Join(dir, card, "device")
		os.MkdirAll(p, 0o755)
		os.WriteFile(filepath.Join(p, "mem_info_vram_total"), []byte(v[0]+"\n"), 0o644)
		os.WriteFile(filepath.Join(p, "mem_info_vram_used"), []byte(v[1]+"\n"), 0o644)
		if v[2] != "" {
			os.WriteFile(filepath.Join(p, "gpu_busy_percent"), []byte(v[2]+"\n"), 0o644)
		}
	}
	return dir
}

func TestAMDCardsSum(t *testing.T) {
	const g = 1 << 30
	dir := fakeDRM(t, map[string][3]string{
		"card0": {"536870912", "1", "0"}, // iGPU carve-out, skipped
		"card1": {"12884901888", "3221225472", "40"},
		"card2": {"8589934592", "1073741824", "90"},
	})
	if got := amdVRAM(dir); got != 20*g {
		t.Errorf("amdVRAM = %d", got)
	}
	if c := amdDetect(dir); len(c) != 2 || c[0].Name != "AMD GPU" {
		t.Errorf("cards %+v", c)
	}
	used, total, busy, ok := amdLive(dir)
	if !ok || used != 4*g || total != 20*g || busy != 90 {
		t.Errorf("live %d %d %d %v", used, total, busy, ok)
	}
	// Only small carve-outs: nothing to skip them for.
	if got := amdVRAM(fakeDRM(t, map[string][3]string{"card0": {"536870912", "0", ""}})); got != 536870912 {
		t.Errorf("lone small card = %d", got)
	}
}

func TestParseNvidiaLiveMulti(t *testing.T) {
	out := "Quadro P2000, 1200, 5120, 30\nQuadro P2000, 800, 5120, 0\nNVIDIA GeForce GTX 1050 Ti, 300, 4096, [N/A]\n"
	cards := parseNvidiaLive(out)
	if len(cards) != 3 || cards[0].Name != "Quadro P2000" || cards[2].Busy != -1 || cards[1].Used != 800<<20 {
		t.Fatalf("%+v", cards)
	}
	used, total, busy := sumCards(cards)
	if used != 2300<<20 || total != 14336<<20 || busy != 30 {
		t.Errorf("sum %d %d %d", used, total, busy)
	}
	// Without the name column, as before.
	u, tot, b, ok := parseNvidiaSMI("1024, 8192, 17\n2048, 4096, 5\n")
	if !ok || u != 3072<<20 || tot != 12288<<20 || b != 17 {
		t.Errorf("smi sum %d %d %d %v", u, tot, b, ok)
	}
}

func TestClassifyExtraCards(t *testing.T) {
	one := Memory{VRAM: 14e9, Cards: []Card{{"a", 14e9}}, RAM: 48e9}
	four := Memory{VRAM: 14e9, Cards: []Card{{"a", 4e9}, {"b", 4e9}, {"c", 3e9}, {"d", 3e9}}, RAM: 48e9}
	need := int64(11.5e9) // under 90% of 14e9, over 90% of 14e9 less three cards' overhead
	if got := Classify(need, one).Level; got != FitGPU {
		t.Errorf("one card: %s", got)
	}
	if got := Classify(need, four).Level; got != FitTight {
		t.Errorf("four cards: %s", got)
	}
	// A lone Cards entry changes nothing.
	if Classify(need, one) != Classify(need, Memory{VRAM: 14e9, RAM: 48e9}) {
		t.Error("single card differs from no card list")
	}
}

func TestFakeGPUs(t *testing.T) {
	f := parseFakeGPUs("Quadro P2000:5120:1200,GeForce GTX 1050 Ti:4096:300")
	if len(f) != 2 || f[0].Total != 5120<<20 || f[1].Used != 300<<20 {
		t.Fatalf("%+v", f)
	}
	if parseFakeGPUs("") != nil || parseFakeGPUs("x:y") != nil {
		t.Error("bad input parsed")
	}
	t.Setenv(FakeEnv, "A:1000:10,B:2000:20")
	m := Detect()
	if m.VRAM != 3000<<20 || len(m.Cards) != 2 || m.GPU != "2 GPUs (A, B)" {
		t.Errorf("%+v", m)
	}
	if l := ReadLive(); l.VRAMTotal != 3000<<20 || l.VRAMUsed != 30<<20 || len(l.Cards) != 2 {
		t.Errorf("%+v", l)
	}
}

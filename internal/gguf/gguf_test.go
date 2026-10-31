package gguf

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// w builds a synthetic GGUF file.
type w struct{ bytes.Buffer }

func (b *w) u32(v uint32) { binary.Write(b, binary.LittleEndian, v) }
func (b *w) u64(v uint64) { binary.Write(b, binary.LittleEndian, v) }
func (b *w) str(s string) { b.u64(uint64(len(s))); b.WriteString(s) }
func (b *w) head(ver uint32, tensors, kvs uint64) {
	b.WriteString("GGUF")
	b.u32(ver)
	b.u64(tensors)
	b.u64(kvs)
}
func (b *w) kv(key string, typ uint32, val func()) {
	b.str(key)
	b.u32(typ)
	val()
}
func (b *w) tensor(name string, typ uint32, dims ...uint64) {
	b.str(name)
	b.u32(uint32(len(dims)))
	for _, d := range dims {
		b.u64(d)
	}
	b.u32(typ)
	b.u64(0)
}

func writeTemp(t *testing.T, name string, b []byte) string {
	p := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(p, b, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func synthetic() []byte {
	var b w
	b.head(3, 2, 18)
	b.kv("u8", 0, func() { b.WriteByte(200) })
	b.kv("i8", 1, func() { b.WriteByte(0xFE) })
	b.kv("u16", 2, func() { binary.Write(&b, binary.LittleEndian, uint16(65000)) })
	b.kv("i16", 3, func() { binary.Write(&b, binary.LittleEndian, int16(-5)) })
	b.kv("u32", 4, func() { b.u32(4000000000) })
	b.kv("i32", 5, func() { binary.Write(&b, binary.LittleEndian, int32(-7)) })
	b.kv("f32", 6, func() { binary.Write(&b, binary.LittleEndian, float32(1.5)) })
	b.kv("bool", 7, func() { b.WriteByte(1) })
	b.kv("str", 8, func() { b.str("héllo") })
	b.kv("u64", 10, func() { b.u64(1 << 40) })
	b.kv("i64", 11, func() { binary.Write(&b, binary.LittleEndian, int64(-9)) })
	b.kv("f64", 12, func() { binary.Write(&b, binary.LittleEndian, math.Pi) })
	b.kv("arr", 9, func() { // nested: [[1,2],[3]]
		b.u32(9)
		b.u64(2)
		b.u32(4)
		b.u64(2)
		b.u32(1)
		b.u32(2)
		b.u32(4)
		b.u64(1)
		b.u32(3)
	})
	b.kv("strs", 9, func() {
		b.u32(8)
		b.u64(2)
		b.str("a")
		b.str("bc")
	})
	b.kv("bigints", 9, func() {
		b.u32(4)
		b.u64(MaxArray + 1)
		b.Write(make([]byte, 4*(MaxArray+1)))
	})
	b.kv("bigstrs", 9, func() {
		b.u32(8)
		b.u64(MaxArray + 10)
		for i := 0; i < MaxArray+10; i++ {
			b.str("tok")
		}
	})
	b.kv("bignested", 9, func() {
		b.u32(9)
		b.u64(MaxArray + 1)
		for i := 0; i < MaxArray+1; i++ {
			b.u32(8)
			b.u64(1)
			b.str("x")
		}
	})
	b.kv("last", 8, func() { b.str("after the skipped arrays") })
	b.tensor("a", 0, 10, 20)
	b.tensor("b", 1, 7)
	return b.Bytes()
}

func TestSynthetic(t *testing.T) {
	f, err := Read(writeTemp(t, "m.gguf", synthetic()))
	if err != nil {
		t.Fatal(err)
	}
	if f.Version != 3 {
		t.Errorf("version %d", f.Version)
	}
	want := map[string]any{
		"u8": 200.0, "i8": -2.0, "u16": 65000.0, "i16": -5.0, "u32": 4e9, "i32": -7.0,
		"f32": 1.5, "bool": true, "str": "héllo", "u64": float64(1 << 40), "i64": -9.0,
		"f64": math.Pi, "last": "after the skipped arrays",
	}
	for k, v := range want {
		if f.Meta[k] != v {
			t.Errorf("%s = %v (%T), want %v", k, f.Meta[k], f.Meta[k], v)
		}
	}
	arr := f.Meta["arr"].([]any)
	if len(arr) != 2 || len(arr[0].([]any)) != 2 || arr[1].([]any)[0] != 3.0 {
		t.Errorf("arr = %v", arr)
	}
	if s := f.Meta["strs"].([]any); len(s) != 2 || s[1] != "bc" {
		t.Errorf("strs = %v", s)
	}
	if got := f.Meta["bigints"]; got != (ArrayInfo{Type: 4, Len: MaxArray + 1}) {
		t.Errorf("bigints = %v", got)
	}
	if got := f.Meta["bigstrs"]; got != (ArrayInfo{Type: 8, Len: MaxArray + 10}) {
		t.Errorf("bigstrs = %v", got)
	}
	if got := f.Meta["bignested"]; got != (ArrayInfo{Type: 9, Len: MaxArray + 1}) {
		t.Errorf("bignested = %v", got)
	}
	if len(f.Tensors) != 2 || f.Tensors[0].Name != "a" || f.Tensors[1].Type != 1 {
		t.Errorf("tensors = %+v", f.Tensors)
	}
	if f.ParamCount() != 207 {
		t.Errorf("params = %d", f.ParamCount())
	}
	if f.SizeLabel() != "207" {
		t.Errorf("size label = %q", f.SizeLabel())
	}
	mi := f.ModelInfo()
	if _, ok := mi["bigints"]; ok {
		t.Error("ModelInfo kept an ArrayInfo")
	}
	if mi["u8"] != 200.0 {
		t.Error("ModelInfo lost u8")
	}
}

func model(arch string, extra func(*w, *int), tensors ...uint64) []byte {
	var b w
	n := 0
	var body w
	body.kv("general.architecture", 8, func() { body.str(arch) })
	n++
	if extra != nil {
		extra(&body, &n)
	}
	b.head(3, uint64(len(tensors)), uint64(n))
	b.Write(body.Bytes())
	for i, d := range tensors {
		b.tensor("t"+string(rune('a'+i)), 0, d, 1000)
	}
	return b.Bytes()
}

func TestAccessors(t *testing.T) {
	data := model("llama", func(b *w, n *int) {
		b.kv("general.name", 8, func() { b.str("Tiny") })
		b.kv("general.file_type", 4, func() { b.u32(15) })
		b.kv("llama.context_length", 4, func() { b.u32(8192) })
		b.kv("llama.block_count", 4, func() { b.u32(32) })
		b.kv("tokenizer.chat_template", 8, func() { b.str("{{x}}") })
		*n += 5
	}, 1_000_000, 8_000_000)
	f, err := Read(writeTemp(t, "m.gguf", data))
	if err != nil {
		t.Fatal(err)
	}
	if f.Arch() != "llama" || f.Name() != "Tiny" || f.FileType() != "Q4_K_M" ||
		f.ContextLength() != 8192 || f.BlockCount() != 32 || f.ChatTemplate() != "{{x}}" {
		t.Errorf("accessors: %q %q %q %d %d %q", f.Arch(), f.Name(), f.FileType(), f.ContextLength(), f.BlockCount(), f.ChatTemplate())
	}
	if f.ParamCount() != 9_000_000_000 || f.SizeLabel() != "9.0B" {
		t.Errorf("params %d label %q", f.ParamCount(), f.SizeLabel())
	}
	if f.IsEmbedding() || f.IsProjector() {
		t.Error("llama is neither embedding nor projector")
	}
	if _, ok := f.ModelInfo()["tokenizer.chat_template"]; ok {
		t.Error("ModelInfo kept tokenizer key")
	}
	if f.ArchUint("nothing") != 0 || f.Uint("general.name") != 0 {
		t.Error("missing/non-numeric Uint should be 0")
	}
}

func TestKinds(t *testing.T) {
	read := func(b []byte) *File {
		f, err := Read(writeTemp(t, "m.gguf", b))
		if err != nil {
			t.Fatal(err)
		}
		return f
	}
	if !read(model("bert", nil)).IsEmbedding() {
		t.Error("bert should be an embedding model")
	}
	pooled := read(model("qwen3", func(b *w, n *int) {
		b.kv("qwen3.pooling_type", 4, func() { b.u32(3) })
		*n++
	}))
	if !pooled.IsEmbedding() {
		t.Error("pooling_type should mark an embedding model")
	}
	if !read(model("clip", nil)).IsProjector() {
		t.Error("clip should be a projector")
	}
}

func TestLoRA(t *testing.T) {
	read := func(b []byte) *File {
		f, err := Read(writeTemp(t, "a.gguf", b))
		if err != nil {
			t.Fatal(err)
		}
		return f
	}
	lora := read(model("qwen35", func(b *w, n *int) {
		b.kv("general.type", 8, func() { b.str("adapter") })
		b.kv("adapter.type", 8, func() { b.str("lora") })
		*n += 2
	}))
	if !lora.IsLoRA() || lora.Arch() != "qwen35" {
		t.Errorf("lora adapter: IsLoRA %v arch %q", lora.IsLoRA(), lora.Arch())
	}
	if read(model("qwen35", nil)).IsLoRA() {
		t.Error("a plain model is not an adapter")
	}
	ia3 := read(model("qwen35", func(b *w, n *int) {
		b.kv("general.type", 8, func() { b.str("adapter") })
		b.kv("adapter.type", 8, func() { b.str("ia3") })
		*n += 2
	}))
	if ia3.IsLoRA() {
		t.Error("only LoRA adapters count")
	}
}

func TestFileType(t *testing.T) {
	for id, want := range map[float64]string{0: "F32", 1: "F16", 7: "Q8_0", 32: "BF16", 19: "IQ2_XXS", 38: "MXFP4_MOE", 99: "type 99"} {
		f := &File{Meta: map[string]any{"general.file_type": id}}
		if got := f.FileType(); got != want {
			t.Errorf("%v: got %q want %q", id, got, want)
		}
	}
	if (&File{Meta: map[string]any{}}).FileType() != "" {
		t.Error("missing file type should be empty")
	}
}

func TestSplits(t *testing.T) {
	dir := t.TempDir()
	one := model("llama", nil, 100)
	two := model("ignored", nil, 200, 300)
	p1, p2 := filepath.Join(dir, "m-00001-of-00002.gguf"), filepath.Join(dir, "m-00002-of-00002.gguf")
	os.WriteFile(p1, one, 0o644)
	os.WriteFile(p2, two, 0o644)
	f, err := ReadSplits([]string{p1, p2})
	if err != nil {
		t.Fatal(err)
	}
	if f.Arch() != "llama" || len(f.Tensors) != 3 || f.ParamCount() != 600_000 {
		t.Errorf("arch %q tensors %d params %d", f.Arch(), len(f.Tensors), f.ParamCount())
	}
	if _, err := ReadSplits(nil); err == nil {
		t.Error("no paths should fail")
	}
}

func TestCorrupt(t *testing.T) {
	good := synthetic()
	// Every truncation must be an error, not a panic.
	for i := 0; i < len(good); i += 7 {
		if _, err := parseBytes(good[:i]); err == nil {
			t.Fatalf("truncated at %d: no error", i)
		}
	}
	var v1 w
	v1.head(1, 0, 0)
	cases := map[string][]byte{
		"empty":     nil,
		"bad magic": []byte("GGML\x03\x00\x00\x00" + strings.Repeat("\x00", 16)),
		"v1":        v1.Bytes(),
	}
	var huge w
	huge.head(3, 0, 1)
	huge.u64(1 << 40)
	cases["huge key"] = huge.Bytes()
	var counts w
	counts.head(3, 1<<60, 0)
	cases["huge counts"] = counts.Bytes()
	var dims w
	dims.head(3, 1, 0)
	dims.str("x")
	dims.u32(99)
	cases["many dims"] = dims.Bytes()
	var typ w
	typ.head(3, 0, 1)
	typ.str("k")
	typ.u32(77)
	cases["bad type"] = typ.Bytes()
	var big w
	big.head(3, 0, 1)
	big.str("k")
	big.u32(8)
	big.u64(200 << 20) // plausible length, but the bytes aren't there
	cases["lying string"] = big.Bytes()
	var arr w
	arr.head(3, 0, 1)
	arr.str("k")
	arr.u32(9)
	arr.u32(4)
	arr.u64(1 << 62)
	cases["huge array"] = arr.Bytes()
	for name, b := range cases {
		if _, err := parseBytes(b); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
	if _, err := parseBytes(v1.Bytes()); err == nil || !strings.Contains(err.Error(), "v1") {
		t.Errorf("v1 error = %v", err)
	}
	if _, err := Read(filepath.Join(t.TempDir(), "missing.gguf")); err == nil {
		t.Error("missing file should fail")
	}
}

func parseBytes(b []byte) (*File, error) {
	return parse(bufio.NewReader(bytes.NewReader(b)))
}

const (
	qwenBlob   = "/home/autumn/.ollama/models/blobs/sha256-6f5d30666c2d8ae16a306e616d95341dcf3cc46810df84d7e6f5a7d1e4c1b293"
	mmprojBlob = "/home/autumn/.ollama/models/blobs/sha256-853698ce7aa6c7ba732478bad280240969ddf7b0fcbf93900046f63903a83383"
	gemmaBlob  = "/home/autumn/.ollama/models/blobs/sha256-90fd44e29e0d7cffeb0fd00dc73cfdab9ed0b0e95306ecf7821ea634c940c370"
	// gemma-4-26B-A4B, a mixture-of-experts model, in Assay's store.
	gemmaMoEBlob = "/home/autumn/.local/share/assay/models/blobs/sha256-eeb867f279ea5a3d52a0dc15fe8ada677b3328a328530957f3f9a5da93cb10b8.gguf"
)

func readReal(t *testing.T, path string) *File {
	if _, err := os.Stat(path); err != nil {
		t.Skip("real model not present")
	}
	start := time.Now()
	f, err := Read(path)
	if err != nil {
		t.Fatal(err)
	}
	if d := time.Since(start); d > time.Second {
		t.Errorf("reading the header took %v", d)
	}
	t.Logf("%s: arch=%s type=%s params=%d (%s) ctx=%d blocks=%d tensors=%d", filepath.Base(path),
		f.Arch(), f.FileType(), f.ParamCount(), f.SizeLabel(), f.ContextLength(), f.BlockCount(), len(f.Tensors))
	return f
}

func TestRealQwen(t *testing.T) {
	f := readReal(t, qwenBlob)
	if f.Arch() != "qwen35" || f.FileType() != "Q4_K_M" || f.ContextLength() != 262144 || f.ChatTemplate() == "" {
		t.Errorf("unexpected: %s %s %d", f.Arch(), f.FileType(), f.ContextLength())
	}
	if p := float64(f.ParamCount()) / 1e9; p < 8.5 || p > 9.5 {
		t.Errorf("params %.2fB", p)
	}
	if f.IsProjector() {
		t.Error("not a projector")
	}
}

func TestRealMmproj(t *testing.T) {
	if !readReal(t, mmprojBlob).IsProjector() {
		t.Error("expected a projector")
	}
}

func TestRealGemma(t *testing.T) {
	f := readReal(t, gemmaBlob)
	if f.Arch() != "gemma4" {
		t.Errorf("arch %q", f.Arch())
	}
	if p := float64(f.ParamCount()) / 1e9; p < 11 || p > 13 {
		t.Errorf("params %.2fB", p)
	}
}

func TestRealGemmaMoE(t *testing.T) {
	// The 26B-A4B is mixture-of-experts; the 12B is dense.
	if f := readReal(t, gemmaMoEBlob); f.ExpertCount() == 0 {
		t.Error("gemma-4-26B-A4B should report experts")
	}
}

// Package gguf reads the header of GGUF model files: the metadata and the tensor
// descriptions, never the weights. It replaces what Ollama's /api/show used to tell us.
package gguf

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
)

// MaxArray is the longest array kept in Meta. Longer ones (tokenizer vocabularies,
// merges) are skipped and their key maps to an ArrayInfo instead.
const MaxArray = 4096

const (
	maxString  = 256 << 20 // reject strings longer than this
	maxKV      = 1 << 20
	maxTensors = 1 << 22
	maxDims    = 8
	maxDepth   = 8
)

// File is a GGUF file's header: metadata and tensor descriptions, never the weights.
type File struct {
	Version uint32
	// Meta holds the metadata key/values. Numbers are float64 (all integer and float types),
	// strings are string, bools are bool, arrays are []any of those. Arrays with more than
	// MaxArray elements are skipped: their key maps to ArrayInfo{Type, Len} instead.
	Meta    map[string]any
	Tensors []Tensor
}

// ArrayInfo stands in for an array too long to keep.
type ArrayInfo struct {
	Type uint32
	Len  uint64
}

// Tensor describes one tensor: its name, shape and ggml type id.
type Tensor struct {
	Name string
	Dims []uint64
	Type uint32 // ggml type id
}

// GGUF value types.
const (
	tUint8 = iota
	tInt8
	tUint16
	tInt16
	tUint32
	tInt32
	tFloat32
	tBool
	tString
	tArray
	tUint64
	tInt64
	tFloat64
)

// Read parses the header of one GGUF file.
func Read(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	g, err := parse(bufio.NewReaderSize(f, 1<<20))
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return g, nil
}

// ReadSplits reads a split model (model-00001-of-00003.gguf ...): metadata from the
// first file, tensors from all of them.
func ReadSplits(paths []string) (*File, error) {
	if len(paths) == 0 {
		return nil, errors.New("gguf: no files")
	}
	out, err := Read(paths[0])
	if err != nil {
		return nil, err
	}
	for _, p := range paths[1:] {
		g, err := Read(p)
		if err != nil {
			return nil, err
		}
		out.Tensors = append(out.Tensors, g.Tensors...)
	}
	return out, nil
}

type reader struct{ r *bufio.Reader }

func (r reader) u32() (uint32, error) {
	var b [4]byte
	if _, err := io.ReadFull(r.r, b[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(b[:]), nil
}

func (r reader) u64() (uint64, error) {
	var b [8]byte
	if _, err := io.ReadFull(r.r, b[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(b[:]), nil
}

func (r reader) strLen() (uint64, error) {
	n, err := r.u64()
	if err != nil {
		return 0, err
	}
	if n > maxString {
		return 0, fmt.Errorf("string length %d is too big", n)
	}
	return n, nil
}

func (r reader) str() (string, error) {
	n, err := r.strLen()
	if err != nil {
		return "", err
	}
	if n <= 1<<16 {
		b := make([]byte, n)
		if _, err := io.ReadFull(r.r, b); err != nil {
			return "", err
		}
		return string(b), nil
	}
	// Big strings: don't trust the length until the bytes actually arrive.
	b, err := io.ReadAll(io.LimitReader(r.r, int64(n)))
	if err != nil {
		return "", err
	}
	if uint64(len(b)) != n {
		return "", io.ErrUnexpectedEOF
	}
	return string(b), nil
}

func (r reader) discard(n uint64) error {
	for n > 0 {
		step := n
		if step > 1<<30 {
			step = 1 << 30
		}
		d, err := r.r.Discard(int(step))
		if err != nil {
			return err
		}
		n -= uint64(d)
	}
	return nil
}

// fixedSize is the byte size of a fixed-width value type, or 0.
func fixedSize(t uint32) uint64 {
	switch t {
	case tUint8, tInt8, tBool:
		return 1
	case tUint16, tInt16:
		return 2
	case tUint32, tInt32, tFloat32:
		return 4
	case tUint64, tInt64, tFloat64:
		return 8
	}
	return 0
}

func (r reader) value(t uint32, depth int) (any, error) {
	if sz := fixedSize(t); sz > 0 {
		var b [8]byte
		if _, err := io.ReadFull(r.r, b[:sz]); err != nil {
			return nil, err
		}
		le := binary.LittleEndian
		switch t {
		case tUint8:
			return float64(b[0]), nil
		case tInt8:
			return float64(int8(b[0])), nil
		case tBool:
			return b[0] != 0, nil
		case tUint16:
			return float64(le.Uint16(b[:])), nil
		case tInt16:
			return float64(int16(le.Uint16(b[:]))), nil
		case tUint32:
			return float64(le.Uint32(b[:])), nil
		case tInt32:
			return float64(int32(le.Uint32(b[:]))), nil
		case tFloat32:
			return float64(math.Float32frombits(le.Uint32(b[:]))), nil
		case tUint64:
			return float64(le.Uint64(b[:])), nil
		case tInt64:
			return float64(int64(le.Uint64(b[:]))), nil
		default:
			return math.Float64frombits(le.Uint64(b[:])), nil
		}
	}
	switch t {
	case tString:
		return r.str()
	case tArray:
		if depth >= maxDepth {
			return nil, errors.New("arrays nested too deeply")
		}
		et, err := r.u32()
		if err != nil {
			return nil, err
		}
		n, err := r.u64()
		if err != nil {
			return nil, err
		}
		if n > MaxArray {
			if err := r.skipElems(et, n, depth+1); err != nil {
				return nil, err
			}
			return ArrayInfo{Type: et, Len: n}, nil
		}
		out := make([]any, 0, n)
		for i := uint64(0); i < n; i++ {
			v, err := r.value(et, depth+1)
			if err != nil {
				return nil, err
			}
			out = append(out, v)
		}
		return out, nil
	}
	return nil, fmt.Errorf("unknown value type %d", t)
}

// skipElems moves past n array elements of type t without keeping them.
func (r reader) skipElems(t uint32, n uint64, depth int) error {
	if sz := fixedSize(t); sz > 0 {
		if n > math.MaxUint64/sz {
			return errors.New("array too long")
		}
		return r.discard(n * sz)
	}
	switch t {
	case tString:
		for i := uint64(0); i < n; i++ {
			l, err := r.strLen()
			if err != nil {
				return err
			}
			if err := r.discard(l); err != nil {
				return err
			}
		}
		return nil
	case tArray:
		if depth >= maxDepth {
			return errors.New("arrays nested too deeply")
		}
		for i := uint64(0); i < n; i++ {
			et, err := r.u32()
			if err != nil {
				return err
			}
			c, err := r.u64()
			if err != nil {
				return err
			}
			if err := r.skipElems(et, c, depth+1); err != nil {
				return err
			}
		}
		return nil
	}
	return fmt.Errorf("unknown value type %d", t)
}

func parse(br *bufio.Reader) (*File, error) {
	r := reader{br}
	var magic [4]byte
	if _, err := io.ReadFull(br, magic[:]); err != nil {
		return nil, fmt.Errorf("reading header: %w", err)
	}
	if string(magic[:]) != "GGUF" {
		return nil, errors.New("not a GGUF file")
	}
	ver, err := r.u32()
	if err != nil {
		return nil, err
	}
	switch {
	case ver == 1:
		return nil, errors.New("GGUF v1 isn't supported")
	case ver != 2 && ver != 3:
		return nil, fmt.Errorf("unsupported GGUF version %d", ver)
	}
	nt, err := r.u64()
	if err != nil {
		return nil, err
	}
	nkv, err := r.u64()
	if err != nil {
		return nil, err
	}
	if nt > maxTensors || nkv > maxKV {
		return nil, fmt.Errorf("implausible counts: %d tensors, %d keys", nt, nkv)
	}
	f := &File{Version: ver, Meta: make(map[string]any, nkv)}
	for i := uint64(0); i < nkv; i++ {
		k, err := r.str()
		if err != nil {
			return nil, fmt.Errorf("key %d: %w", i, err)
		}
		t, err := r.u32()
		if err != nil {
			return nil, fmt.Errorf("key %q: %w", k, err)
		}
		v, err := r.value(t, 0)
		if err != nil {
			return nil, fmt.Errorf("key %q: %w", k, err)
		}
		f.Meta[k] = v
	}
	f.Tensors = make([]Tensor, 0, min(nt, 4096))
	for i := uint64(0); i < nt; i++ {
		name, err := r.str()
		if err != nil {
			return nil, fmt.Errorf("tensor %d: %w", i, err)
		}
		nd, err := r.u32()
		if err != nil {
			return nil, fmt.Errorf("tensor %q: %w", name, err)
		}
		if nd > maxDims {
			return nil, fmt.Errorf("tensor %q has %d dimensions", name, nd)
		}
		dims := make([]uint64, nd)
		for j := range dims {
			if dims[j], err = r.u64(); err != nil {
				return nil, fmt.Errorf("tensor %q: %w", name, err)
			}
		}
		typ, err := r.u32()
		if err != nil {
			return nil, fmt.Errorf("tensor %q: %w", name, err)
		}
		if _, err := r.u64(); err != nil { // data offset, not needed
			return nil, fmt.Errorf("tensor %q: %w", name, err)
		}
		f.Tensors = append(f.Tensors, Tensor{Name: name, Dims: dims, Type: typ})
	}
	return f, nil
}

// Arch returns general.architecture.
func (f *File) Arch() string { return f.String("general.architecture") }

// Name returns general.name.
func (f *File) Name() string { return f.String("general.name") }

// String returns a string metadata value, or "" if missing or not a string.
func (f *File) String(key string) string {
	s, _ := f.Meta[key].(string)
	return s
}

// Uint returns a numeric metadata value, or 0 if missing or not numeric.
func (f *File) Uint(key string) uint64 {
	v, ok := f.Meta[key].(float64)
	if !ok || v < 0 {
		return 0
	}
	return uint64(v)
}

// ArchUint returns Uint(Arch()+"."+key), e.g. ArchUint("block_count").
func (f *File) ArchUint(key string) uint64 { return f.Uint(f.Arch() + "." + key) }

// ContextLength returns <arch>.context_length.
func (f *File) ContextLength() int { return int(f.ArchUint("context_length")) }

// BlockCount returns <arch>.block_count.
func (f *File) BlockCount() int { return int(f.ArchUint("block_count")) }

// ExpertCount returns <arch>.expert_count: how many experts a mixture-of-
// experts model routes between. 0 for a dense model.
func (f *File) ExpertCount() int { return int(f.ArchUint("expert_count")) }

// ChatTemplate returns tokenizer.chat_template, or "" if there is none.
func (f *File) ChatTemplate() string { return f.String("tokenizer.chat_template") }

// ParamCount is the total number of weights: the sum of each tensor's dimensions multiplied.
func (f *File) ParamCount() uint64 {
	var total uint64
	for _, t := range f.Tensors {
		n := uint64(1)
		for _, d := range t.Dims {
			n *= d
		}
		total += n
	}
	return total
}

// SizeLabel returns general.size_label if present, else a label like "8.9B" or "671M".
func (f *File) SizeLabel() string {
	if s := f.String("general.size_label"); s != "" {
		return s
	}
	n := float64(f.ParamCount())
	switch {
	case n >= 1e12:
		return fmt.Sprintf("%.1fT", n/1e12)
	case n >= 1e9:
		return fmt.Sprintf("%.1fB", n/1e9)
	case n >= 1e6:
		return fmt.Sprintf("%.0fM", n/1e6)
	case n >= 1e3:
		return fmt.Sprintf("%.0fK", n/1e3)
	}
	return fmt.Sprintf("%.0f", n)
}

var embeddingArchs = map[string]bool{
	"bert": true, "nomic-bert": true, "nomic-bert-moe": true, "jina-bert-v2": true,
	"jina-bert-v3": true, "modern-bert": true, "neo-bert": true, "eurobert": true,
	"t5encoder": true,
}

// IsEmbedding reports whether this is an embedding model: it has a pooling type,
// or its architecture is an encoder-only one like bert.
func (f *File) IsEmbedding() bool {
	if f.IsProjector() {
		return false
	}
	arch := f.Arch()
	if _, ok := f.Meta[arch+".pooling_type"]; ok {
		return true
	}
	return embeddingArchs[arch]
}

// IsProjector reports whether this is a vision/audio projector (mmproj).
func (f *File) IsProjector() bool {
	if f.Arch() == "clip" {
		return true
	}
	for k := range f.Meta {
		if strings.HasPrefix(k, "clip.") {
			return true
		}
	}
	return false
}

// IsLoRA reports whether this is a LoRA adapter (general.type "adapter", adapter.type
// "lora"): a small file llama.cpp applies on top of a base model of the same architecture.
func (f *File) IsLoRA() bool {
	return f.String("general.type") == "adapter" && f.String("adapter.type") == "lora"
}

// ModelInfo is Meta without tokenizer.* keys and without skipped arrays: the shape
// Ollama's /api/show model_info had.
func (f *File) ModelInfo() map[string]any {
	out := make(map[string]any, len(f.Meta))
	for k, v := range f.Meta {
		if strings.HasPrefix(k, "tokenizer.") {
			continue
		}
		if _, skipped := v.(ArrayInfo); skipped {
			continue
		}
		out[k] = v
	}
	return out
}

// VocabSize is the number of tokens in the tokenizer's vocabulary, 0 if
// the file has none.
func (f *File) VocabSize() int {
	switch t := f.Meta["tokenizer.ggml.tokens"].(type) {
	case []any:
		return len(t)
	case ArrayInfo:
		return int(t.Len)
	}
	return 0
}

// SameVocab reports whether two files tokenize alike, which a draft model
// for speculative decoding must: same tokenizer kind and pre-tokenizer,
// same vocabulary size and the same special tokens.
func (f *File) SameVocab(o *File) bool {
	n := f.VocabSize()
	if n == 0 || n != o.VocabSize() {
		return false
	}
	for _, k := range []string{"tokenizer.ggml.model", "tokenizer.ggml.pre"} {
		if f.String(k) != o.String(k) {
			return false
		}
	}
	for _, k := range []string{"tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id"} {
		if f.Uint(k) != o.Uint(k) {
			return false
		}
	}
	return true
}

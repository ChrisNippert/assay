package hw

import (
	"encoding/json"
	"testing"
)

func info(t *testing.T, s string) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal([]byte(s), &m); err != nil {
		t.Fatal(err)
	}
	return m
}

func TestKVCacheHybrid(t *testing.T) {
	// Qwen3.5-9B: 32 layers, attention on every 4th, 4 KV heads of 256+256.
	k, ok := KVCacheFromInfo(info(t, `{
		"general.architecture": "qwen35", "qwen35.block_count": 32,
		"qwen35.attention.head_count": 16, "qwen35.attention.head_count_kv": 4,
		"qwen35.attention.key_length": 256, "qwen35.attention.value_length": 256,
		"qwen35.full_attention_interval": 4}`))
	if !ok || k.PerToken != 8*4*512*2 || k.Window != 0 {
		t.Fatalf("%+v %v", k, ok)
	}
	// Measured on a real GPU: 4k → 32k context added ~0.97 GB.
	if grow := k.Bytes(32768) - k.Bytes(4096); grow < 900e6 || grow > 1000e6 {
		t.Fatalf("growth %d", grow)
	}
}

func TestKVCacheSlidingWindow(t *testing.T) {
	// Gemma 4 style: per-layer KV heads and an explicit SWA pattern.
	heads := `[8,8,8,8,8,1,8,8,8,8,8,1]`
	pattern := `[true,true,true,true,true,false,true,true,true,true,true,false]`
	k, ok := KVCacheFromInfo(info(t, `{
		"general.architecture": "gemma4", "gemma4.block_count": 12,
		"gemma4.attention.head_count": 16, "gemma4.attention.head_count_kv": `+heads+`,
		"gemma4.attention.key_length": 512, "gemma4.attention.value_length": 512,
		"gemma4.attention.key_length_swa": 256, "gemma4.attention.value_length_swa": 256,
		"gemma4.attention.sliding_window": 1024,
		"gemma4.attention.sliding_window_pattern": `+pattern+`}`))
	if !ok || k.PerToken != 2*1*1024*2 || k.WindowPerToken != 10*8*512*2 || k.Window != 1024 {
		t.Fatalf("%+v %v", k, ok)
	}
	// Past the window, only the two full layers keep growing.
	if k.Bytes(8192)-k.Bytes(4096) != 4096*k.PerToken {
		t.Fatal("window layers should stop growing")
	}
}

func TestKVCacheFallbacks(t *testing.T) {
	// No key_length: derived from embedding/heads. No head_count_kv: MHA.
	k, ok := KVCacheFromInfo(info(t, `{
		"general.architecture": "llama", "llama.block_count": 2,
		"llama.attention.head_count": 4, "llama.embedding_length": 256}`))
	if !ok || k.PerToken != 2*4*(64+64)*2 {
		t.Fatalf("%+v %v", k, ok)
	}
	// Gemma 3 has a window but no pattern list: every 6th layer is full.
	k, ok = KVCacheFromInfo(info(t, `{
		"general.architecture": "gemma3", "gemma3.block_count": 12,
		"gemma3.attention.head_count": 8, "gemma3.attention.head_count_kv": 4,
		"gemma3.attention.key_length": 128, "gemma3.attention.value_length": 128,
		"gemma3.attention.sliding_window": 512}`))
	if !ok || k.PerToken != 2*4*256*2 || k.WindowPerToken != 10*4*256*2 {
		t.Fatalf("gemma3: %+v %v", k, ok)
	}
	if _, ok := KVCacheFromInfo(map[string]any{}); ok {
		t.Fatal("empty info should fail")
	}
}

func TestKVScale(t *testing.T) {
	k := KVCache{PerToken: 6400, WindowPerToken: 640, Window: 1024}
	if got := k.Scale(KVTypeFactor("f16")); got != k {
		t.Errorf("f16 changed the cache: %+v", got)
	}
	q8 := k.Scale(KVTypeFactor("q8_0"))
	if q8.PerToken != 3400 || q8.WindowPerToken != 340 || q8.Window != 1024 {
		t.Errorf("q8_0 = %+v", q8)
	}
	if q4 := k.Scale(KVTypeFactor("q4_0")); q4.PerToken != 1800 {
		t.Errorf("q4_0 per token = %d, want 1800", q4.PerToken)
	}
}

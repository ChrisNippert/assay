package hw

// KVCache describes how much memory a model's attention cache needs as the
// context grows. Layers that see every token grow without limit; sliding-
// window layers stop growing at the window size.
type KVCache struct {
	PerToken       int64 // bytes per context token across full-attention layers
	WindowPerToken int64 // bytes per token across sliding-window layers
	Window         int   // sliding-window size in tokens, 0 if none
}

// Bytes is the cache size at a context length.
func (k KVCache) Bytes(ctx int) int64 {
	w := ctx
	if k.Window > 0 && k.Window < w {
		w = k.Window
	}
	return k.PerToken*int64(ctx) + k.WindowPerToken*int64(w)
}

// kvBytesPerValue assumes an f16 cache, Ollama's default. A q8_0 cache
// (OLLAMA_KV_CACHE_TYPE) would be about half.
const kvBytesPerValue = 2

// KVCacheFromInfo works out the cache shape from the GGUF metadata Ollama
// returns as model_info in /api/show. It understands per-layer KV head
// counts, sliding-window layers (Gemma) and hybrid models where only every
// Nth layer uses attention (Qwen3.5/Qwen3-Next). ok is false when the
// metadata is missing the basics.
func KVCacheFromInfo(info map[string]any) (k KVCache, ok bool) {
	arch, _ := info["general.architecture"].(string)
	if arch == "" {
		return k, false
	}
	get := func(key string) (float64, bool) {
		v, ok := info[arch+"."+key].(float64)
		return v, ok
	}
	layers, ok := get("block_count")
	if !ok || layers <= 0 {
		return k, false
	}
	n := int(layers)

	kvHeads := perLayer(info[arch+".attention.head_count_kv"], n)
	if kvHeads == nil {
		// Older models omit it when every head has its own KV (no GQA).
		kvHeads = perLayer(info[arch+".attention.head_count"], n)
	}
	if kvHeads == nil {
		return k, false
	}

	keyLen, ok1 := get("attention.key_length")
	valLen, ok2 := get("attention.value_length")
	if !ok1 || !ok2 {
		emb, ok3 := get("embedding_length")
		heads, ok4 := get("attention.head_count")
		if !ok3 || !ok4 || heads == 0 {
			return k, false
		}
		keyLen, valLen = emb/heads, emb/heads
	}
	keySWA, okK := get("attention.key_length_swa")
	valSWA, okV := get("attention.value_length_swa")
	if !okK || !okV {
		keySWA, valSWA = keyLen, valLen
	}

	window, _ := get("attention.sliding_window")
	swa := slidingLayers(info[arch+".attention.sliding_window_pattern"], arch, n, window > 0)
	interval, _ := get("full_attention_interval")

	for i := 0; i < n; i++ {
		if interval > 1 && (i+1)%int(interval) != 0 {
			continue // a recurrent (SSM) layer with a small fixed state, no KV cache
		}
		if swa[i] {
			k.WindowPerToken += int64(kvHeads[i] * (keySWA + valSWA) * kvBytesPerValue)
		} else {
			k.PerToken += int64(kvHeads[i] * (keyLen + valLen) * kvBytesPerValue)
		}
	}
	if k.WindowPerToken > 0 {
		k.Window = int(window)
	}
	return k, k.PerToken+k.WindowPerToken > 0
}

// perLayer expands a value that's either one number for all layers or a
// list with one number per layer.
func perLayer(v any, n int) []float64 {
	out := make([]float64, n)
	switch v := v.(type) {
	case float64:
		for i := range out {
			out[i] = v
		}
	case []any:
		if len(v) != n {
			return nil
		}
		for i, x := range v {
			f, ok := x.(float64)
			if !ok {
				return nil
			}
			out[i] = f
		}
	default:
		return nil
	}
	return out
}

// slidingLayers reports which layers use a sliding window. GGUF files store
// it as a per-layer bool list; older Gemma files don't, so their fixed
// patterns are filled in. Unknown models count every layer as full
// attention, which over-estimates rather than under-estimates.
func slidingLayers(pattern any, arch string, n int, hasWindow bool) []bool {
	out := make([]bool, n)
	if !hasWindow {
		return out
	}
	if list, ok := pattern.([]any); ok && len(list) == n {
		for i, x := range list {
			out[i], _ = x.(bool)
		}
		return out
	}
	every := map[string]int{"gemma2": 2, "gemma3": 6}[arch] // every Nth layer is full attention
	if every == 0 {
		return out
	}
	for i := range out {
		out[i] = (i+1)%every != 0
	}
	return out
}

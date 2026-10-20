package ollama

import "testing"

func TestParseEnvironment(t *testing.T) {
	got := ParseEnvironment(`PATH=/usr/bin OLLAMA_FLASH_ATTENTION=1 OLLAMA_HOST=0.0.0.0:11434 "OLLAMA_ORIGINS=*" "OLLAMA_MODELS=/srv/my models"`)
	want := map[string]string{
		"OLLAMA_FLASH_ATTENTION": "1",
		"OLLAMA_HOST":            "0.0.0.0:11434",
		"OLLAMA_ORIGINS":         "*",
		"OLLAMA_MODELS":          "/srv/my models",
	}
	if len(got) != len(want) {
		t.Fatalf("got %v", got)
	}
	for k, v := range want {
		if got[k] != v {
			t.Errorf("%s = %q, want %q", k, got[k], v)
		}
	}
}

func TestKVCacheType(t *testing.T) {
	cases := []struct {
		vars map[string]string
		want string
	}{
		{map[string]string{}, "f16"},
		{map[string]string{"OLLAMA_KV_CACHE_TYPE": "q8_0"}, "q8_0"},
		{map[string]string{"OLLAMA_KV_CACHE_TYPE": "q8_0", "OLLAMA_FLASH_ATTENTION": "0"}, "f16"},
		{map[string]string{"OLLAMA_KV_CACHE_TYPE": "Q4_0", "OLLAMA_FLASH_ATTENTION": "1"}, "q4_0"},
	}
	for _, c := range cases {
		if got := (ServerEnv{Vars: c.vars}).KVCacheType(); got != c.want {
			t.Errorf("%v: got %s, want %s", c.vars, got, c.want)
		}
	}
}

func TestHostPort(t *testing.T) {
	for in, want := range map[string]string{"": "11434", "0.0.0.0:11434": "11434", "127.0.0.1:11500": "11500", "http://[::1]:9000": "9000", "localhost": "11434"} {
		if got := hostPort(in); got != want {
			t.Errorf("hostPort(%q) = %q, want %q", in, got, want)
		}
	}
}

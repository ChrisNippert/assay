package engine

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"
)

func put(t *testing.T, p, data string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
}

func sha(c byte) string { return strings.Repeat(string(c), 64) }

// fakeHFCache lays out a hub cache: blobs named by sha256, snapshots of
// symlinks (or real files when link is false).
func fakeHFCache(t *testing.T, link bool) string {
	root := filepath.Join(t.TempDir(), "hub")
	repo := filepath.Join(root, "models--unsloth--Qwen3-8B-GGUF")
	file := func(rev, rel string, h byte, data string) {
		blob := filepath.Join(repo, "blobs", sha(h))
		snap := filepath.Join(repo, "snapshots", rev, rel)
		if link {
			put(t, blob, data)
			os.MkdirAll(filepath.Dir(snap), 0o755)
			if err := os.Symlink(blob, snap); err != nil {
				t.Skip("no symlinks here")
			}
		} else {
			put(t, snap, data)
		}
	}
	file("rev1", "Qwen3-8B-Q4_K_M.gguf", 'a', "q4 weights")
	file("rev2", "Qwen3-8B-Q4_K_M.gguf", 'a', "q4 weights") // same file, newer revision
	file("rev2", "Q8_0/Qwen3-8B-Q8_0-00001-of-00002.gguf", 'b', "q8 part one")
	file("rev2", "Q8_0/Qwen3-8B-Q8_0-00002-of-00002.gguf", 'c', "q8 part two")
	file("rev2", "mmproj-F16.gguf", 'd', "projector")
	file("rev2", "dspark-Qwen3-8B-Q8_0.gguf", 'e', "a draft model")
	file("rev2", "Qwen3-8B-Q2_K-00001-of-00003.gguf", 'f', "incomplete split set")
	put(t, filepath.Join(repo, "blobs", sha('9')+".incomplete"), "half a download")
	put(t, filepath.Join(repo, "refs", "main"), "rev2")
	// An old repo with a file still downloading, and a non-GGUF one.
	put(t, filepath.Join(root, "models--Systran--faster-whisper-base.en", "snapshots", "r", "model.bin"), "x")
	return root
}

func byName(cs []Candidate) map[string]Candidate {
	m := map[string]Candidate{}
	for _, c := range cs {
		m[c.Name] = c
	}
	return m
}

func names(cs []Candidate) []string {
	var out []string
	for _, c := range cs {
		out = append(out, c.Name)
	}
	sort.Strings(out)
	return out
}

func TestScanHFCache(t *testing.T) {
	for _, link := range []bool{true, false} {
		root := fakeHFCache(t, link)
		cs := scanHFCache(root)
		got := strings.Join(names(cs), " ")
		want := "hf.co/unsloth/Qwen3-8B-GGUF:Q4_K_M hf.co/unsloth/Qwen3-8B-GGUF:Q8_0"
		if got != want {
			t.Fatalf("link=%v names = %q, want %q (drafts, partial splits and non-GGUF repos are left out)", link, got, want)
		}
		m := byName(cs)
		q8 := m["hf.co/unsloth/Qwen3-8B-GGUF:Q8_0"]
		if len(q8.Files) != 2 || !strings.HasSuffix(q8.Rels[0], "00001-of-00002.gguf") || q8.Quant != "Q8_0" || q8.Repo != "unsloth/Qwen3-8B-GGUF" {
			t.Errorf("q8 = %+v", q8)
		}
		if q8.Projector == "" || q8.Size != int64(len("q8 part one")+len("q8 part two")+len("projector")) {
			t.Errorf("projector/size = %q %d", q8.Projector, q8.Size)
		}
		q4 := m["hf.co/unsloth/Qwen3-8B-GGUF:Q4_K_M"]
		if link && (q4.SHAs[0] != sha('a') || !strings.HasSuffix(q4.Files[0], sha('a'))) {
			t.Errorf("q4 should resolve to the blob: %+v", q4)
		}
		if !link && q4.SHAs[0] != "" {
			t.Errorf("no-symlink mode has no hash to go by: %+v", q4)
		}
	}
}

func TestScanLlamaCache(t *testing.T) {
	root := t.TempDir()
	put(t, filepath.Join(root, "unsloth_Qwen3-8B-GGUF_Qwen3-8B-Q4_K_M.gguf"), "q4")
	put(t, filepath.Join(root, "unsloth_Qwen3-8B-GGUF_Qwen3-8B-Q4_K_M.gguf.json"),
		`{"url":"https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf","etag":"\"x\""}`)
	put(t, filepath.Join(root, "unsloth_Qwen3-8B-GGUF_Qwen3-8B-Q6_K.gguf"), "half")
	put(t, filepath.Join(root, "unsloth_Qwen3-8B-GGUF_Qwen3-8B-Q6_K.gguf.json"),
		`{"url":"https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q6_K.gguf"}`)
	put(t, filepath.Join(root, "unsloth_Qwen3-8B-GGUF_Qwen3-8B-Q6_K.gguf.downloadInProgress"), "")
	put(t, filepath.Join(root, "old-thing-Q5_K_M.gguf"), "no sidecar")
	// The newer layout inside the same folder.
	put(t, filepath.Join(root, "models--ggml-org--tiny", "snapshots", "r", "tiny-Q8_0.gguf"), "tiny")
	got := strings.Join(names(scanLlamaCache(root)), " ")
	want := "hf.co/ggml-org/tiny:Q8_0 hf.co/unsloth/Qwen3-8B-GGUF:Q4_K_M old-thing:Q5_K_M"
	if got != want {
		t.Errorf("names = %q, want %q", got, want)
	}
}

func TestScanLMStudio(t *testing.T) {
	root := t.TempDir()
	put(t, filepath.Join(root, "lmstudio-community", "gemma-3-4b-it-GGUF", "gemma-3-4b-it-Q4_K_M.gguf"), "g")
	put(t, filepath.Join(root, "lmstudio-community", "gemma-3-4b-it-GGUF", "mmproj-model-f16.gguf"), "p")
	put(t, filepath.Join(root, "my models", "some thing", "Weird_Name.gguf"), "w")
	cs := scanLMStudio(root)
	if got := strings.Join(names(cs), " "); got != "hf.co/lmstudio-community/gemma-3-4b-it-GGUF:Q4_K_M some thing:Weird_Name" {
		t.Fatalf("names = %q", got)
	}
	if byName(cs)["hf.co/lmstudio-community/gemma-3-4b-it-GGUF:Q4_K_M"].Projector == "" {
		t.Error("projector not attached")
	}
}

func TestScanFolder(t *testing.T) {
	root := t.TempDir()
	put(t, filepath.Join(root, "Llama-3.2-1B-Instruct-Q4_K_M.gguf"), "a")
	put(t, filepath.Join(root, "Llama-3.2-1B-Instruct-Q8_0.gguf"), "b") // same folder, other model
	put(t, filepath.Join(root, "sub", "deeper", "phi-4-IQ2_XXS.gguf"), "c")
	put(t, filepath.Join(root, "sub", "big-Q4_K_M-00001-of-00002.gguf"), "d")
	put(t, filepath.Join(root, "sub", "big-Q4_K_M-00002-of-00002.gguf"), "e")
	put(t, filepath.Join(root, "a", "b", "c", "d", "e", "toodeep-Q4_0.gguf"), "f")
	put(t, filepath.Join(root, "notes.txt"), "no")
	got := strings.Join(names(scanFolder(root)), " ")
	want := "Llama-3.2-1B-Instruct:Q4_K_M Llama-3.2-1B-Instruct:Q8_0 big:Q4_K_M phi-4:IQ2_XXS"
	if got != want {
		t.Errorf("names = %q, want %q", got, want)
	}
}

func TestSourceDirs(t *testing.T) {
	dirs := func(env sourceEnv) map[string]string {
		m := map[string]string{}
		for _, d := range sourceDefs(env) {
			m[d.id] = strings.Join(d.dirs, "|")
		}
		return m
	}
	getenv := func(kv ...string) func(string) string {
		return func(k string) string {
			for i := 0; i+1 < len(kv); i += 2 {
				if kv[i] == k {
					return kv[i+1]
				}
			}
			return ""
		}
	}
	p := filepath.Join
	l := dirs(sourceEnv{GOOS: "linux", Home: "/h", Getenv: getenv()})
	if l["hf-cache"] != p("/h", ".cache", "huggingface", "hub") || l["llama.cpp"] != p("/h", ".cache", "llama.cpp") ||
		l["lmstudio"] != p("/h", ".lmstudio", "models")+"|"+p("/h", ".cache", "lm-studio", "models") {
		t.Errorf("linux = %v", l)
	}
	if h := dirs(sourceEnv{GOOS: "linux", Home: "/h", Getenv: getenv("HF_HOME", "/hf", "LLAMA_CACHE", "/lc")}); h["hf-cache"] != p("/hf", "hub") || h["llama.cpp"] != "/lc" {
		t.Errorf("HF_HOME/LLAMA_CACHE = %v", h)
	}
	if h := dirs(sourceEnv{GOOS: "linux", Home: "/h", Getenv: getenv("HF_HOME", "/hf", "HF_HUB_CACHE", "/hub")}); h["hf-cache"] != "/hub" {
		t.Errorf("HF_HUB_CACHE wins: %v", h)
	}
	if m := dirs(sourceEnv{GOOS: "darwin", Home: "/Users/a", Getenv: getenv()}); m["llama.cpp"] != p("/Users/a", "Library", "Caches", "llama.cpp") {
		t.Errorf("darwin = %v", m)
	}
	if w := dirs(sourceEnv{GOOS: "windows", Home: `C:\Users\a`, Getenv: getenv("LOCALAPPDATA", `C:\L`, "XDG_CACHE_HOME", "/ignored")}); w["llama.cpp"] != p(`C:\L`, "llama.cpp") || strings.Contains(w["hf-cache"], "ignored") || strings.Contains(w["lmstudio"], "lm-studio") {
		t.Errorf("windows = %v", w)
	}
}

// tree lists every file under dir with size, mtime and content.
func tree(t *testing.T, dir string) string {
	var b strings.Builder
	filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		info, _ := os.Lstat(p)
		b.WriteString(p + " " + info.Mode().String() + " " + info.ModTime().String())
		if d.Type().IsRegular() {
			data, _ := os.ReadFile(p)
			b.WriteString(" " + string(data))
		}
		b.WriteString("\n")
		return nil
	})
	return b.String()
}

func TestImportFoundHardLink(t *testing.T) {
	root := fakeHFCache(t, true)
	st, _ := OpenStore(t.TempDir())
	before := tree(t, root)
	cs := scanHFCache(root)
	n, err := st.ImportFound("hf-cache", cs)
	if err != nil || n != 2 {
		t.Fatalf("import = %d, %v", n, err)
	}
	if tree(t, root) != before {
		t.Error("the source was changed")
	}
	m, err := st.Get("hf.co/unsloth/Qwen3-8B-GGUF:q4_k_m")
	if err != nil {
		t.Fatal(err)
	}
	if m.Source != "hf-cache" || m.Link != "hard" || m.Quant != "Q4_K_M" || m.Digest != sha('a') || m.Files[0] != "sha256-"+sha('a')+".gguf" {
		t.Errorf("q4 = %+v", m)
	}
	src, _ := os.Stat(filepath.Join(root, "models--unsloth--Qwen3-8B-GGUF", "blobs", sha('a')))
	dst, _ := os.Stat(st.Blob(m.Files[0]))
	if !os.SameFile(src, dst) {
		t.Error("not a hard link of the cache's blob")
	}
	if info, _ := os.Lstat(st.Blob(m.Files[0])); info.Mode()&os.ModeSymlink != 0 {
		t.Error("should not be a symlink on one disk")
	}
	q8, _ := st.Get("hf.co/unsloth/Qwen3-8B-GGUF:Q8_0")
	if len(q8.Files) != 2 || filepath.Base(q8.Files[1]) != "Qwen3-8B-Q8_0-00002-of-00002.gguf" || filepath.Dir(q8.Files[0]) != filepath.Dir(q8.Files[1]) || q8.Projector == "" {
		t.Errorf("q8 = %+v", q8)
	}
	// Imported models no longer show up, nor does anything twice.
	f := NewFinder("")
	f.env = sourceEnv{GOOS: "linux", Home: t.TempDir(), Getenv: func(k string) string {
		if k == "HF_HUB_CACHE" {
			return root
		}
		return ""
	}}
	if s := f.Scan(st, true); len(s) != 0 {
		t.Errorf("already imported, got %+v", s)
	}
	if n, _ := st.ImportFound("hf-cache", cs); n != 0 {
		t.Errorf("second import added %d", n)
	}
	// Deleting only removes ollamui's link.
	if err := st.Delete(m.Name); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(root, "models--unsloth--Qwen3-8B-GGUF", "blobs", sha('a'))); err != nil {
		t.Errorf("the cache's file is gone: %v", err)
	}
	if tree(t, root) != before {
		t.Error("the source changed after delete")
	}
}

func TestImportFoundInPlaceAcrossDisks(t *testing.T) {
	hardLink = func(string, string) error { return errors.New("invalid cross-device link") }
	defer func() { hardLink = os.Link }()
	root := fakeHFCache(t, true)
	st, _ := OpenStore(t.TempDir())
	if n, err := st.ImportFound("hf-cache", scanHFCache(root)); err != nil || n != 2 {
		t.Fatalf("import = %d, %v", n, err)
	}
	m, _ := st.Get("hf.co/unsloth/Qwen3-8B-GGUF:Q4_K_M")
	if m.Link != "in place" || !filepath.IsAbs(m.Files[0]) || !strings.HasPrefix(m.Files[0], root) {
		t.Errorf("link = %q, files = %v", m.Link, m.Files)
	}
	// Used as found: the snapshot's name, not the hash-named blob, so a
	// split model's parts keep the names llama.cpp looks for.
	if filepath.Base(m.Files[0]) == sha('a') {
		t.Errorf("used the resolved blob %s, not the snapshot path", m.Files[0])
	}
	if b, err := os.ReadFile(st.Blob(m.Files[0])); err != nil || string(b) != "q4 weights" {
		t.Errorf("read in place: %q %v", b, err)
	}
	// No symlinks (they need admin rights on Windows), nothing in blobs/.
	if entries, _ := os.ReadDir(filepath.Join(st.Dir(), "blobs")); len(entries) != 0 {
		t.Errorf("blobs/ has %d entries", len(entries))
	}
	if len(st.Missing(m)) != 0 {
		t.Error("nothing is missing yet")
	}
	// Deleting it here forgets it and leaves the other app's file alone.
	src := filepath.Join(root, "models--unsloth--Qwen3-8B-GGUF", "blobs", sha('a'))
	if err := st.Delete(m.Name); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(src); err != nil {
		t.Fatalf("deleting the model removed the other app's file: %v", err)
	}
	// The other app deletes its file: a model still using it shows as missing.
	st.ImportFound("hf-cache", scanHFCache(root))
	m, _ = st.Get("hf.co/unsloth/Qwen3-8B-GGUF:Q4_K_M")
	os.Remove(src)
	if got := st.Missing(m); len(got) != 1 {
		t.Errorf("missing = %v", got)
	}
}

func TestDedupAgainstStore(t *testing.T) {
	root := fakeHFCache(t, true)
	st, _ := OpenStore(t.TempDir())
	// The same weights were downloaded here, under a different name.
	put(t, st.Blob("sha256-"+sha('a')+".gguf"), "q4 weights")
	st.Put(&Model{Name: "qwen3-8b:mine", Files: []string{"sha256-" + sha('a') + ".gguf"}, Size: 10, Source: "hf"})
	cs := scanHFCache(root)
	have := st.fileInfos()
	var left []string
	for _, c := range cs {
		if !st.hasCandidate(c, have) {
			left = append(left, c.Name)
		}
	}
	if strings.Join(left, " ") != "hf.co/unsloth/Qwen3-8B-GGUF:Q8_0" {
		t.Errorf("left = %v (Q4_K_M has the same sha256 as a blob we have)", left)
	}
	// A file hard-linked into the store from elsewhere is recognised too.
	folder := t.TempDir()
	put(t, filepath.Join(folder, "x-Q4_0.gguf"), "same bytes")
	os.Link(filepath.Join(folder, "x-Q4_0.gguf"), st.Blob("whatever.gguf"))
	st.Put(&Model{Name: "x", Files: []string{"whatever.gguf"}, Size: 10, Source: "hf"})
	fc := scanFolder(folder)
	if len(fc) != 1 || !st.hasCandidate(fc[0], st.fileInfos()) {
		t.Errorf("same inode should count as imported: %+v", fc)
	}
}

func TestFolderImportWithoutHash(t *testing.T) {
	folder := t.TempDir()
	put(t, filepath.Join(folder, "m", "Llama-3.2-1B-Q4_K_M.gguf"), "weights")
	before := tree(t, folder)
	st, _ := OpenStore(t.TempDir())
	if n, err := st.ImportFound("folder", scanFolder(folder)); err != nil || n != 1 {
		t.Fatalf("%d %v", n, err)
	}
	m, _ := st.Get("Llama-3.2-1B:Q4_K_M")
	if m == nil || m.Source != "folder" || !strings.HasPrefix(m.Files[0], "link-") || m.Digest == "" || m.Origin == "" {
		t.Fatalf("model = %+v", m)
	}
	if tree(t, folder) != before {
		t.Error("the folder was changed")
	}
}

func TestFinderFolders(t *testing.T) {
	st, _ := OpenStore(t.TempDir())
	file := filepath.Join(t.TempDir(), "cfg", "sources.json")
	f := NewFinder(file)
	folder := t.TempDir()
	put(t, filepath.Join(folder, "a-Q4_K_M.gguf"), "a")
	for _, bad := range []string{"relative/dir", filepath.Join(folder, "missing"), st.Dir(), filepath.Join(st.Dir(), "blobs")} {
		if f.AddFolder(st, bad) == nil {
			t.Errorf("%q should be refused", bad)
		}
	}
	if err := f.AddFolder(st, folder); err != nil {
		t.Fatal(err)
	}
	if got := NewFinder(file).Folders(); len(got) != 1 || got[0] != folder {
		t.Errorf("not remembered: %v", got)
	}
	var found *Source
	for _, s := range f.Scan(st, true) {
		if s.ID == "folder" && s.Dir == folder {
			found = &s
		}
	}
	if found == nil || len(found.Models) != 1 || found.Link != "hard" {
		t.Fatalf("folder source = %+v", found)
	}
	f.RemoveFolder(folder)
	if len(f.Folders()) != 0 {
		t.Error("not removed")
	}
	if _, err := os.Stat(filepath.Join(folder, "a-Q4_K_M.gguf")); err != nil {
		t.Error("removing a folder must leave it alone")
	}
}

func TestScanIsQuick(t *testing.T) {
	root := t.TempDir()
	for i := 0; i < 200; i++ {
		put(t, filepath.Join(root, "d"+string(rune('a'+i%20)), "m"+string(rune('a'+i%26))+string(rune('a'+i/26))+"-Q4_K_M.gguf"), "x")
	}
	start := time.Now()
	if n := len(scanFolder(root)); n != 200 {
		t.Errorf("found %d", n)
	}
	if d := time.Since(start); d > 2*time.Second {
		t.Errorf("slow: %v", d)
	}
}

func TestUnslothHubCaches(t *testing.T) {
	dir := t.TempDir()
	big := filepath.Join(dir, "big", "hub")
	os.MkdirAll(big, 0o755)
	m := filepath.Join(dir, "manifests", "cache-1")
	os.MkdirAll(m, 0o755)
	write := func(name, cache string) {
		os.WriteFile(filepath.Join(m, name), []byte(`{"repo_id":"unsloth/x","hub_cache":"`+cache+`"}`), 0o644)
	}
	write("a.json", big)
	write("b.json", big)
	write("c.json", filepath.Join(dir, "gone")) // not there any more
	got := unslothHubCaches(filepath.Join(dir, "manifests"))
	if len(got) != 1 || got[0] != big {
		t.Errorf("got %v", got)
	}
}

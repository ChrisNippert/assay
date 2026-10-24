package engine

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"sort"
	"strings"
	"sync"
	"time"

	"ollamui/internal/gguf"
	"ollamui/internal/hf"
)

// Candidate is a model found in another app's folder, which can be linked
// into the store. Nothing in the other folder is ever written to.
type Candidate struct {
	// Name is what the model would be called here.
	Name  string `json:"name"`
	Repo  string `json:"repo,omitempty"`
	Quant string `json:"quant,omitempty"`
	// Files are the GGUF files' real paths (symlinks resolved), in order.
	Files []string `json:"-"`
	// Paths are the files as found, before resolving symlinks. Used in
	// place, they keep the names llama.cpp needs to find a split model's
	// other parts (the Hugging Face cache's blobs are named by hash).
	Paths []string `json:"-"`
	// Rels are the files' names as the other app knows them; split models
	// keep them, because llama.cpp finds the other parts by name.
	Rels []string `json:"-"`
	// SHAs holds each file's sha256 when the source's layout gives it away
	// (the Hugging Face cache names blobs by hash), else "".
	SHAs         []string  `json:"-"`
	Projector    string    `json:"-"`
	ProjectorAt  string    `json:"-"` // the projector as found
	ProjectorSHA string    `json:"-"`
	MTP          string    `json:"-"` // the multi-token-prediction head, likewise
	MTPAt        string    `json:"-"`
	MTPSHA       string    `json:"-"`
	Size         int64     `json:"size"`
	Modified     time.Time `json:"modified"`
	// Adapter is set for a LoRA adapter, which is offered as one rather
	// than as a model; Arch is the base architecture it was trained for.
	Adapter bool   `json:"adapter,omitempty"`
	Arch    string `json:"arch,omitempty"`
}

// Source is one place models can be found: an app's folder.
type Source struct {
	ID    string `json:"id"` // "hf-cache", "llama.cpp", "lmstudio" or "folder"
	Title string `json:"title"`
	Dir   string `json:"dir"`
	// Link is what importing from here does: "hard" on the store's disk,
	// else "symlink".
	Link   string      `json:"link"`
	Models []Candidate `json:"models"`
}

// SourceTitle is the human name of a model's Source value.
func SourceTitle(id string) string {
	switch id {
	case "ollama":
		return "Ollama"
	case "hf-cache":
		return "Hugging Face cache"
	case "llama.cpp":
		return "llama.cpp's cache"
	case "lmstudio":
		return "LM Studio"
	case "unsloth":
		return "Unsloth Studio"
	case "folder":
		return "a folder"
	case "file":
		return "a file"
	}
	return ""
}

// sourceEnv is what the default folders depend on; tests fake it.
type sourceEnv struct {
	GOOS   string
	Home   string
	Getenv func(string) string
}

func hostEnv() sourceEnv {
	home, _ := os.UserHomeDir()
	return sourceEnv{GOOS: runtime.GOOS, Home: home, Getenv: os.Getenv}
}

type sourceDef struct {
	id   string
	dirs []string
	scan func(dir string) []Candidate
}

// sourceDefs lists the places to look, with their folders as the apps and
// their environment variables choose them.
func sourceDefs(env sourceEnv) []sourceDef {
	get := env.Getenv
	home := func(p ...string) string { return filepath.Join(append([]string{env.Home}, p...)...) }
	var hfDirs []string
	switch {
	case get("HF_HUB_CACHE") != "":
		hfDirs = []string{get("HF_HUB_CACHE")}
	case get("HF_HOME") != "":
		hfDirs = []string{filepath.Join(get("HF_HOME"), "hub")}
	case get("XDG_CACHE_HOME") != "" && env.GOOS != "windows":
		hfDirs = []string{filepath.Join(get("XDG_CACHE_HOME"), "huggingface", "hub")}
	default:
		hfDirs = []string{home(".cache", "huggingface", "hub")}
	}
	var llamaDirs []string
	switch {
	case get("LLAMA_CACHE") != "":
		llamaDirs = []string{get("LLAMA_CACHE")}
	case env.GOOS == "windows":
		llamaDirs = []string{filepath.Join(get("LOCALAPPDATA"), "llama.cpp")}
	case env.GOOS == "darwin":
		llamaDirs = []string{home("Library", "Caches", "llama.cpp")}
	case get("XDG_CACHE_HOME") != "":
		llamaDirs = []string{filepath.Join(get("XDG_CACHE_HOME"), "llama.cpp")}
	default:
		llamaDirs = []string{home(".cache", "llama.cpp")}
	}
	lm := []string{home(".lmstudio", "models")}
	if env.GOOS != "windows" {
		lm = append(lm, home(".cache", "lm-studio", "models"))
	}
	var unslothDirs []string
	for _, d := range unslothHubCaches(home(".unsloth", "studio", "cache", "hub-state", "manifests")) {
		if !slices.Contains(hfDirs, d) {
			unslothDirs = append(unslothDirs, d)
		}
	}
	return []sourceDef{
		{"hf-cache", hfDirs, scanHFCache},
		{"unsloth", unslothDirs, scanHFCache},
		{"llama.cpp", llamaDirs, scanLlamaCache},
		{"lmstudio", lm, scanLMStudio},
	}
}

// unslothHubCaches lists the Hugging Face caches Unsloth Studio downloaded
// into. Studio can be pointed at a cache of its own (say on a big disk)
// that no environment variable here names; its download records
// (hub-state/manifests/<cache>/<model>.json) say where, as "hub_cache".
func unslothHubCaches(manifests string) []string {
	paths, _ := filepath.Glob(filepath.Join(manifests, "*", "*.json"))
	var out []string
	for _, p := range paths {
		b, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		var m struct {
			HubCache string `json:"hub_cache"`
		}
		if json.Unmarshal(b, &m) != nil || m.HubCache == "" || slices.Contains(out, m.HubCache) {
			continue
		}
		if info, err := os.Stat(m.HubCache); err == nil && info.IsDir() {
			out = append(out, m.HubCache)
		}
	}
	sort.Strings(out)
	return out
}

// ---- scanning ----

var (
	hex64      = regexp.MustCompile(`^[0-9a-f]{64}$`)
	splitPart  = regexp.MustCompile(`-(\d{5})-of-(\d{5})$`)
	repoPart   = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]*$`)
	maxScanDep = 4
)

// gfile is a GGUF file found on disk.
type gfile struct {
	rel  string // slash path inside its unit (repo snapshot, folder...)
	path string // where it was found (maybe a symlink)
	abs  string // real path
	size int64
	sha  string
	mod  time.Time
}

// statReal follows symlinks to a non-empty regular file.
func statReal(p string) (real string, info fs.FileInfo, ok bool) {
	real, err := filepath.EvalSymlinks(p)
	if err != nil {
		return "", nil, false
	}
	info, err = os.Stat(real)
	if err != nil || !info.Mode().IsRegular() || info.Size() == 0 {
		return "", nil, false
	}
	return real, info, true
}

// ggufFiles lists the .gguf files under root, depth folders down at most
// (a name in root itself is depth 0). Symlinked files count; symlinked
// folders aren't followed.
func ggufFiles(root string, depth int) []gfile {
	var out []gfile
	filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		rel, _ := filepath.Rel(root, p)
		if d.IsDir() {
			if rel != "." && strings.Count(filepath.ToSlash(rel), "/") >= depth {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(strings.ToLower(d.Name()), ".gguf") {
			return nil
		}
		real, info, ok := statReal(p)
		if !ok {
			return nil
		}
		f := gfile{rel: filepath.ToSlash(rel), path: p, abs: real, size: info.Size(), mod: info.ModTime()}
		if b := filepath.Base(real); hex64.MatchString(b) {
			f.sha = b
		}
		out = append(out, f)
		return nil
	})
	return out
}

// completeSplit reports whether files (ordered) are a whole model: one
// file, or every part of a split set.
func completeSplit(files []gfile) bool {
	if len(files) == 0 {
		return false
	}
	m := splitPart.FindStringSubmatch(strings.TrimSuffix(path.Base(files[0].rel), path.Ext(files[0].rel)))
	if m == nil {
		return len(files) == 1
	}
	var want int
	fmt.Sscan(m[2], &want)
	return len(files) == want
}

func projectorOf(files []gfile) *gfile {
	want := hf.Projector(hfFiles(files))
	for i := range files {
		if want != "" && files[i].rel == want {
			return &files[i]
		}
	}
	return nil
}

// mtpOf finds the multi-token-prediction head among a repo's files.
func mtpOf(files []gfile) *gfile {
	want := hf.MTP(hfFiles(files))
	for i := range files {
		if want != "" && files[i].rel == want {
			return &files[i]
		}
	}
	return nil
}

func hfFiles(files []gfile) []hf.File {
	out := make([]hf.File, len(files))
	for i, f := range files {
		out[i] = hf.File{Type: "file", Path: f.rel, Size: f.size}
	}
	return out
}

// makeCandidate builds a candidate from a model's files (already ordered).
func makeCandidate(name, repo, quant string, files []gfile, proj, mtp *gfile) Candidate {
	c := Candidate{Name: name, Repo: repo, Quant: quant}
	for _, f := range files {
		c.Files = append(c.Files, f.abs)
		c.Paths = append(c.Paths, f.path)
		c.Rels = append(c.Rels, f.rel)
		c.SHAs = append(c.SHAs, f.sha)
		c.Size += f.size
		if f.mod.After(c.Modified) {
			c.Modified = f.mod
		}
	}
	if proj != nil {
		c.Projector, c.ProjectorAt, c.ProjectorSHA = proj.abs, proj.path, proj.sha
		c.Size += proj.size
	}
	if mtp != nil {
		c.MTP, c.MTPAt, c.MTPSHA = mtp.abs, mtp.path, mtp.sha
		c.Size += mtp.size
	}
	return c
}

// repoCandidates groups one Hugging Face repo's files into models named
// hf.co/<repo>:<QUANT>, as the downloader does. Draft models are left out.
func repoCandidates(repo string, files []gfile) []Candidate {
	byRel := map[string]gfile{}
	for _, f := range files {
		byRel[f.rel] = f
	}
	quants, _ := hf.Quants(repo, hfFiles(files))
	proj, mtp := projectorOf(files), mtpOf(files)
	var out []Candidate
	for _, q := range quants {
		if q.Draft {
			continue
		}
		var fs []gfile
		for _, p := range q.Files {
			fs = append(fs, byRel[p])
		}
		if completeSplit(fs) {
			out = append(out, makeCandidate(hf.OllamaName(repo, q.Name), repo, q.Name, fs, proj, mtp))
		}
	}
	return out
}

// looseCandidates groups files that aren't known to belong to a repo: by
// folder, then by name without the quant and split suffix. name turns the
// base name and quant into the model's name.
func looseCandidates(files []gfile, name func(base, quant string) string) []Candidate {
	type key struct{ dir, base string }
	groups := map[key][]gfile{}
	var order []key
	byDir := map[string][]gfile{}
	for _, f := range files {
		dir, file := path.Split(f.rel)
		byDir[dir] = append(byDir[dir], f)
		k := key{dir, splitPart.ReplaceAllString(strings.TrimSuffix(file, path.Ext(file)), "")}
		if _, ok := groups[k]; !ok {
			order = append(order, k)
		}
		groups[k] = append(groups[k], f)
	}
	sort.Slice(order, func(i, j int) bool {
		return order[i].dir+order[i].base < order[j].dir+order[j].base
	})
	var out []Candidate
	for _, k := range order {
		fs := groups[k]
		sort.Slice(fs, func(i, j int) bool { return fs[i].rel < fs[j].rel })
		lower := strings.ToLower(k.base)
		if strings.Contains(lower, "mmproj") || strings.Contains(lower, "imatrix") || strings.HasPrefix(lower, "mtp-") || hf.IsDraftFile("", k.base+".gguf") {
			continue
		}
		if !completeSplit(fs) {
			continue
		}
		base, quant := k.base, ""
		if qs, _ := hf.Quants("", hfFiles(fs)); len(qs) == 1 && !qs[0].Draft {
			quant = qs[0].Name
			if n := len(quant) + 1; len(base) > n {
				base = base[:len(base)-n]
			}
		}
		out = append(out, makeCandidate(name(base, quant), "", quant, fs, projectorOf(byDir[k.dir]), nil))
	}
	return out
}

// scanHFCache reads a Hugging Face hub cache: models--owner--repo/
// snapshots/<rev>/<path>, each file a symlink to blobs/<sha256> (or a real
// file where the system can't symlink). A file in several revisions counts
// once, from the revision refs/main names, else the newest.
func scanHFCache(root string) []Candidate {
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil
	}
	var out []Candidate
	for _, e := range entries {
		rest, ok := strings.CutPrefix(e.Name(), "models--")
		if !ok {
			continue
		}
		owner, name, ok := strings.Cut(rest, "--")
		if !ok || owner == "" || name == "" {
			continue
		}
		repoDir := filepath.Join(root, e.Name())
		snaps, _ := os.ReadDir(filepath.Join(repoDir, "snapshots"))
		type rev struct {
			name string
			mod  time.Time
		}
		main, _ := os.ReadFile(filepath.Join(repoDir, "refs", "main"))
		head := strings.TrimSpace(string(main))
		var revs []rev
		for _, s := range snaps {
			if i, err := s.Info(); err == nil && s.IsDir() {
				revs = append(revs, rev{s.Name(), i.ModTime()})
			}
		}
		sort.SliceStable(revs, func(i, j int) bool {
			if mi, mj := revs[i].name == head, revs[j].name == head; mi != mj {
				return mi
			}
			return revs[i].mod.After(revs[j].mod)
		})
		seen := map[string]bool{}
		var files []gfile
		for _, r := range revs {
			for _, f := range ggufFiles(filepath.Join(repoDir, "snapshots", r.name), maxScanDep) {
				if !seen[f.rel] {
					seen[f.rel] = true
					files = append(files, f)
				}
			}
		}
		out = append(out, repoCandidates(owner+"/"+name, files)...)
	}
	return out
}

// scanLlamaCache reads llama.cpp's cache. -hf used to save flat files
// "<owner>_<repo>_<file>.gguf" with a .json beside each naming the URL;
// newer builds use the Hugging Face layout inside the same folder. A file
// still downloading doesn't count.
func scanLlamaCache(root string) []Candidate {
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil
	}
	repos := map[string][]gfile{}
	var repoOrder []string
	var loose []gfile
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(strings.ToLower(e.Name()), ".gguf") {
			continue
		}
		p := filepath.Join(root, e.Name())
		if _, err := os.Stat(p + ".downloadInProgress"); err == nil {
			continue
		}
		real, info, ok := statReal(p)
		if !ok {
			continue
		}
		f := gfile{rel: e.Name(), path: p, abs: real, size: info.Size(), mod: info.ModTime()}
		if repo, file := llamaURL(p + ".json"); repo != "" {
			f.rel = file
			if _, ok := repos[repo]; !ok {
				repoOrder = append(repoOrder, repo)
			}
			repos[repo] = append(repos[repo], f)
			continue
		}
		loose = append(loose, f)
	}
	var out []Candidate
	for _, repo := range repoOrder {
		out = append(out, repoCandidates(repo, repos[repo])...)
	}
	out = append(out, looseCandidates(loose, plainName)...)
	if m, _ := filepath.Glob(filepath.Join(root, "models--*")); len(m) > 0 {
		out = append(out, scanHFCache(root)...)
	}
	return out
}

// llamaURL reads the sidecar llama.cpp writes beside a download:
// {"url":"https://huggingface.co/<owner>/<repo>/resolve/<rev>/<file>"}.
func llamaURL(sidecar string) (repo, file string) {
	b, err := os.ReadFile(sidecar)
	if err != nil {
		return "", ""
	}
	var j struct {
		URL string `json:"url"`
	}
	if json.Unmarshal(b, &j) != nil {
		return "", ""
	}
	u, err := url.Parse(j.URL)
	if err != nil || !strings.HasSuffix(u.Host, "huggingface.co") && !strings.HasSuffix(u.Host, "hf.co") {
		return "", ""
	}
	parts := strings.SplitN(strings.TrimPrefix(u.Path, "/"), "/", 5)
	if len(parts) < 5 || parts[2] != "resolve" {
		return "", ""
	}
	return parts[0] + "/" + parts[1], parts[4]
}

func plainName(base, quant string) string {
	if quant == "" {
		return base
	}
	return base + ":" + quant
}

// scanLMStudio reads LM Studio's models folder: <owner>/<repo>/*.gguf.
func scanLMStudio(root string) []Candidate {
	owners, err := os.ReadDir(root)
	if err != nil {
		return nil
	}
	var out []Candidate
	for _, o := range owners {
		repos, _ := os.ReadDir(filepath.Join(root, o.Name()))
		for _, r := range repos {
			if !o.IsDir() || !r.IsDir() {
				continue
			}
			files := ggufFiles(filepath.Join(root, o.Name(), r.Name()), 2)
			if len(files) == 0 {
				continue
			}
			owner, repo := o.Name(), r.Name()
			if repoPart.MatchString(owner) && repoPart.MatchString(repo) {
				if found := repoCandidates(owner+"/"+repo, files); len(found) > 0 {
					out = append(out, found...)
					continue
				}
			}
			out = append(out, looseCandidates(files, func(base, quant string) string {
				if quant == "" {
					return repo + ":" + base
				}
				return plainName(base, quant)
			})...)
		}
	}
	return out
}

// scanFolder reads any folder for .gguf files, a few levels down.
func scanFolder(root string) []Candidate {
	return looseCandidates(ggufFiles(root, maxScanDep), plainName)
}

// ---- the user's own folders, and the cached scan ----

// Finder looks for models in other apps' folders. Scans are cached for a
// short while; the folders the user added are kept in File.
type Finder struct {
	File string // sources.json in the config folder; "" keeps nothing
	env  sourceEnv

	mu      sync.Mutex
	folders []string
	loaded  bool
	cache   []Source
	at      time.Time
	// kinds remembers each file's header (is it a LoRA adapter?), so
	// scans don't read the same headers again.
	kinds map[string]fileKind
}

// fileKind is what a GGUF file's header says, valid while the file keeps
// its size and time.
type fileKind struct {
	size int64
	mod  time.Time
	lora bool
	arch string
}

// markAdapters flags the candidates that are LoRA adapters, by their GGUF
// headers. Called with f.mu held.
func (f *Finder) markAdapters(models []Candidate) {
	if f.kinds == nil {
		f.kinds = map[string]fileKind{}
	}
	for i := range models {
		c := &models[i]
		if len(c.Files) != 1 {
			continue // adapters are single files
		}
		info, err := os.Stat(c.Files[0])
		if err != nil {
			continue
		}
		k, ok := f.kinds[c.Files[0]]
		if !ok || k.size != info.Size() || !k.mod.Equal(info.ModTime()) {
			k = fileKind{size: info.Size(), mod: info.ModTime()}
			if g, err := gguf.Read(c.Files[0]); err == nil && g.IsLoRA() {
				k.lora, k.arch = true, g.Arch()
			}
			f.kinds[c.Files[0]] = k
		}
		if k.lora {
			c.Adapter, c.Arch = true, k.arch
		}
	}
}

// NewFinder returns a Finder that remembers added folders in file.
func NewFinder(file string) *Finder { return &Finder{File: file, env: hostEnv()} }

type sourcesFile struct {
	Folders []string `json:"folders"`
}

func (f *Finder) load() {
	if f.loaded {
		return
	}
	f.loaded = true
	if f.File == "" {
		return
	}
	var c sourcesFile
	if b, err := os.ReadFile(f.File); err == nil && json.Unmarshal(b, &c) == nil {
		f.folders = c.Folders
	}
}

func (f *Finder) save() error {
	if f.File == "" {
		return nil
	}
	b, err := json.MarshalIndent(sourcesFile{Folders: f.folders}, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(f.File), 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(f.File+".tmp", append(b, '\n'), 0o644); err != nil {
		return err
	}
	return os.Rename(f.File+".tmp", f.File)
}

// Folders lists the folders the user added.
func (f *Finder) Folders() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.load()
	return append([]string(nil), f.folders...)
}

// AddFolder remembers a folder to look in. It must exist and must not be
// the store itself.
func (f *Finder) AddFolder(store *Store, dir string) error {
	dir = strings.TrimSpace(dir)
	if rest, ok := strings.CutPrefix(dir, "~"); ok && (rest == "" || rest[0] == '/' || rest[0] == '\\') {
		home, _ := os.UserHomeDir()
		dir = home + rest
	}
	if !filepath.IsAbs(dir) {
		return errors.New("enter the folder's full path")
	}
	dir = filepath.Clean(dir)
	if info, err := os.Stat(dir); err != nil || !info.IsDir() {
		return errors.New("that folder isn't there")
	}
	if real, err := filepath.EvalSymlinks(store.Dir()); err == nil {
		if r, err := filepath.EvalSymlinks(dir); err == nil && (r == real || strings.HasPrefix(r, real+string(filepath.Separator))) {
			return errors.New("that's ollamui's own models folder")
		}
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	f.load()
	if contains(f.folders, dir) {
		return nil
	}
	f.folders = append(f.folders, dir)
	f.at = time.Time{}
	return f.save()
}

// RemoveFolder forgets a folder the user added. The folder itself is left alone.
func (f *Finder) RemoveFolder(dir string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.load()
	var keep []string
	for _, d := range f.folders {
		if d != dir {
			keep = append(keep, d)
		}
	}
	f.folders = keep
	f.at = time.Time{}
	return f.save()
}

const scanCacheFor = 30 * time.Second

// Scan lists what the other apps have that the store doesn't. Apps that
// aren't installed or have nothing new don't appear; folders the user
// added always do.
func (f *Finder) Scan(store *Store, refresh bool) []Source {
	f.mu.Lock()
	f.load()
	if refresh || f.cache == nil || time.Since(f.at) > scanCacheFor {
		f.cache = f.scanAll()
		f.at = time.Now()
	}
	all := f.cache
	f.mu.Unlock()
	return filterSources(store, all)
}

func (f *Finder) scanAll() []Source {
	out := []Source{}
	for _, d := range sourceDefs(f.env) {
		var dirs []string
		var models []Candidate
		for _, dir := range d.dirs {
			if info, err := os.Stat(dir); err == nil && info.IsDir() {
				dirs = append(dirs, dir)
				found := d.scan(dir)
				f.markAdapters(found)
				models = append(models, found...)
			}
		}
		if len(dirs) > 0 {
			out = append(out, Source{ID: d.id, Title: SourceTitle(d.id), Dir: dirs[0], Models: models})
		}
	}
	for _, dir := range f.folders {
		if info, err := os.Stat(dir); err == nil && info.IsDir() {
			models := scanFolder(dir)
			f.markAdapters(models)
			out = append(out, Source{ID: "folder", Title: "Folder", Dir: dir, Models: models})
		}
	}
	return out
}

// filterSources drops what the store already has and says how an import
// would link.
func filterSources(store *Store, all []Source) []Source {
	have := store.fileInfos()
	out := make([]Source, 0, len(all))
	for _, s := range all {
		cp := s
		cp.Models = nil
		for _, c := range s.Models {
			if !store.hasCandidate(c, have) {
				cp.Models = append(cp.Models, c)
			}
		}
		if len(cp.Models) == 0 && cp.ID != "folder" {
			continue
		}
		cp.Link = "hard"
		if !SameDisk(s.Dir, store.Dir()) {
			cp.Link = "in place"
		}
		out = append(out, cp)
	}
	return out
}

// FindCandidate returns the candidate called name from the source with
// that id and folder.
func FindCandidate(sources []Source, id, dir, name string) (Candidate, bool) {
	for _, s := range sources {
		if s.ID == id && (dir == "" || s.Dir == dir) {
			for _, c := range s.Models {
				if c.Name == name {
					return c, true
				}
			}
		}
	}
	return Candidate{}, false
}

// ---- importing ----

// fileInfos stats every file the store's models use.
func (s *Store) fileInfos() []os.FileInfo {
	var out []os.FileInfo
	for _, m := range s.everything() {
		for _, f := range m.AllFiles() {
			if info, err := os.Stat(s.Blob(f)); err == nil {
				out = append(out, info)
			}
		}
	}
	return out
}

// hasCandidate reports whether the store already has this model: by name,
// by the sha256 its blob would carry, or because its first file is one the
// store already uses (the same file on disk, however it got there).
func (s *Store) hasCandidate(c Candidate, have []os.FileInfo) bool {
	if _, err := s.Get(c.Name); err == nil {
		return true
	}
	if len(c.SHAs) == 1 && c.SHAs[0] != "" {
		if _, err := os.Stat(s.Blob("sha256-" + c.SHAs[0] + ".gguf")); err == nil {
			return true
		}
	}
	if len(c.Files) > 0 {
		if info, err := os.Stat(c.Files[0]); err == nil {
			for _, h := range have {
				if os.SameFile(info, h) {
					return true
				}
			}
		}
	}
	return false
}

func pathHash(p string) string {
	h := sha256.Sum256([]byte(p))
	return hex.EncodeToString(h[:8])
}

// ImportFound brings candidates found in a source into the store without
// copying: hard links on the store's disk, and on another disk the files
// are used where they are (their full paths go in the manifest), which
// needs no symlinks and so no admin rights on Windows. The other app's
// files are only read, and deleting the model here never touches them.
// It returns how many models it added.
func (s *Store) ImportFound(source string, cs []Candidate) (int, error) {
	added := 0
	for _, c := range cs {
		if len(c.Files) == 0 {
			continue
		}
		if _, err := s.Get(c.Name); err == nil {
			continue
		}
		m := &Model{Name: c.Name, Repo: c.Repo, Quant: c.Quant, Size: c.Size, Modified: c.Modified, Source: source, Origin: c.Files[0], Link: "hard"}
		if c.Adapter {
			m.Adapter = &AdapterInfo{Arch: c.Arch}
		}
		if m.Modified.IsZero() {
			m.Modified = time.Now()
		}
		dir := ""
		if len(c.Files) > 1 {
			dir = "link-" + pathHash(c.Files[0])
		}
		// All of a model's parts link, or none do: llama.cpp wants a split
		// model's parts side by side.
		var made []string
		for i, f := range c.Files {
			name, fresh, err := s.linkFile(f, c.SHAs[i], dir, path.Base(c.Rels[i]))
			if err != nil {
				for _, n := range made {
					os.Remove(s.Blob(n))
				}
				if dir != "" {
					os.Remove(s.Blob(dir))
				}
				m.Files, m.Link = append([]string(nil), c.Paths...), "in place"
				break
			}
			if fresh {
				made = append(made, name)
			}
			m.Files = append(m.Files, name)
		}
		if c.Projector != "" {
			if name, _, err := s.linkFile(c.Projector, c.ProjectorSHA, "", ""); err == nil {
				m.Projector = name
			} else {
				m.Projector = c.ProjectorAt
			}
		}
		if c.MTP != "" {
			if name, _, err := s.linkFile(c.MTP, c.MTPSHA, "", ""); err == nil {
				m.MTP = name
			} else {
				m.MTP = c.MTPAt
			}
		}
		m.Digest = c.SHAs[0]
		if m.Digest == "" {
			m.Digest = pathHash(c.Files[0])
		}
		if err := s.Put(m); err != nil {
			return added, err
		}
		added++
	}
	return added, nil
}

// linkFile hard-links src into blobs/ and returns its name there, and
// whether it made the link (false: an earlier import already had it).
func (s *Store) linkFile(src, sha, dir, base string) (name string, fresh bool, err error) {
	switch {
	case dir != "":
		name = dir + "/" + base
	case sha != "":
		name = "sha256-" + sha + ".gguf"
	default:
		name = "link-" + pathHash(src) + ".gguf"
	}
	dst := s.Blob(name)
	if _, err := os.Stat(dst); err == nil {
		return name, false, nil
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return "", false, err
	}
	os.Remove(dst) // a dangling link from before
	if err := hardLink(src, dst); err != nil {
		return "", false, err
	}
	return name, true, nil
}

// linkKind says how a file the store uses is held: "hard", "symlink"
// (imports from before files were used in place), or "in place" (a path
// outside the store).
func (s *Store) linkKind(name string) string {
	if filepath.IsAbs(name) {
		return "in place"
	}
	if info, err := os.Lstat(s.Blob(name)); err == nil && info.Mode()&os.ModeSymlink != 0 {
		return "symlink"
	}
	return "hard"
}

// hardLink is os.Link; tests swap it to act like a second disk.
var hardLink = os.Link

func linkKind(p string) string {
	if info, err := os.Lstat(p); err == nil && info.Mode()&os.ModeSymlink != 0 {
		return "symlink"
	}
	return "hard"
}

// Missing lists the model's files that can't be read, e.g. a symlink
// whose target the other app has since removed.
func (s *Store) Missing(m *Model) []string {
	var out []string
	for _, f := range m.AllFiles() {
		if _, err := os.Stat(s.Blob(f)); err != nil {
			out = append(out, f)
		}
	}
	return out
}

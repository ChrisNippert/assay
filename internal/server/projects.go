package server

import (
	"context"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"time"

	"ollamui/internal/chats"
	"ollamui/internal/index"
	"ollamui/internal/ollama"
	"ollamui/internal/projects"
	"ollamui/internal/rag"
)

// Chunking for RAG: about 1,500 characters (~400 tokens) per chunk with a
// little overlap, embedded 16 at a time.
const (
	chunkSize    = 1500
	chunkOverlap = 200
	embedBatch   = 16
	// minScore drops excerpts too far from the question to help.
	minScore = 0.25
)

// ---- sidebar ----

// ProjectGroup is a project in the sidebar with its chats.
type ProjectGroup struct {
	*projects.Project
	Chats []chats.Summary
}

// ProjectGroups lists the projects with their chats, newest first.
func (p *page) ProjectGroups() []ProjectGroup {
	out := make([]ProjectGroup, 0, len(p.Projects))
	for _, pr := range p.Projects {
		g := ProjectGroup{Project: pr}
		for _, c := range p.Chats {
			if c.Project == pr.ID {
				g.Chats = append(g.Chats, c)
			}
		}
		out = append(out, g)
	}
	return out
}

// looseChats are the chats outside any (existing) project.
func (p *page) looseChats() []chats.Summary {
	ids := map[string]bool{}
	for _, pr := range p.Projects {
		ids[pr.ID] = true
	}
	var out []chats.Summary
	for _, c := range p.Chats {
		if !ids[c.Project] {
			out = append(out, c)
		}
	}
	return out
}

// ---- pages ----

type projectPageData struct {
	Project  *projects.Project
	Chats    []chats.Summary
	RAG      projects.Settings
	Embedder bool   // the chosen embedding model is installed
	Stale    bool   // the index was built with another embedding model
	IndexOf  string // the model the index was built with
	Accept   string // file types the upload takes, for <input accept>
}

func (s *Server) handleProjectPage(w http.ResponseWriter, r *http.Request) {
	pr, err := s.projects.Get(r.PathValue("id"))
	if err != nil {
		http.Redirect(w, r, "/chat", http.StatusFound)
		return
	}
	p := s.basePage(r.Context(), "project", pr.Name)
	data := &projectPageData{Project: pr, RAG: s.ragSt.Get(), Accept: rag.Accept()}
	for _, c := range p.Chats {
		if c.Project == pr.ID {
			data.Chats = append(data.Chats, c)
		}
	}
	data.Embedder = data.RAG.EmbedModel != "" && slices.ContainsFunc(p.Models, func(m ollama.Model) bool { return sameModel(m.Name, data.RAG.EmbedModel) })
	if ix, err := s.projects.Index(pr.ID); err == nil && ix.Len() > 0 {
		data.IndexOf = ix.Model()
		data.Stale = data.RAG.EmbedModel != "" && !sameModel(ix.Model(), data.RAG.EmbedModel)
	}
	p.Data = data
	p.ProjectID = pr.ID
	s.render(w, "project", p)
}

// ---- API ----

func (s *Server) projectError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, projects.ErrNotFound):
		writeError(w, http.StatusNotFound, "project not found")
	default:
		writeError(w, http.StatusBadRequest, err.Error())
	}
}

func (s *Server) handleProjectCreate(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Name string `json:"name"`
	}
	if !readJSON(w, r, &req) {
		return
	}
	pr, err := s.projects.Create(req.Name)
	if err != nil {
		s.projectError(w, err)
		return
	}
	writeJSON(w, http.StatusCreated, pr)
}

func (s *Server) handleProjectGet(w http.ResponseWriter, r *http.Request) {
	pr, err := s.projects.Get(r.PathValue("id"))
	if err != nil {
		s.projectError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, pr)
}

// handleProjectUpdate changes any of name, instructions, model and preset.
func (s *Server) handleProjectUpdate(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Name         *string `json:"name"`
		Instructions *string `json:"instructions"`
		Model        *string `json:"model"`
		Preset       *string `json:"preset"`
	}
	if !readJSON(w, r, &req) {
		return
	}
	pr, err := s.projects.Update(r.PathValue("id"), func(p *projects.Project) error {
		if req.Name != nil {
			n := projects.CleanName(*req.Name)
			if n == "" {
				return errors.New("a project needs a name")
			}
			p.Name = n
		}
		if req.Instructions != nil {
			if len(*req.Instructions) > 20000 {
				return errors.New("instructions are too long (20,000 characters at most)")
			}
			p.Instructions = strings.TrimSpace(*req.Instructions)
		}
		if req.Model != nil {
			p.Model = strings.TrimSpace(*req.Model)
		}
		if req.Preset != nil {
			p.Preset = strings.TrimSpace(*req.Preset)
		}
		return nil
	})
	if err != nil {
		s.projectError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, pr)
}

// handleProjectDelete deletes a project and its files; its chats move out
// of it rather than being deleted.
func (s *Server) handleProjectDelete(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if _, err := s.projects.Get(id); err != nil {
		s.projectError(w, err)
		return
	}
	if err := s.chats.LeaveProject(id); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if err := s.projects.Delete(id); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// handleProjectUpload adds files (multipart "file" fields) and queues them
// for indexing. A file with the same name replaces the old one.
func (s *Server) handleProjectUpload(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if _, err := s.projects.Get(id); err != nil {
		s.projectError(w, err)
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, 25*rag.MaxFileSize)
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		writeError(w, http.StatusBadRequest, "upload failed: "+err.Error())
		return
	}
	defer r.MultipartForm.RemoveAll()
	var added []string
	var problems []string
	for _, fh := range r.MultipartForm.File["file"] {
		name := projects.CleanFileName(fh.Filename)
		switch {
		case name == "":
			problems = append(problems, fh.Filename+": not a usable file name")
			continue
		case !rag.Supported(name):
			problems = append(problems, name+": can't read this kind of file")
			continue
		case fh.Size > rag.MaxFileSize:
			problems = append(problems, name+": larger than 20 MB")
			continue
		}
		if err := s.saveUpload(id, name, fh.Open); err != nil {
			problems = append(problems, name+": "+err.Error())
			continue
		}
		size := fh.Size
		_, err := s.projects.Update(id, func(p *projects.Project) error {
			f := projects.File{Name: name, Size: size, Added: time.Now(), Status: projects.FileQueued}
			if old := p.File(name); old != nil {
				*old = f
			} else {
				p.Files = append(p.Files, f)
			}
			return nil
		})
		if err != nil {
			problems = append(problems, name+": "+err.Error())
			continue
		}
		added = append(added, name)
		s.indexQueue <- indexJob{project: id, file: name}
	}
	if len(added) == 0 && len(problems) > 0 {
		writeError(w, http.StatusBadRequest, strings.Join(problems, "; "))
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"added": added, "problems": problems})
}

func (s *Server) saveUpload(id, name string, open func() (multipart.File, error)) error {
	path, err := s.projects.FilePath(id, name)
	if err != nil {
		return err
	}
	src, err := open()
	if err != nil {
		return err
	}
	defer src.Close()
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	tmp := path + ".upload"
	dst, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if _, err := io.Copy(dst, io.LimitReader(src, rag.MaxFileSize+1)); err != nil {
		dst.Close()
		os.Remove(tmp)
		return err
	}
	if err := dst.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// handleProjectFileDelete removes a file and its chunks.
func (s *Server) handleProjectFileDelete(w http.ResponseWriter, r *http.Request) {
	id, name := r.PathValue("id"), r.PathValue("name")
	path, err := s.projects.FilePath(id, name)
	if err != nil {
		s.projectError(w, err)
		return
	}
	if _, err := s.projects.Update(id, func(p *projects.Project) error {
		i := slices.IndexFunc(p.Files, func(f projects.File) bool { return f.Name == name })
		if i < 0 {
			return projects.ErrNotFound
		}
		p.Files = slices.Delete(p.Files, i, i+1)
		return nil
	}); err != nil {
		s.projectError(w, err)
		return
	}
	os.Remove(path)
	if ix, err := s.projects.Index(id); err == nil {
		ix.Remove(name)
	}
	w.WriteHeader(http.StatusNoContent)
}

// handleProjectReindex re-embeds every file, e.g. after switching the
// embedding model (vectors from different models can't be compared).
func (s *Server) handleProjectReindex(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if err := s.reindex(id); err != nil {
		s.projectError(w, err)
		return
	}
	w.WriteHeader(http.StatusAccepted)
}

func (s *Server) reindex(id string) error {
	ix, err := s.projects.Index(id)
	if err != nil {
		return err
	}
	pr, err := s.projects.Update(id, func(p *projects.Project) error {
		for i := range p.Files {
			p.Files[i].Status, p.Files[i].Error, p.Files[i].Done, p.Files[i].Total = projects.FileQueued, "", 0, 0
		}
		return nil
	})
	if err != nil {
		return err
	}
	if err := ix.Clear(); err != nil {
		return err
	}
	for _, f := range pr.Files {
		s.indexQueue <- indexJob{project: id, file: f.Name}
	}
	return nil
}

// handleChatMove puts a chat in a project ("" takes it out).
func (s *Server) handleChatMove(id, project string) error {
	if project != "" {
		if _, err := s.projects.Get(project); err != nil {
			return err
		}
	}
	return s.chats.SetProject(id, project)
}

// ---- RAG settings ----

func (s *Server) handleRAGSettings(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, s.ragSt.Get())
}

// handleRAGSettingsSave saves the settings. Switching the embedding model
// re-indexes every project that has files.
func (s *Server) handleRAGSettingsSave(w http.ResponseWriter, r *http.Request) {
	var in projects.Settings
	if !readJSON(w, r, &in) {
		return
	}
	in.EmbedModel = strings.TrimSpace(in.EmbedModel)
	old := s.ragSt.Get()
	if err := s.ragSt.Set(in); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if !sameModel(old.EmbedModel, in.EmbedModel) && in.EmbedModel != "" {
		if list, err := s.projects.List(); err == nil {
			for _, p := range list {
				if len(p.Files) > 0 {
					s.reindex(p.ID)
				}
			}
		}
	}
	writeJSON(w, http.StatusOK, in)
}

// ---- indexing ----

type indexJob struct{ project, file string }

// errNoEmbedder leaves a file waiting: choosing an embedding model in
// Settings re-indexes every project, which picks it up.
var errNoEmbedder = errors.New("waiting for an embedding model")

// runIndexer indexes queued files one at a time until ctx ends. On start
// it picks up files a restart interrupted.
func (s *Server) runIndexer(ctx context.Context) {
	if list, err := s.projects.List(); err == nil {
		for _, p := range list {
			for _, f := range p.Files {
				if f.Status == projects.FileQueued || f.Status == projects.FileIndexing {
					go func(j indexJob) { s.indexQueue <- j }(indexJob{p.ID, f.Name})
				}
			}
		}
	}
	for {
		select {
		case <-ctx.Done():
			return
		case j := <-s.indexQueue:
			err := s.indexFile(ctx, j)
			status, msg := projects.FileReady, ""
			if errors.Is(err, errNoEmbedder) {
				status = projects.FileQueued
			} else if err != nil {
				status, msg = projects.FileError, err.Error()
				s.log.Warn("index file", "project", j.project, "file", j.file, "err", err)
			}
			s.projects.Update(j.project, func(p *projects.Project) error {
				if f := p.File(j.file); f != nil {
					f.Status, f.Error = status, msg
					if err == nil {
						f.Chunks = f.Total
					}
				}
				return nil
			})
		}
	}
}

func (s *Server) indexFile(ctx context.Context, j indexJob) error {
	model := s.ragSt.Get().EmbedModel
	if model == "" {
		return errNoEmbedder
	}
	path, err := s.projects.FilePath(j.project, j.file)
	if err != nil {
		return err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	xctx, cancel := context.WithTimeout(ctx, 2*time.Minute)
	text, err := rag.Extract(xctx, j.file, data)
	cancel()
	if err != nil {
		return err
	}
	chunks := rag.Split(text, chunkSize, chunkOverlap)
	if len(chunks) == 0 {
		return errors.New("no text found in this file")
	}
	progress := func(done int) {
		s.projects.Update(j.project, func(p *projects.Project) error {
			if f := p.File(j.file); f != nil {
				f.Status, f.Done, f.Total = projects.FileIndexing, done, len(chunks)
			}
			return nil
		})
	}
	progress(0)
	entries := make([]index.Entry, len(chunks))
	vecs := make([][]float32, 0, len(chunks))
	for i, c := range chunks {
		entries[i] = index.Entry{File: j.file, Page: c.Page, Heading: c.Heading, Text: c.Text}
	}
	for i := 0; i < len(chunks); i += embedBatch {
		end := min(i+embedBatch, len(chunks))
		in := make([]string, 0, end-i)
		for _, c := range chunks[i:end] {
			in = append(in, embedText(j.file, c.Heading, c.Text))
		}
		v, err := s.ollama.Embed(ctx, model, in)
		if err != nil {
			if ollama.IsNotFound(err) {
				return fmt.Errorf("the embedding model %s isn't installed", model)
			}
			return fmt.Errorf("embedding: %w", err)
		}
		vecs = append(vecs, v...)
		progress(end)
	}
	ix, err := s.projects.Index(j.project)
	if err != nil {
		return err
	}
	return ix.Put(model, j.file, entries, vecs)
}

// embedText is what gets embedded for a chunk: the file name and heading
// give a chunk the context its own words may lack.
func embedText(file, heading, text string) string {
	head := file
	if heading != "" {
		head += " · " + heading
	}
	return head + "\n\n" + text
}

// ---- retrieval ----

// retrieve finds the excerpts of a project's files closest to the
// question. It returns nothing (and no error) when the project has no
// usable index; problems are logged, not shown, so a chat still works.
func (s *Server) retrieve(ctx context.Context, project, question string) []chats.Source {
	st := s.ragSt.Get()
	ix, err := s.projects.Index(project)
	if err != nil || ix.Len() == 0 || st.EmbedModel == "" || !sameModel(ix.Model(), st.EmbedModel) {
		return nil
	}
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	v, err := s.ollama.Embed(ctx, st.EmbedModel, []string{question})
	if err != nil {
		s.log.Warn("embed question", "err", err)
		return nil
	}
	var out []chats.Source
	for _, h := range ix.Search(v[0], st.TopK) {
		if h.Score < minScore {
			continue
		}
		out = append(out, chats.Source{N: len(out) + 1, File: h.File, Page: h.Page, Heading: h.Heading, Text: h.Text, Score: h.Score})
	}
	return out
}

// sourcesPrompt is the system-prompt section carrying the excerpts.
func sourcesPrompt(src []chats.Source) string {
	if len(src) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("Excerpts from the project's files that may help with the user's latest message. Use them when they're relevant and cite them by number like [1]; if they don't answer the question, say so instead of guessing.\n")
	for _, x := range src {
		fmt.Fprintf(&b, "\n[%d] %s", x.N, x.File)
		if x.Page > 0 {
			fmt.Fprintf(&b, ", page %d", x.Page)
		}
		if x.Heading != "" {
			fmt.Fprintf(&b, " (%s)", x.Heading)
		}
		b.WriteString("\n" + x.Text + "\n")
	}
	return b.String()
}

// ragQuery is what to search with: the latest message, plus the one before
// when it's short ("and in 2025?" means little alone).
func ragQuery(msgs []chats.Message) string {
	var users []string
	for _, m := range msgs {
		if m.Role == "user" {
			users = append(users, m.Content)
		}
	}
	if len(users) == 0 {
		return ""
	}
	q := users[len(users)-1]
	if len(q) < 60 && len(users) > 1 {
		q = users[len(users)-2] + "\n" + q
	}
	return q
}

// Files lists the project's files by name.
func (d *projectPageData) Files() []projects.File {
	out := slices.Clone(d.Project.Files)
	sort.Slice(out, func(i, j int) bool { return strings.ToLower(out[i].Name) < strings.ToLower(out[j].Name) })
	return out
}

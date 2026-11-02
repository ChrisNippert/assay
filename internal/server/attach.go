package server

import (
	"context"
	"encoding/base64"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"assay/internal/chats"
	"assay/internal/rag"
)

// maxAttachment is the largest file a message can carry.
const maxAttachment = 25 << 20

// docShare is how much of the context attached documents may fill.
const docShare = 0.6

var imageTypes = map[string]string{
	"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
	"webp": "image/webp", "gif": "image/gif",
}

// handleAttachmentUpload stages files (multipart "file" fields) for the next
// message. It answers with the files taken and a line for each one refused.
func (s *Server) handleAttachmentUpload(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 8*maxAttachment)
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		writeError(w, http.StatusBadRequest, "upload failed: "+err.Error())
		return
	}
	defer r.MultipartForm.RemoveAll()
	files := []chats.Attachment{}
	problems := []string{}
	for _, fh := range r.MultipartForm.File["file"] {
		name := filepath.Base(strings.ReplaceAll(fh.Filename, "\\", "/"))
		if fh.Size > maxAttachment {
			problems = append(problems, fmt.Sprintf("%s: larger than %d MB", name, maxAttachment>>20))
			continue
		}
		f, err := fh.Open()
		if err != nil {
			problems = append(problems, name+": could not be read")
			continue
		}
		data, err := io.ReadAll(io.LimitReader(f, maxAttachment+1))
		f.Close()
		if err != nil || len(data) > maxAttachment {
			problems = append(problems, name+": could not be read")
			continue
		}
		a, ext, text, err := s.classify(r.Context(), name, data)
		if err != nil {
			problems = append(problems, name+": "+err.Error())
			continue
		}
		a, err = s.chats.Stage(a, ext, data, text)
		if err != nil {
			s.log.Error("stage attachment", "err", err)
			problems = append(problems, name+": could not be saved")
			continue
		}
		files = append(files, a)
	}
	writeJSON(w, 200, map[string]any{"files": files, "problems": problems})
}

// classify decides what an uploaded file is: an image, or a document whose
// text it extracts.
func (s *Server) classify(ctx context.Context, name string, data []byte) (a chats.Attachment, ext, text string, err error) {
	ext = strings.ToLower(strings.TrimPrefix(filepath.Ext(name), "."))
	a = chats.Attachment{Name: name, Size: int64(len(data))}
	if mime, ok := imageTypes[ext]; ok {
		if !strings.HasPrefix(http.DetectContentType(data), "image/") {
			return a, "", "", fmt.Errorf("not a valid %s image", strings.ToUpper(ext))
		}
		a.Kind, a.Mime = "image", mime
		return a, ext, "", nil
	}
	if !rag.Supported(name) {
		return a, "", "", fmt.Errorf("can't read this kind of file (text, code, PDF, Word, ODT, HTML and images work)")
	}
	text, err = rag.Extract(ctx, name, data)
	if err != nil {
		s.log.Info("attachment not readable", "name", name, "err", err)
		return a, "", "", fmt.Errorf("could not read the text in it")
	}
	if strings.TrimSpace(text) == "" {
		return a, "", "", fmt.Errorf("no text found in it")
	}
	a.Kind, a.Mime = "doc", "application/octet-stream"
	return a, ext, text, nil
}

// handleChatFile serves a stored attachment. Images show inline; anything
// else downloads.
func (s *Server) handleChatFile(w http.ResponseWriter, r *http.Request) {
	p, err := s.chats.FilePath(r.PathValue("id"), r.PathValue("file"))
	if err != nil {
		http.NotFound(w, r)
		return
	}
	f, err := os.Open(p)
	if err != nil {
		http.NotFound(w, r)
		return
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		http.NotFound(w, r)
		return
	}
	ct := imageTypes[strings.TrimPrefix(filepath.Ext(p), ".")]
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("Content-Security-Policy", "default-src 'none'; sandbox")
	w.Header().Set("Cache-Control", "private, max-age=31536000, immutable")
	if ct == "" {
		ct = "application/octet-stream"
		w.Header().Set("Content-Disposition", "attachment")
	}
	w.Header().Set("Content-Type", ct)
	http.ServeContent(w, r, "", info.ModTime(), f)
}

// handleVision says whether a model can see images.
func (s *Server) handleVision(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, 200, map[string]bool{"vision": s.hasVision(r.Context(), r.URL.Query().Get("model"))})
}

func (s *Server) hasVision(ctx context.Context, model string) bool {
	if model == "" {
		return false
	}
	cctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	show, err := s.show(cctx, model)
	return err == nil && show.HasCapability("vision")
}

// fence wraps text in a code fence longer than any run of backticks in it.
func fence(text string) string {
	n, run := 3, 0
	for _, r := range text {
		if r == '`' {
			run++
			n = max(n, run+1)
		} else {
			run = 0
		}
	}
	f := strings.Repeat("`", n)
	return f + "\n" + strings.TrimRight(text, "\n") + "\n" + f
}

// withFiles returns msgs with their attachments put in: a document's text
// goes into the message content, an image into Images when vision is set (a
// note says so otherwise). Documents share budget characters (0: no limit),
// the newest first; the names of those that were cut are returned. The
// stored messages are not changed.
func (s *Server) withFiles(chatID string, msgs []chats.Message, vision bool, budget int) ([]chats.Message, map[string]bool) {
	out := make([]chats.Message, len(msgs))
	copy(out, msgs)
	cut := map[string]bool{}
	left := budget
	for i := len(out) - 1; i >= 0; i-- {
		m := out[i]
		if len(m.Attachments) == 0 {
			continue
		}
		m.Images = nil
		var parts []string
		for _, a := range m.Attachments {
			switch a.Kind {
			case "image":
				if !vision {
					parts = append(parts, fmt.Sprintf("[Attached image: %s (not shown: this model can't see images)]", a.Name))
					continue
				}
				p, err := s.chats.FilePath(chatID, a.File)
				if err != nil {
					continue
				}
				b, err := os.ReadFile(p)
				if err != nil {
					parts = append(parts, fmt.Sprintf("[Attached image: %s (file missing)]", a.Name))
					continue
				}
				m.Images = append(m.Images, base64.StdEncoding.EncodeToString(b))
			default:
				text, err := s.chats.FileText(chatID, a.File)
				if err != nil {
					parts = append(parts, fmt.Sprintf("[Attached file: %s (file missing)]", a.Name))
					continue
				}
				r := []rune(text)
				note := ""
				if budget > 0 && len(r) > left {
					cut[a.File] = true
					if left < 200 {
						parts = append(parts, fmt.Sprintf("[Attached file: %s (left out: the conversation is too long to include it)]", a.Name))
						continue
					}
					note = fmt.Sprintf("\n[Cut: only the first %d of %d characters fit in the context.]", left, len(r))
					r = r[:left]
				}
				if budget > 0 {
					left -= len(r)
				}
				parts = append(parts, "Attached file: "+a.Name+"\n"+fence(string(r))+note)
			}
		}
		if len(parts) > 0 {
			body := strings.Join(parts, "\n\n")
			if m.Content != "" {
				body += "\n\n" + m.Content
			}
			m.Content = body
		}
		out[i] = m
	}
	return out, cut
}

// docBudget is how many characters of documents fit in a ctxLen context.
func docBudget(ctxLen int) int {
	if ctxLen <= 0 {
		ctxLen = 8192
	}
	return int(float64(ctxLen) * docShare * 3)
}

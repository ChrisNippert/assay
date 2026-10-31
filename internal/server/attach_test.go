package server

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"mime/multipart"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"assay/internal/chats"
)

// png1 is a 1x1 PNG.
const png1 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg=="

func upload(t *testing.T, s *Server, files map[string][]byte) (got []chats.Attachment, problems []string, code int) {
	t.Helper()
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	for name, data := range files {
		fw, _ := mw.CreateFormFile("file", name)
		fw.Write(data)
	}
	mw.Close()
	req := httptest.NewRequest("POST", "/api/attachments", &buf)
	req.Host = "localhost:8484"
	req.Header.Set("Content-Type", mw.FormDataContentType())
	rec := httptest.NewRecorder()
	s.ServeHTTP(rec, req)
	var out struct {
		Files    []chats.Attachment `json:"files"`
		Problems []string           `json:"problems"`
	}
	json.Unmarshal(rec.Body.Bytes(), &out)
	return out.Files, out.Problems, rec.Code
}

func TestAttachmentsSendAndServe(t *testing.T) {
	s, fake, store := newTestServer(t)
	fake.caps = `"completion","vision"`
	img, _ := base64.StdEncoding.DecodeString(png1)
	files, problems, code := upload(t, s, map[string][]byte{"pic.png": img, "notes.txt": []byte("secret code is 42"), "x.bin": []byte("zz")})
	if code != 200 || len(files) != 2 || len(problems) != 1 || !strings.Contains(problems[0], "x.bin") {
		t.Fatalf("upload: %d %+v %v", code, files, problems)
	}
	var tokens []string
	for _, f := range files {
		tokens = append(tokens, f.File)
	}
	fake.reply = `{"message":{"role":"assistant","content":"ok"},"done":true}` + "\n"
	body, _ := json.Marshal(map[string]any{"model": "qwen3:8b", "content": "what is in these?", "attachments": tokens})
	rec := do(s, "POST", "/api/chat", string(body))
	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	var first event
	json.NewDecoder(rec.Body).Decode(&first)
	c, err := store.Get(first.ID)
	if err != nil || len(c.Messages[0].Attachments) != 2 {
		t.Fatalf("chat = %+v %v", c, err)
	}
	sent := fake.chatReqs[0].Messages[0]
	if !strings.Contains(sent.Content, "Attached file: notes.txt") || !strings.Contains(sent.Content, "secret code is 42") || !strings.HasSuffix(sent.Content, "what is in these?") {
		t.Errorf("content = %q", sent.Content)
	}
	if len(sent.Images) != 1 || sent.Images[0] != png1 {
		t.Errorf("images = %v", sent.Images)
	}

	// The next turn sends the old attachments again.
	body, _ = json.Marshal(map[string]any{"chatId": c.ID, "model": "qwen3:8b", "content": "and now?"})
	do(s, "POST", "/api/chat", string(body))
	h := fake.chatReqs[1].Messages
	if !strings.Contains(h[0].Content, "secret code is 42") || len(h[0].Images) != 1 {
		t.Errorf("history lost its files: %+v", h[0])
	}

	// An edit keeps the message's files.
	body, _ = json.Marshal(map[string]any{"chatId": c.ID, "model": "qwen3:8b", "content": "edited", "fromUser": 0})
	do(s, "POST", "/api/chat", string(body))
	h = fake.chatReqs[2].Messages
	if !strings.Contains(h[0].Content, "secret code is 42") || !strings.HasSuffix(h[0].Content, "edited") {
		t.Errorf("edit lost its files: %+v", h[0])
	}

	// Files are served; names that aren't ours are not.
	img0 := c.Messages[0].Attachments[0]
	if img0.Kind != "image" {
		img0 = c.Messages[0].Attachments[1]
	}
	if r := do(s, "GET", "/api/chats/"+c.ID+"/files/"+img0.File, ""); r.Code != 200 || r.Header().Get("Content-Type") != "image/png" {
		t.Errorf("serve image: %d %s", r.Code, r.Header().Get("Content-Type"))
	}
	for _, bad := range []string{"..%2f..%2fx", "..", "%2e%2e%2f%2e%2e%2fchats.json", img0.File + ".txt", "../" + img0.File} {
		if r := do(s, "GET", "/api/chats/"+c.ID+"/files/"+bad, ""); r.Code == 200 {
			t.Errorf("served %q", bad)
		}
	}

	// Deleting the chat deletes its files.
	if _, err := os.Stat(store.FilesDir(c.ID)); err != nil {
		t.Fatal(err)
	}
	do(s, "DELETE", "/api/chats/"+c.ID, "")
	if _, err := os.Stat(store.FilesDir(c.ID)); !os.IsNotExist(err) {
		t.Errorf("files left behind: %v", err)
	}
}

func TestAttachmentsNeedVisionAndCut(t *testing.T) {
	s, fake, _ := newTestServer(t)
	img, _ := base64.StdEncoding.DecodeString(png1)
	files, _, _ := upload(t, s, map[string][]byte{"pic.png": img})
	body, _ := json.Marshal(map[string]any{"model": "qwen3:8b", "content": "see", "attachments": []string{files[0].File}})
	if rec := do(s, "POST", "/api/chat", string(body)); rec.Code != 400 || !strings.Contains(rec.Body.String(), "can't see images") {
		t.Errorf("image to a text model: %d %s", rec.Code, rec.Body)
	}

	big := strings.Repeat("word ", 30000)
	files, _, _ = upload(t, s, map[string][]byte{"big.txt": []byte(big)})
	fake.reply = `{"message":{"role":"assistant","content":"ok"},"done":true}` + "\n"
	body, _ = json.Marshal(map[string]any{"model": "qwen3:8b", "content": "sum", "attachments": []string{files[0].File}, "options": map[string]any{"num_ctx": 4096}})
	rec := do(s, "POST", "/api/chat", string(body))
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), `"type":"attachments"`) {
		t.Fatalf("no cut event: %d %s", rec.Code, rec.Body)
	}
	got := fake.chatReqs[len(fake.chatReqs)-1].Messages[0].Content
	if len(got) > 4096*3 || !strings.Contains(got, "[Cut: only the first") {
		t.Errorf("document not cut: %d chars", len(got))
	}
}

func TestAttachmentLimits(t *testing.T) {
	s, _, _ := newTestServer(t)
	_, problems, _ := upload(t, s, map[string][]byte{"huge.txt": bytes.Repeat([]byte("a"), maxAttachment+1), "fake.png": []byte("not an image")})
	if len(problems) != 2 {
		t.Errorf("problems = %v", problems)
	}
}

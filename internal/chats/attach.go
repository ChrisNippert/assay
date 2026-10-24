package chats

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

// Attachment is a file sent with a user message. The file itself lives in
// the chat's files directory under File; for a document the text pulled out
// of it is kept beside it as File+".txt" so it isn't extracted every turn.
type Attachment struct {
	Name string `json:"name"`
	Kind string `json:"kind"` // "image" or "doc"
	Mime string `json:"mime,omitempty"`
	Size int64  `json:"size"`
	File string `json:"file"`
	// Cut: the text was shortened to fit the context when it was sent.
	Cut bool `json:"cut,omitempty"`
}

const (
	filesSuffix = ".files"
	pendingDir  = ".pending"
	pendingTTL  = 24 * time.Hour
)

// ErrBadFile is returned for an attachment name that isn't one Stage made.
var ErrBadFile = errors.New("invalid attachment name")

var (
	filePattern = regexp.MustCompile(`^[0-9a-f]{16}(\.[a-z0-9]{1,8})?$`)
	extPattern  = regexp.MustCompile(`^[a-z0-9]{1,8}$`)
)

// ValidFile reports whether name has the shape Stage produces, so nothing
// else reaches the filesystem.
func ValidFile(name string) bool { return filePattern.MatchString(name) }

// FilesDir is where the files of chat id live.
func (s *Store) FilesDir(id string) string { return filepath.Join(s.dir, id+filesSuffix) }

// Stage saves an uploaded file (and its extracted text, for a document) in
// the staging area, ready for Adopt once the message is sent. It returns the
// attachment, whose File is the token to send back with the message. ext is
// the file's extension without the dot.
func (s *Store) Stage(a Attachment, ext string, data []byte, text string) (Attachment, error) {
	dir := filepath.Join(s.dir, pendingDir)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return a, err
	}
	s.sweepPending(dir)
	var b [8]byte
	rand.Read(b[:])
	a.File = hex.EncodeToString(b[:])
	if ext = strings.ToLower(ext); extPattern.MatchString(ext) {
		a.File += "." + ext
	}
	meta, _ := json.Marshal(a)
	if err := os.WriteFile(filepath.Join(dir, a.File), data, 0o600); err != nil {
		return a, err
	}
	if a.Kind == "doc" {
		if err := os.WriteFile(filepath.Join(dir, a.File+".txt"), []byte(text), 0o600); err != nil {
			return a, err
		}
	}
	return a, os.WriteFile(filepath.Join(dir, a.File+".json"), meta, 0o600)
}

// sweepPending removes staged files nobody sent within a day.
func (s *Store) sweepPending(dir string) {
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if info, err := e.Info(); err == nil && time.Since(info.ModTime()) > pendingTTL {
			os.Remove(filepath.Join(dir, e.Name()))
		}
	}
}

// Adopt moves staged files (by token) into chat id's files directory and
// returns their attachments.
func (s *Store) Adopt(id string, tokens []string) ([]Attachment, error) {
	if !ValidID(id) {
		return nil, ErrNotFound
	}
	var out []Attachment
	for _, tok := range tokens {
		if !ValidFile(tok) {
			return nil, ErrBadFile
		}
		src := filepath.Join(s.dir, pendingDir, tok)
		b, err := os.ReadFile(src + ".json")
		if err != nil {
			return nil, fmt.Errorf("an attached file is no longer available; attach it again")
		}
		var a Attachment
		if json.Unmarshal(b, &a) != nil || a.File != tok {
			return nil, ErrBadFile
		}
		if err := os.MkdirAll(s.FilesDir(id), 0o700); err != nil {
			return nil, err
		}
		dst := filepath.Join(s.FilesDir(id), tok)
		if err := os.Rename(src, dst); err != nil {
			return nil, err
		}
		if a.Kind == "doc" {
			if err := os.Rename(src+".txt", dst+".txt"); err != nil {
				return nil, err
			}
		}
		os.Remove(src + ".json")
		out = append(out, a)
	}
	return out, nil
}

// FilePath is the path of a stored attachment file of chat id, or ErrNotFound
// if either name is not valid.
func (s *Store) FilePath(id, file string) (string, error) {
	if !ValidID(id) || !ValidFile(file) {
		return "", ErrNotFound
	}
	return filepath.Join(s.FilesDir(id), file), nil
}

// FileText is the text extracted from a stored document.
func (s *Store) FileText(id, file string) (string, error) {
	p, err := s.FilePath(id, file)
	if err != nil {
		return "", err
	}
	b, err := os.ReadFile(p + ".txt")
	return string(b), err
}

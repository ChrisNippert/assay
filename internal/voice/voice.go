// Package voice talks to a speech service with OpenAI's audio API:
// /v1/audio/transcriptions (speech to text) and /v1/audio/speech (text to
// speech). tools/voice/voice_server.py is one; speaches and others work too.
package voice

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// Client is a speech service.
type Client struct {
	base *url.URL
	// STTModel and TTSModel are sent as "model"; TTSVoice as "voice". Empty
	// leaves the choice to the service.
	STTModel, TTSModel, TTSVoice string
	http                         *http.Client
}

// New returns a client for the service at base, e.g. http://127.0.0.1:8485.
func New(base string) (*Client, error) {
	u, err := url.Parse(strings.TrimRight(base, "/"))
	if err != nil || u.Host == "" || (u.Scheme != "http" && u.Scheme != "https") {
		return nil, fmt.Errorf("invalid speech service URL %q", base)
	}
	return &Client{base: u, http: &http.Client{}}, nil
}

// URL is the service's base URL.
func (c *Client) URL() string { return c.base.String() }

func (c *Client) endpoint(path string) string { return c.base.String() + path }

// Available reports whether the service answers, and why not.
func (c *Client) Available(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, c.endpoint("/health"), nil)
	res, err := c.http.Do(req)
	if err != nil {
		return err
	}
	res.Body.Close()
	if res.StatusCode == http.StatusNotFound {
		return nil // up, just without a health route (not every service has one)
	}
	if res.StatusCode >= 300 {
		return fmt.Errorf("speech service: %s", res.Status)
	}
	return nil
}

// Transcribe turns audio (any format the service decodes; WAV always works)
// into text. prompt, if set, nudges spelling and vocabulary; model, if set,
// overrides STTModel.
func (c *Client) Transcribe(ctx context.Context, audio []byte, filename, prompt, model string) (string, error) {
	if model == "" {
		model = c.STTModel
	}
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	fw, _ := mw.CreateFormFile("file", filename)
	fw.Write(audio)
	if model != "" {
		mw.WriteField("model", model)
	}
	if prompt != "" {
		mw.WriteField("prompt", prompt)
	}
	mw.WriteField("response_format", "json")
	mw.Close()

	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint("/v1/audio/transcriptions"), &body)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	res, err := c.http.Do(req)
	if err != nil {
		return "", err
	}
	defer res.Body.Close()
	if err := check(res); err != nil {
		return "", err
	}
	var out struct {
		Text string `json:"text"`
	}
	if err := json.NewDecoder(res.Body).Decode(&out); err != nil {
		return "", fmt.Errorf("speech service: %w", err)
	}
	return strings.TrimSpace(out.Text), nil
}

// Speak turns text into audio. voice and speed, if set, override TTSVoice
// and the service's default speed. The caller closes the returned body.
func (c *Client) Speak(ctx context.Context, text, voice string, speed float64) (io.ReadCloser, string, error) {
	if voice == "" {
		voice = c.TTSVoice
	}
	in := map[string]any{"input": text, "response_format": "wav"}
	in["model"] = c.TTSModel
	if c.TTSModel == "" {
		in["model"] = "tts-1"
	}
	if voice != "" {
		in["voice"] = voice
	}
	if speed > 0 {
		in["speed"] = speed
	}
	b, _ := json.Marshal(in)
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint("/v1/audio/speech"), bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	res, err := c.http.Do(req)
	if err != nil {
		return nil, "", err
	}
	if err := check(res); err != nil {
		res.Body.Close()
		return nil, "", err
	}
	ct := res.Header.Get("Content-Type")
	if ct == "" {
		ct = "audio/wav"
	}
	return res.Body, ct, nil
}

func check(res *http.Response) error {
	if res.StatusCode < 300 {
		return nil
	}
	msg, _ := io.ReadAll(io.LimitReader(res.Body, 2048))
	var e struct {
		Detail any `json:"detail"`
		Error  any `json:"error"`
	}
	if json.Unmarshal(msg, &e) == nil {
		if e.Detail != nil {
			return fmt.Errorf("speech service: %v", e.Detail)
		}
		if e.Error != nil {
			return fmt.Errorf("speech service: %v", e.Error)
		}
	}
	return fmt.Errorf("speech service: %s: %s", res.Status, strings.TrimSpace(string(msg)))
}

// Voices lists the voices the service offers, if it says (the bundled
// service does; OpenAI's API has no such endpoint).
func (c *Client) Voices(ctx context.Context) ([]string, error) {
	ctx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, c.endpoint("/v1/audio/voices"), nil)
	res, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := check(res); err != nil {
		return nil, err
	}
	var out struct {
		Voices []string `json:"voices"`
	}
	if err := json.NewDecoder(res.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("speech service: %w", err)
	}
	return out.Voices, nil
}

// STTModels are the faster-whisper models worth offering, fastest first.
// English-only ".en" models are faster and more accurate for English.
var STTModels = []struct{ Name, Note string }{
	{"tiny.en", "fastest, rough"},
	{"base.en", "fast"},
	{"small.en", "balanced"},
	{"distil-small.en", "balanced, a little faster"},
	{"medium.en", "accurate, slow on a CPU"},
	{"large-v3-turbo", "most accurate, any language; slow on a CPU"},
}

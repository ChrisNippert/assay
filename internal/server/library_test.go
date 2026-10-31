package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"assay/internal/bench"
	"assay/internal/hw"
)

func TestPrettyModel(t *testing.T) {
	for _, c := range []struct {
		in, title, tags string
	}{
		{"gemma-4-12b-it-qat:ud-q4_k_xl", "Gemma 4 12B", "QAT"},
		{"hf.co/unsloth/Qwen3.5-2B-GGUF:Q4_K_M", "Qwen3.5 2B", ""},
		{"qwen3.5-9b:q8_0", "Qwen3.5 9B", ""},
		{"glm-4.6v-flash:ud-q4_k_xl", "GLM 4.6V Flash", ""},
		{"hf.co/IFM/K2-Horizon-7B-GGUF:Q4_K_M", "K2 Horizon 7B", ""},
		{"gemma-4-e4b-it-qat:ud-q4_k_xl", "Gemma 4 E4B", "QAT"},
		{"qwen3-embedding:0.6b", "Qwen3 Embedding 0.6B", ""},
		{"qwen3-30b-a3b:q4_k_m", "Qwen3 30B A3B", "MoE"},
		{"muse-glimmer-30b-moe:ud-q3_k_xl", "Muse Glimmer 30B", "MoE"},
		{"llama3.1:8b-instruct", "Llama3.1", ""},
		{"-gguf", "-gguf", ""},
	} {
		title, tags := prettyModel(c.in)
		if title != c.title || strings.Join(tags, ",") != c.tags {
			t.Errorf("prettyModel(%q) = %q %v, want %q [%s]", c.in, title, tags, c.title, c.tags)
		}
	}
}

func TestNearScores(t *testing.T) {
	all := []nearEntry{{Model: "a", Score: 50}, {Model: "b", Score: 40}, {Model: "c", Score: 30}, {Model: "d", Score: 20}, {Model: "e", Score: 10}}
	list, rank, n := nearScores(all, nearEntry{Model: "me", Score: 25, You: true})
	var got []string
	for _, e := range list {
		got = append(got, e.Model)
	}
	if strings.Join(got, ",") != "b,c,me,d,e" || rank != 4 || n != 6 {
		t.Errorf("got %v rank %d of %d", got, rank, n)
	}
	list, rank, _ = nearScores(nil, nearEntry{Model: "me", Score: 25, You: true})
	if len(list) != 1 || rank != 1 {
		t.Errorf("alone: %v rank %d", list, rank)
	}
}

func TestBillionsIn(t *testing.T) {
	for in, want := range map[string]float64{"google/gemma-3-12b-it": 12, "qwen/qwen3-30b-a3b": 30, "org/model-0.6b": 0.6, "org/tiny": 0} {
		if got := billionsIn(in); got != want {
			t.Errorf("billionsIn(%q) = %v, want %v", in, got, want)
		}
	}
}

func TestLibraryPage(t *testing.T) {
	s, _, _ := newTestServer(t)
	rec := httptest.NewRecorder()
	s.ServeHTTP(rec, httptest.NewRequest("GET", "http://localhost/library", nil))
	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	body := rec.Body.String()
	for _, want := range []string{`class="lib-stats"`, `data-model="qwen3:8b"`, `>Qwen3 8B<`, `data-measure="qwen3:8b"`, `Measure all`, `GPU memory`} {
		if !strings.Contains(body, want) {
			t.Errorf("page lacks %q", want)
		}
	}
}

func TestLibScoresWithoutHF(t *testing.T) {
	s, _, _ := newTestServer(t)
	rec := httptest.NewRecorder()
	s.ServeHTTP(rec, httptest.NewRequest("GET", "http://localhost/api/library/scores", nil))
	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	var out struct {
		Scores map[string]scoreJSON `json:"scores"`
		URL    string               `json:"url"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if got := out.Scores["qwen3:8b"]; got.Rated || got.Reason != "local" {
		t.Errorf("score = %+v", got)
	}
	if !strings.Contains(out.URL, "artificialanalysis") {
		t.Errorf("no attribution url: %q", out.URL)
	}
	rec = httptest.NewRecorder()
	s.ServeHTTP(rec, httptest.NewRequest("GET", "http://localhost/api/library/nearby?name=qwen3:8b", nil))
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), `"reason":"local"`) {
		t.Errorf("nearby: %d %s", rec.Code, rec.Body)
	}
	rec = httptest.NewRecorder()
	s.ServeHTTP(rec, httptest.NewRequest("GET", "http://localhost/api/library/nearby?name=nope", nil))
	if rec.Code != 404 {
		t.Errorf("unknown model: %d", rec.Code)
	}
}

const threeModels = `{"models":[
 {"name":"small:q4","digest":"d1","size":2000000000,"details":{"parameter_size":"3B","quantization_level":"Q4_K_M"}},
 {"name":"mid:q4","digest":"d2","size":6000000000,"details":{"parameter_size":"9B","quantization_level":"Q4_K_M"}},
 {"name":"huge:q4","digest":"d3","size":60000000000,"details":{"parameter_size":"120B","quantization_level":"Q4_K_M"}},
 {"name":"embed:0.6b","digest":"d4","size":600000000,"details":{"parameter_size":"0.6B"}}]}`

func waitMeasure(t *testing.T, s *Server) measureState {
	t.Helper()
	for range 200 {
		if st := s.measure.state(); !st.Running {
			return st
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("measure job never finished")
	return measureState{}
}

func fakeBench(mu *sync.Mutex, order *[]string, keeps *[]bool) func(context.Context, string, bool) (bench.Result, error) {
	return func(ctx context.Context, name string, keep bool) (bench.Result, error) {
		mu.Lock()
		*order = append(*order, name)
		*keeps = append(*keeps, keep)
		mu.Unlock()
		return bench.Result{Model: name, Digest: map[string]string{"small:q4": "d1", "mid:q4": "d2", "huge:q4": "d3"}[name], GenTPS: 42, At: time.Now()}, nil
	}
}

func TestMeasureAll(t *testing.T) {
	s, fake, _ := newTestServer(t)
	fake.tags = threeModels
	s.bench = bench.Open(t.TempDir())
	var mu sync.Mutex
	var order []string
	var keeps []bool
	s.benchFn = fakeBench(&mu, &order, &keeps)

	st, err := s.startMeasure(context.Background(), measureRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if st.Total != 3 { // the embedding model is left out
		t.Fatalf("total = %d: %+v", st.Total, st.Items)
	}
	st = waitMeasure(t, s)
	if st.Done != 3 || strings.Join(order, ",") != "huge:q4,mid:q4,small:q4" {
		t.Errorf("done %d, order %v", st.Done, order)
	}
	for _, it := range st.Items {
		if it.State != itemDone || it.GenTPS != 42 {
			t.Errorf("item %+v", it)
		}
	}
	if r, ok := s.bench.Get("d2"); !ok || r.GenTPS != 42 {
		t.Errorf("result not saved: %+v %v", r, ok)
	}
	for _, k := range keeps {
		if k {
			t.Error("nothing was pinned, but pinned models were kept")
		}
	}
	// Everything is tested now.
	if _, err := s.startMeasure(context.Background(), measureRequest{}); err == nil {
		t.Error("expected nothing to measure")
	}
}

func TestMeasureKeepsPinned(t *testing.T) {
	s, fake, _ := newTestServer(t)
	fake.tags = threeModels
	s.bench = bench.Open(t.TempDir())
	s.memory = hw.Memory{GPU: "Test", VRAM: 12e9, RAM: 16e9}
	fake.loaded["mid:q4"] = time.Now().Add(200 * 365 * 24 * time.Hour) // pinned, 6 GB on the GPU
	var mu sync.Mutex
	var order []string
	var keeps []bool
	s.benchFn = fakeBench(&mu, &order, &keeps)

	st, err := s.startMeasure(context.Background(), measureRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if len(st.KeptPinned) != 1 {
		t.Errorf("kept pinned = %v", st.KeptPinned)
	}
	st = waitMeasure(t, s)
	states := map[string]measureItem{}
	for _, it := range st.Items {
		states[it.Name] = it
	}
	if states["small:q4"].State != itemDone {
		t.Errorf("small: %+v", states["small:q4"])
	}
	if it := states["huge:q4"]; it.State != itemSkipped || !strings.Contains(it.Reason, "pinned") {
		t.Errorf("huge should be skipped for the pinned model: %+v", it)
	}
	if it := states["mid:q4"]; it.State != itemSkipped || !strings.Contains(it.Reason, "pinned") {
		t.Errorf("the pinned model itself should be skipped: %+v", it)
	}
	if strings.Join(order, ",") != "small:q4" || !keeps[0] {
		t.Errorf("measured %v keeps %v", order, keeps)
	}
	if _, ok := fake.loaded["mid:q4"]; !ok {
		t.Error("the pinned model was evicted")
	}

	// Saying so evicts it.
	order, keeps = nil, nil
	if _, err := s.startMeasure(context.Background(), measureRequest{Names: []string{"mid:q4"}, EvictPinned: true}); err != nil {
		t.Fatal(err)
	}
	waitMeasure(t, s)
	if strings.Join(order, ",") != "mid:q4" || keeps[0] {
		t.Errorf("evict run: %v %v", order, keeps)
	}
}

func TestMeasureCancel(t *testing.T) {
	s, fake, _ := newTestServer(t)
	fake.tags = threeModels
	s.bench = bench.Open(t.TempDir())
	started := make(chan struct{}, 1)
	s.benchFn = func(ctx context.Context, name string, keep bool) (bench.Result, error) {
		select {
		case started <- struct{}{}:
		default:
		}
		<-ctx.Done()
		return bench.Result{}, ctx.Err()
	}
	if _, err := s.startMeasure(context.Background(), measureRequest{}); err != nil {
		t.Fatal(err)
	}
	<-started
	if _, err := s.startMeasure(context.Background(), measureRequest{}); err == nil {
		t.Error("a second run should be refused")
	}
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("DELETE", "http://localhost/api/library/measure", nil)
	s.ServeHTTP(rec, req)
	if rec.Code != 200 {
		t.Fatalf("cancel: %d", rec.Code)
	}
	st := waitMeasure(t, s)
	if !st.Cancelled || st.Done != 0 {
		t.Errorf("state after cancel: %+v", st)
	}
}

func TestMeasureHandlers(t *testing.T) {
	s, _, _ := newTestServer(t)
	s.bench = bench.Open(t.TempDir())
	s.benchFn = func(ctx context.Context, name string, keep bool) (bench.Result, error) {
		return bench.Result{Model: name, Digest: "dq", GenTPS: 7}, nil
	}
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "http://localhost/api/library/measure", strings.NewReader(`{"names":["qwen3:8b"]}`))
	req.Header.Set("Content-Type", "application/json")
	s.ServeHTTP(rec, req)
	if rec.Code != http.StatusAccepted {
		t.Fatalf("start: %d %s", rec.Code, rec.Body)
	}
	waitMeasure(t, s)
	rec = httptest.NewRecorder()
	s.ServeHTTP(rec, httptest.NewRequest("GET", "http://localhost/api/library/measure", nil))
	var st measureState
	json.Unmarshal(rec.Body.Bytes(), &st)
	if st.Running || st.Total != 1 || st.Items[0].GenTPS != 7 {
		t.Errorf("state: %+v", st)
	}
	// An unknown model leaves nothing to do.
	rec = httptest.NewRecorder()
	req = httptest.NewRequest("POST", "http://localhost/api/library/measure", strings.NewReader(`{"names":["nope"]}`))
	req.Header.Set("Content-Type", "application/json")
	s.ServeHTTP(rec, req)
	if rec.Code != http.StatusConflict {
		t.Errorf("unknown: %d", rec.Code)
	}
}

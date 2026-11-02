package server

import (
	"strings"
	"testing"

	"assay/internal/service"
	"assay/internal/update"
)

const (
	local      = "127.0.0.1:5555"
	remoteAddr = "192.168.1.20:5555"
)

func TestServiceAPI(t *testing.T) {
	s, _, _ := newTestServer(t)
	s.service = service.NewFake("/x/app") // never the real service manager
	s.serviceArgs = []string{"-listen", "0.0.0.0:8484"}

	rec := doFrom(s, local, "GET", "/api/service", "")
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), `"enabled":false`) || !strings.Contains(rec.Body.String(), `"supported":true`) {
		t.Fatalf("off: %d %s", rec.Code, rec.Body)
	}
	rec = doFrom(s, local, "PUT", "/api/service", `{"enabled":true}`)
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), `"enabled":true`) || !strings.Contains(rec.Body.String(), `/x/app -listen 0.0.0.0:8484`) {
		t.Fatalf("on: %d %s", rec.Code, rec.Body)
	}
	rec = doFrom(s, local, "PUT", "/api/service", `{"enabled":false}`)
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), `"enabled":false`) {
		t.Fatalf("off again: %d %s", rec.Code, rec.Body)
	}
}

func TestServiceAPIRefusesRemote(t *testing.T) {
	s, _, _ := newTestServer(t)
	fake := service.NewFake("/x/app")
	s.service = fake
	rec := doFrom(s, remoteAddr, "PUT", "/api/service", `{"enabled":true}`)
	if rec.Code != 403 || fake.Status(nil).Enabled {
		t.Fatalf("remote PUT: %d %s", rec.Code, rec.Body)
	}
	// A remote GET learns nothing: not even the path of the binary.
	fake.Install(nil, []string{"-secret"}, false)
	rec = doFrom(s, remoteAddr, "GET", "/api/service", "")
	if rec.Code != 200 || strings.Contains(rec.Body.String(), "secret") || !strings.Contains(rec.Body.String(), `"supported":false`) {
		t.Fatalf("remote GET: %d %s", rec.Code, rec.Body)
	}
}

func TestServiceAbsent(t *testing.T) {
	s, _, _ := newTestServer(t) // no service manager
	if rec := doFrom(s, local, "PUT", "/api/service", `{"enabled":true}`); rec.Code != 403 {
		t.Fatalf("PUT: %d %s", rec.Code, rec.Body)
	}
	if rec := doFrom(s, local, "GET", "/api/service", ""); !strings.Contains(rec.Body.String(), `"supported":false`) {
		t.Fatalf("GET: %s", rec.Body)
	}
}

func TestSettingsShowsServiceRowOnlyLocally(t *testing.T) {
	s, _, _ := newTestServer(t)
	s.service = service.NewFake("/x/app")
	s.version = "v9.9.9"
	if body := doFrom(s, local, "GET", "/settings", "").Body.String(); !strings.Contains(body, `id="svc-on"`) || !strings.Contains(body, "v9.9.9") {
		t.Error("loopback settings lacks the start-on-boot switch or the version")
	}
	if body := doFrom(s, remoteAddr, "GET", "/settings", "").Body.String(); strings.Contains(body, `id="svc-on"`) {
		t.Error("remote settings shows the start-on-boot switch")
	}
}

func TestUpdateAPIHiddenWithoutRepo(t *testing.T) {
	s, _, _ := newTestServer(t)
	rec := doFrom(s, local, "GET", "/api/update", "")
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), `"enabled":false`) || !strings.Contains(rec.Body.String(), `"available":false`) {
		t.Fatalf("%d %s", rec.Code, rec.Body)
	}
	s.updates = update.Open("", "", "v1.0.0")
	rec = doFrom(s, local, "PUT", "/api/update", `{"check":false}`)
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), `"check":false`) {
		t.Fatalf("%d %s", rec.Code, rec.Body)
	}
}

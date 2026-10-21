package server

import (
	"strings"
	"testing"
	"time"

	"ollamui/internal/chats"
	"ollamui/internal/presets"
)

func TestGroupChats(t *testing.T) {
	now := time.Date(2026, 10, 20, 15, 0, 0, 0, time.Local)
	mk := func(id string, d time.Time) chats.Summary { return chats.Summary{ID: id, Updated: d} }
	got := groupChats([]chats.Summary{
		mk("a", now.Add(-time.Hour)),
		mk("b", now.AddDate(0, 0, -1)),
		mk("c", now.AddDate(0, 0, -5)),
		mk("d", now.AddDate(0, 0, -30)),
		mk("e", now.AddDate(0, 0, -40)),
	}, now)
	want := []struct {
		label string
		n     int
	}{{"Today", 1}, {"Yesterday", 1}, {"Previous 7 days", 1}, {"Older", 2}}
	if len(got) != len(want) {
		t.Fatalf("got %d groups", len(got))
	}
	for i, w := range want {
		if got[i].Label != w.label || len(got[i].Chats) != w.n {
			t.Errorf("group %d = %s/%d", i, got[i].Label, len(got[i].Chats))
		}
	}
	if g := groupChats(nil, now); len(g) != 0 {
		t.Errorf("empty input gave groups")
	}
}

func TestRecentPresets(t *testing.T) {
	all := []*presets.Preset{{Name: "a"}, {Name: "b"}, {Name: "c"}, {Name: "d"}}
	list := []chats.Summary{{Preset: "c"}, {Preset: ""}, {Preset: "gone"}, {Preset: "c"}, {Preset: "b"}}
	var names []string
	for _, p := range recentPresets(list, all, 3) {
		names = append(names, p.Name)
	}
	if got := strings.Join(names, ","); got != "c,b,a" {
		t.Fatalf("recentPresets = %s, want c,b,a", got)
	}
	if got := recentPresets(nil, nil, 3); len(got) != 0 {
		t.Fatalf("no presets gave %v", got)
	}
}

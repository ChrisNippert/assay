package remote

import (
	"path/filepath"
	"testing"
	"time"
)

func TestSpend(t *testing.T) {
	path := filepath.Join(t.TempDir(), "spend.json")
	sp, _ := OpenSpend(path)
	sp.Add("api:a", 0.01, 100)
	sp.Add("api:a", 0.02, 50)
	sp.Add("api:b", 0.5, 10)
	// An old day, written as if from before.
	sp.days["2000-01-01"] = map[string]*Tally{"api:a": {Cost: 1, Calls: 3, Tokens: 9}}
	sum := sp.Summary(time.Now())
	if a := sum.Models["api:a"]; a.Today.Calls != 2 || a.Today.Tokens != 150 || a.Month.Cost != 0.03 || a.All.Cost != 1.03 || a.All.Calls != 5 {
		t.Errorf("a %+v", a)
	}
	if sum.Total.Today.Cost != 0.53 || sum.Since != "2000-01-01" {
		t.Errorf("total %+v since %s", sum.Total, sum.Since)
	}
	again, err := OpenSpend(path)
	if err != nil || again.Summary(time.Now()).Total.Today.Calls != 3 {
		t.Errorf("not saved: %v %+v", err, again.Summary(time.Now()).Total)
	}
}

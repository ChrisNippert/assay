package server

import "testing"

func TestTogetherOpts(t *testing.T) {
	g := &together{lead: "qwen", worker: "e2b", slots: 3}
	lead, worker := g.opts(member{name: "qwen"}), g.opts(member{name: "e2b"})
	if lead["num_ctx"] != togetherLeadCtx || lead["num_parallel"] != nil || lead["vision"] != "off" || lead["keep_with"] != "qwen,e2b" {
		t.Errorf("lead %v", lead)
	}
	if worker["num_ctx"] != 3*togetherWorkerCtx || worker["num_parallel"] != 3 || worker["keep_with"] != "qwen,e2b" {
		t.Errorf("worker %v", worker)
	}
	// One model for both: one server, the lead's context at least.
	same := &together{lead: "e2b", worker: "e2b", slots: 2}
	if o := same.opts(member{name: "e2b"}); o["num_ctx"] != togetherLeadCtx || o["num_parallel"] != 2 {
		t.Errorf("same %v", o)
	}
}

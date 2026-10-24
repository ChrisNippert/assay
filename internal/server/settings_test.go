package server

import "testing"

func TestSettingsSummaryExpertKeys(t *testing.T) {
	got := settingsSummary(map[string]any{"n_cpu_moe": int64(14), "load_mode": "none"})
	if got != "read into RAM · experts of 14 layers on the CPU" {
		t.Errorf("summary = %q", got)
	}
	if got := settingsSummary(map[string]any{"n_cpu_moe": int64(-1), "load_mode": "mlock"}); got != "locked in RAM · all experts on the CPU" {
		t.Errorf("summary = %q", got)
	}
}

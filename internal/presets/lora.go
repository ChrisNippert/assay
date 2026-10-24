package presets

import (
	"fmt"
	"strings"
)

// LoRA is a LoRA adapter a model loads with, at a scale from 0 to 2.
type LoRA struct {
	Name  string
	Scale float64
}

// maxLoRA is how many adapters a model may load with.
const maxLoRA = 8

// LoRAs reads a "lora" option: a list of {name, scale} as JSON, TOML or a
// request gives it. ok is false if v isn't one.
func LoRAs(v any) (out []LoRA, ok bool) {
	var items []map[string]any
	switch l := v.(type) {
	case []map[string]any:
		items = l
	case []any:
		for _, x := range l {
			m, isMap := x.(map[string]any)
			if !isMap {
				return nil, false
			}
			items = append(items, m)
		}
	case []LoRA:
		return l, true
	default:
		return nil, false
	}
	for _, m := range items {
		name, _ := m["name"].(string)
		scale, isNum := toFloat(m["scale"])
		if name == "" || !isNum {
			return nil, false
		}
		out = append(out, LoRA{Name: name, Scale: scale})
	}
	return out, true
}

// cleanLoRA validates the lora option and returns it in the form the
// settings are stored in: a list of {name, scale} maps.
func cleanLoRA(v any) ([]map[string]any, error) {
	list, ok := LoRAs(v)
	if !ok {
		return nil, fmt.Errorf("option lora must be a list of {name, scale}")
	}
	if len(list) > maxLoRA {
		return nil, fmt.Errorf("a model loads at most %d adapters", maxLoRA)
	}
	seen := map[string]bool{}
	out := make([]map[string]any, 0, len(list))
	for _, l := range list {
		if l.Name == "" || len(l.Name) > 300 || strings.ContainsAny(l.Name, "\n\r") {
			return nil, fmt.Errorf("adapter name %q isn't valid", l.Name)
		}
		if seen[l.Name] {
			return nil, fmt.Errorf("adapter %s is listed twice", l.Name)
		}
		seen[l.Name] = true
		if !(l.Scale >= 0 && l.Scale <= 2) {
			return nil, fmt.Errorf("adapter %s: scale must be between 0 and 2", l.Name)
		}
		out = append(out, map[string]any{"name": l.Name, "scale": l.Scale})
	}
	return out, nil
}

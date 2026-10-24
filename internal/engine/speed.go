package engine

import (
	"strconv"
	"strings"
)

// Speed-up settings: "speedup" picks what drafts for the model, and
// "spec_ngram" separately turns on guessing from the conversation.
const (
	SpeedAuto = "auto" // the model's MTP head if it has one (a chosen draft model instead), plus n-gram guessing
	SpeedOff  = "off"  // no MTP head or draft model
	SpeedMTP  = "mtp"  // the model's MTP head, even over a chosen draft model
)

// DefaultNGram is the n-gram type "on" means. ngram-mod costs nothing when
// it has nothing to guess: 40.7 vs 40.8 tok/s on prose, and on a
// code-editing prompt 39 -> 79 once the code was in the conversation.
const DefaultNGram = "ngram-mod"

// NGramTypes are the --spec-type values that guess from the conversation
// and need no extra model.
var NGramTypes = map[string]bool{"ngram-mod": true, "ngram-cache": true, "ngram-simple": true, "ngram-map-k": true, "ngram-map-k4v": true}

// planSpeed turns the speed-up settings in o into what runs: MTP, Draft and
// NGram. Auto never picks a draft model by itself (a 2B draft made a 9B
// slower on this GPU, 29 vs 51 tok/s on prose), only an MTP head, which
// roughly doubles code and adds 40% to prose; a draft the user chose
// still wins over the head.
func planSpeed(m *Model, o loadOpts) loadOpts {
	speed := o.Speed
	if speed == "" {
		speed = SpeedAuto
	}
	ng := o.NGram
	switch {
	case ng == "on" || ng == "" && speed == SpeedAuto:
		ng = DefaultNGram
	case !NGramTypes[ng]:
		ng = ""
	}
	o.MTP = false
	switch {
	case o.Embedding || speed == SpeedOff:
		o.Draft = ""
	case speed == SpeedMTP:
		o.Draft, o.MTP = "", m.MTP != ""
	case o.Draft == "":
		o.MTP = m.MTP != ""
	}
	if o.Embedding {
		ng = ""
	}
	o.NGram, o.Speed = ng, ""
	return o
}

// DraftType is the --spec-type for a draft model of this kind (the
// Model.Draft of a downloaded draft, "" for an ordinary small model).
func DraftType(kind string) string {
	switch kind {
	case "dspark", "dflash", "eagle3":
		return "draft-" + kind
	}
	return "draft-simple"
}

// specArgs are the llama-server flags for planned speculation. One -md file
// serves as the draft: the MTP head, or the draft model.
func (e *Engine) specArgs(m *Model, o loadOpts) []string {
	var a, types []string
	switch {
	case o.MTP && m.MTP != "":
		a = append(a, "-md", e.store.Blob(m.MTP))
		types = append(types, "draft-mtp")
	case o.Draft != "":
		d, err := e.store.Get(o.Draft)
		if err != nil {
			break
		}
		// Its GPU layers follow --fit, like the target's.
		a = append(a, "-md", e.store.Blob(d.Files[0]))
		types = append(types, DraftType(d.Draft))
	}
	if len(a) > 0 && o.DraftMax > 0 {
		a = append(a, "--spec-draft-n-max", strconv.Itoa(o.DraftMax))
	}
	if o.NGram != "" {
		types = append(types, o.NGram)
	}
	if len(types) > 0 {
		a = append(a, "--spec-type", strings.Join(types, ","))
	}
	return a
}

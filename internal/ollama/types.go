package ollama

import "time"

// Details describes a model's format and quantization.
type Details struct {
	ParentModel       string   `json:"parent_model"`
	Format            string   `json:"format"`
	Family            string   `json:"family"`
	Families          []string `json:"families"`
	ParameterSize     string   `json:"parameter_size"`
	QuantizationLevel string   `json:"quantization_level"`
}

// Model is an installed model, as listed by /api/tags.
type Model struct {
	Name       string    `json:"name"`
	Model      string    `json:"model"`
	ModifiedAt time.Time `json:"modified_at"`
	Size       int64     `json:"size"`
	Digest     string    `json:"digest"`
	Details    Details   `json:"details"`
}

// RunningModel is a model loaded in memory, as listed by /api/ps.
type RunningModel struct {
	Name          string    `json:"name"`
	Model         string    `json:"model"`
	Size          int64     `json:"size"`
	SizeVRAM      int64     `json:"size_vram"`
	Digest        string    `json:"digest"`
	ExpiresAt     time.Time `json:"expires_at"`
	ContextLength int       `json:"context_length"`
	Details       Details   `json:"details"`
}

// ShowResponse is the result of /api/show.
type ShowResponse struct {
	Modelfile    string         `json:"modelfile"`
	Parameters   string         `json:"parameters"`
	Template     string         `json:"template"`
	System       string         `json:"system"`
	Details      Details        `json:"details"`
	ModelInfo    map[string]any `json:"model_info"`
	Capabilities []string       `json:"capabilities"`
	ModifiedAt   time.Time      `json:"modified_at"`
}

// Message is one turn in a chat.
type Message struct {
	Role     string   `json:"role"`
	Content  string   `json:"content"`
	Thinking string   `json:"thinking,omitempty"`
	Images   []string `json:"images,omitempty"`
}

// ChatRequest is the body of /api/chat.
type ChatRequest struct {
	Model    string         `json:"model"`
	Messages []Message      `json:"messages"`
	Stream   *bool          `json:"stream,omitempty"`
	Think    any            `json:"think,omitempty"` // bool, or "low"/"medium"/"high" for some models
	Options  map[string]any `json:"options,omitempty"`
	// KeepAlive is how long the model stays loaded after the request.
	KeepAlive any `json:"keep_alive,omitempty"`
}

// ChatResponse is one streamed chunk from /api/chat. The last chunk has
// Done set and carries the timing stats.
type ChatResponse struct {
	Model              string    `json:"model"`
	CreatedAt          time.Time `json:"created_at"`
	Message            Message   `json:"message"`
	Done               bool      `json:"done"`
	DoneReason         string    `json:"done_reason"`
	TotalDuration      int64     `json:"total_duration"`
	LoadDuration       int64     `json:"load_duration"`
	PromptEvalCount    int       `json:"prompt_eval_count"`
	PromptEvalDuration int64     `json:"prompt_eval_duration"`
	EvalCount          int       `json:"eval_count"`
	EvalDuration       int64     `json:"eval_duration"`
}

// PullProgress is one streamed update from /api/pull.
type PullProgress struct {
	Status    string `json:"status"`
	Digest    string `json:"digest,omitempty"`
	Total     int64  `json:"total,omitempty"`
	Completed int64  `json:"completed,omitempty"`
}

// CreateRequest is the body of /api/create. With From set to an existing
// model, Ollama reuses its weights and only stores the new settings.
type CreateRequest struct {
	Model      string         `json:"model"`
	From       string         `json:"from,omitempty"`
	Parameters map[string]any `json:"parameters,omitempty"`
	System     string         `json:"system,omitempty"`
	Template   string         `json:"template,omitempty"`
	Stream     *bool          `json:"stream,omitempty"`
}

// Status is a streamed status line from /api/create.
type Status struct {
	Status string `json:"status"`
}

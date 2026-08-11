package ollamaclient

import (
	"fmt"
	"os"
	"time"
)

type StatusError struct {
	Status       string `json:"status,omitempty"`
	ErrorMessage string `json:"error"`
	StatusCode   int    `json:"code,omitempty"`
}

func (e StatusError) Error() string {
	switch {
	case e.Status != "" && e.ErrorMessage != "":
		return fmt.Sprintf("%s: %s", e.Status, e.ErrorMessage)
	case e.Status != "":
		return e.Status
	case e.ErrorMessage != "":
		return e.ErrorMessage
	default:
		// this should not happen
		return "something went wrong, please see the ollama server logs for details"
	}
}

type GenerateRequest struct {
	Model     string `json:"model"`
	Prompt    string `json:"prompt"`
	System    string `json:"system"`
	Template  string `json:"template"`
	Context   []int  `json:"context,omitempty"`
	Stream    *bool  `json:"stream"`
	KeepAlive string `json:"keep_alive,omitempty"`

	Options Options `json:"options"`
}

type ImageData []byte

type Message struct {
	Role    string      `json:"role"` // one of ["system", "user", "assistant", "tool"]
	Content string      `json:"content"`
	Images  []ImageData `json:"images,omitempty"`

	// Thinking 是思考模型（Qwen3 等）在 think 开启时返回的推理内容。
	// 流式时逐 chunk 到达（此时 content 为空），思考结束后才输出 content。
	Thinking string `json:"thinking,omitempty"`

	// ToolCalls 是 assistant 消息里模型发起的工具调用（Ollama /api/chat 工具调用格式）。
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	// ToolCallID 用于 role="tool" 的结果消息，关联对应的 assistant tool_calls。
	// Ollama 原生 /api/chat 主要按顺序匹配工具结果，可能忽略此字段；保留以便兼容支持该字段的版本。
	ToolCallID string `json:"tool_call_id,omitempty"`
}

// Tool 是 /api/chat 请求里的工具定义，采用 OpenAI 兼容形态。
type Tool struct {
	Type     string       `json:"type"` // 固定为 "function"
	Function ToolFunction `json:"function"`
}

// ToolFunction 是工具的名称、描述与参数 JSON Schema。
type ToolFunction struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	// Parameters 是参数 JSON Schema（对象）。用 any 直接透传上层 llms.FunctionDefinition.Parameters。
	Parameters any `json:"parameters,omitempty"`
}

// ToolCall 是 assistant 消息里模型发起的一次工具调用。
// Ollama 原生格式只含 function（无顶层 id/type 字段）。
type ToolCall struct {
	Function ToolCallFunction `json:"function"`
}

// ToolCallFunction 是工具调用的名称与参数对象。
type ToolCallFunction struct {
	Name string `json:"name"`
	// Arguments 是参数对象（Ollama 用 JSON 对象，而非 OpenAI 的 JSON 字符串）。
	Arguments map[string]any `json:"arguments"`
}

type ChatRequest struct {
	Model     string     `json:"model"`
	Messages  []*Message `json:"messages"`
	Stream    bool       `json:"stream,omitempty"`
	Format    string     `json:"format"`
	KeepAlive string     `json:"keep_alive,omitempty"`

	// Tools 声明可供模型调用的工具列表；为空时 omitempty 不发送该字段。
	Tools []Tool `json:"tools,omitempty"`

	// Think 是 Ollama 0.9.0+ 的思考开关，必须放在请求【顶层】（不是 options 里——
	// 实测放 options 内会被忽略）。用 *bool：nil=省略（用模型默认）；&true=开；&false=关。
	// Qwen3 等模型默认开启思考，ThinkingModeNone 时必须显式 &false，否则模型一直输出
	// reasoning 直到 num_predict 截断、正文为空。
	Think *bool `json:"think,omitempty"`

	Options Options `json:"options"`
}

type Metrics struct {
	TotalDuration      time.Duration `json:"total_duration,omitempty"`
	LoadDuration       time.Duration `json:"load_duration,omitempty"`
	PromptEvalCount    int           `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration time.Duration `json:"prompt_eval_duration,omitempty"`
	EvalCount          int           `json:"eval_count,omitempty"`
	EvalDuration       time.Duration `json:"eval_duration,omitempty"`
}

type EmbeddingRequest struct {
	Model     string  `json:"model"`
	Input     string  `json:"input"`
	Options   Options `json:"options"`
	KeepAlive string  `json:"keep_alive,omitempty"`
}

type EmbeddingResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
}

type GenerateResponse struct {
	CreatedAt          time.Time     `json:"created_at"`
	Model              string        `json:"model"`
	Response           string        `json:"response"`
	Context            []int         `json:"context,omitempty"`
	TotalDuration      time.Duration `json:"total_duration,omitempty"`
	LoadDuration       time.Duration `json:"load_duration,omitempty"`
	PromptEvalCount    int           `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration time.Duration `json:"prompt_eval_duration,omitempty"`
	EvalCount          int           `json:"eval_count,omitempty"`
	EvalDuration       time.Duration `json:"eval_duration,omitempty"`
	Done               bool          `json:"done"`
}

type ChatResponse struct {
	Model     string    `json:"model"`
	CreatedAt time.Time `json:"created_at"`
	Message   *Message  `json:"message,omitempty"`

	Done bool `json:"done"`
	// DoneReason 是停止原因："stop"（自然结束）/ "length"（命中 num_predict 上限被截断）等。
	DoneReason string `json:"done_reason,omitempty"`

	Metrics
}

func (r *GenerateResponse) Summary() {
	if r.TotalDuration > 0 {
		fmt.Fprintf(os.Stderr, "total duration:       %v\n", r.TotalDuration)
	}

	if r.LoadDuration > 0 {
		fmt.Fprintf(os.Stderr, "load duration:        %v\n", r.LoadDuration)
	}

	if r.PromptEvalCount > 0 {
		fmt.Fprintf(os.Stderr, "prompt eval count:    %d token(s)\n", r.PromptEvalCount)
	}

	if r.PromptEvalDuration > 0 {
		fmt.Fprintf(os.Stderr, "prompt eval duration: %s\n", r.PromptEvalDuration)
		fmt.Fprintf(os.Stderr, "prompt eval rate:     %.2f tokens/s\n",
			float64(r.PromptEvalCount)/r.PromptEvalDuration.Seconds())
	}

	if r.EvalCount > 0 {
		fmt.Fprintf(os.Stderr, "eval count:           %d token(s)\n", r.EvalCount)
	}

	if r.EvalDuration > 0 {
		fmt.Fprintf(os.Stderr, "eval duration:        %s\n", r.EvalDuration)
		fmt.Fprintf(os.Stderr, "eval rate:            %.2f tokens/s\n", float64(r.EvalCount)/r.EvalDuration.Seconds())
	}
}

type Runner struct {
	NumCtx             int     `json:"num_ctx,omitempty"`
	NumBatch           int     `json:"num_batch,omitempty"`
	NumGQA             int     `json:"num_gqa,omitempty"`
	NumGPU             int     `json:"num_gpu,omitempty"`
	MainGPU            int     `json:"main_gpu,omitempty"`
	NumThread          int     `json:"num_thread,omitempty"`
	RopeFrequencyBase  float32 `json:"rope_frequency_base,omitempty"`
	RopeFrequencyScale float32 `json:"rope_frequency_scale,omitempty"`
	LogitsAll          bool    `json:"logits_all,omitempty"`
	VocabOnly          bool    `json:"vocab_only,omitempty"`
	UseMMap            bool    `json:"use_mmap,omitempty"`
	UseMLock           bool    `json:"use_mlock,omitempty"`
	EmbeddingOnly      bool    `json:"embedding_only,omitempty"`
	UseNUMA            bool    `json:"numa,omitempty"`
	F16KV              bool    `json:"f16_kv,omitempty"`
	LowVRAM            bool    `json:"low_vram,omitempty"`
}

type Options struct {
	Stop []string `json:"stop,omitempty"`
	Runner
	RepeatLastN      int     `json:"repeat_last_n,omitempty"`
	Seed             int     `json:"seed,omitempty"`
	TopK             int     `json:"top_k,omitempty"`
	NumKeep          int     `json:"num_keep,omitempty"`
	Mirostat         int     `json:"mirostat,omitempty"`
	NumPredict       int     `json:"num_predict,omitempty"`
	Temperature      float32 `json:"temperature"`
	TypicalP         float32 `json:"typical_p,omitempty"`
	RepeatPenalty    float32 `json:"repeat_penalty,omitempty"`
	PresencePenalty  float32 `json:"presence_penalty,omitempty"`
	FrequencyPenalty float32 `json:"frequency_penalty,omitempty"`
	TFSZ             float32 `json:"tfs_z,omitempty"`
	MirostatTau      float32 `json:"mirostat_tau,omitempty"`
	MirostatEta      float32 `json:"mirostat_eta,omitempty"`
	TopP             float32 `json:"top_p,omitempty"`
	PenalizeNewline  bool    `json:"penalize_newline,omitempty"`
	// 注意：思考开关 think 不在此处，而是 ChatRequest 的【顶层】字段 Think。
	// 实测 ollama 会忽略 options 内的 think，只认顶层 think。
}

type PullRequest struct {
	Model  string `json:"model"`
	Stream bool   `json:"stream,omitempty"`
}

type PullResponse struct {
	Status          string  `json:"status"`
	Digest          string  `json:"digest,omitempty"`
	Total           int64   `json:"total,omitempty"`
	Completed       int64   `json:"completed,omitempty"`
	DownloadPercent float64 `json:"percent,omitempty"`
	Error           string  `json:"error,omitempty"`
}

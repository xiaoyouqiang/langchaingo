package ollama

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/tmc/langchaingo/callbacks"
	"github.com/tmc/langchaingo/httputil"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/ollama/internal/ollamaclient"
)

var (
	ErrEmptyResponse       = errors.New("no response")
	ErrIncompleteEmbedding = errors.New("not all input got embedded")
	ErrPullError           = errors.New("ollama model pull error")
	ErrPullTimeout         = errors.New("ollama model pull deadline exceeded")
)

// LLM is a ollama LLM implementation.
type LLM struct {
	CallbacksHandler callbacks.Handler
	client           *ollamaclient.Client
	options          options
}

var (
	_ llms.Model          = (*LLM)(nil)
	_ llms.ReasoningModel = (*LLM)(nil)
)

// New creates a new ollama LLM implementation.
func New(opts ...Option) (*LLM, error) {
	o := options{}
	for _, opt := range opts {
		opt(&o)
	}

	if o.httpClient == nil && o.retryConfig != nil {
		o.httpClient = httputil.NewClientWithRetry(o.retryConfig)
	}

	client, err := ollamaclient.NewClient(o.ollamaServerURL, o.httpClient)
	if err != nil {
		return nil, err
	}

	return &LLM{client: client, options: o}, nil
}

// SupportsReasoning implements the ReasoningModel interface.
// Returns true if the current model supports reasoning/thinking.
func (o *LLM) SupportsReasoning() bool {
	// Check if the model supports reasoning based on model name patterns
	model := strings.ToLower(o.options.model)

	// Ollama models that support reasoning/thinking:
	// - deepseek-r1 models (DeepSeek reasoning models)
	// - qwq models (Alibaba's QwQ reasoning models)
	// - Models with "reasoning" or "thinking" in the name
	if strings.Contains(model, "deepseek-r1") ||
		strings.Contains(model, "qwq") ||
		strings.Contains(model, "reasoning") ||
		strings.Contains(model, "thinking") {
		return true
	}

	// Future: could check model capabilities via Ollama API when available
	return false
}

// Call Implement the call interface for LLM.
func (o *LLM) Call(ctx context.Context, prompt string, options ...llms.CallOption) (string, error) {
	return llms.GenerateFromSinglePrompt(ctx, o, prompt, options...)
}

// GenerateContent implements the Model interface.
func (o *LLM) GenerateContent(ctx context.Context, messages []llms.MessageContent, options ...llms.CallOption) (*llms.ContentResponse, error) { // nolint: lll, cyclop, funlen
	if o.CallbacksHandler != nil {
		o.CallbacksHandler.HandleLLMGenerateContentStart(ctx, messages)
	}

	opts := llms.CallOptions{}
	for _, opt := range options {
		opt(&opts)
	}

	// Check if context caching is enabled
	var contextCache *ContextCache
	if opts.Metadata != nil {
		if cache, ok := opts.Metadata["context_cache"].(*ContextCache); ok {
			contextCache = cache
		}
	}

	// Override LLM model if set as llms.CallOption
	model := o.options.model
	if opts.Model != "" {
		model = opts.Model
	}

	// Pull model if enabled
	if o.options.pullModel {
		if err := o.pullModelIfNeeded(ctx, model); err != nil {
			return nil, fmt.Errorf("%w: %w", ErrPullError, err)
		}
	}

	// Our input is a sequence of MessageContent, each of which potentially has
	// a sequence of Part that could be text, images etc.
	// We have to convert it to a format Ollama undestands: ChatRequest, which
	// has a sequence of Message, each of which has a role and content - single
	// text + potential images.
	chatMsgs := make([]*ollamaclient.Message, 0, len(messages))
	for _, mc := range messages {
		// 工具结果消息：role="tool"，每个 ToolCallResponse 生成一条独立消息，
		// 携带 tool_call_id 以关联对应的 assistant tool_calls。
		if mc.Role == llms.ChatMessageTypeTool {
			for _, p := range mc.Parts {
				if tcr, ok := p.(llms.ToolCallResponse); ok {
					chatMsgs = append(chatMsgs, &ollamaclient.Message{
						Role:       "tool",
						Content:    tcr.Content,
						ToolCallID: tcr.ToolCallID,
					})
				}
			}
			continue
		}

		msg := &ollamaclient.Message{Role: typeToRole(mc.Role)}

		// Look at all the parts in mc; expect to find a single Text part and
		// any number of binary parts. Assistant 消息还可能携带 ToolCall parts。
		var text string
		foundText := false
		var images []ollamaclient.ImageData
		var toolCalls []ollamaclient.ToolCall

		for _, p := range mc.Parts {
			switch pt := p.(type) {
			case llms.TextContent:
				if foundText {
					return nil, errors.New("expecting a single Text content")
				}
				foundText = true
				text = pt.Text
			case llms.BinaryContent:
				images = append(images, ollamaclient.ImageData(pt.Data))
			case llms.ToolCall:
				// assistant 消息里模型发起的工具调用。Ollama 要参数对象，
				// 而 llms.FunctionCall.Arguments 是 JSON 字符串，这里反序列化成 map。
				tc := ollamaclient.ToolCall{}
				if pt.FunctionCall != nil {
					tc.Function.Name = pt.FunctionCall.Name
					if pt.FunctionCall.Arguments != "" {
						args := map[string]any{}
						if err := json.Unmarshal([]byte(pt.FunctionCall.Arguments), &args); err != nil {
							return nil, fmt.Errorf("ollama: invalid tool call arguments for %q: %w", pt.FunctionCall.Name, err)
						}
						tc.Function.Arguments = args
					}
				}
				toolCalls = append(toolCalls, tc)
			default:
				return nil, errors.New("only support Text, BinaryContent and ToolCall parts right now")
			}
		}

		msg.Content = text
		msg.Images = images
		msg.ToolCalls = toolCalls
		chatMsgs = append(chatMsgs, msg)
	}

	format := o.options.format
	if opts.JSONMode {
		format = "json"
	}

	// Get our ollamaOptions from llms.CallOptions（数值类参数；think 开关单独走顶层）
	ollamaOptions := makeOllamaOptionsFromOptions(o.options.ollamaOptions, opts)

	// Convert llms tools to ollama /api/chat tool definitions (OpenAI 兼容形态)。
	var ollamaTools []ollamaclient.Tool
	for _, t := range opts.Tools {
		if t.Function == nil {
			continue
		}
		ollamaTools = append(ollamaTools, ollamaclient.Tool{
			Type: "function",
			Function: ollamaclient.ToolFunction{
				Name:        t.Function.Name,
				Description: t.Function.Description,
				Parameters:  t.Function.Parameters,
			},
		})
	}

	req := &ollamaclient.ChatRequest{
		Model:    model,
		Format:   format,
		Messages: chatMsgs,
		Options:  ollamaOptions,
		// 任一流式回调存在即开启流式（content-only 或 reasoning+toolcall）。
		Stream: opts.StreamingFunc != nil || opts.StreamingReasoningFuncAndToolCall != nil,
		Tools:  ollamaTools,
		// think 必须在请求顶层（ollama 忽略 options.think）。
		// 优先用单次请求的 thinking_config（ThinkingModeNone → 显式 false），
		// 没有则回落到提供商级默认 WithThink。
		Think: thinkFlagFromOptions(opts),
	}
	if req.Think == nil {
		req.Think = o.options.think
	}

	keepAlive := o.options.keepAlive
	if keepAlive != "" {
		req.KeepAlive = keepAlive
	}

	var fn ollamaclient.ChatResponseFunc
	streamedResponse := ""
	streamedThinking := ""
	var resp ollamaclient.ChatResponse
	var respToolCalls []ollamaclient.ToolCall

	fn = func(response ollamaclient.ChatResponse) error {
		if response.Message != nil {
			reasoningChunk := response.Message.Thinking
			contentChunk := response.Message.Content

			// 优先走 reasoning+toolcall 回调（能把思考内容透传给上层）；
			// 未设置时回落到 content-only 的 StreamingFunc。
			if opts.StreamingReasoningFuncAndToolCall != nil {
				var chunkToolCalls []llms.ToolCall
				for i, tc := range response.Message.ToolCalls {
					argsBytes, _ := json.Marshal(tc.Function.Arguments)
					chunkToolCalls = append(chunkToolCalls, llms.ToolCall{
						ID:   fmt.Sprintf("call_%d", i),
						Type: "function",
						FunctionCall: &llms.FunctionCall{
							Name:      tc.Function.Name,
							Arguments: string(argsBytes),
						},
					})
				}
				if err := opts.StreamingReasoningFuncAndToolCall(ctx, []byte(reasoningChunk), []byte(contentChunk), chunkToolCalls); err != nil {
					return err
				}
			} else if opts.StreamingFunc != nil {
				if err := opts.StreamingFunc(ctx, []byte(contentChunk)); err != nil {
					return err
				}
			}

			streamedResponse += contentChunk
			streamedThinking += reasoningChunk
			// 累积模型发起的工具调用。工具调用场景下 Ollama 通常在最终的
			// done 消息里一次性返回 tool_calls；流式场景逐条累积以兼容。
			if len(response.Message.ToolCalls) > 0 {
				respToolCalls = append(respToolCalls, response.Message.ToolCalls...)
			}
		}
		if !req.Stream || response.Done {
			resp = response
			resp.Message = &ollamaclient.Message{
				Role:      "assistant",
				Content:   streamedResponse,
				Thinking:  streamedThinking,
				ToolCalls: respToolCalls,
			}
		}
		return nil
	}

	err := o.client.GenerateChat(ctx, req, fn)
	if err != nil {
		if o.CallbacksHandler != nil {
			o.CallbacksHandler.HandleLLMError(ctx, err)
		}
		return nil, err
	}

	// Handle case where Message might be nil (e.g., context cancelled during streaming)
	content := ""
	thinking := ""
	if resp.Message != nil {
		content = resp.Message.Content
		thinking = resp.Message.Thinking
	}

	// Build generation info with standardized fields
	genInfo := map[string]any{
		"CompletionTokens": resp.EvalCount,
		"PromptTokens":     resp.PromptEvalCount,
		"TotalTokens":      resp.EvalCount + resp.PromptEvalCount,
		// 思考内容（think 开启时）；与 openai provider 的 ThinkingContent 字段对齐。
		"ThinkingContent": thinking,
		"ThinkingTokens":  0, // Ollama 不单独统计思考 token
	}

	// If context caching is enabled, track cache usage
	if contextCache != nil {
		if cacheEntry, hit := contextCache.Get(messages); hit {
			// Cache hit - we reused cached context
			genInfo["CachedTokens"] = cacheEntry.ContextTokens
			genInfo["CacheHit"] = true
		} else {
			// Cache miss - store for future use
			contextCache.Put(messages, resp.PromptEvalCount)
			genInfo["CachedTokens"] = 0
			genInfo["CacheHit"] = false
		}
	}

	// Note: Ollama may include thinking in the main content when Think mode is enabled
	// Future versions may provide separate thinking content
	if req.Think != nil && *req.Think {
		genInfo["ThinkingEnabled"] = true
	}

	// Convert ollama tool calls to langchaingo tool calls.
	// Ollama 原生不返回 tool_call id，这里合成稳定 id（call_0、call_1…），
	// 让上层能用同一 id 关联 assistant tool_calls 与随后的 tool 结果消息。
	// Ollama 的 arguments 是对象，langchaingo 要 JSON 字符串，故重新序列化。
	var choiceToolCalls []llms.ToolCall
	for i, tc := range respToolCalls {
		argsBytes, err := json.Marshal(tc.Function.Arguments)
		if err != nil {
			argsBytes = []byte("{}")
		}
		choiceToolCalls = append(choiceToolCalls, llms.ToolCall{
			ID:   fmt.Sprintf("call_%d", i),
			Type: "function",
			FunctionCall: &llms.FunctionCall{
				Name:      tc.Function.Name,
				Arguments: string(argsBytes),
			},
		})
	}

	choices := []*llms.ContentChoice{
		{
			Content:        content,
			GenerationInfo: genInfo,
			ToolCalls:      choiceToolCalls,
			// 思考内容（think 开启时）；ReasoningContent 是跨提供商的标准字段。
			ReasoningContent: thinking,
			// 停止原因："stop" 自然结束 / "length" 命中 num_predict 被截断
			// （思考吃光预算时会出现 length + 正文为空/截断）。
			StopReason: resp.DoneReason,
		},
	}
	// FuncCall 保留首个工具调用，向后兼容只读 FuncCall 的调用方。
	if len(choiceToolCalls) > 0 {
		choices[0].FuncCall = choiceToolCalls[0].FunctionCall
	}

	response := &llms.ContentResponse{Choices: choices}

	if o.CallbacksHandler != nil {
		o.CallbacksHandler.HandleLLMGenerateContentEnd(ctx, response)
	}

	return response, nil
}

func (o *LLM) CreateEmbedding(ctx context.Context, inputTexts []string) ([][]float32, error) {
	// Pull model if enabled
	if o.options.pullModel {
		if err := o.pullModelIfNeeded(ctx, o.options.model); err != nil {
			return nil, err
		}
	}

	embeddings := [][]float32{}

	for _, input := range inputTexts {
		req := &ollamaclient.EmbeddingRequest{
			Input: input,
			Model: o.options.model,
		}
		if o.options.keepAlive != "" {
			req.KeepAlive = o.options.keepAlive
		}

		embedding, err := o.client.CreateEmbedding(ctx, req)
		if err != nil {
			return nil, err
		}

		if len(embedding.Embeddings) == 0 {
			return nil, ErrEmptyResponse
		}

		embeddings = append(embeddings, embedding.Embeddings...)
	}

	if len(inputTexts) != len(embeddings) {
		return embeddings, ErrIncompleteEmbedding
	}

	return embeddings, nil
}

func typeToRole(typ llms.ChatMessageType) string {
	switch typ {
	case llms.ChatMessageTypeSystem:
		return "system"
	case llms.ChatMessageTypeAI:
		return "assistant"
	case llms.ChatMessageTypeHuman:
		fallthrough
	case llms.ChatMessageTypeGeneric:
		return "user"
	case llms.ChatMessageTypeFunction:
		return "function"
	case llms.ChatMessageTypeTool:
		return "tool"
	}
	return ""
}

func makeOllamaOptionsFromOptions(ollamaOptions ollamaclient.Options, opts llms.CallOptions) ollamaclient.Options {
	// Load back CallOptions as ollamaOptions
	ollamaOptions.NumPredict = opts.MaxTokens
	ollamaOptions.Temperature = float32(opts.Temperature)
	ollamaOptions.Stop = opts.StopWords
	ollamaOptions.TopK = opts.TopK
	ollamaOptions.TopP = float32(opts.TopP)
	ollamaOptions.Seed = opts.Seed
	ollamaOptions.RepeatPenalty = float32(opts.RepetitionPenalty)
	ollamaOptions.FrequencyPenalty = float32(opts.FrequencyPenalty)
	ollamaOptions.PresencePenalty = float32(opts.PresencePenalty)

	// 注意：思考开关 think 是 ChatRequest 顶层字段，不在 options 里，
	// 由 GenerateContent 单独计算并赋给 req.Think（ollama 忽略 options.think）。

	return ollamaOptions
}

// thinkFlagFromOptions 根据 llms.CallOptions 里的 thinking_config 计算请求顶层的 think 开关。
//   mode != None → &true；mode == None → &false（显式关闭，Qwen3 等默认开思考）；
//   没有 thinking_config → nil（不下发，用模型默认）。
func thinkFlagFromOptions(opts llms.CallOptions) *bool {
	if opts.Metadata == nil {
		return nil
	}
	config, ok := opts.Metadata["thinking_config"].(*llms.ThinkingConfig)
	if !ok {
		return nil
	}
	thinking := config.Mode != llms.ThinkingModeNone
	return &thinking
}

// pullModelIfNeeded pulls the model if it's not already available.
func (o *LLM) pullModelIfNeeded(ctx context.Context, model string) error {
	// Try to use the model first. If it fails with a model not found error,
	// then pull the model.
	// This is a simple implementation. In production, you might want to
	// implement a more sophisticated check (e.g., using a list endpoint).

	// Apply timeout if configured
	pullCtx := ctx
	if o.options.pullTimeout > 0 {
		var cancel context.CancelFunc
		pullCtx, cancel = context.WithTimeoutCause(ctx, o.options.pullTimeout, ErrPullTimeout)
		defer func() {
			if cancel != nil {
				cancel()
			}
		}()
	}

	// For now, we'll just pull the model without checking.
	// This ensures the model is available but may result in unnecessary pulls.
	req := &ollamaclient.PullRequest{
		Model:  model,
		Stream: false,
	}

	err := o.client.Pull(pullCtx, req)
	if err != nil {
		// Check if the error is due to context timeout
		if errors.Is(err, context.DeadlineExceeded) {
			return err
		}
		// Check if the context has a cause
		if cause := context.Cause(pullCtx); cause != nil {
			return fmt.Errorf("%w: %w", cause, err)
		}
	}
	return err
}

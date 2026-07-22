package openaiclient

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParseStreamingChatResponse_FinishReason(t *testing.T) {
	ctx := context.Background()
	t.Parallel()
	mockBody := `data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"hello"},"finish_reason":"stop"}]}`
	r := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(bytes.NewBufferString(mockBody)),
	}

	req := &ChatRequest{
		StreamingFunc: func(_ context.Context, _ []byte) error {
			return nil
		},
	}

	resp, err := parseStreamingChatResponse(ctx, r, req)

	require.NoError(t, err)
	assert.NotNil(t, resp)
	assert.Equal(t, FinishReason("stop"), resp.Choices[0].FinishReason)
}

func TestParseStreamingChatResponse_ReasoningContent(t *testing.T) {
	ctx := context.Background()
	t.Parallel()
	mockBody := `data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"final answer","reasoning_content":"step-by-step reasoning"},"finish_reason":"stop"}]}`
	r := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(bytes.NewBufferString(mockBody)),
	}

	req := &ChatRequest{
		StreamingFunc: func(_ context.Context, _ []byte) error {
			return nil
		},
	}

	resp, err := parseStreamingChatResponse(ctx, r, req)

	require.NoError(t, err)
	assert.NotNil(t, resp)
	assert.Equal(t, "final answer", resp.Choices[0].Message.Content)
	assert.Equal(t, "step-by-step reasoning", resp.Choices[0].Message.ReasoningContent)
	assert.Equal(t, FinishReason("stop"), resp.Choices[0].FinishReason)
}

func TestParseStreamingChatResponse_ReasoningFunc(t *testing.T) {
	ctx := context.Background()
	t.Parallel()
	mockBody := `
data: {"id":"fa7e4fc5-a05d-4e7b-9a66-a2dd89e91a4e","object":"chat.completion.chunk","created":1738492867,"model":"deepseek-reasoner","system_fingerprint":"fp_7e73fd9a08","choices":[{"index":0,"delta":{"content":null,"reasoning_content":"Okay"},"logprobs":null,"finish_reason":null}]}
`
	r := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(bytes.NewBufferString(mockBody)),
	}

	req := &ChatRequest{
		StreamingReasoningFunc: func(_ context.Context, reasoningChunk, chunk []byte) error {
			t.Logf("reasoningChunk: %s", string(reasoningChunk))
			t.Logf("chunk: %s", string(chunk))
			return nil
		},
	}

	resp, err := parseStreamingChatResponse(ctx, r, req)

	require.NoError(t, err)
	assert.NotNil(t, resp)
	assert.Equal(t, "", resp.Choices[0].Message.Content)
	assert.Equal(t, "Okay", resp.Choices[0].Message.ReasoningContent)
	assert.Equal(t, FinishReason(""), resp.Choices[0].FinishReason)
}

func TestChatMessage_MarshalUnmarshal(t *testing.T) {
	t.Parallel()
	msg := ChatMessage{
		Role:    "assistant",
		Content: "hello",
		FunctionCall: &FunctionCall{
			Name:      "test",
			Arguments: "func",
		},
	}
	text, err := json.Marshal(msg)
	require.NoError(t, err)
	require.Equal(t, `{"role":"assistant","content":"hello","function_call":{"name":"test","arguments":"func"}}`, string(text)) // nolint: lll

	var msg2 ChatMessage
	err = json.Unmarshal(text, &msg2)
	require.NoError(t, err)
	require.Equal(t, msg, msg2)
}

func TestChatMessage_MarshalUnmarshal_WithReasoning(t *testing.T) {
	t.Parallel()
	msg := ChatMessage{
		Role:             "assistant",
		Content:          "final answer",
		ReasoningContent: "step-by-step reasoning",
	}
	text, err := json.Marshal(msg)
	require.NoError(t, err)
	require.Equal(t, `{"role":"assistant","content":"final answer","reasoning_content":"step-by-step reasoning"}`, string(text))

	var msg2 ChatMessage
	err = json.Unmarshal(text, &msg2)
	require.NoError(t, err)
	require.Equal(t, msg, msg2)
}

// TestUpdateToolCalls_Sequential verifies updateToolCalls when the model streams
// parallel tool calls in order (tool 0 fully, then tool 1 fully). This is the
// common case for OpenAI and the current implementation handles it correctly.
func TestUpdateToolCalls_Sequential(t *testing.T) {
	t.Parallel()

	var tools []ToolCall
	// Simulate SSE deltas. In real OpenAI streams each delta carries an `index`
	// field identifying which tool call it belongs to; the ToolCall struct here
	// does not parse that field, so we annotate the intended index in comments.
	deltas := [][]*ToolCall{
		// index=0: new tool A (get_weather)
		{{ID: "call_A", Type: ToolTypeFunction, Function: ToolFunction{Name: "get_weather"}}},
		// index=0: args chunk 1
		{{Type: "", Function: ToolFunction{Arguments: `{"loc`}}},
		// index=0: args chunk 2
		{{Type: "", Function: ToolFunction{Arguments: `ation":"Paris"}`}}},
		// index=1: new tool B (get_time)
		{{ID: "call_B", Type: ToolTypeFunction, Function: ToolFunction{Name: "get_time"}}},
		// index=1: args chunk 1
		{{Type: "", Function: ToolFunction{Arguments: `{"tz`}}},
	}

	for _, d := range deltas {
		_, tools = updateToolCalls(tools, d)
	}

	require.Len(t, tools, 2, "should have two tool calls")
	assert.Equal(t, "call_A", tools[0].ID)
	assert.Equal(t, "get_weather", tools[0].Function.Name)
	assert.Equal(t, `{"location":"Paris"}`, tools[0].Function.Arguments)
	assert.Equal(t, "call_B", tools[1].ID)
	assert.Equal(t, "get_time", tools[1].Function.Name)
	assert.Equal(t, `{"tz`, tools[1].Function.Arguments)
}

// TestUpdateToolCalls_Interleaved verifies updateToolCalls when the model
// interleaves argument deltas across multiple parallel tool calls. The OpenAI
// streaming protocol permits this (each delta carries an `index` field), but
// the current implementation ignores the index and always appends argument
// chunks to the LAST tool in the slice. As a result tool[0]'s arguments are
// incorrectly appended to tool[1].
//
// This test documents the bug and is expected to FAIL against the current
// implementation.
func TestUpdateToolCalls_Interleaved(t *testing.T) {
	t.Parallel()

	var tools []ToolCall
	deltas := [][]*ToolCall{
		// index=0: new tool A
		{{ID: "call_A", Type: ToolTypeFunction, Function: ToolFunction{Name: "get_weather"}}},
		// index=1: new tool B
		{{ID: "call_B", Type: ToolTypeFunction, Function: ToolFunction{Name: "get_time"}}},
		// index=0: args for A
		{{Type: "", Function: ToolFunction{Arguments: `{"loc`}}},
		// index=1: args for B
		{{Type: "", Function: ToolFunction{Arguments: `{"tz`}}},
	}

	for _, d := range deltas {
		_, tools = updateToolCalls(tools, d)
	}

	require.Len(t, tools, 2, "should have two tool calls")

	// Expected (correct) behavior: arguments land on the tool they belong to.
	assert.Equal(t, "call_A", tools[0].ID)
	assert.Equal(t, "get_weather", tools[0].Function.Name)
	assert.Equal(t, `{"loc`, tools[0].Function.Arguments, "tool A should receive its own args")

	assert.Equal(t, "call_B", tools[1].ID)
	assert.Equal(t, "get_time", tools[1].Function.Name)
	assert.Equal(t, `{"tz`, tools[1].Function.Arguments, "tool B should receive its own args")
}

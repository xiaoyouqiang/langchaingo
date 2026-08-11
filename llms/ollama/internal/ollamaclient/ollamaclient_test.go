package ollamaclient

import (
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tmc/langchaingo/internal/httprr"
)

// getOllamaTestURL returns the Ollama server URL to use for testing.
// It uses OLLAMA_HOST if set, otherwise defaults to localhost:11434.
func getOllamaTestURL(t *testing.T, rr *httprr.RecordReplay) string {
	t.Helper()

	// Default to localhost
	baseURL := "http://localhost:11434"

	// Use environment variable if set and we're recording
	if envURL := os.Getenv("OLLAMA_HOST"); envURL != "" && rr.Recording() {
		baseURL = envURL
	}

	return baseURL
}

// checkOllamaEndpoint performs a lightweight health check on the Ollama endpoint.
// Returns true if the endpoint is available, false otherwise.
func checkOllamaEndpoint(baseURL string) bool {
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get(baseURL + "/api/tags")
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

func TestClient_Generate(t *testing.T) {
	ctx := context.Background()

	rr := httprr.OpenForTest(t, http.DefaultTransport)
	defer rr.Close()

	// No auth scrubbing needed for Ollama as it doesn't use API keys

	baseURL := getOllamaTestURL(t, rr)

	// If recording and endpoint is not available, skip the test
	if rr.Recording() && !checkOllamaEndpoint(baseURL) {
		t.Skipf("Ollama endpoint not available at %s", baseURL)
	}

	// Skip if no recording exists and we're not recording
	if rr.Replaying() {
		httprr.SkipIfNoCredentialsAndRecordingMissing(t)
	}

	parsedURL, err := url.Parse(baseURL)
	require.NoError(t, err)

	client, err := NewClient(parsedURL, rr.Client())
	require.NoError(t, err)

	stream := false
	req := &GenerateRequest{
		Model:  "gemma3:1b",
		Prompt: "Hello, how are you?",
		Stream: &stream,
		Options: Options{
			Temperature: 0.0,
			NumPredict:  100,
		},
	}

	var response *GenerateResponse
	err = client.Generate(ctx, req, func(resp GenerateResponse) error {
		response = &resp
		return nil
	})
	require.NoError(t, err)
	assert.NotNil(t, response)
	assert.NotEmpty(t, response.Response)
	assert.True(t, response.Done)
}

func TestClient_GenerateStream(t *testing.T) {
	ctx := context.Background()

	rr := httprr.OpenForTest(t, http.DefaultTransport)
	defer rr.Close()

	baseURL := getOllamaTestURL(t, rr)

	// If recording and endpoint is not available, skip the test
	if rr.Recording() && !checkOllamaEndpoint(baseURL) {
		t.Skipf("Ollama endpoint not available at %s", baseURL)
	}

	// Skip if no recording exists and we're not recording
	if rr.Replaying() {
		httprr.SkipIfNoCredentialsAndRecordingMissing(t)
	}

	parsedURL, err := url.Parse(baseURL)
	require.NoError(t, err)

	client, err := NewClient(parsedURL, rr.Client())
	require.NoError(t, err)

	stream := true
	req := &GenerateRequest{
		Model:  "gemma3:1b",
		Prompt: "Count from 1 to 5",
		Stream: &stream,
		Options: Options{
			Temperature: 0.0,
			NumPredict:  50,
		},
	}

	var responses []GenerateResponse
	err = client.Generate(ctx, req, func(resp GenerateResponse) error {
		responses = append(responses, resp)
		return nil
	})
	require.NoError(t, err)
	assert.NotEmpty(t, responses)
	assert.True(t, responses[len(responses)-1].Done)
}

func TestClient_GenerateChat(t *testing.T) {
	ctx := context.Background()

	rr := httprr.OpenForTest(t, http.DefaultTransport)
	defer rr.Close()

	baseURL := getOllamaTestURL(t, rr)

	// If recording and endpoint is not available, skip the test
	if rr.Recording() && !checkOllamaEndpoint(baseURL) {
		t.Skipf("Ollama endpoint not available at %s", baseURL)
	}

	// Skip if no recording exists and we're not recording
	if rr.Replaying() {
		httprr.SkipIfNoCredentialsAndRecordingMissing(t)
	}

	parsedURL, err := url.Parse(baseURL)
	require.NoError(t, err)

	client, err := NewClient(parsedURL, rr.Client())
	require.NoError(t, err)

	req := &ChatRequest{
		Model: "gemma3:1b",
		Messages: []*Message{
			{
				Role:    "user",
				Content: "Hello, how are you?",
			},
		},
		Stream: false,
		Options: Options{
			Temperature: 0.0,
			NumPredict:  50,
		},
	}

	var response *ChatResponse
	err = client.GenerateChat(ctx, req, func(resp ChatResponse) error {
		response = &resp
		return nil
	})
	require.NoError(t, err)
	assert.NotNil(t, response)
	assert.NotNil(t, response.Message)
	assert.NotEmpty(t, response.Message.Content)
	assert.True(t, response.Done)
}

func TestClient_GenerateChatStream(t *testing.T) {
	ctx := context.Background()

	rr := httprr.OpenForTest(t, http.DefaultTransport)
	defer rr.Close()

	baseURL := getOllamaTestURL(t, rr)

	// If recording and endpoint is not available, skip the test
	if rr.Recording() && !checkOllamaEndpoint(baseURL) {
		t.Skipf("Ollama endpoint not available at %s", baseURL)
	}

	// Skip if no recording exists and we're not recording
	if rr.Replaying() {
		httprr.SkipIfNoCredentialsAndRecordingMissing(t)
	}

	parsedURL, err := url.Parse(baseURL)
	require.NoError(t, err)

	client, err := NewClient(parsedURL, rr.Client())
	require.NoError(t, err)

	req := &ChatRequest{
		Model: "gemma3:1b",
		Messages: []*Message{
			{
				Role:    "user",
				Content: "Count from 1 to 5",
			},
		},
		Stream: true,
		Options: Options{
			Temperature: 0.0,
			NumPredict:  50,
		},
	}

	var responses []ChatResponse
	err = client.GenerateChat(ctx, req, func(resp ChatResponse) error {
		responses = append(responses, resp)
		return nil
	})
	require.NoError(t, err)
	assert.NotEmpty(t, responses)
	assert.True(t, responses[len(responses)-1].Done)
}

func TestClient_CreateEmbedding(t *testing.T) {
	ctx := context.Background()

	rr := httprr.OpenForTest(t, http.DefaultTransport)
	defer rr.Close()

	baseURL := getOllamaTestURL(t, rr)

	// If recording and endpoint is not available, skip the test
	if rr.Recording() && !checkOllamaEndpoint(baseURL) {
		t.Skipf("Ollama endpoint not available at %s", baseURL)
	}

	// Skip if no recording exists and we're not recording
	if rr.Replaying() {
		httprr.SkipIfNoCredentialsAndRecordingMissing(t)
	}

	parsedURL, err := url.Parse(baseURL)
	require.NoError(t, err)

	client, err := NewClient(parsedURL, rr.Client())
	require.NoError(t, err)

	req := &EmbeddingRequest{
		Model: "nomic-embed-text",
		Input: "Hello world",
		Options: Options{
			Temperature: 0.0,
		},
	}

	resp, err := client.CreateEmbedding(ctx, req)
	if err != nil && strings.Contains(err.Error(), "does not support embeddings") {
		t.Skipf("Model %s does not support embeddings", req.Model)
	}
	require.NoError(t, err)
	assert.NotNil(t, resp)
	assert.NotEmpty(t, resp.Embeddings)
}

func TestClient_GenerateChatWithThink(t *testing.T) {
	ctx := context.Background()

	rr := httprr.OpenForTest(t, http.DefaultTransport)
	defer rr.Close()

	baseURL := getOllamaTestURL(t, rr)

	// If recording and endpoint is not available, skip the test
	if rr.Recording() && !checkOllamaEndpoint(baseURL) {
		t.Skipf("Ollama endpoint not available at %s", baseURL)
	}

	// Skip if no recording exists and we're not recording
	if rr.Replaying() {
		httprr.SkipIfNoCredentialsAndRecordingMissing(t)
	}

	parsedURL, err := url.Parse(baseURL)
	require.NoError(t, err)

	client, err := NewClient(parsedURL, rr.Client())
	require.NoError(t, err)

	thinkEnabled := true
	req := &ChatRequest{
		Model: "gemma3:1b",
		Messages: []*Message{
			{
				Role:    "user",
				Content: "What is 2+2? Show your reasoning.",
			},
		},
		Stream: false,
		Think:  &thinkEnabled, // 顶层 think 开关（ollama 忽略 options.think）
		Options: Options{
			Temperature: 0.0,
			NumPredict:  100,
		},
	}

	var response *ChatResponse
	err = client.GenerateChat(ctx, req, func(resp ChatResponse) error {
		response = &resp
		return nil
	})
	require.NoError(t, err)
	assert.NotNil(t, response)
	assert.NotNil(t, response.Message)
	assert.NotEmpty(t, response.Message.Content)
	assert.True(t, response.Done)

	// The think parameter should be included in the request
	// This test verifies that the parameter is properly serialized
}

func TestOptionsJSONMarshalWithThink(t *testing.T) {
	// think 是 ChatRequest 的【顶层】字段（ollama 忽略 options.think），这里验证其序列化三态。
	thinkOn := true
	reqOn := ChatRequest{Model: "m", Think: &thinkOn}

	data, err := json.Marshal(reqOn)
	require.NoError(t, err)

	var result map[string]interface{}
	err = json.Unmarshal(data, &result)
	require.NoError(t, err)

	// Verify think field exists and is true
	think, exists := result["think"]
	assert.True(t, exists, "think field should exist in JSON")
	assert.Equal(t, true, think, "think field should be true")

	// 关键回归：think:false 必须显式出现在 JSON 里。Qwen3 等模型默认开启思考，
	// 若 omitempty 把 false 丢掉，模型会一直输出 reasoning 直到 num_predict 截断、正文为空。
	thinkOff := false
	reqOff := ChatRequest{Model: "m", Think: &thinkOff}
	dataOff, err := json.Marshal(reqOff)
	require.NoError(t, err)
	var resultOff map[string]interface{}
	require.NoError(t, json.Unmarshal(dataOff, &resultOff))
	thinkVal, exists := resultOff["think"]
	assert.True(t, exists, "think:false must be explicitly present in JSON")
	assert.Equal(t, false, thinkVal, "think field should be false")

	// nil 指针则应省略 think 字段（用模型默认）。
	reqNil := ChatRequest{Model: "m"}
	dataNil, err := json.Marshal(reqNil)
	require.NoError(t, err)
	var resultNil map[string]interface{}
	require.NoError(t, json.Unmarshal(dataNil, &resultNil))
	_, exists = resultNil["think"]
	assert.False(t, exists, "nil think should be omitted from JSON")
}

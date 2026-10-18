/*
 * Copyright 2024 CloudWeGo Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package gemini

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/cloudwego/eino/callbacks"
	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
	"google.golang.org/genai"
)

const usageTailTextEvent = `{"candidates":[{"index":0,"content":{"role":"model","parts":[{"text":"ok"}]}}],"usageMetadata":{"promptTokenCount":100,"totalTokenCount":100}}`
const usageTailStopEvent = `{"candidates":[{"index":0,"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":7,"thoughtsTokenCount":3,"totalTokenCount":110}}`
const usageTailUsageEvent = `{"candidates":[],"usageMetadata":{"promptTokenCount":100,"cachedContentTokenCount":60,"candidatesTokenCount":8,"thoughtsTokenCount":4,"totalTokenCount":112},"p":"opaque"}`

func TestStreamUsageTrailer(t *testing.T) {
	for _, toolCall := range []bool{false, true} {
		t.Run(fmt.Sprint(toolCall), func(t *testing.T) {
			first := usageTailTextEvent
			if toolCall {
				first = `{"candidates":[{"index":0,"content":{"role":"model","parts":[{"functionCall":{"name":"lookup","args":{"query":"q"}}}]}}]}`
			}
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
				payload := "data: " + first + "\r\n\r\ndata: " + usageTailStopEvent + "\r\n\r\n"
				payload += "data: " + usageTailUsageEvent + "\r\n\r\n"
				for len(payload) > 0 {
					n := min(17, len(payload))
					fmt.Fprint(w, payload[:n])
					w.(http.Flusher).Flush()
					payload = payload[n:]
				}
			}))
			defer srv.Close()
			m, err := newUsageTailTestModel(context.Background(), srv.URL)
			if err != nil {
				t.Fatal(err)
			}
			callbackUsage := make(chan *model.TokenUsage, 1)
			handler := callbacks.NewHandlerBuilder().OnEndWithStreamOutputFn(func(ctx context.Context, _ *callbacks.RunInfo, stream *schema.StreamReader[callbacks.CallbackOutput]) context.Context {
				go func() {
					defer stream.Close()
					var last *model.TokenUsage
					for {
						out, err := stream.Recv()
						if err == io.EOF {
							callbackUsage <- last
							return
						}
						if err != nil {
							callbackUsage <- nil
							return
						}
						if usage := out.(*model.CallbackOutput).TokenUsage; usage != nil {
							last = usage
						}
					}
				}()
				return ctx
			}).Build()
			ctx := callbacks.InitCallbacks(context.Background(), nil, handler)
			s, err := m.Stream(ctx, []*schema.Message{schema.UserMessage("test")})
			if err != nil {
				t.Fatal(err)
			}
			defer s.Close()
			var chunks []*schema.Message
			for {
				chunk, err := s.Recv()
				if err == io.EOF {
					break
				}
				if err != nil {
					t.Fatal(err)
				}
				chunks = append(chunks, chunk)
			}
			last := chunks[len(chunks)-1]
			if last.Content != "" || len(last.ToolCalls) != 0 || last.ResponseMeta.Usage.CompletionTokensDetails.ReasoningTokens != 4 {
				t.Fatalf("invalid usage-only message: %+v", last)
			}
			select {
			case usage := <-callbackUsage:
				if usage == nil || usage.TotalTokens != 112 || usage.CompletionTokensDetails.ReasoningTokens != 4 || usage.PromptTokenDetails.CachedTokens != 60 {
					t.Fatalf("callback lost usage: %+v", usage)
				}
			case <-time.After(2 * time.Second):
				t.Fatal("callback did not finish")
			}
			msg, err := schema.ConcatMessages(chunks)
			if err != nil {
				t.Fatal(err)
			}
			if toolCall {
				if len(msg.ToolCalls) != 1 || msg.ToolCalls[0].Function.Name != "lookup" {
					t.Fatalf("lost tool call: %+v", msg)
				}
			} else if msg.Content != "ok" {
				t.Fatalf("text changed: %q", msg.Content)
			}
			if msg.ResponseMeta == nil || msg.ResponseMeta.FinishReason != "STOP" || msg.ResponseMeta.Usage == nil {
				t.Fatalf("lost metadata: %+v", msg)
			}
			u := msg.ResponseMeta.Usage
			if u.PromptTokens != 100 || u.CompletionTokens != 12 || u.TotalTokens != 112 || u.PromptTokenDetails.CachedTokens != 60 {
				t.Fatalf("usage lost or double counted: %+v", u)
			}
		})
	}
}

func TestStreamInvalidUsageTrailers(t *testing.T) {
	for _, tc := range []struct {
		name   string
		events []string
	}{
		{"only_usage", []string{usageTailUsageEvent}},
		{"before_stop", []string{usageTailTextEvent, usageTailUsageEvent}},
		{"stop_without_content", []string{usageTailStopEvent, usageTailUsageEvent}},
		{"empty_object", []string{usageTailTextEvent, usageTailStopEvent, `{}`}},
		{"no_usage", []string{usageTailTextEvent, usageTailStopEvent, `{"candidates":[]}`}},
		{"null_usage", []string{usageTailTextEvent, usageTailStopEvent, `{"candidates":[],"usageMetadata":null}`}},
		{"empty_usage", []string{usageTailTextEvent, usageTailStopEvent, `{"candidates":[],"usageMetadata":{}}`}},
		{"blocked", []string{usageTailTextEvent, usageTailStopEvent, `{"candidates":[],"usageMetadata":{"promptTokenCount":100},"promptFeedback":{"blockReason":"SAFETY"}}`}},
		{"provider_error", []string{usageTailTextEvent, usageTailStopEvent, `{"error":{"code":500,"message":"upstream failure","status":"INTERNAL"}}`}},
		{"malformed", []string{usageTailTextEvent, usageTailStopEvent, `{"candidates":`}},
		{"safety_finish", []string{usageTailTextEvent, `{"candidates":[{"finishReason":"SAFETY"}]}`, usageTailUsageEvent}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				for _, event := range tc.events {
					fmt.Fprintf(w, "data: %s\n\n", event)
				}
			}))
			defer srv.Close()
			m, err := newUsageTailTestModel(context.Background(), srv.URL)
			if err != nil {
				t.Fatal(err)
			}
			s, err := m.Stream(context.Background(), []*schema.Message{schema.UserMessage("test")})
			if err == nil {
				defer s.Close()
				_, err = schema.ConcatMessageStream(s)
			}
			if err == nil {
				t.Fatal("invalid response was accepted")
			}
		})
	}
}

func TestStreamUsageDoesNotBufferUntilEOF(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, "data: %s\n\n", usageTailTextEvent)
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	defer srv.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	m, err := newUsageTailTestModel(ctx, srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	s, err := m.Stream(ctx, []*schema.Message{schema.UserMessage("test")})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	msg, err := s.Recv()
	if err != nil || msg.Content != "ok" {
		t.Fatalf("first event buffered: %v, %v", msg, err)
	}
	cancel()
	_, err = s.Recv()
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation lost: %v", err)
	}
}

func TestGenerateRejectsUsageOnlyResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, usageTailUsageEvent)
	}))
	defer srv.Close()
	cm, err := newUsageTailTestModel(context.Background(), srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = cm.Generate(context.Background(), []*schema.Message{schema.UserMessage("test")}); err == nil {
		t.Fatal("Generate accepted empty output")
	}
}

func newUsageTailTestModel(ctx context.Context, baseURL string) (*ChatModel, error) {
	client, err := genai.NewClient(ctx, &genai.ClientConfig{APIKey: "test-key", Backend: genai.BackendGeminiAPI, HTTPOptions: genai.HTTPOptions{BaseURL: baseURL}})
	if err != nil {
		return nil, err
	}
	return NewChatModel(ctx, &Config{Client: client, Model: "test-model"})
}

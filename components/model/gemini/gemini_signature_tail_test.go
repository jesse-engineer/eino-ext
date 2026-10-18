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
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/cloudwego/eino/schema"
	"github.com/stretchr/testify/require"
	"google.golang.org/genai"
)

func TestStandaloneThoughtSignatureRoundTrip(t *testing.T) {
	sig := []byte("standalone-signature")
	call := &genai.Part{FunctionCall: &genai.FunctionCall{Name: "lookup", Args: map[string]any{"query": "q"}}, ThoughtSignature: []byte("call-signature")}
	for _, tc := range []struct {
		name  string
		parts []*genai.Part
	}{
		{"newline_tail", []*genai.Part{{Text: "\n"}, {ThoughtSignature: sig}}},
		{"text_tail", []*genai.Part{{Text: "answer"}, {ThoughtSignature: sig}}},
		{"only_signature", []*genai.Part{{ThoughtSignature: sig}}},
		{"tool_tail", []*genai.Part{call, {ThoughtSignature: sig}}},
		{"text_tool_tail", []*genai.Part{{Text: "looking up"}, call, {ThoughtSignature: sig}}},
		{"two_tails", []*genai.Part{{Text: "answer"}, {ThoughtSignature: []byte("first")}, {ThoughtSignature: sig}}},
		{"thinking_tail", []*genai.Part{{Text: "thinking", Thought: true}, {Text: "answer"}, {ThoughtSignature: sig}}},
		{"text_then_signed_text", []*genai.Part{{Text: "first"}, {Text: "answer", ThoughtSignature: sig}}},
		{"text_attached", []*genai.Part{{Text: "answer", ThoughtSignature: sig}}},
		{"tool_attached", []*genai.Part{call}},
		{"image_attached", []*genai.Part{{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte("image")}, ThoughtSignature: sig}}},
	} {
		for _, streaming := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream=%t", tc.name, streaming), func(t *testing.T) {
				requests := make(chan []*genai.Content, 2)
				srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					var req struct {
						Contents []*genai.Content `json:"contents"`
					}
					if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
						t.Error(err)
						w.WriteHeader(400)
						return
					}
					requests <- req.Contents
					if streaming {
						w.Header().Set("Content-Type", "text/event-stream")
						for i, part := range tc.parts {
							c := &genai.Candidate{Content: &genai.Content{Role: "model", Parts: []*genai.Part{part}}}
							if i == len(tc.parts)-1 {
								c.FinishReason = genai.FinishReasonStop
							}
							b, _ := json.Marshal(&genai.GenerateContentResponse{Candidates: []*genai.Candidate{c}})
							fmt.Fprintf(w, "data: %s\n\n", b)
						}
					} else {
						w.Header().Set("Content-Type", "application/json")
						_ = json.NewEncoder(w).Encode(&genai.GenerateContentResponse{Candidates: []*genai.Candidate{{Content: &genai.Content{Role: "model", Parts: tc.parts}, FinishReason: genai.FinishReasonStop}}})
					}
				}))
				defer srv.Close()
				ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
				defer cancel()
				m, err := newUsageTailTestModel(ctx, srv.URL)
				require.NoError(t, err)
				invoke := func(input []*schema.Message) *schema.Message {
					if !streaming {
						out, err := m.Generate(ctx, input)
						require.NoError(t, err)
						return out
					}
					stream, err := m.Stream(ctx, input)
					require.NoError(t, err)
					defer stream.Close()
					out, err := schema.ConcatMessageStream(stream)
					require.NoError(t, err)
					return out
				}
				input := []*schema.Message{schema.UserMessage("test")}
				response := invoke(input)
				<-requests
				// Exercise session persistence, which converts []byte extras to base64 strings.
				b, err := json.Marshal(response)
				require.NoError(t, err)
				var restored schema.Message
				require.NoError(t, json.Unmarshal(b, &restored))
				invoke(append(input, &restored, schema.UserMessage("继续")))
				sent := <-requests
				require.Len(t, sent, 3)
				require.Equal(t, tc.parts, sent[1].Parts)
			})
		}
	}
}

func TestLegacyStandaloneThoughtSignature(t *testing.T) {
	for _, withTool := range []bool{false, true} {
		for _, withPart := range []bool{false, true} {
			t.Run(fmt.Sprintf("tool=%t/part=%t", withTool, withPart), func(t *testing.T) {
				msg := &schema.Message{Role: schema.Assistant}
				if withPart {
					msg.Content = "\n"
					msg.AssistantGenMultiContent = []schema.MessageOutputPart{{Type: schema.ChatMessagePartTypeText, Text: "\n"}}
				}
				if withTool {
					msg.ToolCalls = []schema.ToolCall{{Function: schema.FunctionCall{Name: "lookup", Arguments: `{}`}}}
				}
				setMessageThoughtSignature(msg, []byte("legacy-tail"))
				b, err := json.Marshal(msg)
				require.NoError(t, err)
				require.NoError(t, json.Unmarshal(b, msg))
				content, err := convSchemaMessage(msg)
				require.NoError(t, err)
				require.NotEmpty(t, content.Parts)
				require.Equal(t, []byte("legacy-tail"), content.Parts[len(content.Parts)-1].ThoughtSignature)
				require.Empty(t, content.Parts[len(content.Parts)-1].Text)
			})
		}
	}
	// Some persisted callers keep both locations; do not duplicate the signature.
	part := schema.MessageOutputPart{Type: schema.ChatMessagePartTypeText, Text: "answer"}
	setMessageOutputPartThoughtSignature(&part, []byte("same"))
	msg := &schema.Message{Role: schema.Assistant, AssistantGenMultiContent: []schema.MessageOutputPart{part}}
	setMessageThoughtSignature(msg, []byte("same"))
	content, err := convSchemaMessage(msg)
	require.NoError(t, err)
	require.Len(t, content.Parts, 1)
}

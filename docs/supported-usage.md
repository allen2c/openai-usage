# Supported Usage Types

`Usage.from_openai()` accepts every usage object below. Each is normalized into
the same `Usage` shape with `requests=1`.

| OpenAI source | Type passed to `from_openai` | What is captured |
| --- | --- | --- |
| **Chat Completions** | `openai.types.completion_usage.CompletionUsage` | input/output tokens, cached, reasoning |
| **Responses API** | `openai.types.responses.ResponseUsage` | input/output tokens, cached, reasoning |
| **Agents SDK** | `agents.Usage` | input/output tokens, cached, reasoning |
| **Agents SDK (context)** | `agents.RunContextWrapper` | unwrapped to its `.usage` |
| **Realtime** | `openai.types.realtime.RealtimeResponseUsage` | input/output tokens, cached, **audio**, text |
| **Batch** | `openai.types.batch_usage.BatchUsage` | input/output tokens, cached, reasoning |
| **Embeddings** | `openai.types.create_embedding_response.Usage` | input tokens only (no output) |
| **Images** (gpt-image-1) | `openai.types.images_response.Usage` | input/output tokens, **image**, text |
| **Transcription (tokens)** | `openai.types.audio.transcription.UsageTokens` | input/output tokens, **audio**, text |
| **Transcription (duration)** | `openai.types.audio.transcription.UsageDuration` | **`seconds`** only |
| **Assistants run** | `openai.types.beta.threads.run.Usage` | prompt/completion tokens |
| **Assistants run step** | `openai.types.beta.threads.runs.run_step.Usage` | prompt/completion tokens |

All of these types are also re-exported from the package root for convenience and
type hints:

```python
from openai_usage import (
    BatchUsage,
    EmbeddingUsage,
    ImagesUsage,
    RealtimeResponseUsage,
    RunStepUsage,
    RunUsage,
    TranscriptionUsageDuration,
    TranscriptionUsageTokens,
    OpenAIUsage,  # the Union of everything from_openai accepts
)
```

## Examples by modality

### Realtime (audio)

```python
from openai_usage import Usage

# event is a `response.done` event from a realtime session
usage = Usage.from_openai(event.response.usage)

print(usage.input_tokens_details.audio_tokens)
print(usage.output_tokens_details.audio_tokens)
```

### Embeddings

```python
response = client.embeddings.create(model="text-embedding-3-small", input="hello")
usage = Usage.from_openai(response.usage)

assert usage.output_tokens == 0   # embeddings have no generated output
```

### Image generation

```python
result = client.images.generate(model="gpt-image-1", prompt="a red bicycle")
usage = Usage.from_openai(result.usage)

print(usage.input_tokens_details.image_tokens)
print(usage.output_tokens_details.image_tokens)
```

### Transcription

```python
from openai_usage import Usage

# gpt-4o-transcribe → billed by tokens
usage = Usage.from_openai(transcription.usage)        # UsageTokens
print(usage.input_tokens_details.audio_tokens)

# whisper-1 → billed by duration
usage = Usage.from_openai(transcription.usage)        # UsageDuration
print(usage.seconds)
```

!!! info "Not LLM usage? Not included."
    OpenAI's organization-level **Usage API** (`/organization/usage/*`) reports
    aggregated billing buckets rather than per-call token counts, so it is
    intentionally out of scope for this library.

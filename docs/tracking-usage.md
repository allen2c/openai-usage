# Tracking Usage

The `Usage` model is the heart of the library. It holds the token counts for one
or many API calls and knows how to build itself from any OpenAI usage object.

## The `Usage` model

```python
from openai_usage import Usage

usage = Usage(
    requests=1,
    input_tokens=100,
    output_tokens=200,
    total_tokens=300,
)
```

| Field | Type | Description |
| --- | --- | --- |
| `requests` | `int` | Number of billable calls represented. |
| `input_tokens` | `int` | Total input (prompt) tokens. |
| `input_tokens_details` | `UsageInputTokensDetails` | Breakdown: `cached`, `audio`, `image`, `text`. |
| `output_tokens` | `int` | Total output (completion) tokens. |
| `output_tokens_details` | `UsageOutputTokensDetails` | Breakdown: `reasoning`, `audio`, `image`, `text`. |
| `total_tokens` | `int` | Input + output tokens. |
| `seconds` | `float` | Audio duration for duration-billed transcription. |
| `model` | `str \| None` | Optional model id, used as the default for cost estimation. |
| `cost` | `str \| float \| None` | Optional slot to stash an estimated cost. |

The two detail models extend OpenAI's own schema, so the familiar fields keep
working while the new per-modality counts default to `0`:

```python
usage.input_tokens_details.cached_tokens   # from the OpenAI schema
usage.input_tokens_details.audio_tokens    # extension, defaults to 0
usage.input_tokens_details.image_tokens    # extension, defaults to 0
usage.output_tokens_details.reasoning_tokens
```

## Building from OpenAI objects

`Usage.from_openai()` accepts any supported SDK usage object and returns a
normalized `Usage` with `requests=1`. See
[Supported Usage Types](supported-usage.md) for the full list.

```python
from openai import OpenAI
from openai_usage import Usage

client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)

usage = Usage.from_openai(completion.usage)
print(usage.input_tokens, usage.output_tokens)
```

It works the same way across APIs — the call below is identical in spirit for the
Responses API, the Agents SDK, Realtime, Batch, Embeddings, Images, and
Transcription:

```python
# Responses API
usage = Usage.from_openai(response.usage)

# Agents SDK — pass the Usage or the RunContextWrapper
usage = Usage.from_openai(result.context_wrapper)

# Realtime — from a response.done event
usage = Usage.from_openai(event.response.usage)
```

!!! tip "`inplace` for hot loops"
    By default `from_openai` returns a fresh, fully re-validated instance. When
    you are aggregating large volumes and trust the input, pass `inplace=True`
    to skip the extra serialization round-trip.

    ```python
    usage = Usage.from_openai(completion.usage, inplace=True)
    ```

## Combining usage

`Usage.add()` accumulates another `Usage` into the current one — requests,
tokens, every token-detail breakdown, and `seconds`. This is how you roll up the
spend of an entire agent run, a batch job, or a whole day of traffic.

```python
from openai_usage import Usage

total = Usage()
for completion in completions:
    total.add(Usage.from_openai(completion.usage))

print(f"Requests:      {total.requests}")
print(f"Input tokens:  {total.input_tokens}")
print(f"Output tokens: {total.output_tokens}")
print(f"Cached tokens: {total.input_tokens_details.cached_tokens}")
print(f"Audio tokens:  {total.input_tokens_details.audio_tokens}")
```

!!! note "`add` mutates in place"
    `add` modifies the receiver and returns `None`. Start from a fresh `Usage()`
    accumulator rather than mutating a value you still need.

## Serialization

`Usage` is a plain Pydantic model, so it serializes and restores cleanly:

```python
blob = usage.model_dump_json()
restored = Usage.model_validate_json(blob)
```

Payloads serialized by older versions of the library — before the audio, image,
and duration fields existed — still validate. The new fields simply default to
`0`, so stored history keeps working after an upgrade.

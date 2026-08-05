# Estimating Costs

Once you have a `Usage` object, you can estimate what it cost against any model's
pricing.

```python
from openai_usage import Usage

usage = Usage(requests=1, input_tokens=1000, output_tokens=2000, total_tokens=3000)

cost = usage.estimate_cost("gpt-4o")
print(f"${cost:.6f}")
```

## Two return types

| Method | Returns | Use when |
| --- | --- | --- |
| `estimate_cost(...)` | `float` | You want a number to display or compare. |
| `estimate_cost_str(...)` | `str` | You need exact decimal precision (it returns the raw `Decimal` as a string, with no float rounding). |

```python
usage.estimate_cost("gpt-4o")        # 0.0225
usage.estimate_cost_str("gpt-4o")    # "0.0225"
```

## Choosing the model

The model can come from three places, in order of precedence:

1. The argument: `usage.estimate_cost("anthropic/claude-3-haiku")`.
2. The `Usage.model` field, if the argument is omitted.
3. A `gpt-4o-mini` fallback if neither is set (a warning is logged).

```python
usage.model = "gpt-4o"
usage.estimate_cost()          # uses usage.model

usage.estimate_cost("o1")      # argument wins over usage.model
```

## Pricing data

Pricing comes from two catalogs, both fetched at runtime:

| Source | Covers |
| --- | --- |
| [LiteLLM](https://github.com/BerriAI/litellm) | ~3,000 models across every provider, including the embedding, image, audio and transcription models OpenRouter never proxies |
| [OpenRouter](https://openrouter.ai/) | The chat models OpenRouter proxies, under `vendor/model` names |

A name is resolved in three steps:

1. **LiteLLM exact key.** LiteLLM files models under their native ids
   (`gpt-4o`, `whisper-1`, `claude-sonnet-4-5`), so most names hit here
   immediately.
2. **OpenRouter relaxed matching.** Handles `vendor/model` names, aliases, and
   separator differences (`claude-haiku-4-5` finds `anthropic/claude-haiku-4.5`).
3. **LiteLLM relaxed matching**, as a last resort.

You can also pass a fully resolved model object instead of a name.

### Caching

Both catalogs are cached on disk so the download happens at most once a day.

| Setting | Default | Environment variable |
| --- | --- | --- |
| Cache directory | `./.cache/openai-usage/` | `OPENAI_USAGE_CACHE_DIR` |
| Time to live | `86400` (one day) | `OPENAI_USAGE_CACHE_TTL` |
| Disable network | off | `OPENAI_USAGE_OFFLINE` |

Passing `realtime_pricing=True` ignores the TTL and re-downloads both catalogs:

```python
usage.estimate_cost("anthropic/claude-3-haiku", realtime_pricing=True)
```

### When the network is unavailable

Nothing raises. Each step falls through to the next:

```
fresh cache -> network fetch -> stale cache -> bundled snapshot
```

The bundled snapshot is an OpenRouter catalog shipped inside the package, so
pricing keeps working offline for the models it covers. LiteLLM has no bundled
snapshot; without a cache entry its lookups return nothing and resolution falls
back to OpenRouter.

Set `OPENAI_USAGE_OFFLINE=1` to skip the network entirely — useful in airgapped
environments and in tests. Connect and read timeouts are deliberately short, so
an unreachable network fails fast instead of stalling a caller the bundled
snapshot could have served.

## How the estimate is computed

Tokens are split into disjoint buckets and each is priced at its own rate:

| Bucket | Rate |
| --- | --- |
| Cached input tokens | cache-read price |
| Cache-write input tokens | cache-creation price |
| Audio input tokens | input audio price |
| Audio output tokens | output audio price |
| Image input tokens | input image price |
| Image output tokens | output image price |
| Reasoning output tokens | reasoning price, falling back to the completion price |
| Remaining input tokens | prompt price |
| Remaining output tokens | completion price |
| Requests | per-request price |
| Seconds | per-second price |

When a usage object carries no audio, image, cached, or reasoning detail — an
ordinary chat completion, say — every special bucket is `0` and the math reduces
to the familiar `prompt × input + completion × output`.

## Unknown models

By default, an unrecognized model name yields a `0.0` cost and a logged warning,
so a bad id never crashes an aggregation pipeline:

```python
usage.estimate_cost("not-a-real-model")          # 0.0 (warns)
```

Set `ignore_not_found=False` to raise instead:

```python
usage.estimate_cost("not-a-real-model", ignore_not_found=False)  # raises ValueError
```

## Limitations

!!! warning "Video tokens are billed as text"
    `video_tokens` are recorded for reporting, but neither catalog publishes a
    per-token video rate, so they stay in the plain-text remainder and are
    billed at the ordinary input rate. That matches how the providers reporting
    them bill today, but verify it if video is a large share of your spend.

!!! warning "Detail buckets are assumed disjoint"
    Cached, cache-write, audio and image counts are subtracted from the totals
    to derive the plain-text remainder. If a provider reports overlapping
    buckets — a cached audio token counted in both — that token is subtracted
    twice and the estimate comes out low.

!!! note "OpenRouter publishes one rate per modality"
    OpenRouter has a single `audio` and a single `image` rate per model, so a
    model resolved through OpenRouter uses that one rate for both input and
    output. LiteLLM prices them separately and is tried first, so this normally
    only applies to models LiteLLM does not carry.

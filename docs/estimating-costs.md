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

Model names are matched flexibly against the OpenRouter catalog — exact matches
win, and otherwise the closest non-greedy match is used. You can also pass a
fully resolved `OpenRouterModel` directly.

## Pricing data

Pricing comes from [OpenRouter](https://openrouter.ai/).

- **Default** — a snapshot bundled with the package. Fast and offline.
- **`realtime_pricing=True`** — fetches the latest prices from the OpenRouter API.

```python
usage.estimate_cost("anthropic/claude-3-haiku", realtime_pricing=True)
```

## How the estimate is computed

Tokens are split into disjoint buckets and each is priced at its own rate:

| Bucket | Rate |
| --- | --- |
| Cached input tokens | cache-read price |
| Audio tokens (input + output) | audio price |
| Image tokens (input + output) | image price |
| Reasoning output tokens | internal-reasoning price |
| Remaining input tokens | prompt price |
| Remaining output tokens | completion price |
| Requests | per-request price |

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

!!! warning "Audio and image rates are approximate"
    OpenRouter exposes a single `audio` rate and a single `image` rate per
    model. The estimator applies that one rate to **both** input and output
    tokens of that modality. Realtime output audio in particular is billed at a
    higher rate by OpenAI, so audio-heavy realtime sessions can be
    **underestimated**. Cached and audio/image tokens are also assumed to be
    disjoint subsets of the totals. For exact figures, price against OpenAI's
    official rate card.

!!! warning "Duration billing is not priced"
    Duration-billed transcription (e.g. `whisper-1`, which charges per second)
    populates `Usage.seconds`, but OpenRouter has no per-second rate, so this
    time **does not contribute to the cost estimate**. The seconds are tracked
    for your own reporting; compute their cost separately if you need it.

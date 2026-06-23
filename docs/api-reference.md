# API Reference

A concise tour of the public surface. Everything below is importable from
`openai_usage` unless noted otherwise.

## `Usage`

The core model. See [Tracking Usage](tracking-usage.md) for the field table.

### `Usage.from_openai(openai_usage, *, inplace=False) -> Usage`

Build a normalized `Usage` (with `requests=1`) from any
[supported usage type](supported-usage.md).

- **`openai_usage`** — any object in the `OpenAIUsage` union.
- **`inplace`** — when `True`, return the constructed instance directly; when
  `False` (default), return a re-validated copy.
- **Raises** `ValueError` for an unsupported type.

### `Usage.add(other) -> None`

Accumulate `other` into `self`, in place. Sums `requests`, `input_tokens`,
`output_tokens`, `total_tokens`, `seconds`, and every field of both token-detail
breakdowns (cached, reasoning, audio, image, text).

### `Usage.estimate_cost(model=None, *, realtime_pricing=False, ignore_not_found=True) -> float`

Estimate the cost of this usage as a `float`. See
[Estimating Costs](estimating-costs.md).

- **`model`** — an `OpenRouterModel`, a model-name `str`, or `None` to fall back
  to `Usage.model` (then to `gpt-4o-mini`).
- **`realtime_pricing`** — fetch live OpenRouter pricing instead of the bundled
  snapshot.
- **`ignore_not_found`** — return `0.0` for an unknown model (default) instead of
  raising `ValueError`.

### `Usage.estimate_cost_str(...) -> str`

Same parameters as `estimate_cost`, but returns the exact `Decimal` cost as a
string with no float rounding.

## Token detail models

### `UsageInputTokensDetails`

Extends `openai.types.responses.InputTokensDetails`.

| Field | Default | Source |
| --- | --- | --- |
| `cached_tokens` | — | OpenAI schema |
| `audio_tokens` | `0` | extension |
| `text_tokens` | `0` | extension |
| `image_tokens` | `0` | extension |

### `UsageOutputTokensDetails`

Extends `openai.types.responses.OutputTokensDetails`.

| Field | Default | Source |
| --- | --- | --- |
| `reasoning_tokens` | — | OpenAI schema |
| `audio_tokens` | `0` | extension |
| `text_tokens` | `0` | extension |
| `image_tokens` | `0` | extension |

## `OpenAIUsage`

A `typing.Union` of every usage type accepted by `from_openai`. Handy for typing
your own functions:

```python
from openai_usage import OpenAIUsage, Usage

def track(raw: OpenAIUsage) -> Usage:
    return Usage.from_openai(raw)
```

## Pricing helpers — `openai_usage.extra.open_router`

For working with the OpenRouter catalog directly.

### `get_model(model_name, *, realtime_pricing=False) -> OpenRouterModel | None`

Resolve a model name to an `OpenRouterModel` using flexible matching. Returns
`None` when nothing matches.

### `get_models(realtime_pricing=False) -> GetOpenRouterModelsResponse`

Return the full model catalog (bundled snapshot, or live when
`realtime_pricing=True`).

### `OpenRouterModel` / `OpenRouterPricing`

The pricing schema. `OpenRouterPricing` exposes per-token rate properties used by
the cost estimator, including `price_per_audio_token` and
`price_per_image_token`.

```python
from openai_usage.extra.open_router import get_model

model = get_model("gpt-4o")
print(model.pricing.price_per_input_token_without_cached)
print(model.pricing.price_per_audio_token)
```

# API Reference

A concise tour of the public surface. Everything below is importable from
`openai_usage` unless noted otherwise.

## `Usage`

The core model. See [Tracking Usage](tracking-usage.md) for the field table.

### `Usage.from_openai(openai_usage, *, inplace=False) -> Usage`

Build a normalized `Usage` (with `requests=1`) from any
[supported usage type](supported-usage.md). This is the only entry point — it
accepts the OpenAI SDK types plus google-genai and litellm usage.

- **`openai_usage`** — any object in the `AnyUsage` union.
- **`inplace`** — when `True`, return the constructed instance directly; when
  `False` (default), return a re-validated copy.
- **Raises** `ValueError` for an unsupported type.

### `Usage.add(other) -> None`

Accumulate `other` into `self`, in place. Sums `requests`, `input_tokens`,
`output_tokens`, `total_tokens`, `seconds`, and every field of both token-detail
breakdowns (cached, cache-write, reasoning, audio, image, text, video).

### `Usage.estimate_cost(model=None, *, realtime_pricing=False, ignore_not_found=True) -> float`

Estimate the cost of this usage as a `float`. See
[Estimating Costs](estimating-costs.md).

- **`model`** — a resolved model object, a model-name `str`, or `None` to fall
  back to `Usage.model` (then to `gpt-4o-mini`).
- **`realtime_pricing`** — ignore the cache TTL and re-download both catalogs.
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
| `cache_write_tokens` | `0` | OpenAI schema |
| `audio_tokens` | `0` | extension |
| `text_tokens` | `0` | extension |
| `image_tokens` | `0` | extension |
| `video_tokens` | `0` | extension |

`cache_write_tokens` mirrors the field newer OpenAI SDKs add to
`InputTokensDetails`. It is redefined with a `0` default so the model stays
constructible on both older SDKs (which lack the field) and newer ones (which
make it required).

### `UsageOutputTokensDetails`

Extends `openai.types.responses.OutputTokensDetails`.

| Field | Default | Source |
| --- | --- | --- |
| `reasoning_tokens` | — | OpenAI schema |
| `audio_tokens` | `0` | extension |
| `text_tokens` | `0` | extension |
| `image_tokens` | `0` | extension |
| `video_tokens` | `0` | extension |

## Type aliases

### `AnyUsage`

A `typing.Union` of every usage type accepted by `from_openai`, including the
optional google-genai and litellm types. Use it to type your own functions:

```python
from openai_usage import AnyUsage, Usage

def track(raw: AnyUsage) -> Usage:
    return Usage.from_openai(raw)
```

### `OpenAIUsage`

The OpenAI-only subset of `AnyUsage`, kept for backward compatibility.

## Pricing — `openai_usage.pricing`

### `get_model(model_name, *, realtime_pricing=False) -> PricedModel | None`

Resolve a model name across both catalogs. Returns `None` when nothing matches.

```python
from openai_usage.pricing import get_model

model = get_model("gpt-4o")
print(model.pricing.price_per_input_token_without_cached)
print(model.pricing.price_per_output_audio_token)
```

### `ModelPricing`

A runtime-checkable `Protocol` describing the rates the estimator needs, so both
catalogs can be priced through one interface:

`price_per_request`, `price_per_input_token_without_cached`,
`price_per_input_token_with_cached`, `price_per_input_cache_write_token`,
`price_per_output_not_reasoning_token`, `price_per_output_reasoning_token`,
`price_per_input_audio_token`, `price_per_output_audio_token`,
`price_per_input_image_token`, `price_per_output_image_token`,
`price_per_second`.

Every property returns a `decimal.Decimal`, and is `0` when the catalog has no
figure for it.

### `PricedModel`

A runtime-checkable `Protocol` with `id: str` and `pricing: ModelPricing`.
`OpenRouterModel` and `LiteLLMModel` both satisfy it.

### Cache controls

`openai_usage.pricing.cache_dir()`, `ttl_seconds()` and `is_offline()` report the
effective cache settings. See [Estimating Costs](estimating-costs.md#caching) for
the environment variables.

## Per-catalog modules

### `openai_usage.pricing.litellm`

`get_models()`, `get_model(name)`, `get_model_exact(name)`,
`get_model_loose(name)`, plus `LiteLLMModel` and `LiteLLMPricing`. Returns an
empty catalog rather than raising when the source is unreachable.

### `openai_usage.pricing.open_router`

`get_models(realtime_pricing=False)`, `get_model(name)`, `matches_loosely()`,
plus `OpenRouterModel`, `OpenRouterPricing` and the rest of the catalog schema.

`OpenRouterPricing` keeps its original `price_per_audio_token` and
`price_per_image_token` properties alongside the split input/output ones.

!!! note "`openai_usage.extra.open_router` is deprecated"
    It still re-exports everything it did in v0.9.x, so existing imports keep
    working. New code should import from `openai_usage.pricing.open_router`.

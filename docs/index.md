# openai-usage

[![PyPI](https://img.shields.io/pypi/v/openai-usage.svg)](https://pypi.org/project/openai-usage/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/allen2c/openai-usage/blob/main/LICENSE)

**One usage model for every OpenAI API.**

`openai-usage` collapses the dozen different `Usage` shapes scattered across the
OpenAI Python SDK into a single, additive `Usage` object — then estimates what it
costs against live model pricing.

---

## Why

The OpenAI SDK reports usage in a different shape for almost every endpoint:
Chat Completions, the Responses API, the Agents SDK, Realtime, Batch, Embeddings,
Images, and Transcription each have their own `Usage` type with their own field
names. Aggregating spend across them means writing the same adapter code again
and again.

This library does that adapting once. You hand it whatever the SDK gave you, and
you get back a uniform `Usage` you can add up and price.

```python
from openai_usage import Usage

total = Usage()
total.add(Usage.from_openai(chat_completion.usage))      # Chat Completions
total.add(Usage.from_openai(response.usage))             # Responses API
total.add(Usage.from_openai(realtime_event.response.usage))  # Realtime

print(total.input_tokens, total.output_tokens, total.total_tokens)
print(f"${total.estimate_cost('gpt-4o'):.6f}")
```

## Highlights

<div class="grid cards" markdown>

-   :material-merge:{ .lg .middle } **Unify every usage type**

    ---

    A single `Usage.from_openai(...)` accepts Chat Completions, Responses,
    Agents SDK, Realtime, Batch, Embeddings, Images, Transcription, and
    Assistants runs.

-   :material-plus-box:{ .lg .middle } **Add it all up**

    ---

    `Usage.add()` accumulates requests, tokens, and per-modality token details
    across calls — including cached, reasoning, audio, image, and text tokens.

-   :material-cash:{ .lg .middle } **Estimate cost**

    ---

    Price usage against any model using live OpenRouter pricing, with separate
    rates for cached, reasoning, audio, and image tokens.

-   :material-language-python:{ .lg .middle } **Pydantic-native**

    ---

    `Usage` is a `pydantic.BaseModel` — serialize it, store it, and validate it
    anywhere, with full backward compatibility for older payloads.

</div>

## Next steps

- [Installation](installation.md)
- [Tracking Usage](tracking-usage.md) — build and combine `Usage` objects
- [Estimating Costs](estimating-costs.md) — price usage against a model
- [Supported Usage Types](supported-usage.md) — the full compatibility matrix
- [API Reference](api-reference.md)

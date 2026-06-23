# openai-usage

[![PyPI](https://img.shields.io/pypi/v/openai-usage.svg)](https://pypi.org/project/openai-usage/)
[![Tests](https://github.com/allen2c/openai-usage/actions/workflows/test.yml/badge.svg)](https://github.com/allen2c/openai-usage/actions/workflows/test.yml)
[![Docs](https://github.com/allen2c/openai-usage/actions/workflows/docs.yml/badge.svg)](https://allen2c.github.io/openai-usage/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**One usage model for every OpenAI API.**

The OpenAI Python SDK reports usage in a different shape for almost every
endpoint — Chat Completions, Responses, Agents SDK, Realtime, Batch, Embeddings,
Images, Transcription. `openai-usage` collapses them all into a single, additive
`Usage` object and estimates what it costs against live model pricing.

📖 **Full documentation: <https://allen2c.github.io/openai-usage/>**

## Installation

```bash
pip install openai-usage
```

Requires Python 3.11+.

## Quickstart

```python
from openai_usage import Usage

# Normalize usage from any OpenAI API into one shape
total = Usage()
total.add(Usage.from_openai(chat_completion.usage))          # Chat Completions
total.add(Usage.from_openai(response.usage))                 # Responses API
total.add(Usage.from_openai(realtime_event.response.usage))  # Realtime

print(total.input_tokens, total.output_tokens, total.total_tokens)

# Estimate the cost against any model
print(f"${total.estimate_cost('gpt-4o'):.6f}")
```

## What it does

- **Unify** — `Usage.from_openai(...)` accepts Chat Completions, Responses,
  Agents SDK, Realtime, Batch, Embeddings, Images, Transcription, and Assistants
  run usage.
- **Aggregate** — `Usage.add()` accumulates requests, tokens, and per-modality
  detail (cached, reasoning, audio, image, text) across calls.
- **Estimate** — `Usage.estimate_cost()` prices usage against any model using
  OpenRouter pricing, with separate rates for cached, reasoning, audio, and image
  tokens.
- **Serialize** — `Usage` is a Pydantic model with full backward compatibility
  for older payloads.

See the [Supported Usage Types](https://allen2c.github.io/openai-usage/supported-usage/)
table for the full compatibility matrix, and
[Estimating Costs](https://allen2c.github.io/openai-usage/estimating-costs/) for
pricing details and limitations.

## Documentation

The docs are built with [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
and deployed to GitHub Pages on every push to `main`.

```bash
make mkdocs        # serve locally at http://localhost:8000
```

## License

MIT — see [LICENSE](LICENSE).

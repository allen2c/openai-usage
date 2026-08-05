# Installation

`openai-usage` requires **Python 3.11+**.

=== "pip"

    ```bash
    pip install openai-usage
    ```

=== "poetry"

    ```bash
    poetry add openai-usage
    ```

=== "uv"

    ```bash
    uv add openai-usage
    ```

The `openai` and `openai-agents` SDKs are installed as dependencies, so the usage
types you pass to [`Usage.from_openai`](tracking-usage.md) are available out of
the box.

## Optional extras

Install an extra only if you want to hand `Usage.from_openai` that SDK's usage
object. Neither is imported unless you already imported it yourself, so skipping
them costs you nothing.

=== "pip"

    ```bash
    pip install "openai-usage[google-genai]"
    pip install "openai-usage[litellm]"
    pip install "openai-usage[all]"
    ```

=== "poetry"

    ```bash
    poetry add "openai-usage[all]"
    ```

=== "uv"

    ```bash
    uv add "openai-usage[all]"
    ```

| Extra | Adds support for |
| --- | --- |
| `google-genai` | `GenerateContentResponseUsageMetadata` (Gemini) |
| `litellm` | `litellm.types.utils.Usage` |
| `all` | Both |

## Verify

```python
import openai_usage

print(openai_usage.__version__)
```

## What gets installed

| Dependency | Purpose |
| --- | --- |
| `openai` | Source of the SDK usage types (`CompletionUsage`, `ResponseUsage`, …). |
| `openai-agents` | Agents SDK usage (`agents.Usage`, `agents.RunContextWrapper`). |
| `pydantic` | `Usage` is a Pydantic model. |
| `requests` | Fetching the pricing catalogs. |

Model pricing is fetched at runtime from LiteLLM and OpenRouter, then cached
under `./.cache/openai-usage/` for a day. A bundled snapshot keeps things working
offline. See [Estimating Costs](estimating-costs.md).

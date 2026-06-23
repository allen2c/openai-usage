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

Model pricing is fetched from [OpenRouter](https://openrouter.ai/) at runtime
when you opt in with `realtime_pricing=True`; otherwise a bundled snapshot is
used. See [Estimating Costs](estimating-costs.md).

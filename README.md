# openai-usage

[![PyPI](https://img.shields.io/pypi/v/openai-usage.svg)](https://pypi.org/project/openai-usage/)

Utilities to track and estimate API usage costs.

## Installation

```bash
pip install openai-usage
```

## Usage

### Tracking Token Usage

Track usage manually or create `Usage` objects from OpenAI's response objects.

```python
from openai_usage import Usage
from openai.types.completion_usage import CompletionUsage

# Manually create a Usage object
manual_usage = Usage(requests=1, input_tokens=100, output_tokens=200, total_tokens=300)

# Create from an OpenAI CompletionUsage object
openai_completion_usage = CompletionUsage(prompt_tokens=50, completion_tokens=150, total_tokens=200)
usage_from_openai = Usage.from_openai(openai_completion_usage)

# Add usage objects together
combined_usage = Usage()
combined_usage.add(manual_usage)
combined_usage.add(usage_from_openai)

print(f"Total requests: {combined_usage.requests}")
print(f"Total input tokens: {combined_usage.input_tokens}")
print(f"Total output tokens: {combined_usage.output_tokens}")
```

### Estimating Costs

Estimate the cost of your API usage for any supported model.

```python
from openai_usage import Usage

# Create a usage object
usage = Usage(requests=1, input_tokens=1000, output_tokens=2000, total_tokens=3000)

# Estimate cost for a model
# The library fetches pricing data from an external API
cost = usage.estimate_cost("anthropic/claude-3-haiku")
print(f"Estimated cost for the model: ${cost:.6f}")

# For the most up-to-date pricing, use realtime_pricing=True
cost_realtime = usage.estimate_cost("anthropic/claude-3-haiku", realtime_pricing=True)
print(f"Estimated real-time cost for the model: ${cost_realtime:.6f}")
```

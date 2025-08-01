# openai-usage

[![PyPI](https://img.shields.io/pypi/v/openai-usage.svg)](https://pypi.org/project/openai-usage/)

Utilities to track OpenAI API usage.

## Installation

```bash
pip install openai-usage
```

## Usage

```python
from openai_usage import Usage

# Track usage manually
usage = Usage()
usage.add(Usage(requests=1, input_tokens=10, output_tokens=20, total_tokens=30))

# Create from OpenAI object
from openai.types.completion_usage import CompletionUsage
openai_usage = CompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
usage = Usage.from_openai(openai_usage)
```

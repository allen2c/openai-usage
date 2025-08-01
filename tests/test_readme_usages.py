from openai.types.completion_usage import CompletionUsage

from openai_usage import Usage


def test_usage_initialization():
    usage = Usage()
    assert usage.requests == 0
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0
    assert usage.total_tokens == 0


def test_from_openai_completion_usage():
    openai_usage = CompletionUsage(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
    )
    usage = Usage.from_openai(openai_usage)
    assert usage.requests == 1
    assert usage.input_tokens == 10
    assert usage.output_tokens == 20
    assert usage.total_tokens == 30


def test_add_usage():
    usage1 = Usage(requests=1, input_tokens=10, output_tokens=20, total_tokens=30)
    usage2 = Usage(requests=2, input_tokens=15, output_tokens=25, total_tokens=40)
    usage1.add(usage2)
    assert usage1.requests == 3
    assert usage1.input_tokens == 25
    assert usage1.output_tokens == 45
    assert usage1.total_tokens == 70

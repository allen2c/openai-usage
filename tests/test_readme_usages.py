import agents
import pytest
from openai.types.completion_usage import (
    CompletionTokensDetails,
    CompletionUsage,
    PromptTokensDetails,
)
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponseUsage,
)

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


def test_from_openai_completion_usage_with_details():
    openai_usage = CompletionUsage(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=5),
        completion_tokens_details=CompletionTokensDetails(reasoning_tokens=3),
    )
    usage = Usage.from_openai(openai_usage)
    assert usage.requests == 1
    assert usage.input_tokens == 10
    assert usage.output_tokens == 20
    assert usage.total_tokens == 30
    assert usage.input_tokens_details.cached_tokens == 5
    assert usage.output_tokens_details.reasoning_tokens == 3


def test_from_openai_response_usage():
    openai_usage = ResponseUsage(
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
        input_tokens_details=InputTokensDetails(cached_tokens=5),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=8),
    )
    usage = Usage.from_openai(openai_usage)
    assert usage.requests == 1
    assert usage.input_tokens == 10
    assert usage.output_tokens == 20
    assert usage.total_tokens == 30
    assert usage.input_tokens_details.cached_tokens == 5
    assert usage.output_tokens_details.reasoning_tokens == 8


def test_from_openai_inplace():
    openai_usage = CompletionUsage(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
    )
    usage = Usage.from_openai(openai_usage, inplace=True)
    assert usage.requests == 1
    assert usage.input_tokens == 10
    assert usage.output_tokens == 20
    assert usage.total_tokens == 30

    usage.model = "gpt-4o"
    usage.cost = usage.estimate_cost()
    print(f"cost: {round(usage.cost, 6)} USD")


def test_add_usage():
    usage1 = Usage(requests=1, input_tokens=10, output_tokens=20, total_tokens=30)
    usage2 = Usage(requests=2, input_tokens=15, output_tokens=25, total_tokens=40)
    usage1.add(usage2)
    assert usage1.requests == 3
    assert usage1.input_tokens == 25
    assert usage1.output_tokens == 45
    assert usage1.total_tokens == 70


def test_from_agents_usage():
    agents_usage = agents.Usage(
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
        input_tokens_details=InputTokensDetails(cached_tokens=5),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=8),
    )
    usage = Usage.from_openai(agents_usage)
    assert usage.requests == 1
    assert usage.input_tokens == 10
    assert usage.output_tokens == 20
    assert usage.total_tokens == 30
    assert usage.input_tokens_details.cached_tokens == 5
    assert usage.output_tokens_details.reasoning_tokens == 8


def test_from_agents_run_context_wrapper():
    agents_usage = agents.Usage(
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
        input_tokens_details=InputTokensDetails(cached_tokens=5),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=8),
    )
    context_wrapper = agents.RunContextWrapper(context=None, usage=agents_usage)
    usage = Usage.from_openai(context_wrapper)
    assert usage.requests == 1
    assert usage.input_tokens == 10
    assert usage.output_tokens == 20
    assert usage.total_tokens == 30
    assert usage.input_tokens_details.cached_tokens == 5
    assert usage.output_tokens_details.reasoning_tokens == 8


def test_estimate_cost():
    usage = Usage(requests=1, input_tokens=1000, output_tokens=2000, total_tokens=3000)
    cost = usage.estimate_cost("gpt-4o-mini")
    assert isinstance(cost, float)
    assert cost > 0


def test_estimate_cost_realtime():
    usage = Usage(requests=1, input_tokens=1000, output_tokens=2000, total_tokens=3000)
    cost = usage.estimate_cost("anthropic/claude-3-haiku", realtime_pricing=True)
    assert isinstance(cost, float)
    assert cost > 0


def test_estimate_cost_no_model():
    usage = Usage(requests=1, input_tokens=1000, output_tokens=2000, total_tokens=3000)
    cost = usage.estimate_cost()
    assert isinstance(cost, float)
    assert cost > 0


def test_estimate_cost_not_found():
    usage = Usage(requests=1, input_tokens=1000, output_tokens=2000, total_tokens=3000)
    with pytest.raises(ValueError, match="No model found for 'not-a-real-model'"):
        usage.estimate_cost("not-a-real-model")

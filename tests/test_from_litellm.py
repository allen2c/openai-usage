import pytest

from openai_usage import Usage

litellm_utils = pytest.importorskip(
    "litellm.types.utils", reason="the litellm extra is not installed"
)
LiteLLMUsage = litellm_utils.Usage
PromptTokensDetailsWrapper = litellm_utils.PromptTokensDetailsWrapper
CompletionTokensDetailsWrapper = litellm_utils.CompletionTokensDetailsWrapper


def test_plain_usage():
    usage = Usage.from_openai(
        LiteLLMUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    )
    assert usage.requests == 1
    assert usage.input_tokens == 10
    assert usage.output_tokens == 20
    assert usage.total_tokens == 30


def test_from_openai_routes_litellm_before_completion_usage():
    """litellm's Usage subclasses CompletionUsage, which would swallow it."""
    from openai.types.completion_usage import CompletionUsage

    raw = LiteLLMUsage(
        prompt_tokens=800,
        completion_tokens=50,
        total_tokens=850,
        cache_creation_input_tokens=400,
        cache_read_input_tokens=300,
    )
    assert isinstance(raw, CompletionUsage)

    usage = Usage.from_openai(raw)
    assert usage.input_tokens_details.cache_write_tokens == 400
    assert usage.input_tokens_details.cached_tokens == 300


def test_anthropic_cache_buckets():
    raw = LiteLLMUsage(
        prompt_tokens=800,
        completion_tokens=50,
        total_tokens=850,
        cache_creation_input_tokens=400,
        cache_read_input_tokens=300,
    )
    usage = Usage.from_openai(raw)

    # litellm folds both cache buckets into prompt_tokens, so they stay
    # disjoint subsets of the input total.
    assert usage.input_tokens == 800
    assert usage.input_tokens_details.cached_tokens == 300
    assert usage.input_tokens_details.cache_write_tokens == 400


def test_deepseek_style_cache_hit_tokens():
    raw = LiteLLMUsage(
        prompt_tokens=100,
        completion_tokens=10,
        total_tokens=110,
        prompt_cache_hit_tokens=60,
    )
    assert Usage.from_openai(raw).input_tokens_details.cached_tokens == 60


def test_modality_details_are_carried_over():
    raw = LiteLLMUsage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=100, text_tokens=700, image_tokens=150, video_tokens=50
        ),
        completion_tokens_details=CompletionTokensDetailsWrapper(
            reasoning_tokens=200, audio_tokens=60, text_tokens=240
        ),
    )
    usage = Usage.from_openai(raw)

    assert usage.input_tokens_details.audio_tokens == 100
    assert usage.input_tokens_details.text_tokens == 700
    assert usage.input_tokens_details.image_tokens == 150
    assert usage.input_tokens_details.video_tokens == 50
    assert usage.output_tokens_details.reasoning_tokens == 200
    assert usage.output_tokens_details.audio_tokens == 60
    assert usage.output_tokens_details.text_tokens == 240


def test_absent_detail_attributes_do_not_raise():
    """litellm deletes wrapper attributes whose value is None."""
    raw = LiteLLMUsage(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=5),
    )
    usage = Usage.from_openai(raw)
    assert usage.input_tokens_details.cached_tokens == 5
    assert usage.input_tokens_details.image_tokens == 0
    assert usage.input_tokens_details.video_tokens == 0


def test_missing_detail_objects_do_not_raise():
    raw = LiteLLMUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    raw.prompt_tokens_details = None
    raw.completion_tokens_details = None

    usage = Usage.from_openai(raw)
    assert usage.input_tokens_details.cached_tokens == 0
    assert usage.output_tokens_details.reasoning_tokens == 0


def test_cost_is_carried_over_when_present():
    raw = LiteLLMUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    raw.cost = 0.00123
    assert Usage.from_openai(raw).cost == 0.00123


def test_none_token_counts_become_zero():
    raw = LiteLLMUsage(prompt_tokens=None, completion_tokens=None, total_tokens=None)
    usage = Usage.from_openai(raw)
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0
    assert usage.total_tokens == 0


def test_inplace_skips_the_serialization_round_trip():
    raw = LiteLLMUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    assert Usage.from_openai(raw, inplace=True).input_tokens == 10


def test_result_aggregates_with_openai_usage():
    from openai.types.completion_usage import CompletionUsage

    total = Usage()
    total.add(
        Usage.from_openai(
            CompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        )
    )
    total.add(
        Usage.from_openai(
            LiteLLMUsage(prompt_tokens=5, completion_tokens=7, total_tokens=12)
        )
    )
    assert total.requests == 2
    assert total.input_tokens == 15
    assert total.output_tokens == 27


def test_result_can_be_priced():
    raw = LiteLLMUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
    assert Usage.from_openai(raw).estimate_cost("gpt-4o") > 0

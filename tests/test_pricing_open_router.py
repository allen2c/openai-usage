import decimal

import pytest

from openai_usage.pricing.open_router import OpenRouterPricing


def _pricing(**kwargs) -> OpenRouterPricing:
    kwargs.setdefault("prompt", "0.000003")
    kwargs.setdefault("completion", "0.000015")
    return OpenRouterPricing(**kwargs)


def test_reasoning_falls_back_to_the_completion_rate():
    """``internal_reasoning`` is a surcharge, not the whole rate.

    It reads "0" for nearly every model, which used to make thinking tokens
    free instead of billing them like ordinary output.
    """
    pricing = _pricing(internal_reasoning="0")
    assert pricing.price_per_output_reasoning_token == decimal.Decimal("0.000015")


def test_absent_reasoning_rate_falls_back_too():
    pricing = _pricing()
    assert pricing.price_per_output_reasoning_token == decimal.Decimal("0.000015")


def test_explicit_reasoning_surcharge_wins():
    pricing = _pricing(internal_reasoning="0.00009")
    assert pricing.price_per_output_reasoning_token == decimal.Decimal("0.00009")


def test_reasoning_matches_litellms_fallback():
    """Both catalogs must agree, or the price depends on which one resolved."""
    from openai_usage.pricing.litellm import LiteLLMPricing

    open_router = _pricing()
    lite = LiteLLMPricing(output_cost_per_token=1.5e-05)
    assert (
        open_router.price_per_output_reasoning_token
        == lite.price_per_output_reasoning_token
    )


def test_cached_input_falls_back_to_the_prompt_rate():
    pricing = _pricing()
    assert pricing.price_per_input_token_with_cached == decimal.Decimal("0.000003")


def test_cache_write_rate_is_exposed():
    pricing = _pricing(input_cache_write="0.00000375")
    assert pricing.price_per_input_cache_write_token == decimal.Decimal("0.00000375")


def test_cache_write_is_zero_when_absent():
    assert _pricing().price_per_input_cache_write_token == decimal.Decimal(0)


def test_audio_and_image_share_one_rate_across_input_and_output():
    """OpenRouter publishes a single rate per modality, unlike LiteLLM."""
    pricing = _pricing(audio="0.00004", image="0.00001")
    assert pricing.price_per_input_audio_token == pricing.price_per_output_audio_token
    assert pricing.price_per_input_audio_token == decimal.Decimal("0.00004")
    assert pricing.price_per_input_image_token == pricing.price_per_output_image_token
    assert pricing.price_per_input_image_token == decimal.Decimal("0.00001")


def test_legacy_rate_properties_are_kept():
    """Documented in the API reference, so they must keep working."""
    pricing = _pricing(audio="0.00004", image="0.00001")
    assert pricing.price_per_audio_token == decimal.Decimal("0.00004")
    assert pricing.price_per_image_token == decimal.Decimal("0.00001")


def test_duration_rate_is_zero():
    """OpenRouter has no per-second rate at all."""
    assert _pricing().price_per_second == decimal.Decimal(0)


@pytest.mark.parametrize(
    "field",
    ["request", "image", "audio", "internal_reasoning", "input_cache_write"],
)
def test_optional_rates_tolerate_none(field):
    pricing = _pricing(**{field: None})
    assert isinstance(pricing.price_per_request, decimal.Decimal)

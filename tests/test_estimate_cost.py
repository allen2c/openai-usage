import decimal

import pydantic
import pytest

from openai_usage import Usage


class FakePricing(pydantic.BaseModel):
    """A catalog-independent rate card, so the arithmetic is checked alone."""

    request: decimal.Decimal = decimal.Decimal(0)
    input_token: decimal.Decimal = decimal.Decimal(0)
    cached_input_token: decimal.Decimal = decimal.Decimal(0)
    cache_write_token: decimal.Decimal = decimal.Decimal(0)
    output_token: decimal.Decimal = decimal.Decimal(0)
    reasoning_token: decimal.Decimal = decimal.Decimal(0)
    input_audio_token: decimal.Decimal = decimal.Decimal(0)
    output_audio_token: decimal.Decimal = decimal.Decimal(0)
    input_image_token: decimal.Decimal = decimal.Decimal(0)
    output_image_token: decimal.Decimal = decimal.Decimal(0)
    second: decimal.Decimal = decimal.Decimal(0)

    @property
    def price_per_request(self) -> decimal.Decimal:
        return self.request

    @property
    def price_per_input_token_without_cached(self) -> decimal.Decimal:
        return self.input_token

    @property
    def price_per_input_token_with_cached(self) -> decimal.Decimal:
        return self.cached_input_token

    @property
    def price_per_input_cache_write_token(self) -> decimal.Decimal:
        return self.cache_write_token

    @property
    def price_per_output_not_reasoning_token(self) -> decimal.Decimal:
        return self.output_token

    @property
    def price_per_output_reasoning_token(self) -> decimal.Decimal:
        return self.reasoning_token

    @property
    def price_per_input_audio_token(self) -> decimal.Decimal:
        return self.input_audio_token

    @property
    def price_per_output_audio_token(self) -> decimal.Decimal:
        return self.output_audio_token

    @property
    def price_per_input_image_token(self) -> decimal.Decimal:
        return self.input_image_token

    @property
    def price_per_output_image_token(self) -> decimal.Decimal:
        return self.output_image_token

    @property
    def price_per_second(self) -> decimal.Decimal:
        return self.second


class FakeModel(pydantic.BaseModel):
    id: str = "fake/model"
    pricing: FakePricing = pydantic.Field(default_factory=FakePricing)


def test_plain_text_usage():
    usage = Usage(requests=1, input_tokens=1000, output_tokens=500)
    model = FakeModel(
        pricing=FakePricing(
            input_token=decimal.Decimal(2), output_token=decimal.Decimal(3)
        )
    )
    assert usage.estimate_cost(model) == pytest.approx(1000 * 2 + 500 * 3)


def test_per_request_rate_is_applied():
    usage = Usage(requests=4)
    model = FakeModel(pricing=FakePricing(request=decimal.Decimal(5)))
    assert usage.estimate_cost(model) == pytest.approx(20)


def test_cache_write_tokens_are_billed_at_their_own_rate():
    """They used to be dropped, and double-counted as plain input text."""
    usage = Usage(requests=1, input_tokens=1000)
    usage.input_tokens_details.cache_write_tokens = 400
    model = FakeModel(
        pricing=FakePricing(
            input_token=decimal.Decimal(2), cache_write_token=decimal.Decimal(10)
        )
    )
    # 600 text tokens plus 400 cache-write tokens, not 1000 text tokens.
    assert usage.estimate_cost(model) == pytest.approx(600 * 2 + 400 * 10)


def test_cached_and_cache_write_are_disjoint_from_text():
    usage = Usage(requests=1, input_tokens=1000)
    usage.input_tokens_details.cached_tokens = 300
    usage.input_tokens_details.cache_write_tokens = 200
    model = FakeModel(
        pricing=FakePricing(
            input_token=decimal.Decimal(2),
            cached_input_token=decimal.Decimal(1),
            cache_write_token=decimal.Decimal(10),
        )
    )
    assert usage.estimate_cost(model) == pytest.approx(500 * 2 + 300 * 1 + 200 * 10)


def test_audio_input_and_output_use_separate_rates():
    usage = Usage(requests=1, input_tokens=1000, output_tokens=1000)
    usage.input_tokens_details.audio_tokens = 800
    usage.output_tokens_details.audio_tokens = 900
    model = FakeModel(
        pricing=FakePricing(
            input_token=decimal.Decimal(1),
            output_token=decimal.Decimal(2),
            input_audio_token=decimal.Decimal(10),
            output_audio_token=decimal.Decimal(20),
        )
    )
    expected = 200 * 1 + 800 * 10 + 100 * 2 + 900 * 20
    assert usage.estimate_cost(model) == pytest.approx(expected)


def test_image_input_and_output_use_separate_rates():
    usage = Usage(requests=1, input_tokens=100, output_tokens=100)
    usage.input_tokens_details.image_tokens = 60
    usage.output_tokens_details.image_tokens = 70
    model = FakeModel(
        pricing=FakePricing(
            input_token=decimal.Decimal(1),
            output_token=decimal.Decimal(1),
            input_image_token=decimal.Decimal(5),
            output_image_token=decimal.Decimal(9),
        )
    )
    expected = 40 * 1 + 60 * 5 + 30 * 1 + 70 * 9
    assert usage.estimate_cost(model) == pytest.approx(expected)


def test_duration_billing_is_priced():
    """``seconds`` had no rate before, so whisper usage always cost zero."""
    usage = Usage(requests=1, seconds=120.0)
    model = FakeModel(pricing=FakePricing(second=decimal.Decimal("0.0001")))
    assert usage.estimate_cost(model) == pytest.approx(0.012)


def test_fractional_seconds_do_not_raise():
    """Decimal times float is a TypeError, so seconds must be converted."""
    usage = Usage(requests=1, seconds=1.5)
    model = FakeModel(pricing=FakePricing(second=decimal.Decimal("0.1")))
    assert usage.estimate_cost(model) == pytest.approx(0.15)


def test_video_tokens_are_billed_at_the_input_rate():
    """Neither catalog prices video per token, so it stays in the remainder."""
    usage = Usage(requests=1, input_tokens=1000)
    usage.input_tokens_details.video_tokens = 400
    model = FakeModel(pricing=FakePricing(input_token=decimal.Decimal(2)))
    assert usage.estimate_cost(model) == pytest.approx(1000 * 2)


def test_reasoning_tokens_are_carved_out_of_output_text():
    usage = Usage(requests=1, output_tokens=2000)
    usage.output_tokens_details.reasoning_tokens = 1500
    model = FakeModel(
        pricing=FakePricing(
            output_token=decimal.Decimal(2), reasoning_token=decimal.Decimal(3)
        )
    )
    assert usage.estimate_cost(model) == pytest.approx(500 * 2 + 1500 * 3)


def test_detail_counts_exceeding_the_total_clamp_at_zero():
    usage = Usage(requests=1, input_tokens=100)
    usage.input_tokens_details.audio_tokens = 500
    model = FakeModel(pricing=FakePricing(input_token=decimal.Decimal(1000)))
    assert usage.estimate_cost(model) == pytest.approx(500 * 0)


def test_estimate_cost_str_keeps_full_precision():
    usage = Usage(requests=1, input_tokens=3)
    model = FakeModel(pricing=FakePricing(input_token=decimal.Decimal("0.1")))
    assert usage.estimate_cost_str(model) == "0.3"


def test_unknown_model_returns_zero_by_default():
    usage = Usage(requests=1, input_tokens=1000)
    assert usage.estimate_cost("definitely-not-a-real-model-xyz") == 0.0


def test_unknown_model_can_raise():
    usage = Usage(requests=1, input_tokens=1000)
    with pytest.raises(ValueError):
        usage.estimate_cost("definitely-not-a-real-model-xyz", ignore_not_found=False)


def test_video_tokens_accumulate():
    a = Usage(requests=1, input_tokens=10)
    a.input_tokens_details.video_tokens = 3
    a.output_tokens_details.video_tokens = 4
    b = Usage(requests=1, input_tokens=10)
    b.input_tokens_details.video_tokens = 5
    b.output_tokens_details.video_tokens = 6

    a.add(b)
    assert a.input_tokens_details.video_tokens == 8
    assert a.output_tokens_details.video_tokens == 10

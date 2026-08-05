import decimal
import json
import time

import pytest

from openai_usage.pricing import litellm

CATALOG = {
    "sample_spec": {
        "input_cost_per_token": 0.0,
        "litellm_provider": "one of the providers",
    },
    "fallback_generalizations": {"rules": []},
    "gpt-4o": {
        "input_cost_per_token": 2.5e-06,
        "output_cost_per_token": 1e-05,
        "cache_read_input_token_cost": 1.25e-06,
        "litellm_provider": "openai",
        "mode": "chat",
        "search_context_cost_per_query": {"search_context_size_low": 0.035},
    },
    "whisper-1": {
        "input_cost_per_second": 0.0001,
        "output_cost_per_second": 0.0001,
        "litellm_provider": "openai",
        "mode": "audio_transcription",
    },
    "claude-sonnet-4-5": {
        "input_cost_per_token": 3e-06,
        "output_cost_per_token": 1.5e-05,
        "cache_read_input_token_cost": 3e-07,
        "cache_creation_input_token_cost": 3.75e-06,
        "litellm_provider": "anthropic",
        "mode": "chat",
    },
    "gpt-4o-audio-preview": {
        "input_cost_per_token": 2.5e-06,
        "output_cost_per_token": 1e-05,
        "input_cost_per_audio_token": 4e-05,
        "output_cost_per_audio_token": 8e-05,
        "litellm_provider": "openai",
        "mode": "chat",
    },
    "free-model": {
        "input_cost_per_token": 0,
        "output_cost_per_token": 0,
        "litellm_provider": "someone",
        "mode": "chat",
    },
}


@pytest.fixture(autouse=True)
def catalog(monkeypatch):
    """Serve a fixed catalog and reset the in-process memo around each test."""
    litellm._models_cache.clear()
    monkeypatch.setattr(
        litellm,
        "fetch_text",
        lambda url, filename, force_refresh=False: json.dumps(CATALOG),
    )
    yield
    litellm._models_cache.clear()


def test_non_model_entries_are_skipped():
    models = litellm.get_models()
    assert "sample_spec" not in models
    assert "fallback_generalizations" not in models
    assert set(models) == {
        "gpt-4o",
        "whisper-1",
        "claude-sonnet-4-5",
        "gpt-4o-audio-preview",
        "free-model",
    }


def test_unknown_keys_are_ignored():
    """Entries carry ~150 distinct keys, including nested ones."""
    model = litellm.get_models()["gpt-4o"]
    assert model.litellm_provider == "openai"
    assert model.mode == "chat"


def test_rates_avoid_binary_float_noise():
    """Decimal(2.5e-06) would expand the float's exact binary value."""
    rate = litellm.get_models()["gpt-4o"].pricing.price_per_input_token_without_cached
    assert rate == decimal.Decimal("2.5e-06")
    assert str(rate) == "0.0000025"
    assert rate != decimal.Decimal(2.5e-06)
    assert len(str(decimal.Decimal(2.5e-06))) > 20


def test_cached_input_falls_back_to_the_plain_input_rate():
    pricing = litellm.get_models()["gpt-4o-audio-preview"].pricing
    assert pricing.price_per_input_token_with_cached == decimal.Decimal("2.5e-06")


def test_cache_write_uses_the_cache_creation_rate():
    pricing = litellm.get_models()["claude-sonnet-4-5"].pricing
    assert pricing.price_per_input_cache_write_token == decimal.Decimal("3.75e-06")


def test_cache_write_is_zero_when_unpriced():
    pricing = litellm.get_models()["gpt-4o"].pricing
    assert pricing.price_per_input_cache_write_token == decimal.Decimal(0)


def test_audio_input_and_output_rates_are_separate():
    pricing = litellm.get_models()["gpt-4o-audio-preview"].pricing
    assert pricing.price_per_input_audio_token == decimal.Decimal("4e-05")
    assert pricing.price_per_output_audio_token == decimal.Decimal("8e-05")


def test_reasoning_falls_back_to_the_output_rate():
    pricing = litellm.get_models()["gpt-4o"].pricing
    assert pricing.price_per_output_reasoning_token == decimal.Decimal("1e-05")


def test_duration_rate_is_exposed():
    pricing = litellm.get_models()["whisper-1"].pricing
    assert pricing.price_per_second == decimal.Decimal("0.0001")


def test_missing_rates_are_zero_not_none():
    pricing = litellm.get_models()["whisper-1"].pricing
    assert pricing.price_per_input_token_without_cached == decimal.Decimal(0)
    assert pricing.price_per_request == decimal.Decimal(0)
    assert pricing.price_per_input_image_token == decimal.Decimal(0)


def test_zero_rates_stay_zero():
    pricing = litellm.get_models()["free-model"].pricing
    assert pricing.price_per_input_token_without_cached == decimal.Decimal(0)
    assert pricing.price_per_output_reasoning_token == decimal.Decimal(0)


def test_exact_lookup_is_case_insensitive():
    assert litellm.get_model_exact("GPT-4o") is not None
    assert litellm.get_model_exact("gpt-4o") is not None


def test_exact_lookup_misses_are_none():
    assert litellm.get_model_exact("gpt-4") is None


def test_unavailable_catalog_yields_an_empty_map(monkeypatch):
    litellm._models_cache.clear()
    monkeypatch.setattr(
        litellm, "fetch_text", lambda url, filename, force_refresh=False: None
    )
    assert litellm.get_models() == {}
    assert litellm.get_model("gpt-4o") is None


def test_unparseable_catalog_yields_an_empty_map(monkeypatch):
    litellm._models_cache.clear()
    monkeypatch.setattr(
        litellm, "fetch_text", lambda url, filename, force_refresh=False: "{not json"
    )
    assert litellm.get_models() == {}


def test_catalog_url_is_the_documented_one():
    assert litellm.MODELS_URL == (
        "https://raw.githubusercontent.com/BerriAI/litellm/main/"
        "model_prices_and_context_window.json"
    )


def test_a_free_cache_read_rate_is_not_replaced_by_the_input_rate():
    """Decimal(0) is falsy, so `or` would silently bill the full rate.

    Fifteen live catalog entries price cache reads at 0 alongside a non-zero
    input rate.
    """
    pricing = litellm.LiteLLMPricing(
        input_cost_per_token=3e-07, cache_read_input_token_cost=0.0
    )
    assert pricing.price_per_input_token_with_cached == decimal.Decimal(0)
    assert pricing.price_per_input_token_without_cached == decimal.Decimal("3e-07")


def test_a_free_reasoning_rate_is_not_replaced_by_the_output_rate():
    pricing = litellm.LiteLLMPricing(
        output_cost_per_token=1e-05, output_cost_per_reasoning_token=0.0
    )
    assert pricing.price_per_output_reasoning_token == decimal.Decimal(0)


def test_a_failed_fetch_is_retried_soon_rather_than_pinned_for_a_day(monkeypatch):
    """One blip near startup must not cost a process every LiteLLM price."""
    litellm._models_cache.clear()
    calls = {"n": 0}

    def flaky(url, filename, force_refresh=False):
        calls["n"] += 1
        return None if calls["n"] == 1 else json.dumps(CATALOG)

    monkeypatch.setattr(litellm, "fetch_text", flaky)

    assert litellm.get_models() == {}

    # Still inside the short degraded window: no retry yet.
    assert litellm.get_models() == {}
    assert calls["n"] == 1

    now = time.monotonic()
    monkeypatch.setattr(
        time, "monotonic", lambda: now + litellm.DEGRADED_TTL_SECONDS + 1
    )
    assert "gpt-4o" in litellm.get_models()
    assert calls["n"] == 2


def test_degraded_window_is_much_shorter_than_the_normal_ttl():
    assert litellm.DEGRADED_TTL_SECONDS < 3600


@pytest.mark.parametrize("payload", ["[1, 2, 3]", '"a string"', "123", "null"])
def test_a_non_object_payload_does_not_crash(monkeypatch, payload):
    """GitHub raw can serve an error page that still parses as JSON."""
    litellm._models_cache.clear()
    monkeypatch.setattr(
        litellm, "fetch_text", lambda url, filename, force_refresh=False: payload
    )
    assert litellm.get_models() == {}

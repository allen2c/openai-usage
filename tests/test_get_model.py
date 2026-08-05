import pytest

import openai_usage.pricing
from openai_usage.pricing import cache, litellm, open_router
from openai_usage.pricing.open_router import get_model, get_models


@pytest.fixture
def bundled_catalog(monkeypatch):
    """Resolve against the bundled snapshot rather than the live catalog.

    Upstream catalogs change daily, so behavioural assertions run against the
    snapshot shipped with the package to stay deterministic and offline-safe.
    """
    monkeypatch.setenv(cache.OFFLINE_ENV, "1")
    monkeypatch.setenv(cache.CACHE_DIR_ENV, "/nonexistent-cache-for-tests")
    open_router._models_cache.clear()
    yield
    open_router._models_cache.clear()


@pytest.fixture
def live_litellm():
    """Skip when the LiteLLM catalog cannot be reached; it has no snapshot."""
    if not litellm.get_models():
        pytest.skip("LiteLLM pricing catalog unavailable")


@pytest.mark.parametrize(
    ("model_name", "expected_id"),
    [
        ("gpt-3.5-turbo", "openai/gpt-3.5-turbo"),
        ("gpt-4.1-nano", "openai/gpt-4.1-nano"),
        ("voyageai/voyage-3-large", "voyageai/voyage-3-large"),
        ("google/gemma-3-27b-it", "google/gemma-3-27b-it"),
        ("anthropic/claude-haiku-4-5", "anthropic/claude-haiku-4.5"),
        ("anthropic/claude-haiku-4.5", "anthropic/claude-haiku-4.5"),
        ("kimi-k2:thinking", "moonshotai/kimi-k2-thinking"),
    ],
)
def test_get_model(bundled_catalog, model_name, expected_id):
    model = get_model(model_name)
    assert model is not None
    assert model.id == expected_id


def test_get_model_resolves_every_catalog_id(bundled_catalog):
    """Every id in the bundled catalog must resolve back to itself."""
    for model in get_models().data:
        resolved = get_model(model.id)
        assert resolved is not None, model.id
        assert resolved.id == model.id


def test_supplemental_models_are_merged_in(bundled_catalog):
    """Voyage models exist only in the hand-maintained file."""
    ids = {model.id for model in get_models().data}
    assert "voyageai/voyage-3-large" in ids


@pytest.mark.parametrize("model_name", ["x[y", "(", "gpt-4o (preview)", "a+b"])
def test_get_model_tolerates_regex_metacharacters(bundled_catalog, model_name):
    """Names are matched literally, so metacharacters must not raise."""
    assert get_model(model_name) is None


@pytest.mark.parametrize(
    "model_name",
    [
        "whisper-1",
        "gpt-image-1",
        "text-embedding-3-small",
        "gpt-4o-audio-preview",
    ],
)
def test_litellm_covers_openrouter_catalog_gaps(live_litellm, model_name):
    """OpenRouter proxies chat models only; LiteLLM fills in the rest."""
    resolved = openai_usage.pricing.get_model(model_name)
    assert resolved is not None
    assert resolved.id == model_name


@pytest.mark.parametrize(
    "model_name",
    [
        "whisper-1",
        "gpt-image-1",
        "text-embedding-3-small",
        "gpt-4o-audio-preview",
    ],
)
def test_openrouter_alone_cannot_resolve_those(bundled_catalog, model_name):
    assert get_model(model_name) is None


def test_exact_litellm_key_wins_over_openrouter(live_litellm):
    """LiteLLM keys models under native ids, so it resolves first."""
    resolved = openai_usage.pricing.get_model("gpt-4o")
    assert resolved is not None
    assert resolved.id == "gpt-4o"


def test_openrouter_still_resolves_its_own_naming(live_litellm):
    """Vendor-prefixed names have no exact LiteLLM key, so OpenRouter wins."""
    resolved = openai_usage.pricing.get_model("voyageai/voyage-3-large")
    assert resolved is not None
    assert resolved.id == "voyageai/voyage-3-large"


def test_unknown_names_resolve_to_none(live_litellm):
    assert openai_usage.pricing.get_model("definitely-not-a-real-model-xyz") is None

import pytest

from openai_usage.extra.open_router import get_model


@pytest.mark.parametrize(
    "model_name",
    [
        "gpt-3.5-turbo",
        "gpt-4.1-nano",
        "google/gemma-3-27b-it",
        "anthropic/claude-haiku-4-5",
        "anthropic/claude-haiku-4.5",
        "claude-3.7-sonnet-thinking",
        "kimi-k2:thinking",
    ],
)
def test_get_model(model_name):
    model = get_model(model_name)
    assert model is not None

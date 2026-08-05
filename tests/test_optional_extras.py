import subprocess
import sys

import openai_usage
from openai_usage import Usage

PROBE = """
import sys
import openai_usage
from openai_usage import Usage
from openai.types.completion_usage import CompletionUsage

usage = Usage.from_openai(
    CompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
)
assert usage.input_tokens == 10

print("litellm" in sys.modules, "google.genai" in sys.modules)
"""


def test_optional_sdks_are_never_imported():
    """Importing litellm costs over a second, and neither extra is required.

    A caller can only hand us one of their objects if the SDK is already
    loaded, so detection goes through ``sys.modules`` rather than an import.
    """
    result = subprocess.run(
        [sys.executable, "-c", PROBE],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False False", result.stdout


def test_detection_helpers_report_false_when_not_loaded(monkeypatch):
    monkeypatch.delitem(sys.modules, "litellm.types.utils", raising=False)
    monkeypatch.delitem(sys.modules, "google.genai.types", raising=False)

    assert openai_usage._is_litellm_usage(object()) is False
    assert openai_usage._is_genai_usage(object()) is False


def test_detection_ignores_a_module_without_the_type(monkeypatch):
    """A partially initialized module must not be mistaken for the real one."""
    monkeypatch.setitem(sys.modules, "litellm.types.utils", object())
    assert openai_usage._is_litellm_usage(object()) is False


def test_unsupported_types_still_raise():
    try:
        Usage.from_openai(object())  # type: ignore[arg-type]
    except ValueError as error:
        assert "Unsupported usage type" in str(error)
    else:
        raise AssertionError("expected a ValueError")


def test_any_usage_alias_is_exported():
    import openai_usage

    assert hasattr(openai_usage, "AnyUsage")
    assert hasattr(openai_usage, "OpenAIUsage")

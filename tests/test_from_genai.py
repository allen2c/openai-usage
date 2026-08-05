import pytest

from openai_usage import Usage

genai_types = pytest.importorskip(
    "google.genai.types", reason="the google-genai extra is not installed"
)
UsageMetadata = genai_types.GenerateContentResponseUsageMetadata
ModalityTokenCount = genai_types.ModalityTokenCount
MediaModality = genai_types.MediaModality


def _modalities(*pairs):
    return [
        ModalityTokenCount(modality=modality, token_count=count)
        for modality, count in pairs
    ]


def test_plain_usage():
    usage = Usage.from_openai(
        UsageMetadata(
            prompt_token_count=1000,
            candidates_token_count=500,
            total_token_count=1500,
        )
    )
    assert usage.requests == 1
    assert usage.input_tokens == 1000
    assert usage.output_tokens == 500
    assert usage.total_tokens == 1500


def test_every_field_absent():
    """Every field on the Gemini schema is optional."""
    usage = Usage.from_openai(UsageMetadata())
    assert usage.requests == 1
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0
    assert usage.total_tokens == 0


def test_thoughts_count_as_output_and_as_reasoning():
    """Gemini reports thinking separately from candidates_token_count."""
    usage = Usage.from_openai(
        UsageMetadata(
            prompt_token_count=100,
            candidates_token_count=500,
            thoughts_token_count=200,
            total_token_count=800,
        )
    )
    assert usage.output_tokens == 700
    assert usage.output_tokens_details.reasoning_tokens == 200


def test_tool_use_prompt_counts_as_input():
    usage = Usage.from_openai(
        UsageMetadata(prompt_token_count=1000, tool_use_prompt_token_count=50)
    )
    assert usage.input_tokens == 1050


def test_cached_content_is_recorded():
    usage = Usage.from_openai(
        UsageMetadata(prompt_token_count=1000, cached_content_token_count=300)
    )
    assert usage.input_tokens_details.cached_tokens == 300


def test_modality_details_are_split_out():
    usage = Usage.from_openai(
        UsageMetadata(
            prompt_token_count=1000,
            candidates_token_count=500,
            prompt_tokens_details=_modalities(
                (MediaModality.TEXT, 600),
                (MediaModality.AUDIO, 200),
                (MediaModality.VIDEO, 150),
                (MediaModality.IMAGE, 50),
            ),
            candidates_tokens_details=_modalities(
                (MediaModality.TEXT, 400), (MediaModality.AUDIO, 100)
            ),
        )
    )
    assert usage.input_tokens_details.text_tokens == 600
    assert usage.input_tokens_details.audio_tokens == 200
    assert usage.input_tokens_details.video_tokens == 150
    assert usage.input_tokens_details.image_tokens == 50
    assert usage.output_tokens_details.text_tokens == 400
    assert usage.output_tokens_details.audio_tokens == 100


def test_tool_use_modalities_fold_into_the_input_details():
    usage = Usage.from_openai(
        UsageMetadata(
            prompt_token_count=1000,
            tool_use_prompt_token_count=100,
            prompt_tokens_details=_modalities((MediaModality.TEXT, 1000)),
            tool_use_prompt_tokens_details=_modalities((MediaModality.TEXT, 100)),
        )
    )
    assert usage.input_tokens_details.text_tokens == 1100


def test_document_modality_is_left_in_the_text_remainder():
    """Gemini bills documents at the ordinary input rate, and we have no field."""
    usage = Usage.from_openai(
        UsageMetadata(
            prompt_token_count=1000,
            prompt_tokens_details=_modalities(
                (MediaModality.TEXT, 400), (MediaModality.DOCUMENT, 600)
            ),
        )
    )
    assert usage.input_tokens == 1000
    assert usage.input_tokens_details.text_tokens == 400
    assert usage.input_tokens_details.image_tokens == 0


def test_unspecified_modality_is_skipped():
    usage = Usage.from_openai(
        UsageMetadata(
            prompt_token_count=100,
            prompt_tokens_details=_modalities(
                (MediaModality.MODALITY_UNSPECIFIED, 100)
            ),
        )
    )
    assert usage.input_tokens_details.text_tokens == 0


def test_details_with_missing_counts_do_not_raise():
    usage = Usage.from_openai(
        UsageMetadata(
            prompt_token_count=100,
            prompt_tokens_details=[
                ModalityTokenCount(modality=MediaModality.TEXT),
                ModalityTokenCount(token_count=50),
            ],
        )
    )
    assert usage.input_tokens_details.text_tokens == 0


def test_from_openai_dispatches_to_from_genai():
    usage = Usage.from_openai(
        UsageMetadata(
            prompt_token_count=1000,
            candidates_token_count=500,
            thoughts_token_count=200,
            total_token_count=1700,
        )
    )
    assert usage.input_tokens == 1000
    assert usage.output_tokens == 700


def test_inplace_skips_the_serialization_round_trip():
    raw = UsageMetadata(prompt_token_count=10, candidates_token_count=20)
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
        Usage.from_openai(UsageMetadata(prompt_token_count=5, candidates_token_count=7))
    )
    assert total.requests == 2
    assert total.input_tokens == 15
    assert total.output_tokens == 27


def test_result_can_be_priced():
    usage = Usage.from_openai(
        UsageMetadata(prompt_token_count=1000, candidates_token_count=500)
    )
    assert usage.estimate_cost("gemini-2.5-pro") > 0


def test_cached_modalities_are_removed_from_the_prompt_breakdown():
    """The prompt breakdown covers the cached content too.

    Billing the same cached audio token at both the cache rate and the audio
    rate would overcharge, and would push real text out of its bucket.
    """
    usage = Usage.from_openai(
        UsageMetadata(
            prompt_token_count=1000,
            cached_content_token_count=400,
            candidates_token_count=100,
            prompt_tokens_details=_modalities(
                (MediaModality.AUDIO, 400), (MediaModality.TEXT, 600)
            ),
            cache_tokens_details=_modalities((MediaModality.AUDIO, 400)),
        )
    )
    assert usage.input_tokens == 1000
    assert usage.input_tokens_details.cached_tokens == 400
    assert usage.input_tokens_details.audio_tokens == 0
    assert usage.input_tokens_details.text_tokens == 600


def test_cached_modalities_never_go_negative():
    usage = Usage.from_openai(
        UsageMetadata(
            prompt_token_count=100,
            cached_content_token_count=500,
            prompt_tokens_details=_modalities((MediaModality.AUDIO, 100)),
            cache_tokens_details=_modalities((MediaModality.AUDIO, 500)),
        )
    )
    assert usage.input_tokens_details.audio_tokens == 0


def test_total_tokens_falls_back_to_the_parts():
    """total_token_count is optional like every other field."""
    usage = Usage.from_openai(
        UsageMetadata(
            prompt_token_count=100,
            tool_use_prompt_token_count=10,
            candidates_token_count=50,
            thoughts_token_count=20,
        )
    )
    assert usage.input_tokens == 110
    assert usage.output_tokens == 70
    assert usage.total_tokens == 180


def test_reported_total_tokens_wins():
    usage = Usage.from_openai(
        UsageMetadata(
            prompt_token_count=100,
            candidates_token_count=50,
            total_token_count=999,
        )
    )
    assert usage.total_tokens == 999


def test_non_finite_counts_degrade_to_zero():
    metadata = UsageMetadata(prompt_token_count=100)
    object.__setattr__(metadata, "candidates_token_count", float("nan"))

    usage = Usage.from_openai(metadata)
    assert usage.output_tokens == 0

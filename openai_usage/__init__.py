# openai_usage/__init__.py
"""OpenAI usage tracking utilities.

Simple models for tracking and aggregating OpenAI API usage data.
"""

import decimal
import logging
import math
import pathlib
import sys
import typing

import agents
import pydantic
from openai.types.audio.transcription import UsageDuration as TranscriptionUsageDuration
from openai.types.audio.transcription import UsageTokens as TranscriptionUsageTokens
from openai.types.batch_usage import BatchUsage
from openai.types.beta.threads.run import Usage as RunUsage
from openai.types.beta.threads.runs.run_step import Usage as RunStepUsage
from openai.types.completion_usage import CompletionUsage
from openai.types.create_embedding_response import Usage as EmbeddingUsage
from openai.types.images_response import Usage as ImagesUsage
from openai.types.realtime.realtime_response_usage import RealtimeResponseUsage
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponseUsage,
)

if typing.TYPE_CHECKING:
    from google.genai.types import GenerateContentResponseUsageMetadata
    from litellm.types.utils import Usage as LiteLLMUsage

    from openai_usage.pricing import PricedModel

__version__ = pathlib.Path(__file__).parent.joinpath("VERSION").read_text().strip()

logger = logging.getLogger(__name__)

# Usage objects across the OpenAI SDK that map onto a single billable call.
OpenAIUsage: typing.TypeAlias = typing.Union[
    ResponseUsage,
    agents.RunContextWrapper,
    agents.Usage,
    CompletionUsage,
    RealtimeResponseUsage,
    BatchUsage,
    EmbeddingUsage,
    ImagesUsage,
    TranscriptionUsageTokens,
    TranscriptionUsageDuration,
    RunUsage,
    RunStepUsage,
]

# Everything ``Usage.from_openai`` accepts. The two optional SDKs are quoted
# because they are extras, so neither is importable at runtime by default.
AnyUsage: typing.TypeAlias = typing.Union[
    OpenAIUsage,
    "GenerateContentResponseUsageMetadata",
    "LiteLLMUsage",
]

# Gemini's MediaModality values that map onto a token detail field. DOCUMENT
# and MODALITY_UNSPECIFIED have no field of their own and are deliberately left
# in the plain-text remainder, which is the rate they are billed at.
_TRACKED_MODALITIES = ("TEXT", "AUDIO", "IMAGE", "VIDEO")


class UsageInputTokensDetails(InputTokensDetails):
    """Input token breakdown, extended with audio/text/image/video counts.

    Inherits ``cached_tokens`` from the Responses schema so existing data stays
    valid; the extra fields default to ``0`` and only apply to the usage types
    that report them (realtime audio, image generation, transcription, Gemini
    video).

    ``cache_write_tokens`` is redefined with a default so this model stays
    constructible across OpenAI SDK versions: newer releases add it as a
    required field on the base ``InputTokensDetails`` (which would otherwise
    break every ``UsageInputTokensDetails(cached_tokens=...)`` call), while
    older releases lack it entirely.
    """

    cache_write_tokens: int = 0
    audio_tokens: int = 0
    text_tokens: int = 0
    image_tokens: int = 0
    video_tokens: int = 0


class UsageOutputTokensDetails(OutputTokensDetails):
    """Output token breakdown, extended with audio/text/image/video counts.

    Inherits ``reasoning_tokens`` from the Responses schema so existing data
    stays valid; the extra fields default to ``0`` and only apply to the usage
    types that report them (realtime audio, image generation, Gemini video).
    """

    audio_tokens: int = 0
    text_tokens: int = 0
    image_tokens: int = 0
    video_tokens: int = 0


class Usage(pydantic.BaseModel):
    """Usage statistics for OpenAI API calls.

    Tracks token counts and request metrics across API interactions.
    """

    requests: int = 0
    input_tokens: int = 0
    input_tokens_details: UsageInputTokensDetails = pydantic.Field(
        default_factory=lambda: UsageInputTokensDetails(cached_tokens=0)
    )
    output_tokens: int = 0
    output_tokens_details: UsageOutputTokensDetails = pydantic.Field(
        default_factory=lambda: UsageOutputTokensDetails(reasoning_tokens=0)
    )
    total_tokens: int = 0

    # Duration-based usage (e.g. whisper-1 transcription billed per second).
    seconds: float = 0

    # Extra fields from OpenAI schema
    model: str | None = None
    cost: str | float | None = None
    annotations: str | None = None

    @classmethod
    def from_openai(
        cls,
        openai_usage: AnyUsage,
        *,
        inplace: bool = False,
    ) -> "Usage":
        """Create Usage from any supported usage object.

        Accepts every OpenAI SDK usage type, plus google-genai's
        ``usage_metadata`` and litellm's ``Usage`` when those extras are
        installed. Returns a new instance by default.
        """

        if isinstance(openai_usage, agents.RunContextWrapper):
            openai_usage = openai_usage.usage

        # litellm's Usage subclasses CompletionUsage, so it has to be matched
        # before that branch or its extra detail is silently dropped.
        if _is_litellm_usage(openai_usage):
            usage = cls(**_litellm_fields(openai_usage))

        elif _is_genai_usage(openai_usage):
            usage = cls(**_genai_fields(openai_usage))

        # Token + cached/reasoning details: Responses, Agents SDK, Batch.
        elif isinstance(openai_usage, (ResponseUsage, agents.Usage, BatchUsage)):
            usage = cls(
                requests=1,
                input_tokens=openai_usage.input_tokens,
                input_tokens_details=UsageInputTokensDetails(
                    cached_tokens=openai_usage.input_tokens_details.cached_tokens,
                    # Newer SDKs add cache_write_tokens; BatchUsage still omits it.
                    cache_write_tokens=getattr(
                        openai_usage.input_tokens_details, "cache_write_tokens", 0
                    )
                    or 0,
                ),
                output_tokens=openai_usage.output_tokens,
                output_tokens_details=UsageOutputTokensDetails(
                    reasoning_tokens=openai_usage.output_tokens_details.reasoning_tokens
                ),
                total_tokens=openai_usage.total_tokens,
            )

        elif isinstance(openai_usage, CompletionUsage):
            p_details = openai_usage.prompt_tokens_details
            c_details = openai_usage.completion_tokens_details
            usage = cls(
                requests=1,
                input_tokens=openai_usage.prompt_tokens,
                input_tokens_details=UsageInputTokensDetails(
                    cached_tokens=(p_details.cached_tokens or 0) if p_details else 0,
                    # cache_write_tokens is present only on newer SDKs.
                    cache_write_tokens=(
                        (getattr(p_details, "cache_write_tokens", 0) or 0)
                        if p_details
                        else 0
                    ),
                    # Audio input tokens (e.g. gpt-4o-audio-preview via chat).
                    audio_tokens=(p_details.audio_tokens or 0) if p_details else 0,
                ),
                output_tokens=openai_usage.completion_tokens,
                output_tokens_details=UsageOutputTokensDetails(
                    reasoning_tokens=(
                        (c_details.reasoning_tokens or 0) if c_details else 0
                    ),
                    audio_tokens=(c_details.audio_tokens or 0) if c_details else 0,
                ),
                total_tokens=openai_usage.total_tokens,
            )

        # Assistants run / run step: prompt + completion tokens, no details.
        elif isinstance(openai_usage, (RunUsage, RunStepUsage)):
            usage = cls(
                requests=1,
                input_tokens=openai_usage.prompt_tokens,
                output_tokens=openai_usage.completion_tokens,
                total_tokens=openai_usage.total_tokens,
            )

        # Embeddings: prompt tokens only, no generated output.
        elif isinstance(openai_usage, EmbeddingUsage):
            usage = cls(
                requests=1,
                input_tokens=openai_usage.prompt_tokens,
                total_tokens=openai_usage.total_tokens,
            )

        elif isinstance(openai_usage, RealtimeResponseUsage):
            in_details = openai_usage.input_token_details
            out_details = openai_usage.output_token_details
            usage = cls(
                requests=1,
                input_tokens=openai_usage.input_tokens or 0,
                input_tokens_details=UsageInputTokensDetails(
                    cached_tokens=(in_details.cached_tokens or 0) if in_details else 0,
                    audio_tokens=(in_details.audio_tokens or 0) if in_details else 0,
                    text_tokens=(in_details.text_tokens or 0) if in_details else 0,
                    # image_tokens is present only on newer SDKs.
                    image_tokens=(
                        (getattr(in_details, "image_tokens", 0) or 0)
                        if in_details
                        else 0
                    ),
                ),
                output_tokens=openai_usage.output_tokens or 0,
                output_tokens_details=UsageOutputTokensDetails(
                    reasoning_tokens=0,
                    audio_tokens=(out_details.audio_tokens or 0) if out_details else 0,
                    text_tokens=(out_details.text_tokens or 0) if out_details else 0,
                ),
                total_tokens=openai_usage.total_tokens or 0,
            )

        # Image generation (gpt-image-1): image/text token breakdown.
        elif isinstance(openai_usage, ImagesUsage):
            in_details = openai_usage.input_tokens_details
            out_details = openai_usage.output_tokens_details
            usage = cls(
                requests=1,
                input_tokens=openai_usage.input_tokens,
                input_tokens_details=UsageInputTokensDetails(
                    cached_tokens=0,
                    image_tokens=in_details.image_tokens,
                    text_tokens=in_details.text_tokens,
                ),
                output_tokens=openai_usage.output_tokens,
                output_tokens_details=UsageOutputTokensDetails(
                    reasoning_tokens=0,
                    image_tokens=out_details.image_tokens if out_details else 0,
                    text_tokens=out_details.text_tokens if out_details else 0,
                ),
                total_tokens=openai_usage.total_tokens,
            )

        # Transcription billed by tokens (e.g. gpt-4o-transcribe).
        elif isinstance(openai_usage, TranscriptionUsageTokens):
            in_details = openai_usage.input_token_details
            usage = cls(
                requests=1,
                input_tokens=openai_usage.input_tokens,
                input_tokens_details=UsageInputTokensDetails(
                    cached_tokens=0,
                    audio_tokens=(in_details.audio_tokens or 0) if in_details else 0,
                    text_tokens=(in_details.text_tokens or 0) if in_details else 0,
                ),
                output_tokens=openai_usage.output_tokens,
                total_tokens=openai_usage.total_tokens,
            )

        # Transcription billed by audio duration (e.g. whisper-1).
        elif isinstance(openai_usage, TranscriptionUsageDuration):
            usage = cls(requests=1, seconds=openai_usage.seconds)

        else:
            raise ValueError(f"Unsupported usage type: {type(openai_usage)}")

        if inplace:
            return usage
        else:
            return cls.model_validate_json(usage.model_dump_json())

    def add(self, other: "Usage") -> None:
        """Add usage from another Usage instance.

        Accumulates all metrics including token counts and request totals.
        """
        self.requests += other.requests if other.requests else 0
        self.input_tokens += other.input_tokens if other.input_tokens else 0
        self.output_tokens += other.output_tokens if other.output_tokens else 0
        self.total_tokens += other.total_tokens if other.total_tokens else 0
        self.seconds += other.seconds if other.seconds else 0
        self.input_tokens_details = UsageInputTokensDetails(
            cached_tokens=self.input_tokens_details.cached_tokens
            + other.input_tokens_details.cached_tokens,
            cache_write_tokens=self.input_tokens_details.cache_write_tokens
            + other.input_tokens_details.cache_write_tokens,
            audio_tokens=self.input_tokens_details.audio_tokens
            + other.input_tokens_details.audio_tokens,
            text_tokens=self.input_tokens_details.text_tokens
            + other.input_tokens_details.text_tokens,
            image_tokens=self.input_tokens_details.image_tokens
            + other.input_tokens_details.image_tokens,
            video_tokens=self.input_tokens_details.video_tokens
            + other.input_tokens_details.video_tokens,
        )

        self.output_tokens_details = UsageOutputTokensDetails(
            reasoning_tokens=self.output_tokens_details.reasoning_tokens
            + other.output_tokens_details.reasoning_tokens,
            audio_tokens=self.output_tokens_details.audio_tokens
            + other.output_tokens_details.audio_tokens,
            text_tokens=self.output_tokens_details.text_tokens
            + other.output_tokens_details.text_tokens,
            image_tokens=self.output_tokens_details.image_tokens
            + other.output_tokens_details.image_tokens,
            video_tokens=self.output_tokens_details.video_tokens
            + other.output_tokens_details.video_tokens,
        )

    def estimate_cost(
        self,
        model: typing.Union["PricedModel", str, None] = None,
        *,
        realtime_pricing: bool = False,
        ignore_not_found: bool = True,
    ) -> float:
        return float(
            self.estimate_cost_str(
                model,
                realtime_pricing=realtime_pricing,
                ignore_not_found=ignore_not_found,
            )
        )

    def estimate_cost_str(
        self,
        model: typing.Union["PricedModel", str, None] = None,
        *,
        realtime_pricing: bool = False,
        ignore_not_found: bool = True,
    ) -> str:
        """Calculate estimated cost based on usage and model pricing.

        Computes total cost using token counts including cached and reasoning tokens.
        """
        model = model or self.model
        if model is None:
            logger.warning("No model provided, using 'gpt-4o-mini' as default")
            model = "gpt-4o-mini"

        if isinstance(model, str):
            from openai_usage.pricing import get_model

            might_model = get_model(model, realtime_pricing=realtime_pricing)
            if might_model is None:
                if ignore_not_found:
                    logger.warning(f"No model found for '{model}', returning 0.0 cost")
                    return str(decimal.Decimal(0))
                else:
                    raise ValueError(f"No model found for '{model}'")
            else:
                model = might_model

        pricing = model.pricing

        # Cached, cache-write, audio and image tokens are treated as disjoint
        # subsets of the totals and billed at their own rates; the remainder is
        # billed as plain text. Video tokens are left in that remainder because
        # neither catalog prices them per token, and providers that report them
        # bill at the ordinary input rate.
        input_audio = self.input_tokens_details.audio_tokens
        input_image = self.input_tokens_details.image_tokens
        input_cached = self.input_tokens_details.cached_tokens
        input_cache_write = self.input_tokens_details.cache_write_tokens
        input_text = max(
            0,
            self.input_tokens
            - input_audio
            - input_image
            - input_cached
            - input_cache_write,
        )

        output_audio = self.output_tokens_details.audio_tokens
        output_image = self.output_tokens_details.image_tokens
        output_reasoning = self.output_tokens_details.reasoning_tokens
        output_text = max(
            0, self.output_tokens - output_audio - output_image - output_reasoning
        )

        # Decimal(0) * Decimal("Infinity") raises, and a NaN would propagate
        # into the returned string and poison any downstream aggregate.
        seconds = self.seconds
        if not math.isfinite(seconds):
            logger.warning(f"Ignoring non-finite seconds {seconds!r}")
            seconds = 0.0

        cost = (
            +pricing.price_per_request * self.requests
            + pricing.price_per_input_token_without_cached * input_text
            + pricing.price_per_input_token_with_cached * input_cached
            + pricing.price_per_input_cache_write_token * input_cache_write
            + pricing.price_per_output_not_reasoning_token * output_text
            + pricing.price_per_output_reasoning_token * output_reasoning
            + pricing.price_per_input_audio_token * input_audio
            + pricing.price_per_output_audio_token * output_audio
            + pricing.price_per_input_image_token * input_image
            + pricing.price_per_output_image_token * output_image
            + pricing.price_per_second * decimal.Decimal(str(seconds))
        )
        return str(cost)


def _imported_type(module_name: str, attribute: str) -> type | None:
    """Return a type from an already-imported module, without importing it.

    google-genai and litellm are optional and slow to import, but a caller can
    only hand us one of their objects if the SDK is already loaded. Looking the
    module up in ``sys.modules`` therefore costs nothing and never triggers an
    import of an extra the caller did not install.
    """
    module = sys.modules.get(module_name)
    if module is None:
        return None
    candidate = getattr(module, attribute, None)
    return candidate if isinstance(candidate, type) else None


def _is_genai_usage(usage: typing.Any) -> bool:
    """Whether ``usage`` is a google-genai ``usage_metadata``."""
    genai_type = _imported_type(
        "google.genai.types", "GenerateContentResponseUsageMetadata"
    )
    return genai_type is not None and isinstance(usage, genai_type)


def _is_litellm_usage(usage: typing.Any) -> bool:
    """Whether ``usage`` is a litellm ``Usage``."""
    litellm_type = _imported_type("litellm.types.utils", "Usage")
    return litellm_type is not None and isinstance(usage, litellm_type)


def _int(value: typing.Any) -> int:
    """Coerce an optional token count to an int.

    Both optional SDKs leave counts as ``None`` when a modality does not apply,
    and litellm additionally deletes attributes whose value is ``None``. A
    non-finite float would otherwise raise here rather than degrade, so it is
    read as ``0`` alongside the other missing-value cases.
    """
    if isinstance(value, float) and not math.isfinite(value):
        logger.warning(f"Ignoring non-finite token count {value!r}")
        return 0
    return int(value) if value else 0


def _sum_modalities(*detail_lists: typing.Any) -> dict[str, int]:
    """Total up google-genai ``ModalityTokenCount`` entries per modality.

    Reads the enum by value so google-genai never has to be imported, and so an
    unrecognized modality is skipped instead of raising.
    """
    totals = {modality: 0 for modality in _TRACKED_MODALITIES}
    for details in detail_lists:
        for item in details or ():
            modality = getattr(item, "modality", None)
            name = getattr(modality, "value", modality)
            if name in totals:
                totals[name] += _int(getattr(item, "token_count", None))
    return totals


def _genai_fields(genai_usage: typing.Any) -> dict[str, typing.Any]:
    """Map a google-genai ``usage_metadata`` onto ``Usage`` fields.

    Every field on the Gemini schema is optional, so absent counts read as
    ``0``.
    """
    # Gemini reports thinking separately from the visible answer, while
    # ``output_tokens`` here means everything billed as output.
    candidates = _int(genai_usage.candidates_token_count)
    thoughts = _int(genai_usage.thoughts_token_count)

    # Tool-use prompts are billed as input alongside the ordinary prompt.
    prompt = _int(genai_usage.prompt_token_count)
    tool_use_prompt = _int(genai_usage.tool_use_prompt_token_count)

    input_modalities = _sum_modalities(
        genai_usage.prompt_tokens_details,
        genai_usage.tool_use_prompt_tokens_details,
    )
    output_modalities = _sum_modalities(genai_usage.candidates_tokens_details)

    # The prompt breakdown covers the whole prompt, cached content included, so
    # a cached audio segment appears in both the AUDIO bucket and the cached
    # count. The estimator treats those as disjoint, so remove the cached
    # modalities here rather than bill them at two rates at once.
    cached_modalities = _sum_modalities(genai_usage.cache_tokens_details)
    input_modalities = {
        name: max(0, count - cached_modalities[name])
        for name, count in input_modalities.items()
    }

    cached = _int(genai_usage.cached_content_token_count)
    input_tokens = prompt + tool_use_prompt
    output_tokens = candidates + thoughts

    # total_token_count is optional like everything else, and is defined
    # upstream as exactly these parts summed.
    total_tokens = _int(genai_usage.total_token_count) or input_tokens + output_tokens

    return {
        "requests": 1,
        "input_tokens": input_tokens,
        "input_tokens_details": UsageInputTokensDetails(
            cached_tokens=cached,
            audio_tokens=input_modalities["AUDIO"],
            text_tokens=input_modalities["TEXT"],
            image_tokens=input_modalities["IMAGE"],
            video_tokens=input_modalities["VIDEO"],
        ),
        "output_tokens": output_tokens,
        "output_tokens_details": UsageOutputTokensDetails(
            reasoning_tokens=thoughts,
            audio_tokens=output_modalities["AUDIO"],
            text_tokens=output_modalities["TEXT"],
            image_tokens=output_modalities["IMAGE"],
            video_tokens=output_modalities["VIDEO"],
        ),
        "total_tokens": total_tokens,
    }


def _litellm_fields(litellm_usage: typing.Any) -> dict[str, typing.Any]:
    """Map a litellm ``Usage`` onto ``Usage`` fields.

    litellm normalizes every provider onto the Chat Completions shape and then
    widens it, so this reads the extra detail its wrappers add on top of
    ``CompletionUsage``.
    """
    p_details = litellm_usage.prompt_tokens_details
    c_details = litellm_usage.completion_tokens_details

    # litellm folds Anthropic's cache buckets into prompt_tokens, so both stay
    # disjoint subsets of the input total, the way the estimator wants.
    cache_write = _int(getattr(p_details, "cache_creation_tokens", None))
    if not cache_write:
        cache_write = _int(getattr(litellm_usage, "_cache_creation_input_tokens", 0))

    cached = _int(getattr(p_details, "cached_tokens", None))
    if not cached:
        cached = _int(getattr(litellm_usage, "_cache_read_input_tokens", 0))

    return {
        "requests": 1,
        "input_tokens": _int(litellm_usage.prompt_tokens),
        "input_tokens_details": UsageInputTokensDetails(
            cached_tokens=cached,
            cache_write_tokens=cache_write,
            audio_tokens=_int(getattr(p_details, "audio_tokens", None)),
            text_tokens=_int(getattr(p_details, "text_tokens", None)),
            image_tokens=_int(getattr(p_details, "image_tokens", None)),
            video_tokens=_int(getattr(p_details, "video_tokens", None)),
        ),
        "output_tokens": _int(litellm_usage.completion_tokens),
        "output_tokens_details": UsageOutputTokensDetails(
            reasoning_tokens=_int(getattr(c_details, "reasoning_tokens", None)),
            audio_tokens=_int(getattr(c_details, "audio_tokens", None)),
            text_tokens=_int(getattr(c_details, "text_tokens", None)),
            image_tokens=_int(getattr(c_details, "image_tokens", None)),
            video_tokens=_int(getattr(c_details, "video_tokens", None)),
        ),
        "total_tokens": _int(litellm_usage.total_tokens),
        "cost": getattr(litellm_usage, "cost", None),
    }

# openai_usage/__init__.py
"""OpenAI usage tracking utilities.

Simple models for tracking and aggregating OpenAI API usage data.
"""

import decimal
import logging
import pathlib
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
    from openai_usage.extra.open_router import OpenRouterModel

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


class UsageInputTokensDetails(InputTokensDetails):
    """Input token breakdown, extended with audio/text/image counts.

    Inherits ``cached_tokens`` from the Responses schema so existing data stays
    valid; the extra fields default to ``0`` and only apply to the usage types
    that report them (realtime audio, image generation, transcription).
    """

    audio_tokens: int = 0
    text_tokens: int = 0
    image_tokens: int = 0


class UsageOutputTokensDetails(OutputTokensDetails):
    """Output token breakdown, extended with audio/text/image counts.

    Inherits ``reasoning_tokens`` from the Responses schema so existing data
    stays valid; the extra fields default to ``0`` and only apply to the usage
    types that report them (realtime audio, image generation).
    """

    audio_tokens: int = 0
    text_tokens: int = 0
    image_tokens: int = 0


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
        openai_usage: OpenAIUsage,
        *,
        inplace: bool = False,
    ) -> "Usage":
        """Create Usage from OpenAI usage objects.

        Supports multiple OpenAI usage types and returns a new instance by default.
        """

        if isinstance(openai_usage, agents.RunContextWrapper):
            openai_usage = openai_usage.usage

        # Token + cached/reasoning details: Responses, Agents SDK, Batch.
        if isinstance(openai_usage, (ResponseUsage, agents.Usage, BatchUsage)):
            usage = cls(
                requests=1,
                input_tokens=openai_usage.input_tokens,
                input_tokens_details=UsageInputTokensDetails(
                    cached_tokens=openai_usage.input_tokens_details.cached_tokens
                ),
                output_tokens=openai_usage.output_tokens,
                output_tokens_details=UsageOutputTokensDetails(
                    reasoning_tokens=openai_usage.output_tokens_details.reasoning_tokens
                ),
                total_tokens=openai_usage.total_tokens,
            )

        elif isinstance(openai_usage, CompletionUsage):
            usage = cls(
                requests=1,
                input_tokens=openai_usage.prompt_tokens,
                input_tokens_details=UsageInputTokensDetails(
                    cached_tokens=(
                        openai_usage.prompt_tokens_details.cached_tokens or 0
                        if openai_usage.prompt_tokens_details
                        else 0
                    )
                ),
                output_tokens=openai_usage.completion_tokens,
                output_tokens_details=UsageOutputTokensDetails(
                    reasoning_tokens=(
                        openai_usage.completion_tokens_details.reasoning_tokens or 0
                        if openai_usage.completion_tokens_details
                        else 0
                    )
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
            audio_tokens=self.input_tokens_details.audio_tokens
            + other.input_tokens_details.audio_tokens,
            text_tokens=self.input_tokens_details.text_tokens
            + other.input_tokens_details.text_tokens,
            image_tokens=self.input_tokens_details.image_tokens
            + other.input_tokens_details.image_tokens,
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
        )

    def estimate_cost(
        self,
        model: typing.Union["OpenRouterModel", str, None] = None,
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
        model: typing.Union["OpenRouterModel", str, None] = None,
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
            from openai_usage.extra.open_router import get_model

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

        # Audio and image tokens are billed at distinct rates (0 for plain text
        # usage). Cached, audio and image tokens are treated as disjoint subsets
        # of the totals; the remainder is billed as plain text. Duration-based
        # billing (``seconds``) has no OpenRouter rate and is not priced here.
        input_audio = self.input_tokens_details.audio_tokens
        input_image = self.input_tokens_details.image_tokens
        input_cached = self.input_tokens_details.cached_tokens
        input_text = max(
            0, self.input_tokens - input_audio - input_image - input_cached
        )

        output_audio = self.output_tokens_details.audio_tokens
        output_image = self.output_tokens_details.image_tokens
        output_reasoning = self.output_tokens_details.reasoning_tokens
        output_text = max(
            0, self.output_tokens - output_audio - output_image - output_reasoning
        )

        cost = (
            +pricing.price_per_request * self.requests
            + pricing.price_per_input_token_without_cached * input_text
            + pricing.price_per_input_token_with_cached * input_cached
            + pricing.price_per_output_not_reasoning_token * output_text
            + pricing.price_per_output_reasoning_token * output_reasoning
            + pricing.price_per_audio_token * (input_audio + output_audio)
            + pricing.price_per_image_token * (input_image + output_image)
        )
        return str(cost)

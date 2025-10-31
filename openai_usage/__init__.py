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
from openai.types.completion_usage import CompletionUsage
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponseUsage,
)

if typing.TYPE_CHECKING:
    from openai_usage.extra.open_router import OpenRouterModel

__version__ = pathlib.Path(__file__).parent.joinpath("VERSION").read_text().strip()


logger = logging.getLogger(__name__)


class Usage(pydantic.BaseModel):
    """Usage statistics for OpenAI API calls.

    Tracks token counts and request metrics across API interactions.
    """

    requests: int = 0
    input_tokens: int = 0
    input_tokens_details: InputTokensDetails = pydantic.Field(
        default_factory=lambda: InputTokensDetails(cached_tokens=0)
    )
    output_tokens: int = 0
    output_tokens_details: OutputTokensDetails = pydantic.Field(
        default_factory=lambda: OutputTokensDetails(reasoning_tokens=0)
    )
    total_tokens: int = 0

    # Extra fields from OpenAI schema
    model: str | None = None
    cost: str | float | None = None
    annotations: str | None = None

    @classmethod
    def from_openai(
        cls,
        openai_usage: (
            ResponseUsage | agents.RunContextWrapper | agents.Usage | CompletionUsage
        ),
        *,
        inplace: bool = False,
    ) -> "Usage":
        """Create Usage from OpenAI usage objects.

        Supports multiple OpenAI usage types and returns a new instance by default.
        """

        if isinstance(openai_usage, ResponseUsage):
            usage = cls(
                requests=1,
                input_tokens=openai_usage.input_tokens,
                input_tokens_details=openai_usage.input_tokens_details,
                output_tokens=openai_usage.output_tokens,
                output_tokens_details=openai_usage.output_tokens_details,
                total_tokens=openai_usage.total_tokens,
            )
        elif isinstance(openai_usage, agents.RunContextWrapper):
            usage = cls(
                requests=1,
                input_tokens=openai_usage.usage.input_tokens,
                input_tokens_details=openai_usage.usage.input_tokens_details,
                output_tokens=openai_usage.usage.output_tokens,
                output_tokens_details=openai_usage.usage.output_tokens_details,
                total_tokens=openai_usage.usage.total_tokens,
            )
        elif isinstance(openai_usage, agents.Usage):
            usage = cls(
                requests=1,
                input_tokens=openai_usage.input_tokens,
                input_tokens_details=openai_usage.input_tokens_details,
                output_tokens=openai_usage.output_tokens,
                output_tokens_details=openai_usage.output_tokens_details,
                total_tokens=openai_usage.total_tokens,
            )

        elif isinstance(openai_usage, CompletionUsage):
            usage = cls(
                requests=1,
                input_tokens=openai_usage.prompt_tokens,
                input_tokens_details=InputTokensDetails(
                    cached_tokens=(
                        openai_usage.prompt_tokens_details.cached_tokens or 0
                        if openai_usage.prompt_tokens_details
                        else 0
                    )
                ),
                output_tokens=openai_usage.completion_tokens,
                output_tokens_details=OutputTokensDetails(
                    reasoning_tokens=(
                        openai_usage.completion_tokens_details.reasoning_tokens or 0
                        if openai_usage.completion_tokens_details
                        else 0
                    )
                ),
                total_tokens=openai_usage.total_tokens,
            )

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
        self.input_tokens_details = InputTokensDetails(
            cached_tokens=self.input_tokens_details.cached_tokens
            + other.input_tokens_details.cached_tokens
        )

        self.output_tokens_details = OutputTokensDetails(
            reasoning_tokens=self.output_tokens_details.reasoning_tokens
            + other.output_tokens_details.reasoning_tokens
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

        input_tokens_w_cached = self.input_tokens_details.cached_tokens
        input_tokens_wo_cached = self.input_tokens - input_tokens_w_cached

        output_tokens_w_reasoning = self.output_tokens_details.reasoning_tokens
        output_tokens_wo_reasoning = self.output_tokens - output_tokens_w_reasoning

        cost = (
            +pricing.price_per_request * self.requests
            + pricing.price_per_input_token_without_cached * input_tokens_wo_cached
            + pricing.price_per_input_token_with_cached * input_tokens_w_cached
            + (
                pricing.price_per_output_not_reasoning_token
                * output_tokens_wo_reasoning
            )
            + pricing.price_per_output_reasoning_token * output_tokens_w_reasoning
        )
        return str(cost)

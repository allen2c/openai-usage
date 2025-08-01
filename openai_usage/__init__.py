# openai_usage/__init__.py
"""OpenAI usage tracking utilities.

Simple models for tracking and aggregating OpenAI API usage data.
"""

import pathlib

import agents
import pydantic
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponseUsage,
)

__version__ = pathlib.Path(__file__).parent.joinpath("VERSION").read_text().strip()


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

    @classmethod
    def from_openai(
        cls,
        openai_usage: ResponseUsage | agents.RunContextWrapper | agents.Usage,
        *,
        inplace: bool = False,
    ) -> "Usage":
        """Create Usage from OpenAI usage objects.

        Supports ResponseUsage, RunContextWrapper, and agents.Usage types.
        Returns new instance unless inplace=True.
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

import decimal
import enum
import functools
import json
import logging
import pathlib
import re
import typing

import pydantic
import requests

logger = logging.getLogger(__name__)

MODEL_CONFIG_EXTRA: typing.Literal["forbid", "allow", "ignore"] = "ignore"


DASH_TRANS = str.maketrans({ord(":"): "-", ord("_"): "-", ord("."): "-"})
DROP_TRANS = str.maketrans({ord(":"): None, ord("-"): None, ord("."): None})


@functools.cache
def get_models(realtime_pricing: bool = False) -> "GetOpenRouterModelsResponse":
    """Fetch all available models from OpenRouter API.

    Returns cached response for performance.
    """

    if realtime_pricing:
        url = "https://openrouter.ai/api/v1/models"
        try:
            response = requests.get(url)
            response.raise_for_status()
            models = GetOpenRouterModelsResponse.model_validate_json(response.text)
            logger.info(f"There are {len(models.data)} models")
            return models
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch models: {e}")

    logger.info("Using locally cached models")
    return GetOpenRouterModelsResponse.model_validate_json(
        pathlib.Path(__file__).parent.parent.joinpath("models.json").read_text()
    ).merge(
        GetOpenRouterModelsResponse.model_validate_json(
            pathlib.Path(__file__)
            .parent.parent.joinpath("models_voyageai.json")
            .read_text()
        )
    )


def get_model(
    model_name: str, *, realtime_pricing: bool = False
) -> typing.Optional["OpenRouterModel"]:
    """Find a model by name with flexible matching.

    Returns exact match first, then partial match.
    Handles multiple matches by choosing shortest ID.
    """
    all_models = get_models(realtime_pricing=realtime_pricing)

    models: list[OpenRouterModel] = []
    for model in all_models.data:
        # Exact match
        if re.search(f"{model_name}$", model.id, re.IGNORECASE):
            return model
        # Partial match with the same name
        if re.search(model_name, model.id, re.IGNORECASE):
            models.append(model)
        #  Normalize as dash-separated string
        if re.search(
            model_name.translate(DASH_TRANS),
            model.id.translate(DASH_TRANS),
            re.IGNORECASE,
        ):
            models.append(model)
        # Partial match removing colons, dashes and dots
        if re.search(
            model_name.translate(DROP_TRANS),
            model.id.translate(DROP_TRANS),
            re.IGNORECASE,
        ):
            models.append(model)

    if len(models) == 1:
        logger.warning(
            f"Found model '{models[0].id}' for '{model_name}', "
            + "the model name is not strict"
        )
        return models[0]

    elif len(models) > 1:
        logger.warning(
            f"Multiple models found for '{model_name}': "
            + f"{', '.join(m.id for m in models)}, "
            + "choose the most not greedy one"
        )
        models.sort(key=lambda x: len(x.id.split("/")[-1]))
        return models[0]

    logger.debug(f"No model found for '{model_name}'")
    return None


class OpenRouterArchitecture(pydantic.BaseModel):
    """Model architecture details including modalities and tokenizer."""

    modality: str
    input_modalities: list[str]
    output_modalities: list[str]
    tokenizer: str
    instruct_type: str | None

    model_config = pydantic.ConfigDict(extra=MODEL_CONFIG_EXTRA)


class OpenRouterPricing(pydantic.BaseModel):
    """Pricing information for different input/output types."""

    prompt: str
    completion: str
    request: str | None = None
    image: str | None = None
    audio: str | None = None
    web_search: str | None = None
    internal_reasoning: str | None = None
    input_cache_read: str | None = None
    input_cache_write: str | None = None

    model_config = pydantic.ConfigDict(extra=MODEL_CONFIG_EXTRA)

    @property
    def price_per_request(self) -> decimal.Decimal:
        return decimal.Decimal(self.request or 0)

    @property
    def price_per_input_token_without_cached(self) -> decimal.Decimal:
        return decimal.Decimal(self.prompt or 0)

    @property
    def price_per_input_token_with_cached(self) -> decimal.Decimal:
        return (
            decimal.Decimal(self.input_cache_read or 0)
            or self.price_per_input_token_without_cached
        )

    @property
    def price_per_output_not_reasoning_token(self) -> decimal.Decimal:
        return decimal.Decimal(self.completion or 0)

    @property
    def price_per_output_reasoning_token(self) -> decimal.Decimal:
        return decimal.Decimal(self.internal_reasoning or 0)

    @property
    def price_per_audio_token(self) -> decimal.Decimal:
        return decimal.Decimal(self.audio or 0)

    @property
    def price_per_image_token(self) -> decimal.Decimal:
        return decimal.Decimal(self.image or 0)


class OpenRouterTopProvider(pydantic.BaseModel):
    """Provider-specific limits and moderation settings."""

    context_length: int | None = None
    max_completion_tokens: int | None
    is_moderated: bool

    model_config = pydantic.ConfigDict(extra=MODEL_CONFIG_EXTRA)


class OpenRouterPerRequestLimits(pydantic.BaseModel):
    """Token limits for different content types per request."""

    max_tokens: int
    max_completion_tokens: int
    max_prompt_tokens: int
    max_image_tokens: int
    max_audio_tokens: int
    max_web_search_tokens: int

    model_config = pydantic.ConfigDict(extra=MODEL_CONFIG_EXTRA)


class OpenRouterSupportedParameters(pydantic.BaseModel):
    """Supported API parameters and their default values."""

    max_tokens: int
    temperature: float
    top_p: float
    tools: list[str]
    tool_choice: str
    stop: list[str]
    frequency_penalty: float
    presence_penalty: float
    seed: int
    logit_bias: dict

    model_config = pydantic.ConfigDict(extra=MODEL_CONFIG_EXTRA)


class OpenRouterDefaultParameters(pydantic.BaseModel):
    """Provider-suggested default sampling parameters for the model."""

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None

    model_config = pydantic.ConfigDict(extra=MODEL_CONFIG_EXTRA)


class OpenRouterLinks(pydantic.BaseModel):
    """Related API resource links for the model."""

    details: str | None = None

    model_config = pydantic.ConfigDict(extra=MODEL_CONFIG_EXTRA)


class OpenRouterReasoningEffort(str, enum.Enum):
    """Known reasoning effort levels, ordered low to high.

    Different models expose different subsets. Unknown values fall back to a
    plain ``str`` via the ``ReasoningEffort`` alias below.
    """

    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


# Resolve to the enum when the value is known, otherwise keep the raw string.
# ``left_to_right`` is required so the enum is tried before the str fallback.
ReasoningEffort = typing.Annotated[
    OpenRouterReasoningEffort | str,
    pydantic.Field(union_mode="left_to_right"),
]


class OpenRouterReasoning(pydantic.BaseModel):
    """Reasoning / thinking capability metadata for the model."""

    mandatory: bool | None = None
    default_enabled: bool | None = None
    supported_efforts: list[ReasoningEffort] | None = None
    default_effort: ReasoningEffort | None = None
    supports_max_tokens: bool | None = None

    model_config = pydantic.ConfigDict(extra=MODEL_CONFIG_EXTRA)


class OpenRouterDesignArenaBenchmark(pydantic.BaseModel):
    """A single Design Arena benchmark result entry."""

    arena: str | None = None
    category: str | None = None
    elo: float | None = None
    win_rate: float | None = None
    rank: int | None = None

    model_config = pydantic.ConfigDict(extra=MODEL_CONFIG_EXTRA)


class OpenRouterArtificialAnalysisBenchmark(pydantic.BaseModel):
    """Artificial Analysis benchmark index scores."""

    intelligence_index: float | None = None
    coding_index: float | None = None
    agentic_index: float | None = None

    model_config = pydantic.ConfigDict(extra=MODEL_CONFIG_EXTRA)


class OpenRouterBenchmarks(pydantic.BaseModel):
    """Aggregated third-party benchmark results for the model."""

    design_arena: list[OpenRouterDesignArenaBenchmark] | None = None
    artificial_analysis: OpenRouterArtificialAnalysisBenchmark | None = None

    model_config = pydantic.ConfigDict(extra=MODEL_CONFIG_EXTRA)


class OpenRouterModel(pydantic.BaseModel):
    """Complete model information including pricing and capabilities."""

    id: str
    canonical_slug: str
    hugging_face_id: str | None
    name: str
    created: int
    description: str
    context_length: int
    architecture: OpenRouterArchitecture
    pricing: OpenRouterPricing
    top_provider: OpenRouterTopProvider
    per_request_limits: OpenRouterPerRequestLimits | None
    supported_parameters: list[str]

    # Extended metadata from OpenRouter; optional and absent on older snapshots.
    default_parameters: OpenRouterDefaultParameters | None = None
    supported_voices: list[str] | None = None
    knowledge_cutoff: str | None = None
    expiration_date: str | None = None
    links: OpenRouterLinks | None = None
    reasoning: OpenRouterReasoning | None = None
    benchmarks: OpenRouterBenchmarks | None = None

    model_config = pydantic.ConfigDict(extra=MODEL_CONFIG_EXTRA)


class GetOpenRouterModelsResponse(pydantic.BaseModel):
    """API response wrapper containing list of available models."""

    data: list[OpenRouterModel]

    def merge(
        self, other: "GetOpenRouterModelsResponse"
    ) -> "GetOpenRouterModelsResponse":
        data = self.data + other.data
        data_dict = json.loads(
            pydantic.TypeAdapter(list[OpenRouterModel]).dump_json(data)
        )
        return GetOpenRouterModelsResponse.model_validate({"data": data_dict})

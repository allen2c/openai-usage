import functools
import logging
import pathlib
import re
import typing

import pydantic
import requests

logger = logging.getLogger(__name__)

MODEL_CONFIG_EXTRA: typing.Literal["forbid", "allow", "ignore"] = "ignore"


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
        if re.search(f"{model_name}$", model.id, re.IGNORECASE):
            return model  # Exact match
        if re.search(model_name, model.id, re.IGNORECASE):
            models.append(model)
        if re.search(
            model_name.replace(":", "").replace("-", ""),
            model.id.replace(":", "").replace("-", ""),
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
    def price_per_request(self) -> float:
        return float(self.request or 0)

    @property
    def price_per_input_token_without_cached(self) -> float:
        return float(self.prompt or 0)

    @property
    def price_per_input_token_with_cached(self) -> float:
        return (
            float(self.input_cache_read or 0)
            or self.price_per_input_token_without_cached
        )

    @property
    def price_per_output_not_reasoning_token(self) -> float:
        return float(self.completion or 0)

    @property
    def price_per_output_reasoning_token(self) -> float:
        return float(self.internal_reasoning or 0)


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

    model_config = pydantic.ConfigDict(extra=MODEL_CONFIG_EXTRA)


class GetOpenRouterModelsResponse(pydantic.BaseModel):
    """API response wrapper containing list of available models."""

    data: list[OpenRouterModel]

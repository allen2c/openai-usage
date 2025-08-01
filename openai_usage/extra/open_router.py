import functools
import typing

import pydantic
import requests

MODEL_CONFIG_EXTRA: typing.Literal["forbid", "allow", "ignore"] = "ignore"


@functools.cache
def get_models() -> "GetOpenRouterModelsResponse":
    url = "https://openrouter.ai/api/v1/models"
    response = requests.get(url)
    response.raise_for_status()
    return GetOpenRouterModelsResponse.model_validate_json(response.text)


class OpenRouterArchitecture(pydantic.BaseModel):
    modality: str
    input_modalities: list[str]
    output_modalities: list[str]
    tokenizer: str
    instruct_type: str | None

    model_config = pydantic.ConfigDict(extra=MODEL_CONFIG_EXTRA)


class OpenRouterPricing(pydantic.BaseModel):
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


class OpenRouterTopProvider(pydantic.BaseModel):
    context_length: int | None = None
    max_completion_tokens: int | None
    is_moderated: bool

    model_config = pydantic.ConfigDict(extra=MODEL_CONFIG_EXTRA)


class OpenRouterPerRequestLimits(pydantic.BaseModel):
    max_tokens: int
    max_completion_tokens: int
    max_prompt_tokens: int
    max_image_tokens: int
    max_audio_tokens: int
    max_web_search_tokens: int

    model_config = pydantic.ConfigDict(extra=MODEL_CONFIG_EXTRA)


class OpenRouterSupportedParameters(pydantic.BaseModel):
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
    data: list[OpenRouterModel]

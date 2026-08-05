import decimal
import enum
import functools
import logging
import pathlib
import typing

import pydantic

from openai_usage.pricing.cache import DEGRADED_TTL_SECONDS, TTLCache, fetch_text

logger = logging.getLogger(__name__)

MODEL_CONFIG_EXTRA: typing.Literal["forbid", "allow", "ignore"] = "ignore"

MODELS_URL = "https://openrouter.ai/api/v1/models"
CACHE_FILENAME = "open_router_models.json"

DASH_TRANS = str.maketrans({ord(":"): "-", ord("_"): "-", ord("."): "-"})
DROP_TRANS = str.maketrans({ord(":"): None, ord("-"): None, ord("."): None})

_models_cache: "TTLCache[GetOpenRouterModelsResponse]" = TTLCache()


def get_models(realtime_pricing: bool = False) -> "GetOpenRouterModelsResponse":
    """Return the OpenRouter catalog, fetched at runtime and cached on disk.

    The catalog is downloaded once per TTL and reused from ``./.cache/``
    afterwards; ``realtime_pricing`` forces a refresh regardless of TTL. If the
    catalog cannot be fetched or read from cache, the snapshot bundled with the
    package is used instead. The supplemental catalog is always merged in
    because OpenRouter does not list those models at all.
    """
    if not realtime_pricing:
        cached = _models_cache.get()
        if cached is not None:
            return cached

    text = fetch_text(MODELS_URL, CACHE_FILENAME, force_refresh=realtime_pricing)

    models: GetOpenRouterModelsResponse | None = None
    if text is not None:
        try:
            models = GetOpenRouterModelsResponse.model_validate_json(text)
        except pydantic.ValidationError as e:
            logger.warning(f"Could not parse the OpenRouter catalog: {e}")

    # The snapshot is memoized only briefly, so a process that started while
    # the network was down picks up the live catalog once it recovers instead
    # of staying pinned to the shipped data for a whole day.
    degraded_ttl = None
    if models is None:
        logger.info("Using the bundled OpenRouter models snapshot")
        models = _bundled_models()
        degraded_ttl = DEGRADED_TTL_SECONDS

    logger.info(f"There are {len(models.data)} models")
    return _models_cache.set(models.merge(_supplemental_models()), ttl=degraded_ttl)


def normalize_query(model_name: str) -> tuple[str, str, str]:
    """Precompute the three forms a name is matched under.

    Scanning a catalog re-derived these for every entry, so callers that loop
    over thousands of ids compute them once and pass them to
    :func:`matches_normalized` instead.
    """
    return (
        model_name.lower(),
        model_name.translate(DASH_TRANS).lower(),
        model_name.translate(DROP_TRANS).lower(),
    )


def matches_normalized(query: tuple[str, str, str], model_id: str) -> bool:
    """Whether ``model_id`` matches an already-normalized query."""
    verbatim, dashed, dropped = query
    lowered = model_id.lower()
    return (
        verbatim in lowered
        or dashed in lowered.translate(DASH_TRANS)
        or dropped in lowered.translate(DROP_TRANS)
    )


def matches_loosely(model_name: str, model_id: str) -> bool:
    """Whether ``model_id`` matches ``model_name`` under any normalization.

    Tries the name verbatim, then dash-separated, then with every separator
    dropped, so ``claude-haiku-4-5`` still finds ``anthropic/claude-haiku-4.5``.
    """
    return matches_normalized(normalize_query(model_name), model_id)


def get_model(
    model_name: str, *, realtime_pricing: bool = False
) -> typing.Optional["OpenRouterModel"]:
    """Find a model by name with flexible matching.

    Returns exact match first, then partial match.
    Handles multiple matches by choosing shortest ID.
    """
    all_models = get_models(realtime_pricing=realtime_pricing)
    query = normalize_query(model_name)
    lowered_name = query[0]

    # Keyed by id so a model matching under several normalizations is counted
    # once; otherwise a single hit looks like an ambiguous one.
    matches: dict[str, OpenRouterModel] = {}
    for model in all_models.data:
        if model.id.lower().endswith(lowered_name):
            return model
        if matches_normalized(query, model.id):
            matches.setdefault(model.id, model)

    models = list(matches.values())

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
        # Tested against None rather than falsiness: a rate of "0" means the
        # model reads cache for free, which is not the same as "unpriced".
        if self.input_cache_read is None:
            return self.price_per_input_token_without_cached
        return decimal.Decimal(self.input_cache_read)

    @property
    def price_per_output_not_reasoning_token(self) -> decimal.Decimal:
        return decimal.Decimal(self.completion or 0)

    @property
    def price_per_input_cache_write_token(self) -> decimal.Decimal:
        return decimal.Decimal(self.input_cache_write or 0)

    @property
    def price_per_output_reasoning_token(self) -> decimal.Decimal:
        # ``internal_reasoning`` is a surcharge, not a standalone rate, and
        # reads "0" for nearly every model. Unlike the other rates here, a zero
        # therefore means "no surcharge, bill at the completion rate" rather
        # than "free", so falsiness is the correct test.
        return (
            decimal.Decimal(self.internal_reasoning or 0)
            or self.price_per_output_not_reasoning_token
        )

    @property
    def price_per_audio_token(self) -> decimal.Decimal:
        return decimal.Decimal(self.audio or 0)

    @property
    def price_per_image_token(self) -> decimal.Decimal:
        return decimal.Decimal(self.image or 0)

    # OpenRouter publishes one audio and one image rate per model rather than
    # separate input and output rates, so both sides map onto the same value.
    @property
    def price_per_input_audio_token(self) -> decimal.Decimal:
        return self.price_per_audio_token

    @property
    def price_per_output_audio_token(self) -> decimal.Decimal:
        return self.price_per_audio_token

    @property
    def price_per_input_image_token(self) -> decimal.Decimal:
        return self.price_per_image_token

    @property
    def price_per_output_image_token(self) -> decimal.Decimal:
        return self.price_per_image_token

    @property
    def price_per_second(self) -> decimal.Decimal:
        """OpenRouter has no duration-based rate, so this is always zero."""
        return decimal.Decimal(0)


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
        """Combine two catalogs, keeping this one's entry on an id collision."""
        # The already-validated models are reused directly. Round-tripping them
        # back through dump_json and model_validate re-validated the whole
        # upstream catalog just to fold in a handful of supplemental entries.
        by_id: dict[str, OpenRouterModel] = {}
        for model in self.data + other.data:
            by_id.setdefault(model.id, model)
        return GetOpenRouterModelsResponse.model_construct(data=list(by_id.values()))


def _bundled_models() -> "GetOpenRouterModelsResponse":
    """The snapshot shipped with the package, used when the fetch fails."""
    return GetOpenRouterModelsResponse.model_validate_json(
        pathlib.Path(__file__).parent.parent.joinpath("models.json").read_text()
    )


@functools.lru_cache(maxsize=1)
def _supplemental_models() -> "GetOpenRouterModelsResponse":
    """Hand-maintained models OpenRouter's catalog does not carry.

    Cached because the file ships with the package and never changes at
    runtime, yet it was re-read and re-validated on every catalog load.
    """
    return GetOpenRouterModelsResponse.model_validate_json(
        pathlib.Path(__file__)
        .parent.parent.joinpath("models_voyageai.json")
        .read_text()
    )

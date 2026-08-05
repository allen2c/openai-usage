"""Pricing sourced from LiteLLM's ``model_prices_and_context_window.json``.

LiteLLM publishes a flat ``{model_name: metadata}`` map covering roughly ten
times as many models as OpenRouter's catalog, including the embedding, image,
audio and transcription models OpenRouter never proxies. Its rates are also
broken down further: input and output audio are priced separately, reasoning
output has its own rate, and duration-billed models carry a per-second rate.

There is no bundled snapshot for this source; when the catalog cannot be
fetched or read from cache the lookups simply return ``None`` and the caller
falls back to the OpenRouter snapshot.
"""

import decimal
import json
import logging
import typing

import pydantic

from openai_usage.pricing.cache import TTLCache, fetch_text

logger = logging.getLogger(__name__)

MODELS_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)
CACHE_FILENAME = "litellm_model_prices.json"

# Entries that document the schema or configure provider routing rather than
# describing a billable model.
NON_MODEL_KEYS = frozenset({"sample_spec", "fallback_generalizations"})

_models_cache: "TTLCache[dict[str, LiteLLMModel]]" = TTLCache()


def get_models(*, force_refresh: bool = False) -> dict[str, "LiteLLMModel"]:
    """Return the LiteLLM catalog, or an empty map when it is unavailable."""
    if not force_refresh:
        cached = _models_cache.get()
        if cached is not None:
            return cached

    text = fetch_text(MODELS_URL, CACHE_FILENAME, force_refresh=force_refresh)
    if text is None:
        logger.debug("LiteLLM pricing catalog unavailable")
        return _models_cache.set({})

    try:
        models = _parse(text)
    except (json.JSONDecodeError, pydantic.ValidationError) as e:
        logger.warning(f"Could not parse the LiteLLM pricing catalog: {e}")
        return _models_cache.set({})

    logger.info(f"There are {len(models)} LiteLLM models")
    return _models_cache.set(models)


def get_model_exact(
    model_name: str, *, force_refresh: bool = False
) -> typing.Optional["LiteLLMModel"]:
    """Look up a model by its exact catalog key, case-insensitively.

    LiteLLM keys models under their native ids (``gpt-4o``, ``whisper-1``,
    ``claude-sonnet-4-5``), which is what callers normally pass, so this
    resolves most names without any fuzzy matching at all.
    """
    models = get_models(force_refresh=force_refresh)
    model = models.get(model_name)
    if model is not None:
        return model

    lowered = model_name.lower()
    for name, candidate in models.items():
        if name.lower() == lowered:
            return candidate
    return None


def get_model_loose(
    model_name: str, *, force_refresh: bool = False
) -> typing.Optional["LiteLLMModel"]:
    """Look up a model by relaxed matching, mirroring the OpenRouter rules."""
    from openai_usage.pricing.open_router import matches_loosely

    models = get_models(force_refresh=force_refresh)

    matches = [m for name, m in models.items() if matches_loosely(model_name, name)]
    if not matches:
        logger.debug(f"No LiteLLM model found for '{model_name}'")
        return None

    if len(matches) > 1:
        logger.warning(
            f"Multiple LiteLLM models found for '{model_name}': "
            + f"{', '.join(m.id for m in matches[:10])}, "
            + "choose the most not greedy one"
        )
        matches.sort(key=lambda m: len(m.id.split("/")[-1]))

    logger.warning(
        f"Found LiteLLM model '{matches[0].id}' for '{model_name}', "
        + "the model name is not strict"
    )
    return matches[0]


def get_model(
    model_name: str, *, force_refresh: bool = False
) -> typing.Optional["LiteLLMModel"]:
    """Resolve a model name against the LiteLLM catalog, exact match first."""
    exact = get_model_exact(model_name, force_refresh=force_refresh)
    if exact is not None:
        return exact
    return get_model_loose(model_name, force_refresh=force_refresh)


class LiteLLMPricing(pydantic.BaseModel):
    """The subset of LiteLLM's cost fields the estimator can use.

    Rates are per token unless the name says otherwise, and every field is
    optional because coverage varies widely across the catalog.
    """

    input_cost_per_token: float | None = None
    output_cost_per_token: float | None = None
    cache_read_input_token_cost: float | None = None
    cache_creation_input_token_cost: float | None = None
    input_cost_per_audio_token: float | None = None
    output_cost_per_audio_token: float | None = None
    input_cost_per_image_token: float | None = None
    output_cost_per_image_token: float | None = None
    output_cost_per_reasoning_token: float | None = None
    input_cost_per_second: float | None = None
    output_cost_per_second: float | None = None
    input_cost_per_request: float | None = None

    model_config = pydantic.ConfigDict(extra="ignore")

    @property
    def price_per_request(self) -> decimal.Decimal:
        return _decimal(self.input_cost_per_request)

    @property
    def price_per_input_token_without_cached(self) -> decimal.Decimal:
        return _decimal(self.input_cost_per_token)

    @property
    def price_per_input_token_with_cached(self) -> decimal.Decimal:
        return (
            _decimal(self.cache_read_input_token_cost)
            or self.price_per_input_token_without_cached
        )

    @property
    def price_per_input_cache_write_token(self) -> decimal.Decimal:
        return _decimal(self.cache_creation_input_token_cost)

    @property
    def price_per_output_not_reasoning_token(self) -> decimal.Decimal:
        return _decimal(self.output_cost_per_token)

    @property
    def price_per_output_reasoning_token(self) -> decimal.Decimal:
        # Only a few dozen entries price reasoning separately; everywhere else
        # providers bill thinking tokens at the ordinary output rate.
        return (
            _decimal(self.output_cost_per_reasoning_token)
            or self.price_per_output_not_reasoning_token
        )

    @property
    def price_per_input_audio_token(self) -> decimal.Decimal:
        return _decimal(self.input_cost_per_audio_token)

    @property
    def price_per_output_audio_token(self) -> decimal.Decimal:
        return _decimal(self.output_cost_per_audio_token)

    @property
    def price_per_input_image_token(self) -> decimal.Decimal:
        return _decimal(self.input_cost_per_image_token)

    @property
    def price_per_output_image_token(self) -> decimal.Decimal:
        return _decimal(self.output_cost_per_image_token)

    @property
    def price_per_second(self) -> decimal.Decimal:
        return _decimal(self.input_cost_per_second)


class LiteLLMModel(pydantic.BaseModel):
    """A single catalog entry, keyed by the name LiteLLM files it under."""

    id: str
    pricing: LiteLLMPricing
    litellm_provider: str | None = None
    mode: str | None = None

    model_config = pydantic.ConfigDict(extra="ignore")


def _decimal(value: float | None) -> decimal.Decimal:
    """Convert a JSON rate to ``Decimal`` without inheriting binary float noise.

    ``Decimal(2.5e-06)`` expands the float's exact binary value; going through
    ``str`` keeps the rate as written in the catalog.
    """
    if not value:
        return decimal.Decimal(0)
    return decimal.Decimal(str(value))


def _parse(text: str) -> dict[str, LiteLLMModel]:
    """Build the catalog map, skipping the non-model bookkeeping entries."""
    raw = json.loads(text)
    models: dict[str, LiteLLMModel] = {}
    for name, entry in raw.items():
        if name in NON_MODEL_KEYS or not isinstance(entry, dict):
            continue
        models[name] = LiteLLMModel(
            id=name,
            pricing=LiteLLMPricing.model_validate(entry),
            litellm_provider=entry.get("litellm_provider"),
            mode=entry.get("mode"),
        )
    return models

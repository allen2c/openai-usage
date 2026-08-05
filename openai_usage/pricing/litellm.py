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

from openai_usage.pricing.cache import DEGRADED_TTL_SECONDS, TTLCache, fetch_text

logger = logging.getLogger(__name__)

MODELS_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)
CACHE_FILENAME = "litellm_model_prices.json"

# Entries that document the schema or configure provider routing rather than
# describing a billable model.
NON_MODEL_KEYS = frozenset({"sample_spec", "fallback_generalizations"})

# The catalog is held as raw dicts and validated one entry at a time. Building
# all ~3,000 pydantic models up front cost ~30 ms and ~4.5 MB to answer a
# lookup that reads exactly one of them.
_entries_cache: "TTLCache[dict[str, dict]]" = TTLCache()
_built: dict[str, "LiteLLMModel"] = {}


def get_models(*, force_refresh: bool = False) -> dict[str, "LiteLLMModel"]:
    """Return the whole LiteLLM catalog, validated.

    Every entry is built here, so prefer :func:`get_model` when you only need
    one. Returns an empty map when the catalog is unavailable.
    """
    return {name: _build(name) for name in get_entries(force_refresh=force_refresh)}


def get_entries(*, force_refresh: bool = False) -> dict[str, dict]:
    """Return the raw catalog entries, keyed by model name."""
    if not force_refresh:
        cached = _entries_cache.get()
        if cached is not None:
            return cached

    text = fetch_text(MODELS_URL, CACHE_FILENAME, force_refresh=force_refresh)
    if text is None:
        logger.debug("LiteLLM pricing catalog unavailable")
        return _set_entries({}, ttl=DEGRADED_TTL_SECONDS)

    try:
        entries = _parse(text)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Could not parse the LiteLLM pricing catalog: {e}")
        return _set_entries({}, ttl=DEGRADED_TTL_SECONDS)

    logger.info(f"There are {len(entries)} LiteLLM models")
    return _set_entries(entries)


def get_model_exact(
    model_name: str, *, force_refresh: bool = False
) -> typing.Optional["LiteLLMModel"]:
    """Look up a model by its exact catalog key, case-insensitively.

    LiteLLM keys models under their native ids (``gpt-4o``, ``whisper-1``,
    ``claude-sonnet-4-5``), which is what callers normally pass, so this
    resolves most names without any fuzzy matching at all.
    """
    entries = get_entries(force_refresh=force_refresh)
    if model_name in entries:
        return _build(model_name)

    lowered = model_name.lower()
    for name in entries:
        if name.lower() == lowered:
            return _build(name)
    return None


def get_model_loose(
    model_name: str, *, force_refresh: bool = False
) -> typing.Optional["LiteLLMModel"]:
    """Look up a model by relaxed matching, mirroring the OpenRouter rules."""
    from openai_usage.pricing.open_router import matches_normalized, normalize_query

    entries = get_entries(force_refresh=force_refresh)
    query = normalize_query(model_name)

    names = [name for name in entries if matches_normalized(query, name)]
    if not names:
        logger.debug(f"No LiteLLM model found for '{model_name}'")
        return None

    if len(names) > 1:
        logger.warning(
            f"Multiple LiteLLM models found for '{model_name}': "
            + f"{', '.join(names[:10])}, "
            + "choose the most not greedy one"
        )
        names.sort(key=lambda name: len(name.split("/")[-1]))

    logger.warning(
        f"Found LiteLLM model '{names[0]}' for '{model_name}', "
        + "the model name is not strict"
    )
    return _build(names[0])


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
        # Tested against None rather than falsiness: a rate of 0 means the
        # model reads cache for free, which is not the same as "unpriced".
        if self.cache_read_input_token_cost is None:
            return self.price_per_input_token_without_cached
        return _decimal(self.cache_read_input_token_cost)

    @property
    def price_per_input_cache_write_token(self) -> decimal.Decimal:
        return _decimal(self.cache_creation_input_token_cost)

    @property
    def price_per_output_not_reasoning_token(self) -> decimal.Decimal:
        return _decimal(self.output_cost_per_token)

    @property
    def price_per_output_reasoning_token(self) -> decimal.Decimal:
        # Only a few dozen entries price reasoning separately; everywhere else
        # providers bill thinking tokens at the ordinary output rate. An
        # explicit 0 means free, so absence is what triggers the fallback.
        if self.output_cost_per_reasoning_token is None:
            return self.price_per_output_not_reasoning_token
        return _decimal(self.output_cost_per_reasoning_token)

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


def _set_entries(entries: dict[str, dict], *, ttl: float | None = None) -> dict:
    """Store the raw catalog, discarding models built from the previous one."""
    _built.clear()
    return _entries_cache.set(entries, ttl=ttl)


def _build(name: str) -> "LiteLLMModel":
    """Validate one catalog entry, reusing it on later lookups."""
    model = _built.get(name)
    if model is None:
        entry = (_entries_cache.get() or {}).get(name, {})
        model = LiteLLMModel(
            id=name,
            pricing=LiteLLMPricing.model_validate(entry),
            litellm_provider=entry.get("litellm_provider"),
            mode=entry.get("mode"),
        )
        _built[name] = model
    return model


def _decimal(value: float | None) -> decimal.Decimal:
    """Convert a JSON rate to ``Decimal`` without inheriting binary float noise.

    ``Decimal(2.5e-06)`` expands the float's exact binary value; going through
    ``str`` keeps the rate as written in the catalog.
    """
    if not value:
        return decimal.Decimal(0)
    return decimal.Decimal(str(value))


def _parse(text: str) -> dict[str, dict]:
    """Extract the model entries, skipping the non-model bookkeeping ones.

    Raises ``TypeError`` when the payload parses as JSON but is not the
    expected object, which an upstream format change or an error page served
    in place of the file would produce.
    """
    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a JSON object, got {type(raw).__name__}")

    return {
        name: entry
        for name, entry in raw.items()
        if name not in NON_MODEL_KEYS and isinstance(entry, dict)
    }

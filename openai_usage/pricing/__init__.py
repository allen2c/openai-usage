"""Model pricing, resolved across the LiteLLM and OpenRouter catalogs.

Both catalogs are fetched at runtime and cached under ``./.cache/openai-usage/``
for a day; see :mod:`openai_usage.pricing.cache` for the environment variables
that control that. Names resolve against LiteLLM's exact keys first, then
OpenRouter's relaxed matching, then LiteLLM's relaxed matching:

* LiteLLM keys models under their native ids, so an exact hit is both cheap and
  unambiguous, and its rates are the more detailed of the two.
* OpenRouter's relaxed matching handles the ``vendor/model`` names and aliases
  its catalog is organized around, plus the supplemental models bundled here.
* LiteLLM's relaxed matching is the last resort, covering the embedding, image,
  audio and transcription models OpenRouter does not carry.
"""

import decimal
import logging
import typing

from openai_usage.pricing import litellm, open_router
from openai_usage.pricing.cache import cache_dir, is_offline, ttl_seconds
from openai_usage.pricing.litellm import LiteLLMModel, LiteLLMPricing
from openai_usage.pricing.open_router import (
    GetOpenRouterModelsResponse,
    OpenRouterModel,
    OpenRouterPricing,
)

logger = logging.getLogger(__name__)

__all__ = [
    "GetOpenRouterModelsResponse",
    "LiteLLMModel",
    "LiteLLMPricing",
    "ModelPricing",
    "OpenRouterModel",
    "OpenRouterPricing",
    "PricedModel",
    "cache_dir",
    "get_model",
    "is_offline",
    "litellm",
    "open_router",
    "ttl_seconds",
]


def get_model(
    model_name: str, *, realtime_pricing: bool = False
) -> typing.Optional["PricedModel"]:
    """Resolve a model name against both catalogs, or ``None`` if unknown.

    ``realtime_pricing`` bypasses the cache TTL and re-downloads both catalogs.
    """
    exact = litellm.get_model_exact(model_name, force_refresh=realtime_pricing)
    if exact is not None:
        return exact

    from_open_router = open_router.get_model(
        model_name, realtime_pricing=realtime_pricing
    )
    if from_open_router is not None:
        return from_open_router

    return litellm.get_model_loose(model_name, force_refresh=realtime_pricing)


@typing.runtime_checkable
class ModelPricing(typing.Protocol):
    """The rates the cost estimator needs, however a catalog spells them.

    Every rate is per token except ``price_per_request`` and
    ``price_per_second``, and every one is zero when the catalog has no figure
    for it.
    """

    @property
    def price_per_request(self) -> decimal.Decimal: ...

    @property
    def price_per_input_token_without_cached(self) -> decimal.Decimal: ...

    @property
    def price_per_input_token_with_cached(self) -> decimal.Decimal: ...

    @property
    def price_per_input_cache_write_token(self) -> decimal.Decimal: ...

    @property
    def price_per_output_not_reasoning_token(self) -> decimal.Decimal: ...

    @property
    def price_per_output_reasoning_token(self) -> decimal.Decimal: ...

    @property
    def price_per_input_audio_token(self) -> decimal.Decimal: ...

    @property
    def price_per_output_audio_token(self) -> decimal.Decimal: ...

    @property
    def price_per_input_image_token(self) -> decimal.Decimal: ...

    @property
    def price_per_output_image_token(self) -> decimal.Decimal: ...

    @property
    def price_per_second(self) -> decimal.Decimal: ...


@typing.runtime_checkable
class PricedModel(typing.Protocol):
    """A catalog entry the estimator can price against."""

    @property
    def id(self) -> str: ...

    @property
    def pricing(self) -> ModelPricing: ...

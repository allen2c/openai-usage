"""Deprecated import path, kept so existing code keeps working.

This module moved to :mod:`openai_usage.pricing.open_router` when the second
pricing source was added. Import from there instead.
"""

from openai_usage.pricing.open_router import (
    CACHE_FILENAME,
    DASH_TRANS,
    DROP_TRANS,
    MODEL_CONFIG_EXTRA,
    MODELS_URL,
    GetOpenRouterModelsResponse,
    OpenRouterArchitecture,
    OpenRouterArtificialAnalysisBenchmark,
    OpenRouterBenchmarks,
    OpenRouterDefaultParameters,
    OpenRouterDesignArenaBenchmark,
    OpenRouterLinks,
    OpenRouterModel,
    OpenRouterPerRequestLimits,
    OpenRouterPricing,
    OpenRouterReasoning,
    OpenRouterReasoningEffort,
    OpenRouterSupportedParameters,
    OpenRouterTopProvider,
    ReasoningEffort,
    get_model,
    get_models,
    matches_loosely,
)

__all__ = [
    "CACHE_FILENAME",
    "DASH_TRANS",
    "DROP_TRANS",
    "MODELS_URL",
    "MODEL_CONFIG_EXTRA",
    "GetOpenRouterModelsResponse",
    "OpenRouterArchitecture",
    "OpenRouterArtificialAnalysisBenchmark",
    "OpenRouterBenchmarks",
    "OpenRouterDefaultParameters",
    "OpenRouterDesignArenaBenchmark",
    "OpenRouterLinks",
    "OpenRouterModel",
    "OpenRouterPerRequestLimits",
    "OpenRouterPricing",
    "OpenRouterReasoning",
    "OpenRouterReasoningEffort",
    "OpenRouterSupportedParameters",
    "OpenRouterTopProvider",
    "ReasoningEffort",
    "get_model",
    "get_models",
    "matches_loosely",
]

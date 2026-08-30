"""Codex OAuth route detection and effective context-window caps.

Isolated from ``engine.py`` (WS5 seam) so the Codex-specific routing policy —
which model slugs are on the ChatGPT Codex OAuth route and what effective
context window that route enforces — lives in one cohesive place. These are
pure helpers with no engine state; ``engine.py`` imports them and keeps its own
policy constants (for example the gpt-5.5 compaction threshold).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Only these exact normalized bare slugs have a proven 900k Codex OAuth route.
# Keep them separate from the family fallbacks below so suffixes and synthetic
# aliases cannot inherit the larger window.
_CODEX_OAUTH_EXACT_CONTEXT_CAPS: dict[str, int] = {
    "gpt-5.6-terra-900k": 900_000,
    "gpt-5.6-sol-900k": 900_000,
    "gpt-5.6-luna-900k": 900_000,
}

# ChatGPT Codex OAuth exposes provider-enforced context windows that can be
# materially lower than the same model slug on direct OpenAI/OpenRouter routes.
# Hermes Agent resolves these from chatgpt.com/backend-api/codex/models, with
# this table as a compatibility fallback. LCM receives a host-provided
# context_length, but that value can be an explicit override or stale metadata;
# resolve the active provider route here as well so compaction follows the same
# authenticated window Hermes uses for the request.
_CODEX_OAUTH_CONTEXT_CAPS: dict[str, int] = {
    "gpt-5.1-codex-max": 272_000,
    "gpt-5.1-codex-mini": 272_000,
    "gpt-5.3-codex-spark": 128_000,
    "gpt-5.3-codex": 272_000,
    "gpt-5.2-codex": 272_000,
    "gpt-5.4-mini": 272_000,
    "gpt-5.5": 272_000,
    "gpt-5.4": 272_000,
    "gpt-5.2": 272_000,
    "gpt-5.6": 372_000,
    "gpt-5": 272_000,
}


def _bare_model_slug(model: str | None) -> str:
    return (model or "").strip().lower().rsplit("/", 1)[-1]


def _is_openai_codex_route(provider: str | None) -> bool:
    return (provider or "").strip().lower() == "openai-codex"


def _codex_oauth_context_cap(
    model: str | None,
    provider: str | None,
    *,
    api_key: str = "",
) -> int | None:
    """Return the active Hermes-resolved Codex OAuth context window.

    Exact proven 900k route names are authoritative. Other Codex routes use
    Hermes Agent's provider-aware resolver, with the local table retained for
    compatibility when that resolver is unavailable.
    """
    if not _is_openai_codex_route(provider):
        return None
    bare_model = _bare_model_slug(model)
    if not bare_model:
        return None

    exact_cap = _CODEX_OAUTH_EXACT_CONTEXT_CAPS.get(bare_model)
    if exact_cap is not None:
        return exact_cap

    try:
        from agent.model_metadata import get_model_context_length

        resolved = get_model_context_length(
            bare_model,
            api_key=api_key or "",
            provider=provider or "",
        )
        if isinstance(resolved, int) and resolved > 0:
            return resolved
    except Exception:
        logger.debug(
            "Hermes Codex context resolver unavailable for %s; using LCM fallback",
            bare_model,
            exc_info=True,
        )

    for slug, cap in sorted(
        _CODEX_OAUTH_CONTEXT_CAPS.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if slug in bare_model:
            return cap
    return None


def _is_codex_gpt55_route(model: str | None, provider: str | None) -> bool:
    """Return True for gpt-5.5 on ChatGPT Codex OAuth, mirroring Hermes core."""
    if not _is_openai_codex_route(provider):
        return False
    bare_model = _bare_model_slug(model)
    return (
        bare_model == "gpt-5.5"
        or bare_model.startswith("gpt-5.5-")
        or bare_model.startswith("gpt-5.5.")
    )

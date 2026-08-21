"""Regression tests for Codex OAuth route-cap matching."""

import sys
from types import ModuleType

import pytest

import hermes_lcm.codex_routing as codex_routing
import hermes_lcm.engine as lcm_engine

from hermes_lcm.config import LCMConfig
from hermes_lcm.codex_routing import _codex_oauth_context_cap
from hermes_lcm.engine import LCMEngine


EXACT_CODEX_900K_MODELS = (
    "gpt-5.6-terra-900k",
    "gpt-5.6-sol-900k",
    "gpt-5.6-luna-900k",
)


@pytest.mark.parametrize("model", EXACT_CODEX_900K_MODELS)
def test_exact_codex_900k_routes_receive_proven_cap(model):
    assert _codex_oauth_context_cap(model, "openai-codex") == 900_000


def test_codex_900k_route_matches_normalized_bare_slug():
    assert (
        _codex_oauth_context_cap(
            "  openai/GPT-5.6-SOL-900K  ",
            "  OPENAI-CODEX  ",
        )
        == 900_000
    )


@pytest.mark.parametrize("provider", [None, "openai", "openai-codex-proxy"])
def test_codex_900k_routes_require_exact_provider(provider):
    assert _codex_oauth_context_cap("gpt-5.6-sol-900k", provider) is None


@pytest.mark.parametrize(
    ("model", "expected_cap"),
    [
        ("gpt-5.6", 372_000),
        ("gpt-5.6-preview", 372_000),
        ("gpt-5.5", 272_000),
        ("gpt-5.4", 272_000),
        ("gpt-5.3-codex-spark", 128_000),
    ],
)
def test_existing_codex_route_caps_are_preserved(model, expected_cap):
    assert _codex_oauth_context_cap(model, "openai-codex") == expected_cap


@pytest.mark.parametrize(
    ("model", "expected_cap"),
    [
        ("gpt-5.5-900k", 272_000),
        ("gpt-5.6-terra-900k-pro", 372_000),
        ("fake-gpt-5.6-sol-900k", 372_000),
        ("gpt-5.6-luna-900k.fake", 372_000),
        ("gpt-5.6-900k", 372_000),
        ("gpt-5.7-terra-900k", 272_000),
    ],
)
def test_900k_suffix_and_malformed_aliases_do_not_gain_900k_cap(
    model,
    expected_cap,
):
    assert _codex_oauth_context_cap(model, "openai-codex") == expected_cap


def _install_model_metadata_resolver(monkeypatch, resolver):
    import agent

    metadata_module = ModuleType("agent.model_metadata")
    setattr(metadata_module, "get_model_context_length", resolver)
    monkeypatch.setitem(sys.modules, "agent.model_metadata", metadata_module)
    monkeypatch.setattr(agent, "model_metadata", metadata_module, raising=False)


def test_codex_oauth_context_cap_uses_hermes_provider_resolver(monkeypatch):
    calls = {}

    def resolve_context_length(model, *, base_url="", api_key="", provider="", **kwargs):
        calls.update(
            {
                "model": model,
                "base_url": base_url,
                "api_key": api_key,
                "provider": provider,
                "kwargs": kwargs,
            }
        )
        return 900_000

    _install_model_metadata_resolver(monkeypatch, resolve_context_length)

    assert codex_routing._codex_oauth_context_cap(
        "gpt-5.6-luna",
        "openai-codex",
        api_key="oauth-token",
    ) == 900_000
    assert calls == {
        "model": "gpt-5.6-luna",
        "base_url": "",
        "api_key": "oauth-token",
        "provider": "openai-codex",
        "kwargs": {},
    }


def test_codex_oauth_context_cap_uses_lower_live_value_over_static_fallback(monkeypatch):
    _install_model_metadata_resolver(monkeypatch, lambda *args, **kwargs: 272_000)

    assert codex_routing._codex_oauth_context_cap(
        "gpt-5.6-luna",
        "openai-codex",
    ) == 272_000


def test_codex_oauth_context_cap_keeps_existing_fallback_on_resolver_failure(monkeypatch):
    def fail_resolver(*args, **kwargs):
        raise RuntimeError("provider metadata unavailable")

    _install_model_metadata_resolver(monkeypatch, fail_resolver)

    assert codex_routing._codex_oauth_context_cap(
        "gpt-5.6-luna",
        "openai-codex",
    ) == 372_000


def test_engine_uses_resolved_codex_cap_and_forwards_route_credentials(tmp_path, monkeypatch):
    calls = {}

    def resolve_cap(model, provider, *, api_key=""):
        calls.update({"model": model, "provider": provider, "api_key": api_key})
        return 900_000

    monkeypatch.setattr(lcm_engine, "_codex_oauth_context_cap", resolve_cap)
    engine = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "codex-routing.db")),
    )
    try:
        engine.update_model(
            model="gpt-5.6-luna",
            context_length=1_000_000,
            api_key="oauth-token",
            provider="openai-codex",
        )

        assert engine.context_length == 900_000
        assert engine.effective_context_length_cap == 900_000
        assert engine.effective_context_length_reason == "codex_oauth_context_cap"
        assert calls == {
            "model": "gpt-5.6-luna",
            "provider": "openai-codex",
            "api_key": "oauth-token",
        }
    finally:
        engine.shutdown()

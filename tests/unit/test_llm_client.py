"""Unit tests for the provider-agnostic LLM client dispatcher.

Mirrors the design of ml-intern's `agent/core/llm_params.py`: each
provider prefix produces a specific shape of litellm kwargs, and only
the `azure/` branch ever attaches an Entra token provider.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataset_scout import LLMError, ScoutContext
from dataset_scout.llm_client import (
    build_completion_kwargs,
    effective_model_id,
    resolve_llm_params,
)

pytestmark = pytest.mark.unit


_MESSAGES: list[dict[str, str]] = [{"role": "user", "content": "hello"}]


def _aoai_ctx() -> ScoutContext:
    return ScoutContext(
        aoai_endpoint="https://example.openai.azure.com",
        aoai_deployment="gpt-4o-mini",
    )


# ─── effective_model_id ─────────────────────────────────────────────


def test_effective_model_id_returns_none_when_unconfigured() -> None:
    assert effective_model_id(ScoutContext()) is None


def test_effective_model_id_synthesizes_azure_from_aoai_back_compat() -> None:
    """Existing users with only AZURE_OPENAI_* env vars set must
    keep working: the legacy fields synthesize an `azure/<deployment>`
    model id automatically."""
    assert effective_model_id(_aoai_ctx()) == "azure/gpt-4o-mini"


def test_effective_model_id_prefers_ctx_model_over_aoai_synthesis() -> None:
    ctx = ScoutContext(
        model="github_copilot/gpt-5-mini",
        aoai_endpoint="https://example.openai.azure.com",
        aoai_deployment="gpt-4o-mini",
    )
    assert effective_model_id(ctx) == "github_copilot/gpt-5-mini"


def test_effective_model_id_override_wins_over_everything() -> None:
    """Per-call override (CLI flag) beats both ctx.model and AOAI."""
    ctx = ScoutContext(
        model="github_copilot/gpt-5-mini",
        aoai_endpoint="https://example.openai.azure.com",
        aoai_deployment="gpt-4o-mini",
    )
    assert (
        effective_model_id(ctx, "anthropic/claude-sonnet-4-5")
        == "anthropic/claude-sonnet-4-5"
    )


def test_effective_model_id_empty_override_falls_through() -> None:
    """An empty/None override does NOT mask the configured model."""
    ctx = ScoutContext(model="github_copilot/gpt-5-mini")
    assert effective_model_id(ctx, None) == "github_copilot/gpt-5-mini"
    assert effective_model_id(ctx, "") == "github_copilot/gpt-5-mini"


# ─── resolve_llm_params: azure branch ───────────────────────────────


def test_resolve_azure_attaches_token_provider_and_api_base() -> None:
    """azure/<deployment> must build the full Entra-shaped call: model,
    api_base, api_version, azure_ad_token_provider — and never api_key.
    Provided token provider is honored (no DefaultAzureCredential touch)."""
    sentinel = lambda: "fake-bearer"  # noqa: E731

    kwargs = resolve_llm_params(
        "azure/gpt-4o-mini",
        ctx=_aoai_ctx(),
        messages=_MESSAGES,
        token_provider=sentinel,
    )
    assert kwargs["model"] == "azure/gpt-4o-mini"
    assert kwargs["api_base"] == "https://example.openai.azure.com"
    assert kwargs["api_version"] == "2024-10-21"
    assert kwargs["azure_ad_token_provider"] is sentinel
    assert kwargs["messages"] == _MESSAGES
    assert kwargs["timeout"] == 30.0
    assert "api_key" not in kwargs
    assert "response_format" not in kwargs


def test_resolve_azure_calls_make_token_provider_when_none_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the caller doesn't pass token_provider, resolve_llm_params
    invokes make_token_provider() — the Azure-only lazy entry point."""
    called: list[bool] = []

    def fake_provider() -> Any:
        called.append(True)
        return "FAKE"

    monkeypatch.setattr("dataset_scout.llm_client.make_token_provider", fake_provider)
    kwargs = resolve_llm_params("azure/gpt-4o-mini", ctx=_aoai_ctx(), messages=_MESSAGES)
    assert called == [True]
    assert kwargs["azure_ad_token_provider"] == "FAKE"


def test_resolve_azure_raises_when_endpoint_missing() -> None:
    """An `azure/...` model id with no ctx.aoai_endpoint is a config
    error — user explicitly opted into the Azure path but didn't
    finish the config."""
    with pytest.raises(LLMError, match="aoai_endpoint is unset"):
        resolve_llm_params("azure/gpt-4o-mini", ctx=ScoutContext(), messages=_MESSAGES)


# ─── resolve_llm_params: github_copilot / github / openai / anthropic ─


def test_resolve_github_copilot_returns_bare_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The github_copilot branch must NOT call make_token_provider()
    and must NOT attach api_base / api_version / api_key. litellm
    handles the OAuth device-code flow itself."""

    def boom() -> Any:
        raise AssertionError("make_token_provider must NOT be called for github_copilot/")

    monkeypatch.setattr("dataset_scout.llm_client.make_token_provider", boom)
    kwargs = resolve_llm_params(
        "github_copilot/gpt-5-mini",
        ctx=ScoutContext(),  # no AOAI config at all
        messages=_MESSAGES,
    )
    assert kwargs["model"] == "github_copilot/gpt-5-mini"
    assert kwargs["messages"] == _MESSAGES
    assert kwargs["timeout"] == 30.0
    assert "api_base" not in kwargs
    assert "api_version" not in kwargs
    assert "api_key" not in kwargs
    assert "azure_ad_token_provider" not in kwargs


def test_resolve_github_returns_bare_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """The github/ (free GitHub Models tier) branch is the same shape:
    bare model id, litellm reads GITHUB_TOKEN itself."""

    def boom() -> Any:
        raise AssertionError("make_token_provider must NOT be called for github/")

    monkeypatch.setattr("dataset_scout.llm_client.make_token_provider", boom)
    kwargs = resolve_llm_params(
        "github/gpt-4o-mini",
        ctx=ScoutContext(),
        messages=_MESSAGES,
    )
    assert kwargs["model"] == "github/gpt-4o-mini"
    assert "azure_ad_token_provider" not in kwargs


def test_resolve_openai_returns_bare_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> Any:
        raise AssertionError("make_token_provider must NOT be called for openai/")

    monkeypatch.setattr("dataset_scout.llm_client.make_token_provider", boom)
    kwargs = resolve_llm_params(
        "openai/gpt-4o", ctx=ScoutContext(), messages=_MESSAGES
    )
    assert kwargs["model"] == "openai/gpt-4o"
    assert "azure_ad_token_provider" not in kwargs


def test_resolve_anthropic_returns_bare_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> Any:
        raise AssertionError("make_token_provider must NOT be called for anthropic/")

    monkeypatch.setattr("dataset_scout.llm_client.make_token_provider", boom)
    kwargs = resolve_llm_params(
        "anthropic/claude-sonnet-4-5", ctx=ScoutContext(), messages=_MESSAGES
    )
    assert kwargs["model"] == "anthropic/claude-sonnet-4-5"
    assert "azure_ad_token_provider" not in kwargs


# ─── response_format ────────────────────────────────────────────────


class _Marker:
    """Stand-in Pydantic class — only its truthiness matters here."""


def test_response_format_translates_to_json_object_mode() -> None:
    kwargs = resolve_llm_params(
        "github_copilot/gpt-5-mini",
        ctx=ScoutContext(),
        messages=_MESSAGES,
        response_format=_Marker,
    )
    # We never forward the Pydantic class itself — Azure's strict
    # schema validator rejects some Pydantic-generated schemas, and
    # passing json_object mode is universal across providers.
    assert kwargs["response_format"] == {"type": "json_object"}


def test_response_format_omitted_when_none() -> None:
    kwargs = resolve_llm_params(
        "github_copilot/gpt-5-mini",
        ctx=ScoutContext(),
        messages=_MESSAGES,
        response_format=None,
    )
    assert "response_format" not in kwargs


# ─── build_completion_kwargs (the wrapper) ──────────────────────────


def test_build_completion_kwargs_resolves_via_ctx_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The convenience wrapper should pull ctx.model when no override."""

    def boom() -> Any:
        raise AssertionError("must NOT call Azure token provider")

    monkeypatch.setattr("dataset_scout.llm_client.make_token_provider", boom)
    ctx = ScoutContext(model="github_copilot/gpt-5-mini")
    kwargs = build_completion_kwargs(ctx, messages=_MESSAGES)
    assert kwargs["model"] == "github_copilot/gpt-5-mini"


def test_build_completion_kwargs_falls_back_to_synthesized_azure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Back-compat: an AOAI-configured ctx with no .model still routes
    to azure/<deployment> with the Entra token provider attached."""
    monkeypatch.setattr(
        "dataset_scout.llm_client.make_token_provider", lambda: "FAKE"
    )
    kwargs = build_completion_kwargs(_aoai_ctx(), messages=_MESSAGES)
    assert kwargs["model"] == "azure/gpt-4o-mini"
    assert kwargs["azure_ad_token_provider"] == "FAKE"


def test_build_completion_kwargs_raises_when_nothing_configured() -> None:
    with pytest.raises(LLMError, match="No LLM provider configured"):
        build_completion_kwargs(ScoutContext(), messages=_MESSAGES)


def test_build_completion_kwargs_per_call_model_overrides_ctx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dataset_scout.llm_client.make_token_provider", lambda: "AZURE_PROVIDER"
    )
    # ctx is set up for Azure...
    kwargs = build_completion_kwargs(
        _aoai_ctx(),
        messages=_MESSAGES,
        model="github_copilot/gpt-5-mini",  # ...but this overrides.
    )
    assert kwargs["model"] == "github_copilot/gpt-5-mini"
    assert "azure_ad_token_provider" not in kwargs


def test_build_completion_kwargs_back_compat_kwargs_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard: the synthesized Azure path must produce
    exactly the same kwargs shape as the pre-refactor code did, so
    existing AOAI-only users don't see any behavioral drift."""
    monkeypatch.setattr(
        "dataset_scout.llm_client.make_token_provider", lambda: "FAKE_PROVIDER"
    )
    kwargs = build_completion_kwargs(
        _aoai_ctx(),
        messages=_MESSAGES,
        response_format=_Marker,
        timeout_s=60.0,
    )
    assert kwargs == {
        "model": "azure/gpt-4o-mini",
        "messages": _MESSAGES,
        "timeout": 60.0,
        "response_format": {"type": "json_object"},
        "api_base": "https://example.openai.azure.com",
        "api_version": "2024-10-21",
        "azure_ad_token_provider": "FAKE_PROVIDER",
    }

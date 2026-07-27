"""Unit tests for the LLM decomposition module (provider-agnostic via llm_client)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from dataset_scout import (
    DecompositionDirection,
    Intent,
    LLMError,
    ScoutContext,
)
from dataset_scout.decompose import (
    decompose_intent,
    llm_available,
    render_decompose_prompt,
)

pytestmark = pytest.mark.unit


# ─── fakes for litellm + azure-identity ─────────────────────────────


@dataclass
class _Msg:
    content: str


@dataclass
class _Choice:
    message: _Msg


@dataclass
class _Resp:
    choices: list[_Choice]


def _resp(payload: Any) -> _Resp:
    return _Resp(choices=[_Choice(message=_Msg(content=json.dumps(payload)))])


def _good_directions(n: int = 3) -> list[dict[str, Any]]:
    return [
        {
            "name": f"dir_{i}",
            "rationale": f"because reason {i}",
            "keywords": [f"k{i}a", f"k{i}b"],
            "threat_families": ["prompt_injection"],
            "expected_finds": f"useful data {i}",
        }
        for i in range(n)
    ]


@pytest.fixture
def fake_token_provider(monkeypatch: pytest.MonkeyPatch):
    """Replace `llm_client.make_token_provider` with a stub so tests
    don't touch real Azure credentials.

    decompose now delegates to `llm_client.build_completion_kwargs`,
    which calls `make_token_provider()` only when the resolved model
    starts with `azure/`. Patching the shared helper covers both the
    legacy AOAI path and any future Azure-only call site.
    """

    def _stub() -> object:
        return lambda: "fake-bearer-token"

    monkeypatch.setattr("dataset_scout.llm_client.make_token_provider", _stub)


def _ctx(
    *,
    endpoint: str = "https://example.openai.azure.com",
    deployment: str = "gpt-4o-mini",
    api_version: str = "2024-10-21",
) -> ScoutContext:
    return ScoutContext(
        aoai_endpoint=endpoint,
        aoai_deployment=deployment,
        aoai_api_version=api_version,
    )


# ─── render_decompose_prompt ────────────────────────────────────────


def test_render_returns_nonempty_with_key_phrases() -> None:
    p = render_decompose_prompt(Intent(raw_brief="x"))
    assert p
    assert "3-7" in p
    assert "JSON matching" in p
    assert "lexical" in p.lower()


def test_render_renders_none_for_empty_fields() -> None:
    p = render_decompose_prompt(Intent(raw_brief=""))
    assert "Brief: (none)" in p
    assert "Detection target: (none)" in p
    assert "Threat families: (none)" in p
    assert "Deployment context: (none)" in p


def test_render_includes_provided_fields() -> None:
    intent = Intent(
        raw_brief="find prompt injection data",
        detection_target="prompt injection",
        threat_families=["prompt_injection", "jailbreak"],
        deployment_context="rag_pipeline",
    )
    p = render_decompose_prompt(intent)
    assert "Brief: find prompt injection data" in p
    assert "Detection target: prompt injection" in p
    assert "Threat families: prompt_injection, jailbreak" in p
    assert "Deployment context: rag_pipeline" in p


_SNAPSHOT_PATH = Path(__file__).parent / "fixtures" / "decompose_prompt.txt"


def test_render_snapshot_stable() -> None:
    """Prompt drift surfaces as a PR diff. First run writes the snapshot."""
    intent = Intent(
        raw_brief="find prompt injection corpora for our RAG service",
        detection_target="prompt injection",
        threat_families=["prompt_injection", "indirect_injection"],
        deployment_context="rag_pipeline",
    )
    rendered = render_decompose_prompt(intent)

    if not _SNAPSHOT_PATH.exists():
        _SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SNAPSHOT_PATH.write_text(rendered, encoding="utf-8")
    expected = _SNAPSHOT_PATH.read_text(encoding="utf-8")
    assert rendered == expected, (
        "Decomposition prompt drifted from snapshot. If intentional, delete "
        f"{_SNAPSHOT_PATH} and re-run the test."
    )


# ─── llm_available ──────────────────────────────────────────────────


def test_llm_available_false_when_unconfigured() -> None:
    ctx = ScoutContext()
    assert llm_available(ctx) is False


def test_llm_available_false_when_only_endpoint_set() -> None:
    ctx = ScoutContext(aoai_endpoint="https://example.openai.azure.com")
    assert llm_available(ctx) is False


def test_llm_available_false_when_only_deployment_set() -> None:
    ctx = ScoutContext(aoai_deployment="gpt-4o-mini")
    assert llm_available(ctx) is False


def test_llm_available_true_when_both_set() -> None:
    assert llm_available(_ctx()) is True


def test_llm_available_true_when_universal_model_set() -> None:
    """Setting ctx.model alone (no AOAI config) should be enough — that
    is the whole point of the github_copilot / openai / anthropic
    routes: no Azure config required."""
    ctx = ScoutContext(model="github_copilot/gpt-5-mini")
    assert llm_available(ctx) is True


# ─── decompose_intent — call wiring ────────────────────────────────


def test_decompose_routes_via_azure_with_token_provider(
    monkeypatch: pytest.MonkeyPatch, fake_token_provider: None
) -> None:
    captured: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> _Resp:
        captured.update(kwargs)
        return _resp({"directions": _good_directions(3)})

    monkeypatch.setattr("litellm.completion", fake_completion)
    result = decompose_intent(Intent(raw_brief="prompt injection"), ctx=_ctx())

    assert len(result) == 3
    assert all(isinstance(d, DecompositionDirection) for d in result)

    # Azure routing: model prefixed with `azure/` and deployment name.
    assert captured["model"] == "azure/gpt-4o-mini"
    assert captured["api_base"] == "https://example.openai.azure.com"
    assert captured["api_version"] == "2024-10-21"
    # Entra: a token provider callable, NOT an api_key.
    assert callable(captured["azure_ad_token_provider"])
    assert "api_key" not in captured

    assert captured["timeout"] == 60.0
    assert captured["response_format"] == {"type": "json_object"}
    msgs = captured["messages"]
    assert msgs[0]["role"] == "user"
    assert "3-7" in msgs[0]["content"]


def test_decompose_routes_via_github_copilot_when_model_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ctx.model='github_copilot/...' bypasses the Azure path entirely:
    no api_base, no token provider, no api_key. litellm handles auth
    via OAuth device-code flow internally."""
    captured: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> _Resp:
        captured.update(kwargs)
        return _resp({"directions": _good_directions(3)})

    monkeypatch.setattr("litellm.completion", fake_completion)
    # Crucially, NO fake_token_provider fixture — the github_copilot
    # branch must not call make_token_provider() at all.
    ctx = ScoutContext(model="github_copilot/gpt-5-mini")
    result = decompose_intent(Intent(raw_brief="prompt injection"), ctx=ctx)

    assert len(result) == 3
    assert captured["model"] == "github_copilot/gpt-5-mini"
    assert "api_base" not in captured
    assert "api_version" not in captured
    assert "azure_ad_token_provider" not in captured
    assert "api_key" not in captured
    assert captured["response_format"] == {"type": "json_object"}


def test_decompose_model_override_wins_over_ctx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The per-call `model=` arg overrides ctx.model AND ctx.aoai_*."""
    captured: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> _Resp:
        captured.update(kwargs)
        return _resp({"directions": _good_directions(2)})

    monkeypatch.setattr("litellm.completion", fake_completion)
    ctx = ScoutContext(
        model="github/gpt-4o-mini",
        aoai_endpoint="https://example.openai.azure.com",
        aoai_deployment="gpt-4o-mini",
    )
    decompose_intent(
        Intent(raw_brief="x"),
        ctx=ctx,
        model="github_copilot/claude-sonnet-4-5",
    )
    assert captured["model"] == "github_copilot/claude-sonnet-4-5"
    assert "azure_ad_token_provider" not in captured


def test_decompose_raises_when_unconfigured() -> None:
    with pytest.raises(LLMError, match="No LLM provider configured"):
        decompose_intent(Intent(raw_brief="x"), ctx=ScoutContext())


def test_decompose_retries_once_on_validation_error(
    monkeypatch: pytest.MonkeyPatch, fake_token_provider: None
) -> None:
    calls: list[Any] = []

    def fake_completion(**kwargs: Any) -> _Resp:
        calls.append(kwargs)
        if len(calls) == 1:
            return _resp({"oops": "wrong shape"})
        return _resp({"directions": _good_directions(3)})

    monkeypatch.setattr("litellm.completion", fake_completion)
    result = decompose_intent(Intent(raw_brief="x"), ctx=_ctx())
    assert len(calls) == 2
    assert len(result) == 3


def test_decompose_raises_after_two_validation_failures(
    monkeypatch: pytest.MonkeyPatch, fake_token_provider: None
) -> None:
    def fake_completion(**kwargs: Any) -> _Resp:
        return _resp({"oops": "still wrong"})

    monkeypatch.setattr("litellm.completion", fake_completion)
    with pytest.raises(LLMError):
        decompose_intent(Intent(raw_brief="x"), ctx=_ctx())


def test_decompose_clips_to_seven(
    monkeypatch: pytest.MonkeyPatch, fake_token_provider: None
) -> None:
    def fake_completion(**kwargs: Any) -> _Resp:
        return _resp({"directions": _good_directions(9)})

    monkeypatch.setattr("litellm.completion", fake_completion)
    result = decompose_intent(Intent(raw_brief="x"), ctx=_ctx())
    assert len(result) == 7


def test_decompose_empty_directions_returns_empty(
    monkeypatch: pytest.MonkeyPatch, fake_token_provider: None
) -> None:
    def fake_completion(**kwargs: Any) -> _Resp:
        return _resp({"directions": []})

    monkeypatch.setattr("litellm.completion", fake_completion)
    result = decompose_intent(Intent(raw_brief="x"), ctx=_ctx())
    assert result == []


def test_decompose_wraps_completion_exception(
    monkeypatch: pytest.MonkeyPatch, fake_token_provider: None
) -> None:
    def fake_completion(**kwargs: Any) -> _Resp:
        raise RuntimeError("boom")

    monkeypatch.setattr("litellm.completion", fake_completion)
    with pytest.raises(LLMError, match="boom"):
        decompose_intent(Intent(raw_brief="x"), ctx=_ctx())

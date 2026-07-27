"""Unit tests for the coverage report module."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from dataset_scout import (
    Candidate,
    CandidateMetadata,
    DecompositionDirection,
    Intent,
    LLMError,
    Scorecard,
    ScoutContext,
    Strategy,
    StrategyKind,
    TransformSpec,
)
from dataset_scout.coverage import (
    build_coverage_report,
    render_coverage_prompt,
)

pytestmark = pytest.mark.unit


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


@pytest.fixture
def fake_token_provider(monkeypatch: pytest.MonkeyPatch):
    def _stub() -> object:
        return lambda: "fake-bearer-token"

    # The shared client function — both decompose and coverage import from there.
    monkeypatch.setattr("dataset_scout.llm_client.make_token_provider", _stub)
    monkeypatch.setattr("dataset_scout.coverage._make_token_provider", _stub)


def _ctx() -> ScoutContext:
    return ScoutContext(
        aoai_endpoint="https://example.openai.azure.com",
        aoai_deployment="gpt-4o-mini",
    )


def _sample_scorecards() -> list[Scorecard]:
    s = Strategy(
        kind=StrategyKind.DIRECT_USE,
        confidence=0.85,
        rationale="solid match",
        transform=TransformSpec(),
    )
    return [
        Scorecard(
            candidate=Candidate(
                source="huggingface",
                id="org/x",
                revision="r",
                metadata=CandidateMetadata(description="A useful corpus."),
            ),
            strategies=[s],
        )
    ]


def test_render_coverage_prompt_includes_key_pieces():
    intent = Intent(raw_brief="prompt injection corpora")
    directions = [
        DecompositionDirection(name="hard_negs", rationale="balance"),
    ]
    p = render_coverage_prompt(intent, directions, _sample_scorecards())
    assert "prompt injection corpora" in p
    assert "hard_negs" in p
    assert "huggingface:org/x" in p
    assert "direct_use" in p


def test_render_coverage_handles_empty_inputs():
    intent = Intent(raw_brief="")
    p = render_coverage_prompt(intent, [], [])
    assert "(none)" in p
    # Still well-formed JSON-schema instruction.
    assert "gaps" in p


def test_build_coverage_report_happy_path(
    monkeypatch: pytest.MonkeyPatch, fake_token_provider: None
) -> None:
    captured: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> _Resp:
        captured.update(kwargs)
        return _resp(
            {
                "gaps": [
                    {
                        "aspect": "tool_call_outputs",
                        "description": "no candidate covers tool-call output injection",
                        "suggestion": "augment with synthetic tool-call traces",
                    }
                ]
            }
        )

    monkeypatch.setattr("litellm.completion", fake_completion)
    intent = Intent(raw_brief="prompt injection")
    gaps = build_coverage_report(
        intent,
        [DecompositionDirection(name="d", rationale="r")],
        _sample_scorecards(),
        ctx=_ctx(),
    )
    assert len(gaps) == 1
    assert gaps[0].aspect == "tool_call_outputs"
    # Routed via Azure with Entra.
    assert captured["model"] == "azure/gpt-4o-mini"
    assert callable(captured["azure_ad_token_provider"])
    assert "api_key" not in captured


def test_build_coverage_returns_empty_when_llm_says_no_gaps(
    monkeypatch: pytest.MonkeyPatch, fake_token_provider: None
) -> None:
    monkeypatch.setattr("litellm.completion", lambda **kw: _resp({"gaps": []}))
    gaps = build_coverage_report(Intent(raw_brief="x"), [], _sample_scorecards(), ctx=_ctx())
    assert gaps == []


def test_build_coverage_raises_when_unconfigured() -> None:
    with pytest.raises(LLMError, match="No LLM provider configured"):
        build_coverage_report(Intent(raw_brief="x"), [], _sample_scorecards(), ctx=ScoutContext())


def test_build_coverage_retries_once_on_validation_error(
    monkeypatch: pytest.MonkeyPatch, fake_token_provider: None
) -> None:
    calls = []

    def fake(**kw: Any) -> _Resp:
        calls.append(1)
        if len(calls) == 1:
            return _resp({"oops": "wrong"})
        return _resp({"gaps": []})

    monkeypatch.setattr("litellm.completion", fake)
    gaps = build_coverage_report(Intent(raw_brief="x"), [], _sample_scorecards(), ctx=_ctx())
    assert gaps == []
    assert len(calls) == 2


def test_top2_strategies_included_when_close(
    monkeypatch: pytest.MonkeyPatch, fake_token_provider: None
) -> None:
    """When best and second-best confidence are within delta, both appear."""
    s1 = Strategy(
        kind=StrategyKind.DIRECT_USE,
        confidence=0.85,
        rationale="primary",
        transform=TransformSpec(),
    )
    s2 = Strategy(
        kind=StrategyKind.SIGNAL_PROXY,
        confidence=0.78,  # within 0.15 of 0.85
        rationale="secondary",
        transform=TransformSpec(),
    )
    sc = Scorecard(
        candidate=Candidate(
            source="huggingface",
            id="org/y",
            revision="r",
            metadata=CandidateMetadata(),
        ),
        strategies=[s1, s2],
    )
    captured: dict[str, Any] = {}

    def fake(**kw: Any) -> _Resp:
        captured.update(kw)
        return _resp({"gaps": []})

    monkeypatch.setattr("litellm.completion", fake)
    build_coverage_report(Intent(raw_brief="x"), [], [sc], ctx=_ctx())
    msg = captured["messages"][0]["content"]
    # Both rationales should appear in the prompt.
    assert "primary" in msg
    assert "secondary" in msg

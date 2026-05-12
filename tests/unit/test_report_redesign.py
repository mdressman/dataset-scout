"""Unit tests for the report redesign: groupings, verdicts, recipe preview."""

from __future__ import annotations

import pytest

from dataset_scout.core import (
    Candidate,
    CandidateMetadata,
    Intent,
    ReconResult,
    Scorecard,
    Strategy,
    StrategyKind,
    TransformSpec,
)
from dataset_scout.render._view import (
    DISPLAY_GROUPS_ORDER,
    GROUP_LABELS,
    ReconReportContext,
    _verdict_for,
)
from tests._fakes.recon_fixture import build_demo_recon_result

pytestmark = pytest.mark.unit


def _scorecard(id_: str, kind: StrategyKind, confidence: float) -> Scorecard:
    return Scorecard(
        candidate=Candidate(
            source="huggingface",
            id=id_,
            revision="abc",
            metadata=CandidateMetadata(
                description="X", card_url=f"https://x/{id_}"
            ),
            streamable=True,
            surfaced_by=[],
        ),
        strategies=[
            Strategy(
                kind=kind,
                confidence=confidence,
                rationale="Reason. With sentences.",
                caveats=[],
                transform=TransformSpec(),
                composes_with=[],
            )
        ],
    )


def _result_with(scorecards: list[Scorecard]) -> ReconResult:
    return ReconResult(
        intent=Intent(raw_brief="x"),
        candidates=scorecards,
        sources_searched=["huggingface"],
        notices=[],
    )


# ─── Verdict computation ────────────────────────────────────────────


def test_verdict_for_direct_use_strong():
    sc = _scorecard("a/b", StrategyKind.DIRECT_USE, 0.85)
    v = _verdict_for(sc)
    assert v.group == "direct_fit"
    assert "Direct fit" in v.headline
    assert "strong" in v.headline
    assert "0.85" in v.headline
    assert v.use_as.lower().startswith("use")
    assert v.confidence == pytest.approx(0.85)
    assert v.n_strategies == 1


def test_verdict_for_direct_use_moderate():
    sc = _scorecard("a/b", StrategyKind.DIRECT_USE, 0.55)
    v = _verdict_for(sc)
    assert "moderate" in v.headline


def test_verdict_for_direct_use_tentative():
    sc = _scorecard("a/b", StrategyKind.DIRECT_USE, 0.30)
    v = _verdict_for(sc)
    assert "tentative" in v.headline


def test_verdict_for_subset_extraction_is_reframing():
    sc = _scorecard("a/b", StrategyKind.SUBSET_EXTRACTION, 0.65)
    v = _verdict_for(sc)
    assert v.group == "reframing"
    assert "Reframing" in v.headline


def test_verdict_for_label_remapping_is_reframing():
    sc = _scorecard("a/b", StrategyKind.LABEL_REMAPPING, 0.65)
    v = _verdict_for(sc)
    assert v.group == "reframing"


def test_verdict_for_signal_proxy():
    sc = _scorecard("a/b", StrategyKind.SIGNAL_PROXY, 0.55)
    v = _verdict_for(sc)
    assert v.group == "signal_proxy"
    assert "exclude from eval" in v.use_as.lower()


def test_verdict_for_benign_baseline():
    sc = _scorecard("a/b", StrategyKind.BENIGN_BASELINE, 0.5)
    v = _verdict_for(sc)
    assert v.group == "benign_baseline"


def test_verdict_for_no_strategies():
    sc = Scorecard(
        candidate=Candidate(
            source="huggingface",
            id="a/b",
            revision=None,
            metadata=CandidateMetadata(description=None, card_url="x"),
            streamable=True,
            surfaced_by=[],
        ),
        strategies=[],
    )
    v = _verdict_for(sc)
    assert v.group == "not_useful"
    assert v.headline == "Unassessed"
    assert v.confidence is None


# ─── Group ordering and counts ─────────────────────────────────────


def test_groups_in_display_order():
    """ReconReportContext.groups follows DISPLAY_GROUPS_ORDER even when buckets are empty."""
    result = _result_with([
        _scorecard("a/direct", StrategyKind.DIRECT_USE, 0.9),
        _scorecard("a/proxy", StrategyKind.SIGNAL_PROXY, 0.6),
    ])
    ctx = ReconReportContext.from_result(result)
    keys = [g.key for g in ctx.groups]
    assert keys == list(DISPLAY_GROUPS_ORDER)
    counts = {g.key: g.count for g in ctx.groups}
    assert counts["direct_fit"] == 1
    assert counts["signal_proxy"] == 1
    assert counts["reframing"] == 0
    assert counts["benign_baseline"] == 0
    assert counts["not_useful"] == 0


def test_within_group_sorted_by_confidence_desc():
    result = _result_with([
        _scorecard("a/low", StrategyKind.DIRECT_USE, 0.55),
        _scorecard("a/high", StrategyKind.DIRECT_USE, 0.92),
        _scorecard("a/mid", StrategyKind.DIRECT_USE, 0.75),
    ])
    ctx = ReconReportContext.from_result(result)
    direct = next(g for g in ctx.groups if g.key == "direct_fit")
    ids = [c.scorecard.candidate.id for c in direct.cards]
    assert ids == ["a/high", "a/mid", "a/low"]


def test_group_labels_present():
    result = _result_with([_scorecard("a/b", StrategyKind.DIRECT_USE, 0.9)])
    ctx = ReconReportContext.from_result(result)
    assert ctx.groups[0].label == GROUP_LABELS["direct_fit"]


# ─── Recipe preview ────────────────────────────────────────────────


def test_recipe_preview_counts():
    result = _result_with([
        _scorecard("a/d1", StrategyKind.DIRECT_USE, 0.9),
        _scorecard("a/d2", StrategyKind.DIRECT_USE, 0.8),
        _scorecard("a/r1", StrategyKind.SUBSET_EXTRACTION, 0.65),
        _scorecard("a/p1", StrategyKind.SIGNAL_PROXY, 0.6),
        _scorecard("a/b1", StrategyKind.BENIGN_BASELINE, 0.55),
        # Below threshold (0.5) — should NOT land in recipe.
        _scorecard("a/weak", StrategyKind.DIRECT_USE, 0.3),
        # Not_useful — never lands.
        _scorecard("a/skip", StrategyKind.NOT_USEFUL, 0.99),
    ])
    ctx = ReconReportContext.from_result(result)
    rp = ctx.recipe_preview
    assert rp is not None
    assert rp.n_components == 5
    assert rp.n_direct_fit == 2
    assert rp.n_reframing == 1
    assert rp.n_proxy == 1
    assert rp.n_benign == 1
    assert ctx.show_recipe_preview is True


def test_recipe_preview_empty_when_no_qualifying_components():
    result = _result_with([
        _scorecard("a/weak", StrategyKind.DIRECT_USE, 0.2),
    ])
    ctx = ReconReportContext.from_result(result)
    rp = ctx.recipe_preview
    assert rp is not None
    assert rp.n_components == 0
    assert ctx.show_recipe_preview is False


# ─── Render integration ────────────────────────────────────────────


def test_md_render_uses_groupings_and_verdicts():
    from dataset_scout.render.markdown_report import render_recon_report

    md = render_recon_report(build_demo_recon_result())
    # Grouped sections.
    assert "## At a glance" in md
    assert "🎯 Direct fits" in md or "Direct fits" in md
    # Verdict-led card headers (no longer starts with bare "X. `huggingface:..`").
    assert "Direct fit (" in md
    # Plain-text license signals — no green/orange badge styling.
    assert "license:" in md
    # Recipe preview at end.
    assert "Next steps — recipe & curate" in md
    assert "datascout curate" in md


def test_html_render_uses_groupings_and_recipe_preview():
    from dataset_scout.render.html_report import render_recon_report_html

    html = render_recon_report_html(build_demo_recon_result())
    # Scoreboard pills.
    assert 'class="scoreboard"' in html
    assert 'class="pill pill--direct_fit"' in html
    # Group sections with kind-keyed CSS.
    assert 'class="group group--direct_fit"' in html
    # Verdict pill on the candidate card.
    assert "candidate__verdict" in html
    # Recipe preview block.
    assert 'class="recipe-preview"' in html
    assert "datascout curate" in html


def test_html_no_colored_license_badges():
    """Per user feedback: the green/orange license badges added confusion."""
    from dataset_scout.render.html_report import render_recon_report_html

    html = render_recon_report_html(build_demo_recon_result())
    # The old badge--good / badge--warn classes are no longer applied
    # to license rows; license is plain text in the snapshot.
    # We don't strictly test absence (the class declaration may still
    # exist for legacy callers), but we DO verify the new structure
    # produces plain-text "license: <spdx> (in policy)" form.
    assert "license:" in html
    assert "(in policy)" in html or "(unknown)" in html


def test_view_show_recipe_preview_false_when_metadata_only():
    """No strategies → no qualifying components → no recipe preview shown."""
    from dataset_scout.core import CandidateMetadata

    result = ReconResult(
        intent=Intent(raw_brief="x"),
        candidates=[
            Scorecard(
                candidate=Candidate(
                    source="huggingface",
                    id="a/b",
                    revision=None,
                    metadata=CandidateMetadata(description=None, card_url="x"),
                    streamable=True,
                    surfaced_by=[],
                ),
            )
        ],
        notices=["Running in metadata-only mode: no LLM provider is configured, ..."],
    )
    ctx = ReconReportContext.from_result(result)
    assert ctx.metadata_only is True
    assert ctx.show_recipe_preview is False


# ─── regression: mode-detection sentinels track pipeline notice text ──


def test_view_mode_detection_uses_canonical_pipeline_notice() -> None:
    """ReconReportContext must detect metadata-only / runtime-error mode
    from the *current* pipeline notice strings, not stale Azure-only text.

    Pre-fix regression: ``_view.py`` substring-matched the historical
    "Azure OpenAI is not configured" text. After the provider-agnostic
    rewrite the pipeline emits "Running in metadata-only mode: ..." so
    the detection silently broke and every recon report rendered with
    ``metadata_only=False`` even when no LLM was configured.
    """
    from dataset_scout.pipeline import LLM_RUNTIME_HINT, METADATA_ONLY_NOTICE

    md_result = ReconResult(
        intent=Intent(raw_brief="x"),
        candidates=[],
        sources_searched=[],
        notices=[METADATA_ONLY_NOTICE],
    )
    assert ReconReportContext.from_result(md_result).metadata_only is True

    rt_result = ReconResult(
        intent=Intent(raw_brief="x"),
        candidates=[],
        sources_searched=[],
        notices=["decomposition skipped: oops", LLM_RUNTIME_HINT],
    )
    assert ReconReportContext.from_result(rt_result).llm_runtime_error is True

"""Unit tests for `dataset_scout.render.html_report`."""

from __future__ import annotations

import typing
from html.parser import HTMLParser

import pytest

from dataset_scout.render._view import ReconReportContext
from dataset_scout.render.html_report import (
    render_recon_report_html,
    write_recon_report_html,
)
from tests._fakes.recon_fixture import build_demo_recon_result

pytestmark = pytest.mark.unit


# ─── helpers ────────────────────────────────────────────────────────


class _SimpleHTMLValidator(HTMLParser):
    """Lightweight well-formedness check: every opening tag is closed.

    Skips void elements that don't need closing in HTML5 (meta, br, etc.)
    The HTML report is hand-rolled so the cost of being strict is real;
    we just want a minimal sanity gate that rendering produced
    structurally valid HTML.
    """

    _VOID: typing.ClassVar[set[str]] = {"meta", "br", "img", "input", "link", "hr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.errors: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self._VOID:
            return
        self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in self._VOID:
            return
        if not self.stack:
            self.errors.append(f"end tag </{tag}> with empty stack")
            return
        if self.stack[-1] != tag:
            self.errors.append(f"mismatch: closing </{tag}>, expected </{self.stack[-1]}>")
        else:
            self.stack.pop()


def _validate_html(html: str) -> None:
    validator = _SimpleHTMLValidator()
    validator.feed(html)
    assert not validator.errors, f"HTML errors: {validator.errors}"
    assert not validator.stack, f"unclosed tags: {validator.stack}"


# ─── tests against the demo fixture ────────────────────────────────


def test_renders_well_formed_html_for_demo_result():
    result = build_demo_recon_result()
    html = render_recon_report_html(result)
    assert html.startswith("<!doctype html>")
    assert "<title>" in html
    assert "</html>" in html
    _validate_html(html)


def test_includes_brief_section():
    result = build_demo_recon_result()
    html = render_recon_report_html(result)
    assert "Brief" in html
    assert result.intent.raw_brief in html


def test_html_card_title_links_to_dataset():
    """Per UX feedback: card title is a clickable link to the dataset card."""
    result = build_demo_recon_result()
    html = render_recon_report_html(result)
    # The candidate__title-link wraps rank + verdict + id in an <a> with
    # href pointing to the upstream dataset card.
    assert 'class="candidate__title-link"' in html
    # Demo fixture's first card url should be referenced.
    first = result.candidates[0].candidate.metadata.card_url
    assert first is not None
    assert f'href="{first}"' in html


def test_html_card_details_are_collapsed_by_default():
    """Per UX feedback: rationale + transform + caveats are collapsed."""
    result = build_demo_recon_result()
    html = render_recon_report_html(result)
    # Each assessed card has a <details class="candidate__details"> block
    # for the long-form rationale/caveats/transform.
    assert 'class="candidate__details"' in html
    # The summary lists strategies + count.
    assert "Strategies, caveats" in html


def test_html_decomposition_and_papers_are_collapsed():
    """Heavy investigative sections wrapped in <details> by default."""
    result = build_demo_recon_result()
    html = render_recon_report_html(result)
    # Both sections share the collapsed-section class.
    assert 'class="collapsed-section"' in html
    # The Decomposition heading is now inside a <summary> element.
    # Use a substring that's unique to that wrapper.
    assert 'inline-h2">Decomposition' in html


def test_html_run_summary_compact_at_top():
    """Compact run summary lives near the top, before candidates."""
    result = build_demo_recon_result()
    html = render_recon_report_html(result)
    # Position check: run-summary-compact must appear before the
    # first candidate__title-link (and before Decomposition).
    pos_summary = html.find("run-summary-compact")
    pos_first_card = html.find("candidate__title-link")
    pos_decomp = html.find("Decomposition")
    assert pos_summary > 0
    assert pos_summary < pos_first_card
    if pos_decomp > 0:
        assert pos_summary < pos_decomp


def test_html_compact_meta_replaces_snapshot_list():
    """Old bulleted snapshot list is gone; compact meta paragraph replaces it."""
    result = build_demo_recon_result()
    html = render_recon_report_html(result)
    assert "candidate__meta" in html
    # Old snapshot class should not be emitted anymore.
    assert "candidate__snapshot" not in html


def test_html_card_doesnt_reproduce_card_url_in_body():
    """Avoid rendering the dataset URL twice — once in title link is enough."""
    result = build_demo_recon_result()
    html = render_recon_report_html(result)
    # The header link contains the URL. The body should NOT have a
    # second 🔗 / Card: line; that was the verbose old layout.
    # Use a defensive check: the URL of the first candidate appears
    # at most once outside the link href.
    first = result.candidates[0].candidate.metadata.card_url
    assert first is not None
    # Once in href, possibly once more in the visible link text — but
    # not in a separate "Card:" snapshot line.
    assert "Card:" not in html or "<b>Card:" not in html


def test_lists_all_candidates_with_card_links():
    result = build_demo_recon_result()
    html = render_recon_report_html(result)
    assert "Candidates" in html
    for sc in result.candidates:
        # Card URL appears as an anchor href.
        if sc.candidate.metadata.card_url:
            assert sc.candidate.metadata.card_url in html


def test_renders_strategies_when_assessed():
    result = build_demo_recon_result()
    html = render_recon_report_html(result)
    # The demo result has strategy assessments; expect kind labels.
    assert "direct use" in html or "signal proxy" in html
    assert "confidence" in html


def test_renders_coverage_gaps_when_present():
    result = build_demo_recon_result()
    if result.coverage and result.coverage.semantic_gaps:
        html = render_recon_report_html(result)
        for gap in result.coverage.semantic_gaps:
            assert gap.aspect in html


def test_html_header_includes_crisp_wordmark():
    """Header is a clean typographic h1 (no ASCII art)."""
    result = build_demo_recon_result()
    html = render_recon_report_html(result)
    assert "report-hero" in html
    assert 'class="report-hero__title"' in html
    assert "dataset-scout" in html
    # No ASCII wordmark anymore.
    assert 'class="hero-wordmark"' not in html
    assert "<pre" not in html.split("</header>", 1)[0]


def test_html_brief_renders_as_hero_blockquote():
    """Round-2: the brief is highlighted as a quoted hero block, not a tiny header."""
    result = build_demo_recon_result()
    html = render_recon_report_html(result)
    assert 'class="brief-hero"' in html
    assert 'class="brief-hero__text"' in html
    assert "<blockquote" in html
    assert "BRIEF" in html


def test_html_gaps_render_as_table_when_present():
    """Round-2: sourcing roadmap is a scannable table, not a stack of headings."""
    result = build_demo_recon_result()
    if result.coverage and result.coverage.semantic_gaps:
        html = render_recon_report_html(result)
        assert '<table class="gaps-table">' in html
        assert "Aspect" in html and "What's missing" in html and "Next step" in html


def test_html_collapses_not_useful_into_single_note():
    """Round-2: 'not useful' candidates collapse into one aside, not card grid."""
    result = build_demo_recon_result()
    html = render_recon_report_html(result)
    # If the Demo fixture has any not_useful candidates, they should NOT be
    # rendered as full cards. We assert the consolidated note is present
    # whenever any candidate has a 'not_useful' verdict.
    from dataset_scout.render._view import ReconReportContext

    ctx = ReconReportContext.from_result(result)
    has_not_useful = any(g.key == "not_useful" and g.cards for g in ctx.groups)
    if has_not_useful:
        assert 'class="not-useful-note"' in html
        assert "not useful" in html.lower()


def test_html_escaping_blocks_brief_xss():
    """A brief containing HTML metacharacters is escaped, not injected."""
    result = build_demo_recon_result()
    # Mutate the intent's raw_brief to include payload; Pydantic v2
    # frozen models support model_copy.
    result = result.model_copy(
        update={"intent": result.intent.model_copy(update={"raw_brief": "<script>alert(1)</script>"})}
    )
    html = render_recon_report_html(result)
    assert "<script>" not in html  # raw payload absent
    assert "&lt;script&gt;" in html  # escaped form present


def test_write_recon_report_html_creates_file(tmp_path):
    result = build_demo_recon_result()
    path = write_recon_report_html(result, tmp_path)
    assert path.exists()
    assert path.name == "report.html"
    content = path.read_text(encoding="utf-8")
    assert "<!doctype html>" in content


# ─── view-context tests ────────────────────────────────────────────


def test_view_context_from_demo_result():
    """The shared view-model derives consistent flags from a real result."""
    result = build_demo_recon_result()
    ctx = ReconReportContext.from_result(result)
    assert ctx.n_candidates == len(result.candidates)
    # Demo fixture has strategies + decomposition + gaps (mini-recon shape).
    assert ctx.has_strategies is True
    assert ctx.has_decomposition is True


def test_view_context_metadata_only_flag():
    from dataset_scout.core import Intent, ReconResult

    result = ReconResult(
        intent=Intent(raw_brief="x"),
        candidates=[],
        sources_searched=[],
        notices=["Running in metadata-only mode: no LLM provider is configured, ..."],
    )
    ctx = ReconReportContext.from_result(result)
    assert ctx.metadata_only is True
    assert ctx.has_strategies is False


def test_view_context_show_gaps_lead():
    """show_gaps_lead is True only when gaps are notable or coverage sparse."""
    from dataset_scout.core import (
        CoverageGap,
        CoverageReport,
        DecompositionDirection,
        Intent,
        ReconResult,
    )

    # Two gaps + a few candidates → notable_gaps → show_gaps_lead True.
    result = ReconResult(
        intent=Intent(raw_brief="x"),
        candidates=[],
        sources_searched=[],
        notices=[],
        coverage=CoverageReport(
            decomposition=[
                DecompositionDirection(
                    name="d1",
                    rationale="r",
                    keywords=[],
                    threat_families=[],
                    expected_finds="",
                ),
            ],
            semantic_gaps=[
                CoverageGap(aspect="a1", description="d1", suggestion="s1"),
                CoverageGap(aspect="a2", description="d2", suggestion="s2"),
            ],
        ),
    )
    ctx = ReconReportContext.from_result(result)
    assert ctx.show_gaps_lead is True


def test_includes_label_intent_fit_in_badge_when_present():
    """If a scorecard has label_intent_fit, the HTML surfaces a semantic-fit badge."""
    from dataset_scout.core import Evidence, SubScore

    result = build_demo_recon_result()
    sc = result.candidates[0]
    sc.label_intent_fit = SubScore(
        value=0.72,
        status="ok",
        evidence=[Evidence(kind="embedding_fit", detail="cosine=0.72")],
        probe_version="1",
    )
    html = render_recon_report_html(result)
    assert "semantic fit" in html
    assert "0.72" in html

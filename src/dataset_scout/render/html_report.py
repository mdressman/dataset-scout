"""HTML report renderer for `ReconResult`.

Self-contained HTML — embedded CSS, no JS, no external assets. Designed
to be opened directly in any browser, including by reviewers who don't
read Markdown well. Mirrors the section structure of `markdown_report.py`
via the shared `ReconReportContext` so the two renderers stay in sync.

Strategy kinds are color-coded; coverage gaps live in a callout box;
each candidate's badges and strategies are visually grouped.
"""

from __future__ import annotations

from html import escape
from io import StringIO
from pathlib import Path

from dataset_scout.core import (
    PaperReference,
    ReconResult,
    Scorecard,
    Strategy,
    StrategyKind,
    SubScore,
)
from dataset_scout.render._view import (
    CardVerdict,
    ReconReportContext,
    StrategyGroup,
)

# ─── public entry points ───────────────────────────────────────────


def write_recon_report_html(result: ReconResult, out_dir: Path) -> Path:
    """Render and write `<out_dir>/report.html`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "report.html"
    target.write_text(render_recon_report_html(result), encoding="utf-8")
    return target


def render_recon_report_html(result: ReconResult) -> str:
    """Render a ReconResult to a self-contained HTML string.

    Section order is tuned for fast scanning by a reader who hasn't
    seen the report before. Top: what happened. Middle: where the
    data is and what to do with it. Bottom: investigative material
    (decomposition, papers) collapsed in <details> blocks.

      1. Header (wordmark + tagline)
      2. Mode callout (Strategy-assessed / Sparse coverage / etc.)
      3. Brief
      4. Compact run summary (one-line counts)
      5. At-a-glance scoreboard (verdict mix)
      6. Grouped candidate cards
      7. Decomposition (collapsed)
      8. Related papers (collapsed)
      9. Sourcing roadmap / coverage gaps
     10. Recipe preview / next steps
     11. Footer disclaimer
    """
    ctx = ReconReportContext.from_result(result)
    buf = StringIO()
    _write_doc_head(buf, result.intent.raw_brief)
    buf.write('<body>\n<main class="container">\n')
    _write_header(buf, ctx)
    _write_brief_section(buf, result)
    _write_run_summary_compact(buf, result, ctx)
    if ctx.has_strategies:
        _write_at_a_glance(buf, ctx)
    if result.candidates:
        _write_candidates_section(buf, result, ctx)
    if ctx.has_decomposition and result.coverage:
        _write_decomposition_section(buf, result)
    if ctx.show_papers:
        _write_papers_section(buf, result, ctx)
    if ctx.show_gaps_lead and result.coverage:
        _write_gaps_section(buf, result, lead=True)
    elif result.coverage and result.coverage.semantic_gaps:
        _write_gaps_section(buf, result, lead=False)
    if ctx.show_recipe_preview:
        _write_recipe_preview(buf, ctx)
    _write_footer(buf)
    buf.write("</main>\n</body>\n</html>\n")
    return buf.getvalue()


# ─── HTML head + CSS ───────────────────────────────────────────────


_CSS = """
:root {
  --fg: #1f2328;
  --fg-muted: #57606a;
  --bg: #ffffff;
  --bg-muted: #f6f8fa;
  --border: #d0d7de;
  --accent: #0969da;
  --good: #1f883d;
  --warn: #bf8700;
  --bad: #cf222e;
  --neutral: #57606a;
  --proxy: #8250df;
  --benign: #218bff;
  --code-bg: #f6f8fa;
}
@media (prefers-color-scheme: dark) {
  :root {
    --fg: #c9d1d9;
    --fg-muted: #8b949e;
    --bg: #0d1117;
    --bg-muted: #161b22;
    --border: #30363d;
    --accent: #58a6ff;
    --good: #56d364;
    --warn: #e3b341;
    --bad: #ff7b72;
    --neutral: #8b949e;
    --proxy: #d2a8ff;
    --benign: #79c0ff;
    --code-bg: #161b22;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: var(--bg);
  color: var(--fg);
  line-height: 1.55;
  font-size: 16px;
}
.container { max-width: 920px; margin: 0 auto; padding: 32px 24px 80px; }
h1, h2, h3 { line-height: 1.25; margin-top: 1.5em; }
h1 {
  font-size: 2.1rem;
  font-weight: 700;
  border-bottom: 2px solid var(--border);
  padding-bottom: 0.35em;
  letter-spacing: -0.01em;
}
h2 {
  font-size: 1.55rem;
  font-weight: 700;
  border-bottom: 1px solid var(--border);
  padding-bottom: 0.25em;
  letter-spacing: -0.005em;
}
h3 { font-size: 1.15rem; font-weight: 600; }
p { margin: 0.6em 0; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
code, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
              background: var(--code-bg); padding: 0.1em 0.35em; border-radius: 3px;
              font-size: 0.92em; }
.callout {
  border-left: 4px solid var(--accent);
  background: var(--bg-muted);
  padding: 14px 18px;
  border-radius: 6px;
  margin: 1em 0;
}
.callout--warn { border-left-color: var(--warn); }
.callout--bad { border-left-color: var(--bad); }
.callout--good { border-left-color: var(--good); }
.callout--info { border-left-color: var(--accent); }
.callout--proxy { border-left-color: var(--proxy); }
.callout strong { display: block; margin-bottom: 0.25em; }
.muted { color: var(--fg-muted); }
.small { font-size: 0.88em; }

/* ─── Report hero (crisp typographic header) ──────────────────── */
.report-hero {
  margin: 0 0 1.25em;
  padding: 28px 0 20px;
  border-bottom: 1px solid var(--border);
}
.report-hero__title {
  margin: 0;
  font-size: clamp(1.6rem, 2.6vw + 0.6rem, 2.4rem);
  font-weight: 800;
  letter-spacing: -0.02em;
  line-height: 1.1;
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 0.4em;
}
.report-hero__product { color: var(--accent); }
.report-hero__sep {
  color: var(--fg-muted);
  font-weight: 400;
}
.report-hero__verb {
  color: var(--fg);
  font-weight: 700;
  text-transform: uppercase;
  font-size: 0.62em;
  letter-spacing: 0.14em;
  padding: 4px 10px;
  background: var(--bg-muted);
  border-radius: 4px;
}
.report-hero__tagline {
  margin: 8px 0 0;
  color: var(--fg-muted);
  font-size: 0.98em;
}

/* ─── Brief hero (the user's input, prominent) ─────────────────── */
.brief-hero {
  margin: 1.5em 0 2em;
  padding: 22px 26px;
  background: var(--bg-muted);
  border-radius: 8px;
  border-left: 4px solid var(--accent);
  position: relative;
}
.brief-hero__label {
  display: inline-block;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.72em;
  font-weight: 700;
  letter-spacing: 0.18em;
  color: var(--accent);
  background: color-mix(in srgb, var(--accent) 14%, var(--bg));
  padding: 2px 10px;
  border-radius: 3px;
  margin-bottom: 10px;
}
.brief-hero__text {
  margin: 0;
  padding: 0;
  font-size: 1.18em;
  line-height: 1.5;
  font-weight: 500;
  color: var(--fg);
  border: none;
  font-style: normal;
  /* Rely on the side rule of .brief-hero for visual emphasis. */
}
.brief-hero__meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 14px;
}
.brief-hero__pill {
  display: inline-flex; align-items: baseline; gap: 6px;
  padding: 4px 12px;
  border-radius: 999px;
  background: var(--bg);
  border: 1px solid var(--border);
  font-size: 0.88em;
  color: var(--fg);
}
.brief-hero__pill-label {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.82em;
  font-weight: 600;
  color: var(--fg-muted);
  letter-spacing: 0.05em;
}

/* ─── Compact run summary at top ──────────────────────────────── */
.run-summary-compact {
  display: flex; flex-wrap: wrap; gap: 12px; align-items: baseline;
  padding: 10px 14px;
  background: var(--bg-muted);
  border-radius: 6px;
  margin: 0.6em 0;
  font-size: 0.95em;
}
.run-summary-compact b { font-weight: 600; color: var(--fg); }
.notices-details { margin: 6px 0 1em; font-size: 0.92em; }
.notices-details summary { cursor: pointer; }
.notices-details ul { margin: 6px 0 0; padding-left: 22px; }

/* ─── Collapsed sections (Decomposition + Papers) ─────────────── */
.collapsed-section {
  margin: 1.5em 0;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--bg-muted);
  overflow: hidden;
  transition: border-color 0.15s ease;
}
.collapsed-section[open] {
  border-color: var(--accent);
  background: var(--bg);
}
.collapsed-section > summary {
  cursor: pointer;
  list-style: none;
  padding: 14px 18px;
  display: flex;
  align-items: center;
  gap: 12px;
  user-select: none;
  background: var(--bg-muted);
  border-radius: 8px;
}
.collapsed-section[open] > summary {
  border-bottom: 1px solid var(--border);
  border-radius: 8px 8px 0 0;
}
.collapsed-section > summary:hover {
  background: color-mix(in srgb, var(--accent) 8%, var(--bg-muted));
}
.collapsed-section > summary::-webkit-details-marker { display: none; }
.collapsed-section > summary::before {
  content: "▸";
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 22px; height: 22px;
  flex-shrink: 0;
  color: var(--accent);
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 4px;
  font-size: 0.7em;
  transition: transform 0.15s ease;
}
.collapsed-section[open] > summary::before {
  content: "▾";
  background: var(--accent);
  color: var(--bg);
  border-color: var(--accent);
}
.collapsed-section > summary::after {
  content: "Click to expand";
  margin-left: auto;
  font-size: 0.78em;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--accent);
  flex-shrink: 0;
}
.collapsed-section[open] > summary::after { content: "Click to collapse"; }
.collapsed-section > :not(summary) {
  padding: 0 18px;
}
.collapsed-section > :not(summary):last-child { padding-bottom: 16px; }
.collapsed-section > :not(summary):first-of-type { padding-top: 16px; }
.inline-h2 {
  display: inline;
  border: none;
  padding: 0;
  margin: 0;
  font-size: 1.25rem;
}

/* ─── At-a-glance scoreboard ──────────────────────────────────── */
.scoreboard { margin: 1.5em 0; }
.scoreboard__pills {
  display: flex; flex-wrap: wrap; gap: 10px;
  margin: 12px 0;
}
.pill {
  display: inline-flex; align-items: baseline; gap: 6px;
  padding: 6px 14px;
  border-radius: 999px;
  font-size: 0.95em;
  border: 1px solid var(--border);
  background: var(--bg-muted);
  color: var(--fg);
}
.pill b { font-size: 1.1em; }
.pill--direct_fit { border-left: 4px solid var(--good); padding-left: 12px; }
.pill--reframing { border-left: 4px solid var(--accent); padding-left: 12px; }
.pill--signal_proxy { border-left: 4px solid var(--proxy); padding-left: 12px; }
.pill--benign_baseline { border-left: 4px solid var(--benign); padding-left: 12px; }
.pill--not_useful { border-left: 4px solid var(--neutral); padding-left: 12px; opacity: 0.7; }
.scoreboard__note {
  font-size: 0.92em; color: var(--fg-muted); font-style: italic;
}

/* ─── Strategy-grouped sections ───────────────────────────────── */
.group { margin: 2em 0; }
.group__head {
  border-left: 6px solid var(--neutral);
  padding-left: 14px;
  margin-bottom: 18px;
}
.group--direct_fit .group__head { border-left-color: var(--good); }
.group--reframing .group__head { border-left-color: var(--accent); }
.group--signal_proxy .group__head { border-left-color: var(--proxy); }
.group--benign_baseline .group__head { border-left-color: var(--benign); }
.group--not_useful .group__head { border-left-color: var(--neutral); opacity: 0.7; }
.group__title { margin: 0; border: none; padding: 0; font-size: 1.45rem; }
.group__count { color: var(--fg-muted); font-weight: 400; font-size: 0.9em; }
.group__desc { color: var(--fg-muted); font-size: 0.95em; margin: 4px 0 0; }

/* ─── Candidate card ──────────────────────────────────────────── */
.candidate {
  border: 1px solid var(--border);
  border-left: 4px solid var(--neutral);
  border-radius: 6px;
  padding: 18px 22px;
  margin: 14px 0;
  background: var(--bg);
}
.candidate--direct_use { border-left-color: var(--good); }
.candidate--subset_extraction { border-left-color: var(--accent); }
.candidate--label_remapping { border-left-color: var(--accent); }
.candidate--cross_class_repurposing { border-left-color: var(--proxy); }
.candidate--signal_proxy { border-left-color: var(--proxy); }
.candidate--benign_baseline { border-left-color: var(--benign); }
.candidate--not_useful { border-left-color: var(--neutral); opacity: 0.75; }
.candidate--unscored { border-left-color: var(--neutral); }

.candidate__head {
  margin-bottom: 6px;
}
.candidate__title-link {
  display: flex; flex-wrap: wrap; gap: 8px 14px;
  align-items: baseline;
  text-decoration: none;
  color: inherit;
  padding: 4px 0;
  border-radius: 4px;
}
.candidate__title-link:hover {
  text-decoration: none;
}
.candidate__title-link:hover .candidate__id {
  color: var(--accent);
  text-decoration: underline;
}
.candidate__title-link::after {
  content: "↗";
  color: var(--fg-muted);
  font-size: 0.85em;
  margin-left: auto;
  opacity: 0.6;
  align-self: center;
}
.candidate__rank {
  font-weight: 600;
  color: var(--fg-muted);
  font-size: 0.78em;
  font-variant-numeric: tabular-nums;
  margin-left: auto;
  letter-spacing: 0.04em;
}
.candidate__verdict {
  font-weight: 700;
  padding: 3px 10px;
  border-radius: 4px;
  background: var(--bg-muted);
  font-size: 0.78em;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  white-space: nowrap;
}
.candidate__verdict--direct_use { color: var(--good); background: color-mix(in srgb, var(--good) 12%, var(--bg)); }
.candidate__verdict--subset_extraction,
.candidate__verdict--label_remapping { color: var(--accent); background: color-mix(in srgb, var(--accent) 12%, var(--bg)); }
.candidate__verdict--cross_class_repurposing,
.candidate__verdict--signal_proxy { color: var(--proxy); background: color-mix(in srgb, var(--proxy) 12%, var(--bg)); }
.candidate__verdict--benign_baseline { color: var(--benign); background: color-mix(in srgb, var(--benign) 12%, var(--bg)); }
.candidate__verdict--not_useful { color: var(--neutral); background: var(--bg-muted); opacity: 0.85; }
.candidate__id {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 1.18em;
  font-weight: 700;
  color: var(--fg);
  letter-spacing: -0.005em;
}

.candidate__lede {
  font-size: 1.02em;
  margin: 6px 0 4px;
  color: var(--fg);
}
.candidate__use-as {
  font-size: 0.95em;
  margin: 4px 0;
  color: var(--fg);
}
.candidate__use-as b { font-weight: 600; }

.candidate__meta {
  font-size: 0.88em;
  color: var(--fg-muted);
  margin: 4px 0 0;
}

.candidate__details {
  margin-top: 10px;
  font-size: 0.93em;
}
.candidate__details > summary {
  cursor: pointer;
  color: var(--fg-muted);
  padding: 4px 0;
  list-style: none;
  user-select: none;
}
.candidate__details > summary::-webkit-details-marker { display: none; }
.candidate__details > summary::before {
  content: "▸ ";
  color: var(--fg-muted);
}
.candidate__details[open] > summary::before { content: "▾ "; }
.candidate__details > summary:hover { color: var(--fg); }
.candidate__desc {
  font-size: 0.95em;
  color: var(--fg);
  margin: 6px 0;
}

/* Strategy detail (kept; same colour-stripe per kind) */
.strategies { margin: 10px 0; }
.strategy {
  display: block; padding: 8px 12px; margin: 6px 0;
  border-left: 4px solid var(--neutral);
  background: var(--bg-muted);
  border-radius: 0 6px 6px 0;
}
.strategy--direct_use { border-left-color: var(--good); }
.strategy--subset_extraction { border-left-color: var(--accent); }
.strategy--label_remapping { border-left-color: var(--accent); }
.strategy--cross_class_repurposing { border-left-color: var(--proxy); }
.strategy--signal_proxy { border-left-color: var(--proxy); }
.strategy--benign_baseline { border-left-color: var(--benign); }
.strategy--composition_only { border-left-color: var(--neutral); }
.strategy--not_useful { border-left-color: var(--bad); opacity: 0.85; }

.strategy__head { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.strategy__kind { font-weight: 600; }
.strategy__conf { color: var(--fg-muted); font-size: 0.9em; }
.strategy__rationale { margin: 4px 0; }
.strategy__caveats, .strategy__transform { font-size: 0.9em; color: var(--fg-muted); }
.strategy__caveats li { margin: 2px 0; }
.strategy__transform { margin-top: 4px; }

.probe-grid {
  display: grid;
  grid-template-columns: 160px 1fr;
  gap: 4px 12px;
  font-size: 0.9em;
  margin: 8px 0;
}
.probe-grid dt { color: var(--fg-muted); }
.probe-grid dd { margin: 0; }

/* ─── Sourcing roadmap table ──────────────────────────────────── */
.gaps-table {
  width: 100%;
  border-collapse: collapse;
  margin: 1em 0 1.5em;
  font-size: 0.95em;
}
.gaps-table thead th {
  text-align: left;
  padding: 10px 12px;
  background: var(--bg-muted);
  border-bottom: 2px solid var(--border);
  font-weight: 600;
  font-size: 0.85em;
  color: var(--fg-muted);
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.gaps-table tbody td {
  padding: 14px 12px;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
  line-height: 1.5;
}
.gaps-table tbody tr:last-child td { border-bottom: none; }
.gaps-table tbody tr:hover { background: var(--bg-muted); }
.gaps-table__aspect {
  width: 22%;
  color: var(--fg);
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.92em;
}
.gaps-table__gap { width: 38%; color: var(--fg-muted); }
.gaps-table__step {
  width: 40%;
  color: var(--fg);
  border-left: 3px solid color-mix(in srgb, var(--accent) 30%, var(--bg));
  padding-left: 16px;
}
@media (max-width: 700px) {
  /* Stack on narrow screens — table layout breaks down anyway. */
  .gaps-table thead { display: none; }
  .gaps-table, .gaps-table tbody, .gaps-table tr, .gaps-table td { display: block; width: auto; }
  .gaps-table tbody tr { padding: 10px 0; border-bottom: 1px solid var(--border); }
  .gaps-table tbody tr:hover { background: transparent; }
  .gaps-table__aspect { font-weight: 700; padding-bottom: 4px; }
  .gaps-table__step { border-left: none; padding-left: 12px; }
}

/* ─── Consolidated "not useful" note ─────────────────────────── */
.not-useful-note {
  display: flex;
  align-items: center;
  gap: 14px;
  margin: 1.5em 0;
  padding: 12px 16px;
  background: var(--bg-muted);
  border-radius: 6px;
  border-left: 3px solid var(--neutral);
  font-size: 0.93em;
  color: var(--fg-muted);
}
.not-useful-note__count {
  font-size: 1.4em;
  font-weight: 700;
  color: var(--fg);
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  min-width: 1.6em;
  text-align: center;
}

/* ─── Recipe / curate preview ─────────────────────────────────── */
.recipe-preview {
  margin: 2em 0;
  padding: 16px 20px;
  background: var(--bg-muted);
  border-radius: 8px;
  border-left: 4px solid var(--good);
}
.recipe-preview h2 { margin-top: 0; }
.recipe-preview pre {
  background: var(--bg);
  border: 1px solid var(--border);
  padding: 10px 14px;
  border-radius: 4px;
  overflow-x: auto;
}
.recipe-preview__table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.92em;
  margin: 12px 0;
}
.recipe-preview__table th,
.recipe-preview__table td {
  padding: 6px 10px;
  border-bottom: 1px solid var(--border);
  text-align: left;
}
.recipe-preview__table th { font-weight: 600; color: var(--fg-muted); }

.run-summary { background: var(--bg-muted); padding: 12px 16px; border-radius: 6px; }
.run-summary ul { margin: 0; padding-left: 20px; }
.notices { margin-top: 10px; }
.notice { font-size: 0.85em; color: var(--fg-muted); margin: 4px 0; }

footer.disclaimer {
  margin-top: 60px; padding-top: 16px; border-top: 1px solid var(--border);
  font-size: 0.85em; color: var(--fg-muted); text-align: center;
}

.paper {
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px 18px;
  margin: 12px 0;
  background: var(--bg);
}
.paper__title { margin: 0 0 6px; font-size: 1.05em; }
.paper__meta { color: var(--fg-muted); font-size: 0.9em; margin-bottom: 4px; }
.paper__authors { color: var(--fg-muted); font-size: 0.9em; margin-bottom: 4px; }
.paper__abstract { font-size: 0.95em; margin: 6px 0; }
.paper__surfaced { font-size: 0.85em; color: var(--fg-muted); margin: 4px 0; }
.paper__citations { background: var(--bg-muted); padding: 8px 12px; border-radius: 6px;
                    font-size: 0.92em; margin-top: 8px; }
.paper__citations ul { margin: 4px 0 0 0; padding-left: 20px; }
"""


def _write_doc_head(buf: StringIO, title: str) -> None:
    safe_title = escape(title[:120] or "dataset-scout recon report")
    buf.write("<!doctype html>\n<html lang=\"en\">\n<head>\n")
    buf.write("<meta charset=\"utf-8\">\n")
    buf.write("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n")
    buf.write(f"<title>dataset-scout: {safe_title}</title>\n")
    buf.write("<style>\n")
    buf.write(_CSS)
    buf.write("\n</style>\n</head>\n")


# ─── sections ──────────────────────────────────────────────────────


def _write_header(buf: StringIO, ctx: ReconReportContext) -> None:
    """Crisp, bold report header: product name + tagline + mode callout.

    No ASCII art — just a confident wordmark and a single-line tagline
    that orients the reader before the brief itself takes over below.
    """
    buf.write(
        '<header class="report-hero">\n'
        '  <h1 class="report-hero__title">'
        '<span class="report-hero__product">dataset-scout</span>'
        '<span class="report-hero__sep">/</span>'
        '<span class="report-hero__verb">recon report</span>'
        "</h1>\n"
        '  <p class="report-hero__tagline">'
        "Public-dataset reconnaissance &mdash; brief in, audit-ready corpus out."
        "</p>\n"
        "</header>\n"
    )
    # Mode callout immediately below the hero so the framing is set
    # before the user sees the brief.
    if ctx.metadata_only:
        buf.write(
            '<div class="callout callout--warn"><strong>⚠️ Metadata-only mode.</strong>'
            "No LLM provider is configured, so decomposition, strategy assessment, "
            "and coverage gaps were skipped. To enable them, copy <code>.env.example</code> "
            "to <code>.env</code> and set <code>DATASET_SCOUT_MODEL</code> "
            "(e.g. <code>github_copilot/gpt-5-mini</code> or <code>github/gpt-4o-mini</code>), "
            "or set <code>AZURE_OPENAI_ENDPOINT</code> and "
            "<code>AZURE_OPENAI_DEPLOYMENT</code> and run <code>az login</code>."
            "</div>\n"
        )
    elif ctx.llm_runtime_error:
        buf.write(
            '<div class="callout callout--warn"><strong>⚠️ LLM call failed — running in metadata-only mode.</strong>'
            "The LLM provider was configured but a call failed at runtime "
            "(model id, credential, network, or quota — see notices below). "
            "Decomposition, strategy assessment, and coverage gaps were skipped."
            "</div>\n"
        )
    elif ctx.sparse_coverage:
        buf.write(
            f'<div class="callout callout--info"><strong>🗺️ Sparse-coverage territory.</strong>'
            f"HuggingFace returned {ctx.n_candidates} candidate(s) for this brief. "
            "The decomposition + coverage gaps below are the actual sourcing roadmap — "
            "most of the data you need for this detection program lives outside HF."
            "</div>\n"
        )
    elif ctx.no_direct_fits:
        buf.write(
            '<div class="callout callout--proxy"><strong>🧩 No direct-fit corpora — every candidate is a reframing.</strong>'
            "Strategy assessment ran successfully; review the reframings "
            "(signal_proxy / cross_class_repurposing / benign_baseline) before committing. "
            "Direct ground-truth corpora for this brief don't appear to exist on HuggingFace yet."
            "</div>\n"
        )
    elif ctx.has_strategies:
        buf.write(
            f'<div class="callout callout--good"><strong>Strategy-assessed.</strong>'
            f"Search expanded across the original brief and {ctx.n_directions} related "
            "direction(s); shortlisted candidates were assessed for direct fits and reframings. "
            "Listed in best-strategy order; review the strategy + caveats before committing."
            "</div>\n"
        )
    elif ctx.has_decomposition:
        buf.write(
            f'<div class="callout callout--info"><strong>Discovery + decomposition.</strong>'
            f"Search expanded across the original brief and {ctx.n_directions} related "
            "direction(s) proposed by the LLM."
            "</div>\n"
        )
    else:
        buf.write(
            '<div class="callout callout--info"><strong>Pre-fit metadata screening.</strong>'
            "Candidates are returned in source/search relevance order. "
            "Probe outputs are annotations, not a ranking score."
            "</div>\n"
        )


# (No ASCII wordmark — see _write_header for the crisp typographic header.)


def _write_brief_section(buf: StringIO, result: ReconResult) -> None:
    """Render the brief as a prominent quote-styled hero block.

    This is the second visual focus of the page (after the wordmark)
    so users immediately see what the run was about. Detection target,
    threat families, languages render as inline metadata pills below.
    """
    intent = result.intent
    buf.write('<section class="brief-hero">\n')
    buf.write('  <span class="brief-hero__label">BRIEF</span>\n')
    buf.write(
        f'  <blockquote class="brief-hero__text">{escape(intent.raw_brief)}</blockquote>\n'
    )
    meta_pills: list[str] = []
    if intent.detection_target:
        target = intent.detection_target
        if len(target) > 80:
            target = target[:77].rstrip(" ,;:") + "…"
        meta_pills.append(
            f'<span class="brief-hero__pill"><span class="brief-hero__pill-label">target</span>'
            f' {escape(target)}</span>'
        )
    if intent.threat_families:
        meta_pills.append(
            f'<span class="brief-hero__pill"><span class="brief-hero__pill-label">threats</span>'
            f' {escape(", ".join(intent.threat_families))}</span>'
        )
    if intent.languages:
        meta_pills.append(
            f'<span class="brief-hero__pill"><span class="brief-hero__pill-label">languages</span>'
            f' {escape(", ".join(intent.languages))}</span>'
        )
    if intent.deployment_context:
        meta_pills.append(
            f'<span class="brief-hero__pill"><span class="brief-hero__pill-label">deployment</span>'
            f' {escape(intent.deployment_context)}</span>'
        )
    if meta_pills:
        buf.write('  <div class="brief-hero__meta">' + "".join(meta_pills) + "</div>\n")
    buf.write("</section>\n")


def _write_gaps_section(buf: StringIO, result: ReconResult, *, lead: bool) -> None:
    """Render the sourcing roadmap as a scannable table.

    Per UX feedback: cognition is faster on a table than on a stack of
    h3 + prose blocks. Three columns: aspect, what's missing, concrete
    next step. The last column is the actionable bit; the row links
    aspect-name → suggestion → action.
    """
    if result.coverage is None:
        return
    title = "Sourcing roadmap" if lead else "Coverage gaps"
    buf.write(f"<h2>{title}</h2>\n")
    if lead:
        buf.write(
            "<p>Where the data is — and isn't. Specific aspects of your brief that "
            "the candidate set doesn't cover, with concrete next steps to close "
            "each gap:</p>\n"
        )
    else:
        buf.write("<p>Aspects worth augmenting:</p>\n")
    buf.write(
        '<table class="gaps-table">\n'
        "<thead><tr>"
        '<th class="gaps-table__aspect">Aspect</th>'
        '<th class="gaps-table__gap">What\'s missing</th>'
        '<th class="gaps-table__step">Next step</th>'
        "</tr></thead>\n<tbody>\n"
    )
    for gap in result.coverage.semantic_gaps:
        buf.write(
            "<tr>"
            f'<td class="gaps-table__aspect"><b>{escape(gap.aspect)}</b></td>'
            f'<td class="gaps-table__gap">{escape(gap.description)}</td>'
            f'<td class="gaps-table__step">{escape(gap.suggestion)}</td>'
            "</tr>\n"
        )
    buf.write("</tbody></table>\n")


def _write_decomposition_section(buf: StringIO, result: ReconResult) -> None:
    """Decomposition section, collapsed by default.

    Important investigative material but not first-read content. Users
    open it when they want to know which directions surfaced a given
    candidate, or when iterating on the brief.
    """
    if result.coverage is None:
        return
    buf.write(
        '<details class="collapsed-section">\n'
        f'<summary><h2 class="inline-h2">Decomposition '
        f'<span class="muted small">· {len(result.coverage.decomposition)} directions</span>'
        "</h2></summary>\n"
    )
    buf.write(
        "<p>The LLM proposed these search directions in addition to the original brief:</p>\n<ul>\n"
    )
    for d in result.coverage.decomposition:
        buf.write(f"<li><b>{escape(d.name)}</b> — {escape(d.rationale)}")
        if d.keywords:
            buf.write(
                f"<br><span class=\"muted\">keywords: <code>{escape(', '.join(d.keywords))}</code></span>"
            )
        if d.expected_finds:
            buf.write(
                f'<br><span class="muted">expected: {escape(d.expected_finds)}</span>'
            )
        buf.write("</li>\n")
    buf.write("</ul>\n</details>\n")


def _write_run_summary_compact(
    buf: StringIO, result: ReconResult, ctx: ReconReportContext
) -> None:
    """Compact one-line run summary at the top of the report.

    Bubbles the headline numbers up so a reader knows in 2 seconds
    what the run produced. The legacy full Run summary section is
    expanded only if the reader expands the meta details footer below.
    """
    bits: list[str] = []
    if ctx.has_strategies:
        bits.append(
            f"<b>{ctx.n_strategy_assessed}</b> of <b>{ctx.n_candidates}</b> assessed"
        )
    else:
        bits.append(f"<b>{ctx.n_candidates}</b> candidates")
    if ctx.n_directions:
        bits.append(f"<b>{ctx.n_directions}</b> directions")
    if ctx.n_papers:
        bits.append(f"<b>{ctx.n_papers}</b> papers")
    bits.append(f"{result.elapsed_seconds:.1f}s")
    sources = ", ".join(result.sources_searched) or "(none)"
    buf.write(
        '<aside class="run-summary-compact">\n'
        f"  <span>{' · '.join(bits)}</span>\n"
        f'  <span class="muted small">via {escape(sources)}</span>\n'
        "</aside>\n"
    )
    if result.notices:
        buf.write(
            '<details class="notices-details">\n'
            f'<summary class="muted small">{len(result.notices)} '
            f'notice{"s" if len(result.notices) != 1 else ""}</summary>\n<ul>\n'
        )
        for n in result.notices:
            buf.write(f'<li class="notice">{escape(n)}</li>\n')
        buf.write("</ul>\n</details>\n")


def _write_run_summary(buf: StringIO, result: ReconResult, ctx: ReconReportContext) -> None:
    buf.write("<h2>Run summary</h2>\n")
    buf.write('<div class="run-summary"><ul>\n')
    buf.write(
        f'<li>Sources searched: {escape(", ".join(result.sources_searched) or "(none)")}</li>\n'
    )
    buf.write(f"<li>Candidates returned: <b>{ctx.n_candidates}</b></li>\n")
    if ctx.has_strategies:
        buf.write(f"<li>Strategy-assessed: <b>{ctx.n_strategy_assessed}</b></li>\n")
    buf.write(f"<li>Wall-clock: {result.elapsed_seconds:.2f}s</li>\n")
    buf.write(f"<li>dataset-scout version: <code>{escape(result.scout_version)}</code></li>\n")
    buf.write("</ul>\n")
    if result.notices:
        buf.write('<div class="notices"><b>Notices:</b><ul>\n')
        for n in result.notices:
            buf.write(f'<li class="notice">{escape(n)}</li>\n')
        buf.write("</ul></div>\n")
    buf.write("</div>\n")


def _write_at_a_glance(buf: StringIO, ctx: ReconReportContext) -> None:
    """Top-level scoreboard: counts per strategy bucket as bold pill badges."""
    buf.write('<section class="scoreboard">\n<h2>At a glance</h2>\n')
    buf.write('<div class="scoreboard__pills">\n')
    any_pill = False
    for group in ctx.groups:
        if group.count == 0:
            continue
        any_pill = True
        buf.write(
            f'<span class="pill pill--{group.key}">'
            f'<b>{group.count}</b> {escape(group.label)}'
            "</span>"
        )
    if not any_pill:
        buf.write("<i>No strategy assessments produced for this run.</i>")
    buf.write("\n</div>\n")
    if ctx.no_direct_fits:
        buf.write(
            '<p class="scoreboard__note">'
            "No direct fits — every candidate is a reframing. Review the "
            "rationale + caveats below before committing."
            "</p>\n"
        )
    buf.write("</section>\n")


def _write_candidates_section(
    buf: StringIO, result: ReconResult, ctx: ReconReportContext
) -> None:
    if ctx.has_strategies:
        # Per UX feedback: render the actually-useful groups as full
        # cards, then collapse the "not useful / unassessed" bucket
        # to a single one-liner. The full list is still in
        # results.json for anyone debugging.
        not_useful_count = 0
        for group in ctx.groups:
            if group.count == 0:
                continue
            if group.key == "not_useful":
                not_useful_count = group.count
                continue
            _write_strategy_group(buf, group)
        if not_useful_count > 0:
            buf.write(
                '<aside class="not-useful-note">\n'
                f'  <span class="not-useful-note__count">{not_useful_count}</span>\n'
                "  <span>"
                "additional candidate(s) judged "
                "<b>not useful</b> by the strategy assessor or returned "
                "without an assessment. Hidden from the report to keep the "
                "signal-to-noise ratio high; full list in "
                "<code>results.json</code>."
                "</span>\n"
                "</aside>\n"
            )
        return

    # Fallback: pre-strategy mode (metadata-only / discovery-only).
    buf.write("<h2>Candidates</h2>\n")
    if ctx.has_decomposition:
        buf.write(
            "<p>Listed in search-relevance order, deduped across the original brief "
            "and decomposition directions. The <code>surfaced by</code> annotation "
            "shows which direction(s) found each candidate.</p>\n"
        )
    else:
        buf.write(
            "<p>Listed in <b>search-relevance order from the source</b>. "
            "This is not a fitness ranking.</p>\n"
        )
    for i, sc in enumerate(result.candidates, start=1):
        _write_candidate(buf, i, sc, verdict=None)


def _write_strategy_group(buf: StringIO, group: StrategyGroup) -> None:
    """Render one bucket of cards as a section with pill-styled heading."""
    buf.write(
        f'<section class="group group--{group.key}">\n'
        f'<header class="group__head">'
        f'<h2 class="group__title">{escape(group.label)} '
        f'<span class="group__count">· {group.count}</span></h2>'
        f'<p class="group__desc">{escape(group.description)}</p>'
        f"</header>\n"
    )
    for card in group.cards:
        _write_candidate(buf, card.rank, card.scorecard, verdict=card.verdict)
    buf.write("</section>\n")


def _write_candidate(
    buf: StringIO,
    index: int,
    sc: Scorecard,
    *,
    verdict: CardVerdict | None,
) -> None:
    """Render one candidate card — minimal, scannable.

    Visible by default (the at-a-glance read):
      - Header: clickable link to the dataset card. Wraps rank +
        verdict pill + dataset id.
      - Lede: one-liner pulled from the best strategy's rationale.
      - Use-as: practical "what to do with it" guidance.
      - Compact meta line: license · size · freshness · surfaced by.

    Collapsed in <details> (the deep-dive read):
      - Per-strategy rationale, caveats, and transform spec.
      - Probe signals (cheap + label-intent-fit).
      - Description, revision, requires-auth flag (low priority).

    Per user feedback: "the most important takeaways should be summary
    results... clean and direct representation of datasets... heading
    clickable link to the dataset itself, then a summary analysis,
    then the rest of the coverage analysis."
    """
    cand = sc.candidate
    md = cand.metadata

    kind_class = verdict.primary_kind if verdict and verdict.primary_kind else "unscored"
    buf.write(f'<article class="candidate candidate--{kind_class}">\n')

    # ─── Header: clickable, links to the dataset card ──────────
    title_text = f"{escape(cand.source)}:{escape(cand.id)}"
    head_inner_parts: list[str] = [
        f'<span class="candidate__id">{title_text}</span>',
    ]
    if verdict is not None:
        head_inner_parts.append(
            f'<span class="candidate__verdict candidate__verdict--{kind_class}">'
            f"{escape(verdict.headline)}</span>"
        )
    head_inner_parts.append(f'<span class="candidate__rank">#{index}</span>')
    head_inner = "".join(head_inner_parts)

    if md.card_url:
        # Whole header becomes the link — bigger click target than just
        # the id text. external icon hint via CSS::after.
        buf.write(
            f'<header class="candidate__head">\n'
            f'<a class="candidate__title-link" href="{escape(md.card_url)}" '
            f'target="_blank" rel="noopener">{head_inner}</a>\n'
            f"</header>\n"
        )
    else:
        buf.write(f'<header class="candidate__head">{head_inner}</header>\n')

    # ─── Summary analysis (always visible) ─────────────────────
    if verdict is not None and verdict.one_liner:
        buf.write(f'<p class="candidate__lede">{escape(verdict.one_liner)}</p>\n')

    if verdict is not None and verdict.confidence is not None:
        buf.write(
            f'<p class="candidate__use-as"><b>Use as:</b> {escape(verdict.use_as)}</p>\n'
        )

    # ─── Compact meta one-liner (always visible) ────────────────
    # License · size · freshness · surfaced by — separated by middots,
    # not a bulleted list. Less visual weight than the old snapshot
    # block, gets the key facts in one scan.
    meta_bits = list(_render_plain_signals(sc))
    if cand.surfaced_by:
        meta_bits.append(f'surfaced by: {", ".join(cand.surfaced_by)}')
    if cand.requires_auth:
        meta_bits.append("🔒 gated")
    if meta_bits:
        buf.write(
            '<p class="candidate__meta">'
            + " · ".join(escape(b) for b in meta_bits)
            + "</p>\n"
        )

    # ─── Collapsible deep-dive (rationale + caveats + probes) ───
    has_details = bool(sc.strategies) or bool(sc.cheap_probes) or bool(md.description)
    if has_details:
        n_strats = len(sc.strategies)
        summary_label = (
            "Strategies, caveats & transform"
            + (f" · {n_strats}" if n_strats else "")
        )
        buf.write(
            '<details class="candidate__details">\n'
            f'<summary>{summary_label}</summary>\n'
        )
        if md.description:
            desc = md.description.strip().splitlines()[0][:300]
            buf.write(f'<p class="candidate__desc">{escape(desc)}</p>\n')
        if cand.revision:
            buf.write(
                f'<p class="muted small">revision <code>{escape(cand.revision[:12])}</code></p>\n'
            )
        if sc.strategies:
            _write_strategies(buf, sc.strategies)
        _write_probe_signals(buf, sc)
        buf.write("</details>\n")

    buf.write("</article>\n")


def _render_plain_signals(sc: Scorecard) -> list[str]:
    """Plain-text snapshot signals: license, size, freshness, languages.

    Mirrors markdown_report._render_plain_signals. No colour codes;
    license is stated factually.
    """
    bits: list[str] = []
    license_sub = sc.cheap_probes.get("license")
    if license_sub is not None:
        spdx = _evidence_detail(license_sub, "license_spdx")
        match = _evidence_detail(license_sub, "policy_match")
        if spdx and match == "allow":
            bits.append(f"license: {spdx} (in policy)")
        elif spdx and match == "warn_only":
            bits.append(f"license: {spdx} (review)")
        elif spdx:
            bits.append(f"license: {spdx} (outside policy)")
        elif license_sub.status == "low_confidence":
            raw = _evidence_detail(license_sub, "license_raw") or "?"
            bits.append(f"license: {raw} (unknown SPDX)")
        else:
            bits.append("license: missing")

    size_sub = sc.cheap_probes.get("size")
    if size_sub is not None and size_sub.status == "ok":
        rows = _evidence_detail(size_sub, "rows")
        bytes_ = _evidence_detail(size_sub, "bytes")
        downloads = _evidence_detail(size_sub, "downloads")
        if rows:
            bits.append(f"{rows} rows")
        elif bytes_:
            bits.append(f"{bytes_} bytes")
        if downloads:
            bits.append(f"{downloads} downloads")

    fresh_sub = sc.cheap_probes.get("freshness")
    if fresh_sub is not None and fresh_sub.status == "ok":
        bucket = _evidence_detail(fresh_sub, "bucket") or "?"
        bits.append(f"freshness: {bucket}")

    lang_sub = sc.cheap_probes.get("languages")
    if lang_sub is not None and lang_sub.status == "ok":
        declared = _evidence_detail(lang_sub, "declared") or "?"
        bits.append(f"languages: {declared}")

    if sc.label_intent_fit is not None and sc.label_intent_fit.value is not None:
        bits.append(f"semantic fit: {sc.label_intent_fit.value:.2f}")

    return bits


def _write_recipe_preview(buf: StringIO, ctx: ReconReportContext) -> None:
    """End-of-report bridge: what `curate` would produce, and how to run it."""
    rp = ctx.recipe_preview
    if rp is None:
        return
    buf.write('<section class="recipe-preview">\n<h2>Next steps — recipe & curate</h2>\n')
    buf.write(
        f"<p><b>{rp.n_components} component(s)</b> would land in the draft recipe "
        "(min strategy confidence ≥ 0.5):</p>\n<ul>\n"
    )
    if rp.n_direct_fit:
        buf.write(f"<li><b>{rp.n_direct_fit}</b> direct-fit</li>\n")
    if rp.n_reframing:
        buf.write(f"<li><b>{rp.n_reframing}</b> reframing</li>\n")
    if rp.n_proxy:
        buf.write(f"<li><b>{rp.n_proxy}</b> signal proxy</li>\n")
    if rp.n_benign:
        buf.write(f"<li><b>{rp.n_benign}</b> benign baseline</li>\n")
    buf.write(
        f"</ul>\n<p>Estimated row count: <b>~{rp.estimated_rows:,}</b> "
        "(after the 5,000-row auto-cap on <code>take: all</code> components).</p>\n"
    )
    if rp.components:
        buf.write(
            '<table class="recipe-preview__table">\n'
            "<thead><tr>"
            "<th>#</th><th>Component</th><th>Strategy</th>"
            "<th>Confidence</th><th>Take</th><th>Label kind</th>"
            "</tr></thead>\n<tbody>\n"
        )
        for i, comp in enumerate(rp.components, start=1):
            take_str = str(comp.take) if comp.take is not None else "all"
            label = comp.label_kind or "—"
            buf.write(
                f"<tr><td>{i}</td>"
                f"<td><code>{escape(comp.candidate_id)}</code></td>"
                f"<td>{escape(comp.primary_kind)}</td>"
                f"<td>{comp.confidence:.2f}</td>"
                f"<td>{escape(take_str)}</td>"
                f"<td>{escape(label)}</td></tr>\n"
            )
        buf.write("</tbody></table>\n")
    buf.write(
        f'<p><b>Run:</b></p>\n<pre><code>{escape(rp.next_command)}</code></pre>\n'
        "<p>You'll get <code>train.jsonl</code> / <code>val.jsonl</code> / "
        "<code>test.jsonl</code> (leakage-aware splits), "
        "<code>recipe.lock.yaml</code> (audit trail), "
        "<code>report.md</code> (5-second scorecard), and "
        "<code>usage.md</code> (3-line snippets for HF datasets / pandas / raw JSONL).</p>\n"
        "<p>Need to refine first? Edit <code>recipe.draft.yaml</code> "
        "(drop weak components, tune <code>take</code>, add "
        "<code>filter</code> expressions), or re-run with "
        "<code>datascout recon ... --review</code> to edit the search "
        "directions before discovery.</p>\n"
        "</section>\n"
    )


def _write_strategies(buf: StringIO, strategies: list[Strategy]) -> None:
    buf.write('<div class="strategies">\n')
    for s in strategies:
        kind_class = s.kind.value
        buf.write(f'<div class="strategy strategy--{kind_class}">\n')
        buf.write(
            f'<div class="strategy__head">'
            f'<span class="strategy__kind">{escape(_strategy_label(s.kind))}</span>'
            f'<span class="strategy__conf">confidence {s.confidence:.2f}</span>'
            f"</div>\n"
        )
        buf.write(f'<div class="strategy__rationale">{escape(s.rationale)}</div>\n')
        if s.caveats:
            buf.write('<ul class="strategy__caveats">\n')
            for cav in s.caveats:
                buf.write(f"<li>⚠ {escape(cav)}</li>\n")
            buf.write("</ul>\n")
        bits = _transform_bits(s)
        if bits:
            buf.write(f'<div class="strategy__transform">transform: {bits}</div>\n')
        buf.write("</div>\n")
    buf.write("</div>\n")


def _strategy_label(kind: StrategyKind) -> str:
    return {
        StrategyKind.DIRECT_USE: "✅ direct use",
        StrategyKind.SUBSET_EXTRACTION: "🔎 subset extraction",
        StrategyKind.LABEL_REMAPPING: "🔁 label remapping",
        StrategyKind.CROSS_CLASS_REPURPOSING: "🔀 cross-class repurposing",
        StrategyKind.SIGNAL_PROXY: "📡 signal proxy",
        StrategyKind.BENIGN_BASELINE: "🧊 benign baseline",
        StrategyKind.COMPOSITION_ONLY: "🧩 composition-only",
        StrategyKind.NOT_USEFUL: "❌ not useful",
    }.get(kind, kind.value)


def _transform_bits(s: Strategy) -> str:
    t = s.transform
    bits: list[str] = []
    if t.text_column:
        bits.append(f"text=<code>{escape(t.text_column)}</code>")
    if t.label_column:
        bits.append(f"label=<code>{escape(t.label_column)}</code>")
    if t.label_value_map:
        kv = ", ".join(
            f"<code>{escape(k)}</code>→{escape(v)}" for k, v in t.label_value_map.items()
        )
        bits.append("values={" + kv + "}")
    if t.filter:
        bits.append(f"filter=<code>{escape(t.filter)}</code>")
    if t.take != "all":
        bits.append(f"take={escape(str(t.take))}")
    return " · ".join(bits)


# ─── badges (parallel to markdown_report._render_badges) ───────────


def _evidence_detail(sub: SubScore, kind: str) -> str | None:
    for e in sub.evidence:
        if e.kind == kind:
            return e.detail
    return None


def _render_badges(sc: Scorecard) -> list[tuple[str, str]]:
    """(css-class, text) tuples for the candidate's badges."""
    badges: list[tuple[str, str]] = []
    license_sub = sc.cheap_probes.get("license")
    if license_sub is not None:
        match = _evidence_detail(license_sub, "policy_match")
        spdx = _evidence_detail(license_sub, "license_spdx")
        if spdx and match == "allow":
            badges.append(("good", f"license: {spdx} ✅"))
        elif spdx and match == "warn_only":
            badges.append(("warn", f"license: {spdx} ⚠️"))
        elif spdx:
            badges.append(("bad", f"license: {spdx} ❗"))
        elif license_sub.status == "low_confidence":
            raw = _evidence_detail(license_sub, "license_raw") or "?"
            badges.append(("warn", f"license: {raw} (?)"))
        else:
            badges.append(("warn", "license: missing"))

    fresh_sub = sc.cheap_probes.get("freshness")
    if fresh_sub is not None and fresh_sub.status == "ok":
        bucket = _evidence_detail(fresh_sub, "bucket") or "?"
        badges.append(("", f"freshness: {bucket}"))

    size_sub = sc.cheap_probes.get("size")
    if size_sub is not None and size_sub.status == "ok":
        rows = _evidence_detail(size_sub, "rows")
        bytes_ = _evidence_detail(size_sub, "bytes")
        downloads = _evidence_detail(size_sub, "downloads")
        if rows:
            badges.append(("", f"rows: {rows}"))
        elif bytes_:
            badges.append(("", f"bytes: {bytes_}"))
        if downloads:
            badges.append(("", f"downloads: {downloads}"))

    lang_sub = sc.cheap_probes.get("languages")
    if lang_sub is not None and lang_sub.status == "ok":
        declared = _evidence_detail(lang_sub, "declared") or "?"
        badges.append(("", f"languages: {declared}"))

    if sc.label_intent_fit is not None and sc.label_intent_fit.value is not None:
        v = sc.label_intent_fit.value
        cls = "good" if v >= 0.6 else ("warn" if v >= 0.4 else "")
        badges.append((cls, f"semantic fit: {v:.2f}"))

    return badges


def _write_probe_signals(buf: StringIO, sc: Scorecard) -> None:
    """Compact per-probe details below badges. Kept brief; the report
    leads with strategies — probe signals are supporting evidence."""
    if not sc.cheap_probes and sc.label_intent_fit is None:
        return
    buf.write('<dl class="probe-grid">\n')
    for name, sub in sorted(sc.cheap_probes.items()):
        if sub.status not in ("ok", "low_confidence"):
            continue
        details = "; ".join(
            f"{e.kind}={escape(e.detail)}" for e in sub.evidence[:3]
        )
        if not details:
            continue
        buf.write(f"<dt>{escape(name)}</dt><dd>{details}</dd>\n")
    if sc.label_intent_fit is not None and sc.label_intent_fit.evidence:
        e = sc.label_intent_fit.evidence[0]
        buf.write(
            f"<dt>label_intent_fit</dt><dd>{escape(e.detail)}</dd>\n"
        )
    buf.write("</dl>\n")


def _write_footer(buf: StringIO) -> None:
    buf.write(
        '<footer class="disclaimer">'
        "License signals are an SPDX-best-effort guess. "
        "Always read the upstream card before redistributing. This is not legal advice."
        "</footer>\n"
    )


def _write_papers_section(
    buf: StringIO, result: ReconResult, ctx: ReconReportContext
) -> None:
    """Render the academic-paper discovery section, collapsed by default.

    Important investigative material — papers often cite datasets the
    HF channel missed — but not first-read content. Wrap in <details>
    so it doesn't add scroll-weight to a 100-card report.
    """
    citations = ctx.n_paper_dataset_citations
    citation_label = (
        f" · {citations} dataset citation(s) extracted"
        if citations > 0
        else " · no dataset URLs in abstracts"
    )
    buf.write(
        '<details class="collapsed-section">\n'
        f'<summary><h2 class="inline-h2">Related papers '
        f'<span class="muted small">· {ctx.n_papers}{citation_label}</span>'
        "</h2></summary>\n"
    )
    for p in result.papers:
        _write_paper(buf, p)
    buf.write("</details>\n")


def _write_paper(buf: StringIO, p: PaperReference) -> None:
    venue = p.venue or "?"
    cite_count = (
        f' · {p.citation_count} citation(s)' if p.citation_count else ""
    )
    buf.write('<article class="paper">\n')
    buf.write(f'<h3 class="paper__title"><a href="{escape(p.url)}">{escape(p.title)}</a></h3>\n')
    authors = ", ".join(p.authors[:5]) + ("…" if len(p.authors) > 5 else "")
    buf.write(
        f'<div class="paper__meta"><span class="badge">{escape(venue)} {p.year}</span>'
        f"{escape(cite_count)}</div>\n"
    )
    if authors:
        buf.write(f'<div class="paper__authors">{escape(authors)}</div>\n')
    if p.abstract:
        snippet = p.abstract.strip().replace("\n", " ")[:280]
        if len(p.abstract) > 280:
            snippet += "…"
        buf.write(f'<p class="paper__abstract">{escape(snippet)}</p>\n')
    if p.surfaced_by:
        buf.write(
            f'<div class="paper__surfaced">🧭 surfaced by: '
            f"{escape(', '.join(p.surfaced_by))}</div>\n"
        )
    if p.referenced_datasets:
        buf.write('<div class="paper__citations"><b>Datasets cited:</b><ul>\n')
        for d in p.referenced_datasets:
            buf.write(
                f'<li><code>{escape(d.source)}:{escape(d.identifier)}</code> — '
                f'<a href="{escape(d.url)}">{escape(d.url)}</a></li>\n'
            )
        buf.write("</ul></div>\n")
    buf.write("</article>\n")


__all__ = ["render_recon_report_html", "write_recon_report_html"]

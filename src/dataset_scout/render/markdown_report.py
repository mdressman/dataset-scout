"""Markdown report renderer.

Three framings depending on what ran:

- **Metadata-only.** No LLM configured. Discovery-only language.
- **Discovery + decomposition.** LLM ran the decomposer; candidates
  surfaced from multiple directions but no per-candidate strategies.
- **Strategy-assessed.** Per-candidate strategies + (optional)
  coverage gaps. Report leads with the strongest fits.

Receipts everywhere — every claim links back to a card URL or
specific evidence.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

from dataset_scout.core import (
    Evidence,
    PaperReference,
    ReconResult,
    Scorecard,
    Strategy,
    StrategyKind,
    SubScore,
)
from dataset_scout.render._view import CardVerdict, ReconReportContext


def write_recon_report(result: ReconResult, out_dir: Path) -> Path:
    """Render and write `<out_dir>/report.md`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "report.md"
    target.write_text(render_recon_report(result), encoding="utf-8")
    return target


def _has_strategies(result: ReconResult) -> bool:
    return any(sc.strategies for sc in result.candidates)


def render_recon_report(result: ReconResult) -> str:
    """Render a ReconResult to a Markdown string."""
    ctx = ReconReportContext.from_result(result)
    buf = StringIO()
    intent = result.intent
    metadata_only = ctx.metadata_only
    llm_runtime_error = ctx.llm_runtime_error
    has_strategies = ctx.has_strategies
    has_decomposition = ctx.has_decomposition
    has_gaps = ctx.has_gaps
    notable_gaps = ctx.notable_gaps
    sparse_coverage = ctx.sparse_coverage
    no_direct_fits = ctx.no_direct_fits

    buf.write("# dataset-scout recon report\n\n")
    if metadata_only:
        buf.write(
            "> ⚠️ **Metadata-only mode.**  \n"
            "> No LLM provider is configured, so decomposition, strategy\n"
            "> assessment, and coverage gaps were skipped. To enable them,\n"
            "> copy `.env.example` to `.env` and set `DATASET_SCOUT_MODEL`\n"
            "> (e.g. `github_copilot/gpt-5-mini` or `github/gpt-4o-mini`),\n"
            "> or set `AZURE_OPENAI_ENDPOINT` + `AZURE_OPENAI_DEPLOYMENT`\n"
            "> and run `az login`.\n\n"
        )
    elif llm_runtime_error:
        buf.write(
            "> ⚠️ **LLM call failed — running in metadata-only mode.**  \n"
            "> The LLM provider was configured but a call failed at runtime\n"
            "> (model id, credential, network, or quota — see notices\n"
            "> below). Decomposition, strategy assessment, and coverage\n"
            "> gaps were skipped.\n\n"
        )
    elif sparse_coverage:
        buf.write(
            "> 🗺️ **Sparse-coverage territory.**  \n"
            f"> HuggingFace returned {len(result.candidates)} candidate(s) "
            "for this brief. The decomposition + coverage gaps below\n"
            "> are the actual sourcing roadmap — most of the data you need\n"
            "> for this detection program lives outside HF.\n\n"
        )
    elif no_direct_fits:
        buf.write(
            "> 🧩 **No direct-fit corpora — every candidate is a reframing.**  \n"
            "> Strategy assessment ran successfully; review the reframings\n"
            "> (signal_proxy / cross_class_repurposing / benign_baseline) before\n"
            "> committing. Direct ground-truth corpora for this brief don't\n"
            "> appear to exist on HuggingFace yet.\n\n"
        )
    elif has_strategies:
        n_dirs = len(result.coverage.decomposition) if result.coverage else 0
        buf.write(
            "> **Strategy-assessed.**  \n"
            "> Search expanded across the original brief and "
            f"{n_dirs} related direction(s); shortlisted candidates were "
            "assessed for direct fits and reframings. Listed here in\n"
            "> best-strategy order; review the strategy + caveats before "
            "committing.\n\n"
        )
    elif has_decomposition and result.coverage is not None:
        buf.write(
            "> **Discovery + decomposition.**  \n"
            "> Search expanded across the original brief and "
            f"{len(result.coverage.decomposition)} related direction(s) "
            "proposed by the LLM.\n\n"
        )
    else:
        buf.write(
            "> **Pre-fit metadata screening.**  \n"
            "> Candidates are returned in source/search relevance order.\n"
            "> Probe outputs are annotations, not a ranking score.\n\n"
        )

    buf.write("## Brief\n\n")
    buf.write(f"**Raw brief:** {intent.raw_brief}\n\n")
    if intent.detection_target:
        buf.write(f"**Detection target:** {intent.detection_target}\n\n")
    if intent.threat_families:
        buf.write(f"**Threat families:** {', '.join(intent.threat_families)}\n\n")
    buf.write(f"**Languages requested:** {', '.join(intent.languages)}\n\n")

    # ─── Sourcing roadmap (LEAD section when gaps are present) ──
    # Per recommendation A: gaps + decomposition together are the
    # primary deliverable for novel briefs. Lead with them.
    if has_gaps and result.coverage and (notable_gaps or sparse_coverage):
        buf.write("## Sourcing roadmap\n\n")
        buf.write(
            "Where the data is — and isn't. The LLM identified specific aspects\n"
            "of your brief that the candidate set doesn't cover, with concrete\n"
            "next steps:\n\n"
        )
        for gap in result.coverage.semantic_gaps:
            buf.write(f"### {gap.aspect}\n\n")
            buf.write(f"{gap.description}\n\n")
            buf.write(f"**→ Next step:** {gap.suggestion}\n\n")

    # ─── Decomposition audit ────────────────────────────────────
    if has_decomposition and result.coverage:
        buf.write("## Decomposition\n\n")
        buf.write("The LLM proposed these search directions in addition to the original brief:\n\n")
        for d in result.coverage.decomposition:
            buf.write(f"- **{d.name}** — {d.rationale}\n")
            if d.keywords:
                buf.write(f"  - keywords: `{', '.join(d.keywords)}`\n")
            if d.expected_finds:
                buf.write(f"  - expected: {d.expected_finds}\n")
        buf.write("\n")

    if result.candidates:
        if has_strategies:
            _render_at_a_glance(buf, ctx)
            _render_grouped_candidates(buf, ctx)
        elif has_decomposition:
            buf.write("## Candidates\n\n")
            buf.write(
                "Listed in search-relevance order, deduped across the "
                "original brief and decomposition directions. The "
                "`surfaced by` annotation\n"
                "on each candidate shows which direction(s) found it.\n\n"
            )
            for i, sc in enumerate(result.candidates, start=1):
                _render_candidate(buf, i, sc, verdict=None)
        else:
            buf.write("## Candidates\n\n")
            buf.write(
                "Listed in **search-relevance order from the source.** "
                "This is not a fitness ranking — embedding fit and the "
                "strategy assessor land in a follow-up milestone.\n\n"
            )
            for i, sc in enumerate(result.candidates, start=1):
                _render_candidate(buf, i, sc, verdict=None)
    else:
        buf.write(
            "_No HuggingFace candidates were returned — see notices "
            "above. The decomposition, coverage gaps, and related papers "
            "below are your sourcing roadmap; for novel territory, the "
            "data often lives outside HF (academic repositories, vendor "
            "telemetry, web archives)._\n\n"
        )

    # When gaps exist but weren't notable enough to lead the report,
    # render them after candidates. Also covers the empty-candidates
    # path: the gaps section runs regardless.
    if (
        result.coverage
        and result.coverage.semantic_gaps
        and not notable_gaps
        and not sparse_coverage
    ):
        buf.write("## Coverage gaps\n\n")
        buf.write("Aspects worth augmenting:\n\n")
        for gap in result.coverage.semantic_gaps:
            buf.write(f"- **{gap.aspect}** — {gap.description}\n")
            buf.write(f"  - *Suggestion:* {gap.suggestion}\n")
        buf.write("\n")

    # ─── Related papers + dataset citations (academic channel) ──
    if ctx.show_papers:
        _render_papers_section(buf, result, ctx)

    # ─── Next steps: recipe + curate preview ────────────────────
    if ctx.show_recipe_preview:
        _render_recipe_preview(buf, ctx)

    # ─── Run summary (footer-position) ──────────────────────────
    buf.write("## Run summary\n\n")
    buf.write(f"- Sources searched: {', '.join(result.sources_searched) or '(none)'}\n")
    buf.write(f"- Candidates returned: **{len(result.candidates)}**\n")
    if has_strategies:
        assessed = sum(1 for sc in result.candidates if sc.strategies)
        buf.write(f"- Strategy-assessed: **{assessed}**\n")
    buf.write(f"- Wall-clock: {result.elapsed_seconds:.2f}s\n")
    buf.write(f"- dataset-scout version: {result.scout_version}\n")
    if result.notices:
        buf.write("\n### Notices\n\n")
        for n in result.notices:
            buf.write(f"- {n}\n")
    buf.write("\n")

    buf.write("\n---\n\n")
    buf.write(
        "_License signals are an SPDX-best-effort guess. Always read the "
        "upstream card before redistributing. This is not legal advice._\n"
    )
    return buf.getvalue()


def _render_at_a_glance(buf: StringIO, ctx: ReconReportContext) -> None:
    """Top-level scoreboard: how many cards in each strategy bucket.

    Lets users see at a glance whether to expect direct fits, only
    reframings, or a sourcing-roadmap outcome — without scrolling.
    """
    buf.write("## At a glance\n\n")
    counts = [(g.label, g.count) for g in ctx.groups if g.count > 0]
    if not counts:
        buf.write("_No strategy assessments produced for this run._\n\n")
        return
    line = "  ·  ".join(f"**{n}** {label}" for label, n in counts)
    buf.write(line + "\n\n")
    if ctx.no_direct_fits:
        buf.write(
            "_No direct fits — every candidate is a reframing. Review the "
            "rationale + caveats below before committing._\n\n"
        )


def _render_grouped_candidates(buf: StringIO, ctx: ReconReportContext) -> None:
    """Render scorecards grouped by strategy kind.

    Direct fits → reframings → proxies → benign → not-useful. Each
    section has a one-line description and lists its cards in
    descending-confidence order.
    """
    for group in ctx.groups:
        if group.count == 0:
            continue
        buf.write(f"## {group.label}  ·  {group.count}\n\n")
        buf.write(f"_{group.description}_\n\n")
        for card in group.cards:
            _render_candidate(buf, card.rank, card.scorecard, verdict=card.verdict)


def _render_candidate(
    buf: StringIO,
    index: int,
    sc: Scorecard,
    *,
    verdict: CardVerdict | None,
) -> None:
    """Render one candidate as a Markdown card.

    When `verdict` is supplied (strategy-assessed runs), the header
    bubbles up the at-a-glance answer ("Direct fit (strong, 0.85)")
    and a one-line use-as guide, so users don't have to read three
    nested strategy bullets to decide whether to use the dataset.
    """
    cand = sc.candidate
    meta = cand.metadata

    # ─── Header ────────────────────────────────────────────────
    if verdict is not None:
        # New: verdict-led header.
        buf.write(f"### #{index} — {verdict.headline}  ·  `{cand.source}:{cand.id}`\n\n")
        if verdict.one_liner:
            buf.write(f"> {verdict.one_liner}\n\n")
    else:
        # Fallback for non-assessed runs.
        buf.write(f"### {index}. `{cand.source}:{cand.id}`\n\n")

    # ─── Verdict block (only when assessed) ────────────────────
    if verdict is not None and verdict.confidence is not None:
        buf.write(f"**Use as:** {verdict.use_as}\n\n")

    # ─── Snapshot block (everything pragmatic in one place) ───
    snapshot_lines: list[str] = []
    if meta.description:
        desc = meta.description.strip().splitlines()[0][:200]
        snapshot_lines.append(f"📝 {desc}")
    if meta.card_url:
        snapshot_lines.append(f"🔗 [{meta.card_url}]({meta.card_url})")
    if cand.revision:
        snapshot_lines.append(f"📌 revision `{cand.revision[:12]}`")

    # Plain-text license + size + freshness signals (no colour codes —
    # license is a yes/no/needs-review fact, not a quality dial).
    plain_signals = list(_render_plain_signals(sc))
    if plain_signals:
        snapshot_lines.append(" · ".join(plain_signals))

    if cand.requires_auth:
        snapshot_lines.append("🔒 gated — requires authentication")

    if cand.surfaced_by:
        snapshot_lines.append(f"🧭 surfaced by: {', '.join(cand.surfaced_by)}")

    for line in snapshot_lines:
        buf.write(f"- {line}\n")
    if snapshot_lines:
        buf.write("\n")

    # ─── Strategy detail (existing rationale + caveats + transform) ──
    if sc.strategies:
        _render_strategies(buf, sc.strategies)

    buf.write("\n")
    _render_probe_signals(buf, sc)
    buf.write("\n")


def _render_plain_signals(sc: Scorecard) -> list[str]:
    """Plain-text snapshot signals: license, size, freshness, languages.

    No colour codes. License is a yes/no/needs-review fact, not a
    quality dial — green/orange badges add confusion, per user feedback.
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


def _render_recipe_preview(buf: StringIO, ctx: ReconReportContext) -> None:
    """End-of-report bridge: what `curate` would produce, and how to run it."""
    rp = ctx.recipe_preview
    if rp is None:
        return
    buf.write("## Next steps — recipe & curate\n\n")
    buf.write(
        f"**{rp.n_components} component(s)** would land in the draft recipe "
        f"(min strategy confidence ≥ 0.5):\n\n"
    )
    if rp.n_direct_fit:
        buf.write(f"- **{rp.n_direct_fit}** direct-fit\n")
    if rp.n_reframing:
        buf.write(f"- **{rp.n_reframing}** reframing\n")
    if rp.n_proxy:
        buf.write(f"- **{rp.n_proxy}** signal proxy\n")
    if rp.n_benign:
        buf.write(f"- **{rp.n_benign}** benign baseline\n")
    buf.write(
        f"\nEstimated row count: **~{rp.estimated_rows:,}** "
        "(after the 5,000-row auto-cap on `take: all` components).\n\n"
    )
    if rp.components:
        buf.write("| # | Component | Strategy | Confidence | Take | Label kind |\n")
        buf.write("|---|---|---|---|---|---|\n")
        for i, comp in enumerate(rp.components, start=1):
            take_str = str(comp.take) if comp.take is not None else "all"
            label = comp.label_kind or "-"
            buf.write(
                f"| {i} | `{comp.candidate_id}` | {comp.primary_kind} "
                f"| {comp.confidence:.2f} | {take_str} | {label} |\n"
            )
        buf.write("\n")
    buf.write("**Run:**\n\n")
    buf.write(f"```bash\n{rp.next_command}\n```\n\n")
    buf.write(
        "You'll get `train.jsonl` / `val.jsonl` / `test.jsonl` (leakage-aware "
        "splits), `recipe.lock.yaml` (audit trail), `report.md` (5-second "
        "scorecard), and `usage.md` (3-line snippets for HF datasets / pandas / raw JSONL).\n\n"
    )
    buf.write(
        "Need to refine first? Edit `recipe.draft.yaml` (drop weak components, "
        "tune `take`, add `filter` expressions), or re-run with "
        "`datascout recon ... --review` to edit the search directions before "
        "discovery.\n\n"
    )


def _strategy_badge(kind: StrategyKind) -> str:
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


def _render_strategies(buf: StringIO, strategies: list[Strategy]) -> None:
    buf.write("\n**Strategies:**\n\n")
    for s in strategies:
        badge = _strategy_badge(s.kind)
        buf.write(f"- {badge} — confidence {s.confidence:.2f}  \n  {s.rationale}\n")
        if s.caveats:
            for cav in s.caveats:
                buf.write(f"  - ⚠ {cav}\n")
        t = s.transform
        transform_bits: list[str] = []
        if t.text_column:
            transform_bits.append(f"text=`{t.text_column}`")
        if t.label_column:
            transform_bits.append(f"label=`{t.label_column}`")
        if t.label_value_map:
            transform_bits.append(
                "values={" + ", ".join(f"`{k}`→{v}" for k, v in t.label_value_map.items()) + "}"
            )
        if t.filter:
            transform_bits.append(f"filter=`{t.filter}`")
        if t.take != "all":
            transform_bits.append(f"take={t.take}")
        if transform_bits:
            buf.write("  - transform: " + " · ".join(transform_bits) + "\n")


def _render_badges(sc: Scorecard) -> list[str]:
    """Compact per-candidate badges. Order is informative, not scored."""
    badges: list[str] = []
    license_sub = sc.cheap_probes.get("license")
    if license_sub is not None:
        match = _evidence_detail(license_sub, "policy_match")
        spdx = _evidence_detail(license_sub, "license_spdx")
        if spdx and match == "allow":
            badges.append(f"license: {spdx} ✅")
        elif spdx and match == "warn_only":
            badges.append(f"license: {spdx} ⚠️")
        elif spdx:
            badges.append(f"license: {spdx} ❗")
        elif license_sub.status == "low_confidence":
            raw = _evidence_detail(license_sub, "license_raw") or "?"
            badges.append(f"license: {raw} (?)")
        else:
            badges.append("license: missing")

    fresh_sub = sc.cheap_probes.get("freshness")
    if fresh_sub is not None and fresh_sub.status == "ok":
        bucket = _evidence_detail(fresh_sub, "bucket") or "?"
        badges.append(f"freshness: {bucket}")

    size_sub = sc.cheap_probes.get("size")
    if size_sub is not None and size_sub.status == "ok":
        rows = _evidence_detail(size_sub, "rows")
        bytes_ = _evidence_detail(size_sub, "bytes")
        downloads = _evidence_detail(size_sub, "downloads")
        if rows:
            badges.append(f"rows: {rows}")
        elif bytes_:
            badges.append(f"bytes: {bytes_}")
        if downloads:
            badges.append(f"downloads: {downloads}")

    lang_sub = sc.cheap_probes.get("languages")
    if lang_sub is not None and lang_sub.status == "ok":
        declared = _evidence_detail(lang_sub, "declared") or "?"
        badges.append(f"languages: {declared}")

    return badges


def _render_probe_signals(buf: StringIO, sc: Scorecard) -> None:
    """Per-probe signal block in a Markdown table."""
    rows: list[tuple[str, str, str]] = []
    for name, sub in sc.cheap_probes.items():
        rows.append(
            (
                name,
                sub.status,
                _format_subscore_evidence(sub),
            )
        )
    if not rows:
        return
    buf.write("| Probe | Status | Signal |\n|---|---|---|\n")
    for name, status, sig in rows:
        buf.write(f"| {name} | {status} | {sig} |\n")


def _format_subscore_evidence(sub: SubScore) -> str:
    """One-line summary of a SubScore's evidence."""
    if sub.status != "ok":
        if sub.evidence:
            return _safe_md(sub.evidence[0].detail)
        return "—"
    parts: list[str] = []
    for ev in sub.evidence:
        parts.append(f"{ev.kind}={_safe_md(ev.detail)}")
    if sub.value is not None:
        parts.append(f"value={sub.value:.2f}")
    return "; ".join(parts) or "ok"


def _evidence_detail(sub: SubScore, kind: str) -> str | None:
    for ev in sub.evidence:
        if ev.kind == kind:
            return ev.detail
    return None


def _safe_md(text: str | Evidence) -> str:
    """Trim and escape pipes/newlines so the markdown table stays well-formed."""
    s = text if isinstance(text, str) else str(text)
    s = s.replace("|", "\\|").replace("\n", " ")
    if len(s) > 80:
        s = s[:77] + "..."
    return s


def _render_papers_section(buf: StringIO, result: ReconResult, ctx: ReconReportContext) -> None:
    """Render the academic-paper discovery section.

    Listed in round-robin order from `paper_search`. Papers with explicit
    HF/Kaggle dataset URLs lead each entry's "datasets cited" sub-list;
    GitHub citations follow as best-effort hints.
    """
    buf.write("## Related papers\n\n")
    citations = ctx.n_paper_dataset_citations
    if citations > 0:
        buf.write(
            f"{ctx.n_papers} paper(s) discovered, "
            f"with **{citations} dataset citation(s)** extracted from abstracts.\n\n"
        )
    else:
        buf.write(
            f"{ctx.n_papers} paper(s) discovered. "
            "No dataset URLs found in abstracts — read the paper to find "
            "the dataset directly.\n\n"
        )
    for p in result.papers:
        _render_paper(buf, p)


def _render_paper(buf: StringIO, p: PaperReference) -> None:
    venue = p.venue or "?"
    citations = f" · {p.citation_count} citation(s)" if p.citation_count else ""
    surfaced = f"  - 🧭 *Surfaced by:* {', '.join(p.surfaced_by)}\n" if p.surfaced_by else ""
    buf.write(f"### {p.title}\n\n")
    authors = ", ".join(p.authors[:5]) + ("…" if len(p.authors) > 5 else "")
    buf.write(f"- 👤 **Authors:** {authors or '(unknown)'}\n")
    buf.write(f"- 🎓 **Venue:** {venue} {p.year}{citations}\n")
    buf.write(f"- 🔗 **Link:** {p.url}\n")
    if p.abstract:
        snippet = p.abstract.strip().replace("\n", " ")[:280]
        if len(p.abstract) > 280:
            snippet += "…"
        buf.write(f"- 📝 **Abstract:** {snippet}\n")
    buf.write(surfaced)
    if p.referenced_datasets:
        buf.write("- 📦 **Datasets cited:**\n")
        for d in p.referenced_datasets:
            buf.write(f"  - `{d.source}:{d.identifier}` — {d.url}\n")
    buf.write("\n")

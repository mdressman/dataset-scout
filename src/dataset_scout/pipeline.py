"""The recon pipeline.

M2a: parse → (decompose if LLM available) → multi-direction search →
cheap probes → ReconResult.

Mode is decided once at the top by `decompose.llm_available(ctx)` and
the `explore` flag. When the LLM is available and `explore=True`, the
pipeline runs the full decomposition + multi-direction search. When
not, it runs in metadata-only mode and emits an explicit notice in the
result so the report is honest about the difference.

Strategy assessor + coverage report land in M2b.
"""

from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING, Any

from dataset_scout.context import ScoutContext
from dataset_scout.core import (
    Candidate,
    CoverageGap,
    CoverageReport,
    DecompositionDirection,
    PaperReference,
    ReconResult,
    Scorecard,
    Strategy,
    SubScore,
)
from dataset_scout.errors import LLMError
from dataset_scout.events import ProgressEvent, ProgressEventKind
from dataset_scout.intent import HeuristicIntentParser, brief_smell_warnings
from dataset_scout.probes import cheap_probes
from dataset_scout.probes.base import Probe, ProbeRegistry
from dataset_scout.sources.base import Budget

if TYPE_CHECKING:
    from dataset_scout.core import Intent
    from dataset_scout.sources.base import Source


_DEFAULT_MAX_CANDIDATES = 100


# Single source of truth for the metadata-only-mode notice. Used by both
# the stderr emitter and the report header so wording stays in sync.
METADATA_ONLY_NOTICE = (
    "Running in metadata-only mode: no LLM provider is configured, so "
    "decomposition, strategy assessment, and coverage gaps were skipped. "
    "To enable them, copy .env.example to .env and set DATASET_SCOUT_MODEL "
    "(e.g. 'github_copilot/gpt-5-mini' or 'github/gpt-4o-mini'), or set "
    "AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_DEPLOYMENT (and run `az login` "
    "for Entra auth)."
)


# Companion hint shown when the LLM IS configured but a call failed at
# runtime — the model name is wrong, the token couldn't be acquired, the
# network is unreachable, etc. The specific error is already in the
# notice list above this; this just orients the user.
LLM_RUNTIME_HINT = (
    "The LLM provider was configured but the call failed (see error "
    "above). Common causes: wrong model id (DATASET_SCOUT_MODEL or "
    "AZURE_OPENAI_DEPLOYMENT), expired credential (`az login` for Azure "
    "Entra; re-authorize the device-code flow for github_copilot/), "
    "missing API key (GITHUB_TOKEN, OPENAI_API_KEY, ANTHROPIC_API_KEY), "
    "network issue, or quota exhausted. Pipeline continued in metadata-"
    "only mode."
)


def _build_sources(ctx: ScoutContext) -> list[Source]:
    """Instantiate concrete sources from the context.

    Thin wrapper around `sources.factory.build_sources` — kept here as
    a name the rest of `pipeline.py` can call without importing the
    factory module twice.
    """
    from dataset_scout.sources.factory import build_sources

    return build_sources(ctx)


def _score_candidate(
    candidate: Candidate,
    intent: Intent,
    probes: ProbeRegistry,
) -> Scorecard:
    """Run every applicable cheap probe against a single candidate."""
    cheap: dict[str, SubScore] = {}
    for probe in probes:
        if not probe.applies(candidate, intent):
            continue
        cheap[probe.name] = probe.run(candidate, intent)
    return Scorecard(candidate=candidate, cheap_probes=cheap)


def _merge_or_register(
    pool: dict[tuple[str, str], Candidate],
    cand: Candidate,
) -> bool:
    """Insert `cand` into `pool` keyed by (source, id), or merge surfaced_by.

    Returns True if this is a newly-seen candidate, False if it merged
    into an existing one.
    """
    key = (cand.source, cand.id)
    existing = pool.get(key)
    if existing is None:
        pool[key] = cand
        return True
    merged: list[str] = list(existing.surfaced_by)
    for d in cand.surfaced_by:
        if d not in merged:
            merged.append(d)
    if merged != existing.surfaced_by:
        pool[key] = existing.model_copy(update={"surfaced_by": merged})
    return False


def run_recon(
    brief: str,
    *,
    ctx: ScoutContext,
    parser_overrides: dict[str, object] | None = None,
    max_candidates: int = _DEFAULT_MAX_CANDIDATES,
    events: list[ProgressEvent] | None = None,
    probes: ProbeRegistry | None = None,
    sources: list[Source] | None = None,
    explore: bool = True,
    directions_override: list[DecompositionDirection] | None = None,
    paper_search_fn: Any = None,
) -> ReconResult:
    """Run the discovery pipeline and return a `ReconResult`.

    Parameters
    ----------
    brief
        Natural-language brief.
    ctx
        Explicit ScoutContext (no global state).
    parser_overrides
        CLI-flag-derived overrides applied to the parsed Intent.
    max_candidates
        Upper bound on candidates surfaced in the report.
    events
        If provided, ProgressEvent entries are appended for each stage.
    probes / sources
        Dependency-injection hooks. Defaults wire the cheap probe set
        and HuggingFaceSource.
    explore
        If False, skip decomposition unconditionally (debug). The CLI
        wires this to a hidden `--no-explore` flag.
    directions_override
        Use the supplied DecompositionDirection list instead of calling
        the LLM. Lets `--decomposition-from <path>` reuse a hand-edited
        decomposition.yaml without paying for a fresh LLM call.
    paper_search_fn
        Injection hook for the academic-paper discovery stage. Default
        (None) wires `dataset_scout.paper_search.find_papers_and_promote`.
        Tests pass a no-op or a respx-backed stub. Set to a callable
        with the same signature as `find_papers_and_promote` to override.
        Pass `False` to disable the stage entirely (returns no papers).
    """
    overrides = parser_overrides or {}
    notices: list[str] = []

    # Open the cache once for the whole run; pass to LLM call sites so
    # repeat runs (same brief, same candidates) don't re-pay. Best-
    # effort: if the cache fails to open we degrade silently — the
    # pipeline must not be blocked by infrastructure.
    cache: Any = None
    try:
        from dataset_scout.cache import open_cache as _open_cache

        cache_cm = _open_cache(ctx.cache_dir)
        cache = cache_cm.__enter__()
    except Exception as exc:  # pragma: no cover - defensive
        cache_cm = None
        notices.append(f"cache disabled this run: {exc}")
    else:
        pass

    def _emit(kind: ProgressEventKind, stage: str, message: str = "", **data: object) -> None:
        if events is not None:
            events.append(ProgressEvent(kind=kind, stage=stage, message=message, data=dict(data)))

    started = time.monotonic()

    _emit(ProgressEventKind.STAGE_STARTED, stage="parse")
    intent = HeuristicIntentParser().parse(brief, **overrides)
    _emit(
        ProgressEventKind.STAGE_FINISHED,
        stage="parse",
        message="parsed brief into Intent",
        threat_families=list(intent.threat_families),
        languages=list(intent.languages),
    )

    # Surface brief-style hints (detector-spec / kitchen-sink patterns).
    for warning in brief_smell_warnings(brief):
        notices.append(warning)

    directions: list[DecompositionDirection] = []
    use_llm = explore
    llm_runtime_error: bool = False  # True iff configured but call failed

    # If the caller provided directions (e.g. --decomposition-from),
    # use them verbatim — skip the decompose LLM call entirely.
    # The strategy assessor + coverage still depend on llm_available(ctx).
    if directions_override is not None:
        directions = list(directions_override)
        for d in directions:
            _emit(
                ProgressEventKind.DIRECTION_PROPOSED,
                stage="decompose",
                name=d.name,
                keywords=list(d.keywords),
                source="reused-from-file",
            )
        # Strategy assessment / coverage still need an LLM; if none is
        # configured we degrade quietly and search-only.
        from dataset_scout.decompose import llm_available

        if not llm_available(ctx):
            use_llm = False
            notices.append(
                "Reused decomposition.yaml; no LLM provider is configured "
                "so strategy assessment and coverage gaps are skipped."
            )
    elif use_llm:
        from dataset_scout.decompose import decompose_intent, llm_available

        if not llm_available(ctx):
            use_llm = False
        else:
            _emit(ProgressEventKind.STAGE_STARTED, stage="decompose")
            try:
                directions = decompose_intent(intent, ctx=ctx, cache=cache)
            except LLMError as exc:
                notices.append(f"decomposition skipped: {exc}")
                notices.append(LLM_RUNTIME_HINT)
                use_llm = False
                llm_runtime_error = True
                directions = []
            else:
                for d in directions:
                    _emit(
                        ProgressEventKind.DIRECTION_PROPOSED,
                        stage="decompose",
                        name=d.name,
                        keywords=list(d.keywords),
                    )
            _emit(
                ProgressEventKind.STAGE_FINISHED,
                stage="decompose",
                message=f"proposed {len(directions)} direction(s)",
            )

    # Only emit the metadata-only notice when no LLM is genuinely
    # configured AND the user didn't supply pre-computed directions
    # (in which case we already emitted a more specific notice above).
    if not use_llm and not llm_runtime_error and directions_override is None:
        notices.append(METADATA_ONLY_NOTICE)

    # ─── Keyword expansion (translates academic decomposition into ───
    # HF-uploader-style compound nouns; one extra LLM call, cached).
    # Empirically this is the difference between zero HF candidates and
    # several genuinely-relevant ones for frontier-territory briefs:
    # the decomposer thinks like a paper writer, but HF dataset
    # uploaders use simple compound nouns ("mental health chat") that
    # the academic decomposition never produces.
    if use_llm and directions:
        from dataset_scout.keyword_expansion import expand_dataset_keywords

        _emit(ProgressEventKind.STAGE_STARTED, stage="keyword_expansion")
        try:
            directions = expand_dataset_keywords(intent, directions, ctx=ctx, cache=cache)
        except Exception as exc:  # pragma: no cover - defensive
            notices.append(f"keyword expansion skipped: {exc}")
        n_with_expansions = sum(1 for d in directions if d.dataset_keywords)
        _emit(
            ProgressEventKind.STAGE_FINISHED,
            stage="keyword_expansion",
            message=f"expanded {n_with_expansions}/{len(directions)} direction(s)",
        )

    if sources is None:
        sources = _build_sources(ctx)
    probes = probes if probes is not None else cheap_probes()
    budget = Budget()

    _emit(ProgressEventKind.STAGE_STARTED, stage="search")
    pool: dict[tuple[str, str], Candidate] = {}
    for source in sources:
        try:
            stream = source.search(intent, directions, budget=budget)
        except Exception as exc:  # defensive: misbehaving source must not kill the run
            notices.append(f"source '{source.name}' failed: {exc}")
            continue
        for cand in stream:
            is_new = _merge_or_register(pool, cand)
            if is_new:
                _emit(
                    ProgressEventKind.CANDIDATE_FOUND,
                    stage="search",
                    source=cand.source,
                    id=cand.id,
                    surfaced_by=list(cand.surfaced_by),
                )
            if len(pool) >= max_candidates:
                break
        if len(pool) >= max_candidates:
            break

    candidates = list(pool.values())[:max_candidates]
    _emit(
        ProgressEventKind.STAGE_FINISHED,
        stage="search",
        message=f"found {len(candidates)} unique candidate(s)",
    )

    if not candidates:
        if directions:
            notices.append(
                "No HuggingFace candidates matched the brief or any decomposition "
                "direction. The decomposition + coverage gaps in the report are "
                "your sourcing roadmap — for novel briefs, the data often lives "
                "outside HF (academic repositories, vendor telemetry, web archives)."
            )
        else:
            notices.append(
                "No candidates returned. Try broadening the brief or check source connectivity."
            )

    _emit(ProgressEventKind.STAGE_STARTED, stage="probe")
    scorecards: list[Scorecard] = []
    for cand in candidates:
        sc = _score_candidate(cand, intent, probes)
        scorecards.append(sc)
        _emit(
            ProgressEventKind.CANDIDATE_SCORED,
            stage="probe",
            id=cand.id,
            probes=list(sc.cheap_probes),
        )
    _emit(
        ProgressEventKind.STAGE_FINISHED,
        stage="probe",
        message=f"scored {len(scorecards)} candidate(s) with {len(probes)} probes",
    )

    # ─── Academic paper discovery (new dataset-lookup channel) ───
    # Search NeurIPS / ICML / ICLR / SaTML for papers relevant to the
    # brief. Surfaces:
    #   1. PaperReferences with abstract-extracted dataset citations,
    #      attached to ReconResult.papers for the report.
    #   2. Promoted Candidates for HF / Kaggle datasets cited by those
    #      papers, merged into the existing pool with paper provenance
    #      via surfaced_by.
    # Failures are non-blocking — the rest of recon proceeds.
    papers: list[PaperReference] = []
    ps_callable: Any
    if paper_search_fn is False:
        # Explicitly disabled (tests).
        ps_callable = None
    elif paper_search_fn is None:
        from dataset_scout.paper_search import find_papers_and_promote

        ps_callable = find_papers_and_promote
    else:
        ps_callable = paper_search_fn

    if ps_callable is not None and (directions or intent.threat_families or intent.raw_brief.strip()):
        _emit(ProgressEventKind.STAGE_STARTED, stage="papers")
        try:
            papers, promoted = ps_callable(
                intent,
                directions,
                cache=cache,
            )
        except Exception as exc:  # pragma: no cover - defensive
            notices.append(f"paper-search stage failed: {exc}")
            papers, promoted = [], []
        for cand in promoted:
            existed = not _merge_or_register(pool, cand)
            if not existed:
                # Newly-promoted candidate; score it with the cheap probes
                # so the report has consistent annotations.
                sc = _score_candidate(cand, intent, probes)
                scorecards.append(sc)
                _emit(
                    ProgressEventKind.CANDIDATE_FOUND,
                    stage="papers",
                    source=cand.source,
                    id=cand.id,
                    surfaced_by=list(cand.surfaced_by),
                )
            else:
                # Existing candidate — _merge_or_register already merged
                # surfaced_by; reflect the update on its scorecard so the
                # report shows the paper provenance.
                merged = pool[(cand.source, cand.id)]
                for sc in scorecards:
                    if sc.candidate.source == cand.source and sc.candidate.id == cand.id:
                        sc.candidate = merged
                        break
        _emit(
            ProgressEventKind.STAGE_FINISHED,
            stage="papers",
            message=f"found {len(papers)} paper(s); promoted {len(promoted)} dataset(s)",
        )

    # ─── Embedding label-intent fit (between probes and shortlist) ─
    # Populates Scorecard.label_intent_fit when an embedding backend is
    # available (sbert by default; AOAI when DATASET_SCOUT_EMBEDDING_BACKEND=
    # aoai and the deployment is configured). No-op otherwise. Runs
    # before the shortlist so future rankers can incorporate the
    # signal — today it's surfaced as a report annotation.
    #
    # NOTE: Intentionally not gated on `use_llm` — the default backend
    # is local sentence-transformers and needs no LLM provider. Users
    # in metadata-only mode (no LLM configured, or LLM call failed)
    # still benefit from the embedding-fit signal. `assess_label_intent_fit`
    # itself no-ops cleanly when no embedder is available, so this
    # branch is safe even when sentence-transformers isn't installed.
    if scorecards:
        from dataset_scout.embedding_fit import assess_label_intent_fit

        _emit(ProgressEventKind.STAGE_STARTED, stage="embedding_fit")
        try:
            updated = assess_label_intent_fit(
                scorecards,
                intent,
                ctx=ctx,
                source_index={s.name: s for s in sources},
                directions=directions,
                cache=cache,
            )
        except Exception as exc:  # pragma: no cover - defensive
            notices.append(f"embedding-fit stage failed: {exc}")
            updated = 0
        _emit(
            ProgressEventKind.STAGE_FINISHED,
            stage="embedding_fit",
            message=f"updated {updated} candidate(s)",
        )

    # ─── Strategy assessment + coverage (M2b) ──
    semantic_gaps: list[CoverageGap] = []
    if use_llm and scorecards:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from dataset_scout.shortlist import select_top_for_assessor
        from dataset_scout.strategy import assess_strategies

        recalled_names: list[str] = []
        for d in directions:
            for n in d.recalled_dataset_names or ():
                if n and n.strip():
                    recalled_names.append(n)

        shortlist = select_top_for_assessor(scorecards, recalled_names=recalled_names)
        source_index: dict[str, Source] = {s.name: s for s in sources}
        _emit(
            ProgressEventKind.STAGE_STARTED,
            stage="assess",
            message=f"assessing {len(shortlist)} candidate(s)",
        )

        def _assess_one(sc: Scorecard) -> tuple[Scorecard, list[Strategy] | str]:
            try:
                strategies = assess_strategies(
                    sc.candidate,
                    intent,
                    ctx=ctx,
                    source=source_index.get(sc.candidate.source),
                    cache=cache,
                )
                return sc, strategies
            except LLMError as exc:
                return sc, str(exc)

        workers = min(5, len(shortlist))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_assess_one, sc): sc for sc in shortlist}
            for fut in as_completed(futures):
                sc, result = fut.result()
                if isinstance(result, str):
                    notices.append(
                        f"strategy assessment skipped for {sc.candidate.source}:"
                        f"{sc.candidate.id}: {result}"
                    )
                    continue
                sc.strategies = result
                _emit(
                    ProgressEventKind.STRATEGY_ASSESSED,
                    stage="assess",
                    id=sc.candidate.id,
                    strategies=[s.kind.value for s in sc.strategies],
                )
        _emit(ProgressEventKind.STAGE_FINISHED, stage="assess")

    # Coverage gap synthesis runs whenever the LLM is available and we
    # have directions — including when zero candidates returned.
    # For frontier-territory briefs the gap analysis IS the deliverable;
    # gating it on `any(sc.strategies)` would silently swallow the most
    # useful artefact for exactly the briefs that need it most.
    if use_llm and directions:
        from dataset_scout.coverage import build_coverage_report

        _emit(ProgressEventKind.STAGE_STARTED, stage="coverage")
        try:
            semantic_gaps = build_coverage_report(
                intent, directions, scorecards, ctx=ctx, cache=cache
            )
        except LLMError as exc:
            notices.append(f"coverage report skipped: {exc}")
            semantic_gaps = []
        _emit(
            ProgressEventKind.STAGE_FINISHED,
            stage="coverage",
            message=f"identified {len(semantic_gaps)} gap(s)",
        )

        # Re-rank scorecards by best_strategy + kind bonus so the report
        # leads with the strongest fits. Candidates without an assessed
        # strategy keep their relative order at the bottom. (No-op when
        # there are no strategies, which is the empty-candidates path.)
        if any(sc.strategies for sc in scorecards):
            scorecards.sort(key=_strategy_rank_key, reverse=True)

    elapsed = time.monotonic() - started

    coverage: CoverageReport | None = None
    if directions or semantic_gaps:
        coverage = CoverageReport(decomposition=directions, semantic_gaps=semantic_gaps)

    if cache_cm is not None:
        with contextlib.suppress(Exception):  # pragma: no cover
            cache_cm.__exit__(None, None, None)

    return ReconResult(
        intent=intent,
        candidates=scorecards,
        sources_searched=[s.name for s in sources],
        coverage=coverage,
        papers=papers,
        elapsed_seconds=round(elapsed, 3),
        notices=notices,
    )


# Strategy-kind bonus for re-ranking. Direct fits float to the top;
# proxies and benign baselines sink. Cosmetic ordering only — every
# strategy is still rendered.
_KIND_BONUS = {
    "direct_use": 0.20,
    "subset_extraction": 0.10,
    "label_remapping": 0.05,
    "cross_class_repurposing": 0.0,
    "signal_proxy": -0.05,
    "benign_baseline": -0.10,
    "not_useful": -1.0,
}


def _strategy_rank_key(sc: Scorecard) -> float:
    best = sc.best_strategy
    if best is None:
        return -100.0
    return best.confidence + _KIND_BONUS.get(best.kind.value, 0.0)


__all__ = ["METADATA_ONLY_NOTICE", "Probe", "run_recon"]

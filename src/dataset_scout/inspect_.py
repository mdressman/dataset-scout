"""`datascout inspect` — single-candidate deep-dive (M3).

Pulls a candidate's metadata + a small streamed sample from its source,
runs the cheap probes, optionally assesses strategies via the
configured LLM provider, and returns a typed `InspectResult` the CLI
renders to stdout.

Usage shapes:

  datascout inspect huggingface:deepset/prompt-injections
  datascout inspect huggingface:org/x@<revision>
  datascout inspect huggingface:org/x --intent-from datascout-out/results.json
  datascout inspect huggingface:org/x --brief "find prompt injection corpora"

Inspect re-uses M2b's `assess_strategies` so a single Intent (the one
that produced the most recent recon, or one parsed from a fresh brief)
drives both recon and inspect — the strategy assessment stays consistent.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from dataset_scout.context import ScoutContext
from dataset_scout.core import (
    Candidate,
    CandidateMetadata,
    ColumnInfo,
    InspectResult,
    Intent,
    LabelBucket,
    LengthStats,
    LicenseSummary,
    Strategy,
)
from dataset_scout.errors import DatasetScoutError, LLMError
from dataset_scout.intent import HeuristicIntentParser
from dataset_scout.licenses import guess_spdx
from dataset_scout.sources.base import Source

_DEFAULT_SAMPLE_N = 50

# Heuristic candidates for the "main text" column when card metadata
# doesn't tell us. First match wins.
_TEXT_COLUMN_HEURISTICS: tuple[str, ...] = (
    "text",
    "content",
    "prompt",
    "input",
    "instruction",
    "sentence",
    "utterance",
    "question",
)

_LABEL_COLUMN_HEURISTICS: tuple[str, ...] = (
    "label",
    "labels",
    "class",
    "category",
    "target",
    "is_injection",
    "is_jailbreak",
)


# ─── target parsing ──────────────────────────────────────────────────


def parse_target(target: str) -> tuple[str, str, str | None]:
    """Parse `<source>:<id>[@<revision>]`.

    Returns (source, id, revision_or_None). When the user omits the
    `<source>:` prefix we default to `huggingface` since that's the
    only wired source today.
    """
    rev: str | None = None
    if "@" in target:
        target, rev = target.rsplit("@", 1)
    if ":" in target:
        source, sid = target.split(":", 1)
    else:
        source, sid = "huggingface", target
    if not source or not sid:
        raise DatasetScoutError(
            f"could not parse target {target!r} — expected <source>:<id>[@<revision>]"
        )
    return source, sid, rev


# ─── intent reuse ────────────────────────────────────────────────────


def load_intent_from(path: Path) -> Intent:
    """Re-hydrate the Intent embedded in a recon `results.json`."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    intent_block = payload.get("intent") if isinstance(payload, dict) else None
    if not isinstance(intent_block, dict):
        raise DatasetScoutError(f"{path} is not a recon results.json (missing 'intent' block).")
    return Intent.model_validate(intent_block)


# ─── source dispatch ─────────────────────────────────────────────────


def _build_source(ctx: ScoutContext, name: str) -> Source:
    from dataset_scout.sources.factory import build_source_index

    index = build_source_index(ctx)
    if name in index:
        return index[name]
    raise DatasetScoutError(
        f"source '{name}' is not configured. Available: "
        f"{', '.join(sorted(index)) or '(none)'}."
    )


# ─── statistics helpers ──────────────────────────────────────────────


def _infer_columns(rows: list[dict[str, Any]]) -> list[ColumnInfo]:
    """Infer (name, dtype) tuples by walking the first row.

    Naive but cheap: the source-side type system is much richer than
    what we need here. The dtype string is informative-only.
    """
    if not rows:
        return []
    head = rows[0]
    out: list[ColumnInfo] = []
    for name, value in head.items():
        out.append(ColumnInfo(name=str(name), dtype=type(value).__name__))
    return out


def _pick_text_column(rows: list[dict[str, Any]], hint: str | None) -> str | None:
    if not rows:
        return None
    head_keys = list(rows[0])
    if hint and hint in head_keys:
        return hint
    for guess in _TEXT_COLUMN_HEURISTICS:
        if guess in head_keys:
            return guess
    # Fall back to the first string-typed column.
    for k, v in rows[0].items():
        if isinstance(v, str):
            return k
    return None


def _pick_label_column(rows: list[dict[str, Any]], hint: str | None) -> str | None:
    if not rows:
        return None
    head_keys = list(rows[0])
    if hint and hint in head_keys:
        return hint
    for guess in _LABEL_COLUMN_HEURISTICS:
        if guess in head_keys:
            return guess
    return None


def _label_distribution(rows: list[dict[str, Any]], column: str) -> list[LabelBucket]:
    """Compute label distribution + Wilson 95% CIs."""
    from dataset_scout.stats import wilson_ci

    counts: dict[str, int] = {}
    for row in rows:
        v = row.get(column)
        if v is None:
            continue
        key = str(v)
        counts[key] = counts.get(key, 0) + 1

    n = sum(counts.values())
    buckets: list[LabelBucket] = []
    for raw_value, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        lo, hi = wilson_ci(count, n) if n > 0 else (0.0, 1.0)
        buckets.append(
            LabelBucket(
                raw_value=raw_value,
                count=count,
                fraction=count / n if n > 0 else 0.0,
                ci_low=lo,
                ci_high=hi,
            )
        )
    return buckets


def _length_stats(rows: list[dict[str, Any]], column: str) -> LengthStats | None:
    lengths: list[int] = []
    for row in rows:
        v = row.get(column)
        if isinstance(v, str):
            lengths.append(len(v))
    if not lengths:
        return None
    return LengthStats(
        column=column,
        n=len(lengths),
        min=min(lengths),
        median=int(statistics.median(lengths)),
        max=max(lengths),
    )


def _license_summary(meta: CandidateMetadata) -> LicenseSummary | None:
    if not meta.license_raw and not meta.license_spdx:
        return None
    return LicenseSummary(
        spdx_guess=meta.license_spdx or guess_spdx(meta.license_raw),
        raw_string=meta.license_raw or "",
        nonstandard_clauses_detected=False,
    )


# ─── orchestrator ────────────────────────────────────────────────────


def run_inspect(
    target: str,
    *,
    ctx: ScoutContext,
    intent: Intent | None = None,
    sample_size: int = _DEFAULT_SAMPLE_N,
    source_override: Source | None = None,
    candidate_override: Candidate | None = None,
) -> InspectResult:
    """Run a single-candidate deep-dive.

    Parameters
    ----------
    target
        ``<source>:<id>[@<revision>]``. When `candidate_override` is
        provided, this is informational only.
    ctx
        Explicit ScoutContext (no global state).
    intent
        Reused from a prior recon (`load_intent_from(results.json)`)
        or freshly parsed from a brief. When omitted, the LLM strategy
        assessment step is skipped.
    sample_size
        Rows to stream for the schema / label-distribution / sample
        section.
    source_override / candidate_override
        Test seams. When both are given the source name still has to
        match `target`'s source.
    """
    started = time.monotonic()
    notices: list[str] = []

    source_name, source_id, revision = parse_target(target)

    # Build the Source plugin.
    source = source_override or _build_source(ctx, source_name)

    # Build (or accept) the Candidate.
    if candidate_override is not None:
        candidate = candidate_override
    else:
        candidate = _build_candidate(source, source_name, source_id, revision, notices)

    # Stream the sample.
    sample_rows = _stream_sample_rows(source, candidate, sample_size, notices)

    columns = _infer_columns(sample_rows)
    text_col = _pick_text_column(sample_rows, candidate.metadata.text_column_guess)
    label_col = _pick_label_column(sample_rows, candidate.metadata.label_column_guess)

    label_dist: list[LabelBucket] = []
    if label_col is not None:
        label_dist = _label_distribution(sample_rows, label_col)

    length_stats: LengthStats | None = None
    if text_col is not None:
        length_stats = _length_stats(sample_rows, text_col)

    # Strategy assessment (optional — needs intent + an LLM provider).
    strategies = _maybe_assess_strategies(candidate, intent, ctx, notices)

    return InspectResult(
        candidate=candidate,
        license_summary=_license_summary(candidate.metadata),
        sample_size=len(sample_rows),
        columns=columns,
        label_distribution=label_dist,
        label_column_used=label_col,
        length_stats=length_stats,
        sample_rows=sample_rows[:5],  # only keep a few in the result for rendering
        strategies=strategies,
        intent_used=intent,
        elapsed_seconds=round(time.monotonic() - started, 3),
        notices=notices,
    )


def _build_candidate(
    source: Source,
    source_name: str,
    source_id: str,
    revision: str | None,
    notices: list[str],
) -> Candidate:
    """Fetch full metadata for the candidate via the source.

    Falls back to a bare-minimum `Candidate` (no metadata) if metadata
    fetch fails — `inspect` is still useful in that degraded mode.
    """
    bare = Candidate(
        source=source_name,
        id=source_id,
        revision=revision,
        metadata=CandidateMetadata(),
    )
    fetcher = getattr(source, "fetch_metadata", None)
    if fetcher is None:
        return bare
    try:
        fetcher(bare)  # we don't use the return; metadata population is via search
    except Exception as exc:
        notices.append(f"metadata fetch failed: {exc}")
        return bare

    # For HuggingFaceSource we can re-build metadata from the HfApi
    # response. To avoid coupling, do it via the source's own helpers
    # when present; else accept the bare candidate.
    builder = getattr(source, "_build_metadata", None)
    if source_name == "huggingface" and builder is None:
        try:
            from huggingface_hub import HfApi

            api: Any = getattr(source, "_api", HfApi())
            info = api.dataset_info(source_id, revision=revision)
            from dataset_scout.sources.huggingface import _build_metadata

            return Candidate(
                source=source_name,
                id=source_id,
                revision=getattr(info, "sha", revision),
                metadata=_build_metadata(info),
            )
        except Exception as exc:
            notices.append(f"hf metadata enrich failed: {exc}")
    return bare


def _stream_sample_rows(
    source: Source,
    candidate: Candidate,
    sample_size: int,
    notices: list[str],
) -> list[dict[str, Any]]:
    """Stream `sample_size` rows for inspect's schema/label/length sections.

    Tries `stream_rows` first (preferred for HF; deterministic order).
    Falls back to `stream_sample`. Empty list on failure with a notice.
    """
    rows: list[dict[str, Any]] = []
    try:
        for row in source.stream_rows(candidate, take=sample_size):
            rows.append(row)
    except NotImplementedError:
        try:
            for row in source.stream_sample(candidate, sample_size, seed=42):
                rows.append(row)
        except Exception as exc:
            notices.append(f"sample streaming failed: {exc}")
    except Exception as exc:
        notices.append(f"row streaming failed: {exc}")
    return rows


def _maybe_assess_strategies(
    candidate: Candidate,
    intent: Intent | None,
    ctx: ScoutContext,
    notices: list[str],
) -> list[Strategy]:
    if intent is None:
        return []
    if not ctx.llm_configured:
        notices.append(
            "Strategy assessment skipped: no LLM provider configured "
            "(set DATASET_SCOUT_MODEL, e.g. 'github_copilot/gpt-5-mini', "
            "or AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_DEPLOYMENT)."
        )
        return []
    from dataset_scout.strategy import assess_strategies

    try:
        return assess_strategies(candidate, intent, ctx=ctx)
    except LLMError as exc:
        notices.append(f"strategy assessment failed: {exc}")
        return []


# ─── public helper for CLI/library to build an Intent on the fly ────


def make_intent(*, brief: str | None, intent_from: Path | None) -> Intent | None:
    """Resolve the Intent input for `inspect`.

    Precedence: `intent_from` > `brief` > None. Library callers can
    skip both and pass intent=None to disable strategy assessment.
    """
    if intent_from is not None:
        return load_intent_from(intent_from)
    if brief:
        return HeuristicIntentParser().parse(brief)
    return None

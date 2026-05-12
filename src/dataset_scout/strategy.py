"""LLM-driven per-candidate strategy assessment.

Given a `Candidate` and the user's `Intent`, ask the configured LLM
provider for 1-4 ranked strategies for using the candidate (direct
use, subset extraction, label remapping, cross-class repurposing,
signal proxy, benign baseline, or "not useful"). The pipeline calls
this once per shortlisted candidate; failures fall back to skipping
assessment for that candidate (the pipeline catches the `LLMError`).

Composition pairing (`composes_with`) is a portfolio-level decision
made elsewhere; the prompt explicitly forbids `composition_only` here
and any such strategy that slips through is dropped with a logged
warning.

Network-free at import time. Provider routing (litellm + per-provider
auth) lives in `dataset_scout.llm_client` and is imported lazily.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from dataset_scout.core import Candidate, Intent, Strategy, StrategyKind, TransformSpec
from dataset_scout.errors import LLMError
from dataset_scout.llm_client import (
    build_completion_kwargs,
    effective_model_id,
    extract_content,
    import_litellm,
)

if TYPE_CHECKING:
    from dataset_scout.cache import Cache
    from dataset_scout.context import ScoutContext
    from dataset_scout.sources.base import Source

# Bumped when prompt or response handling changes in a way that would
# invalidate cached assessments. v3 introduces effective-model-id
# keying (was ctx.aoai_deployment) so cross-provider runs don't
# pollute each other.
ASSESSOR_VERSION = "3"

# Column names commonly used for the supervision label. Used to summarise
# distinct values in the SAMPLE ROWS section so the LLM can fill in
# `label_value_map` with real source-side values.
_LABEL_COLUMN_CANDIDATES = ("label", "labels", "class", "category", "target")

# Per-value truncation limit for sample-row rendering. Keeps the prompt
# bounded when datasets contain long text fields.
_VALUE_PREVIEW_LEN = 200

# Mirrors the prompt's hard cap.
_MAX_STRATEGIES = 4

_log = logging.getLogger(__name__)


# ─── response model ─────────────────────────────────────────────────


class _StrategyEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: StrategyKind
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    caveats: list[str] = Field(default_factory=list)
    transform: TransformSpec


class AssessorResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    strategies: list[_StrategyEntry]


# ─── prompt rendering ───────────────────────────────────────────────


# The literal JSON-schema block contains `{` and `}`, so we use the
# same sentinel-replacement pattern as `decompose.py` rather than
# str.format / f-strings.
_PROMPT_TEMPLATE = """\
You are helping an AI security engineer assess whether a public
dataset is useful for their detection task. Their goal is described
below. The candidate dataset's metadata is provided.

USER INTENT
-----------
Brief: <<RAW_BRIEF>>
Detection target: <<DETECTION_TARGET>>
Threat families: <<THREAT_FAMILIES>>
Deployment context: <<DEPLOYMENT_CONTEXT>>

CANDIDATE DATASET
-----------------
Source: <<SOURCE>>
Id: <<CANDIDATE_ID>>
Card URL: <<CARD_URL>>
Description: <<DESCRIPTION>>
License (raw / SPDX guess): <<LICENSE_RAW>> / <<LICENSE_SPDX>>
Declared languages: <<LANGUAGES>>
Declared task categories: <<TASK_CATEGORIES>>
Tags: <<TAGS>>
Surfaced by direction(s): <<SURFACED_BY>>

<<SAMPLE_SECTION>>

YOUR TASK
---------
For this candidate, produce a ranked list of plausible STRATEGIES
for using it in the user's task. Use only the strategy kinds below;
do not invent new ones. Be conservative-but-creative: stretches get
low confidence. Include at most 4 strategies; return fewer if fewer
plausibly apply. If the candidate is genuinely not useful, return a
single strategy with kind = "not_useful" and explain why.

DO NOT use the kind "composition_only" — composition is a
portfolio-level decision evaluated separately from per-candidate
strategy assessment. Stick to the seven kinds below.

Strategy kinds:
  - direct_use:               labels and content map cleanly
  - subset_extraction:        only some rows are relevant; filter to subset
  - label_remapping:          same data, different label semantics
  - cross_class_repurposing:  positives -> hard-negatives, etc.
  - signal_proxy:             adjacent threat as proxy positive
  - benign_baseline:          no relevant positives, useful as benign
  - not_useful:               nothing relevant

Use ACTUAL column names and label values from the SAMPLE ROWS section
when filling out the transform. Do NOT invent placeholder names like
"label_or_equivalent". If the sample rows section is empty, set
text_column and label_column to null and explain in caveats.

The `filter` field, when non-null, MUST be a sandboxed expression
that evaluates to a boolean per row. It is NOT a free-text note.
Use real column names from the sample rows. Allowed primitives:
  - column references (bare names, e.g. `label`, `text`)
  - comparison operators: == != < <= > >= in
  - boolean operators: and or not
  - parentheses
  - functions: len(...), contains_pattern(col, regex), lower(...),
    startswith(col, prefix), endswith(col, suffix), int(...), str(...)
NOT allowed: attribute access (`.upper()`), subscript (`x[0]`),
lambdas, comprehensions, prose, configuration directives.
Examples of GOOD filter values:
  - "label == 'jailbreak'"
  - "len(text) > 50 and category in ('prompt_injection', 'override')"
  - "contains_pattern(prompt, '(?i)ignore previous')"
If you only need a subset based on a config/split that is NOT a row
column, set filter to null and describe the constraint in caveats —
do not put prose in the filter field.

Return JSON matching this schema:
{
  "strategies": [
    {
      "kind": "...",
      "confidence": 0.0-1.0,
      "rationale": "1-3 sentences",
      "caveats": ["...", "..."],
      "transform": {
        "text_column": "string or null",
        "label_column": "string or null",
        "label_value_map": {"src_value": "positive"|"benign"|"hard_negative", ...},
        "label_kind_map":  {"src_value": "ground_truth"|"remapped"|"proxy"|"subset_extracted", ...},
        "filter": "string or null",
        "take": int or "all"
      }
    },
    ...
  ]
}
"""


def _none_or_csv(values: list[str]) -> str:
    return ", ".join(values) if values else "(none)"


def _none_or_str(value: str | None) -> str:
    return value if value else "(none)"


def _stringify_value(value: Any, max_len: int = _VALUE_PREVIEW_LEN) -> str:
    """Coerce an arbitrary row value into a short, JSON-safe preview.

    Mirrors the spirit of `curate._jsonable`: bytes / arrays / nested
    objects are stringified rather than dropped. Truncation keeps the
    prompt bounded for long text fields.
    """
    if value is None:
        s = "None"
    elif isinstance(value, bool):
        s = "true" if value else "false"
    elif isinstance(value, (int, float, str)):
        s = str(value)
    elif isinstance(value, bytes):
        s = f"<bytes len={len(value)}>"
    else:
        try:
            s = json.dumps(value, default=repr, ensure_ascii=False)
        except Exception:
            s = repr(value)
    if len(s) > max_len:
        s = s[:max_len] + "...(truncated)"
    return s


def _render_sample_section(
    rows: list[dict[str, Any]] | None,
    *,
    note: str | None = None,
) -> str:
    """Render the SAMPLE ROWS prompt section.

    Empty / None rows degrade to a clear "no rows available" notice so
    the LLM is told to fall back to metadata. ``note`` carries the
    underlying reason (fetch failure, no Source plugin, empty stream)
    so the model can be honest about why.
    """
    if not rows:
        body = "(no rows available — assessor working from metadata only)"
        if note:
            body += f"\nNotice: {note}"
        return "SAMPLE ROWS\n-----------\n" + body

    columns = list(rows[0].keys())
    column_list = ", ".join(columns) if columns else "(none)"

    label_lines: list[str] = []
    for col in _LABEL_COLUMN_CANDIDATES:
        seen: list[str] = []
        for r in rows:
            if col not in r:
                continue
            sv = _stringify_value(r[col], max_len=80)
            if sv not in seen:
                seen.append(sv)
        if seen:
            label_lines.append(f"  {col}: {', '.join(seen)}")
    label_distribution = (
        "\n".join(label_lines)
        if label_lines
        else "  (no standard label columns detected among label/labels/class/category/target)"
    )

    row_lines: list[str] = []
    for i, r in enumerate(rows):
        row_lines.append(f"Row {i + 1}:")
        for k, v in r.items():
            row_lines.append(f"  {k} = {_stringify_value(v)}")
    sample_rows_block = "\n".join(row_lines)

    header = f"SAMPLE ROWS (first {len(rows)} rows from the source)"
    underline = "-" * len(header)
    return (
        f"{header}\n{underline}\n"
        f"Available columns: {column_list}\n"
        f"Distinct values seen in candidate label columns:\n"
        f"{label_distribution}\n"
        f"\nRows:\n{sample_rows_block}"
    )


def render_assessor_prompt(
    candidate: Candidate,
    intent: Intent,
    *,
    sample_rows: list[dict[str, Any]] | None = None,
    sample_note: str | None = None,
) -> str:
    """Render the exact prompt sent to the model. Pure; no I/O.

    Empty / None fields render as ``(none)`` so the prompt stays
    well-formed and snapshot-stable across candidates.

    ``sample_rows`` is a small list of rows fetched from the candidate's
    Source by the caller. When None or empty, the SAMPLE ROWS section
    degrades to a "no rows available" notice — the model is then
    instructed to set text/label columns to null and explain in caveats
    rather than invent placeholder names.
    """
    md = candidate.metadata
    sample_section = _render_sample_section(sample_rows, note=sample_note)
    return (
        _PROMPT_TEMPLATE.replace("<<RAW_BRIEF>>", _none_or_str(intent.raw_brief))
        .replace("<<DETECTION_TARGET>>", _none_or_str(intent.detection_target))
        .replace("<<THREAT_FAMILIES>>", _none_or_csv(intent.threat_families))
        .replace("<<DEPLOYMENT_CONTEXT>>", _none_or_str(intent.deployment_context))
        .replace("<<SOURCE>>", _none_or_str(candidate.source))
        .replace("<<CANDIDATE_ID>>", _none_or_str(candidate.id))
        .replace("<<CARD_URL>>", _none_or_str(md.card_url))
        .replace("<<DESCRIPTION>>", _none_or_str(md.description))
        .replace("<<LICENSE_RAW>>", _none_or_str(md.license_raw))
        .replace("<<LICENSE_SPDX>>", _none_or_str(md.license_spdx))
        .replace("<<LANGUAGES>>", _none_or_csv(md.languages_declared))
        .replace("<<TASK_CATEGORIES>>", _none_or_csv(md.task_categories))
        .replace("<<TAGS>>", _none_or_csv(md.tags))
        .replace("<<SURFACED_BY>>", _none_or_csv(candidate.surfaced_by))
        .replace("<<SAMPLE_SECTION>>", sample_section)
    )


# ─── parsing ────────────────────────────────────────────────────────


def _parse_response(content: str) -> AssessorResponse:
    """Parse and validate. Raises ValidationError on schema mismatch,
    json.JSONDecodeError on malformed JSON; both are retryable."""
    payload = json.loads(content)
    return AssessorResponse.model_validate(payload)


def _verify_columns(
    transform: TransformSpec,
    sample_rows: list[dict[str, Any]] | None,
) -> bool | None:
    """Check whether transform column names exist in the sample rows.

    Returns True if all referenced columns were found, False if any were
    missing, or None if verification was not possible (no sample rows).
    """
    if not sample_rows:
        return None
    available = set(sample_rows[0].keys())
    for col in (transform.text_column, transform.label_column):
        if col is not None and col not in available:
            return False
    return True


def _to_strategy(
    entry: _StrategyEntry,
    sample_rows: list[dict[str, Any]] | None = None,
) -> Strategy:
    """Convert a parsed strategy entry to a Strategy with column verification."""
    verified = _verify_columns(entry.transform, sample_rows)
    return Strategy(
        kind=entry.kind,
        confidence=entry.confidence,
        rationale=entry.rationale,
        caveats=list(entry.caveats),
        transform=entry.transform,
        composes_with=[],
        columns_verified=verified,
    )


# ─── assess_strategies ─────────────────────────────────────────────


def _fetch_sample_rows(
    source: Source | None,
    candidate: Candidate,
    sample_n: int,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Best-effort sample fetch for the assessor.

    Returns ``(rows, note)``. ``rows`` is None on error or absent
    Source; an empty list signals "fetched but stream was empty".
    ``note`` carries a short human-readable reason that is rendered
    into the prompt's SAMPLE ROWS section.
    """
    if source is None:
        return None, "no Source plugin available"
    if sample_n <= 0:
        return None, "sample_n <= 0"
    try:
        rows: list[dict[str, Any]] = []
        for row in source.stream_rows(candidate, config=None, split="train", take=sample_n):
            rows.append(dict(row))
            if len(rows) >= sample_n:
                break
    except Exception as exc:  # defensive: a misbehaving source must not break assessment
        _log.warning(
            "sample fetch failed for %s/%s: %s",
            candidate.source,
            candidate.id,
            exc,
        )
        return None, f"row fetch failed: {exc}"
    if not rows:
        return [], "source returned no rows"
    return rows, None


def assess_strategies(
    candidate: Candidate,
    intent: Intent,
    *,
    ctx: ScoutContext,
    timeout_s: float = 60.0,
    source: Source | None = None,
    sample_n: int = 8,
    cache: Cache | None = None,
) -> list[Strategy]:
    """Ask the configured LLM provider for 1-4 ranked strategies for a candidate.

    Single completion call; one retry on Pydantic validation failure
    using the same prompt. Any other failure (network, no LLM
    configured, missing credentials, repeated validation failure)
    raises ``LLMError`` so the pipeline can skip assessment for this
    candidate.

    `composition_only` strategies are dropped with a logged warning —
    composition is decided at portfolio level, not per candidate.

    When ``source`` is provided, up to ``sample_n`` rows are streamed
    from it and rendered into the prompt so the LLM can fill out
    `transform.text_column` / `label_column` / `label_value_map` with
    real source-side names and values. Sample-fetch failures are
    logged and the assessor degrades to the metadata-only path with a
    notice in the prompt; they never raise.

    When ``cache`` is provided, identical (rendered prompt,
    ASSESSOR_VERSION) inputs return without an LLM call. The prompt
    encodes intent + candidate + sampled rows, so identical inputs
    deterministically hit the cache.

    Returns strategies sorted by confidence descending. Capped at 4.
    """
    sample_rows_for_verify: list[dict[str, Any]] | None = None

    if not ctx.llm_configured:
        raise LLMError(
            "No LLM provider configured. Set DATASET_SCOUT_MODEL "
            "(e.g. 'github_copilot/gpt-5-mini' or 'github/gpt-4o-mini'), "
            "or AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_DEPLOYMENT (and run "
            "`az login` for Entra auth)."
        )

    sample_rows, sample_note = _fetch_sample_rows(source, candidate, sample_n)
    sample_rows_for_verify = sample_rows
    prompt = render_assessor_prompt(
        candidate,
        intent,
        sample_rows=sample_rows,
        sample_note=sample_note,
    )

    cache_key: str | None = None
    if cache is not None:
        resolved = effective_model_id(ctx) or ""
        cache_key = hashlib.sha256(
            (ASSESSOR_VERSION + "\n" + resolved + "\n" + prompt).encode("utf-8")
        ).hexdigest()
        cached = cache.get_json("strategy", cache_key)
        if cached is not None:
            try:
                payload = AssessorResponse.model_validate(cached)
            except ValidationError:
                pass
            else:
                return _finalise_strategies(payload, candidate)

    litellm = import_litellm()

    messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
    completion_kwargs: dict[str, Any] = build_completion_kwargs(
        ctx,
        messages=messages,
        response_format=AssessorResponse,
        timeout_s=timeout_s,
    )

    last_parse_error: Exception | None = None
    parsed: AssessorResponse | None = None
    for attempt in range(2):
        try:
            response = litellm.completion(**completion_kwargs)
        except Exception as exc:
            err_text = str(exc).lower()
            if attempt == 0 and ("timeout" in err_text or "timed out" in err_text):
                continue
            raise LLMError(f"LLM call failed: {exc}") from exc

        content = extract_content(response)
        try:
            parsed = _parse_response(content)
            break
        except (ValidationError, json.JSONDecodeError) as exc:
            last_parse_error = exc
            parsed = None
            continue

    if parsed is None:
        msg = f"LLM returned invalid JSON twice: {last_parse_error}"
        raise LLMError(msg) from last_parse_error

    if cache is not None and cache_key is not None:
        cache.set_json("strategy", cache_key, parsed.model_dump(mode="json"))

    return _finalise_strategies(parsed, candidate, sample_rows=sample_rows_for_verify)


def _finalise_strategies(
    parsed: AssessorResponse,
    candidate: Candidate,
    sample_rows: list[dict[str, Any]] | None = None,
) -> list[Strategy]:
    """Drop composition_only, verify columns, sort by confidence desc, cap at _MAX_STRATEGIES."""
    kept: list[_StrategyEntry] = []
    for entry in parsed.strategies:
        if entry.kind is StrategyKind.COMPOSITION_ONLY:
            _log.warning(
                "dropping composition_only strategy from assessor response "
                "(candidate=%s/%s); composition is portfolio-level",
                candidate.source,
                candidate.id,
            )
            continue
        kept.append(entry)

    kept.sort(key=lambda e: e.confidence, reverse=True)
    strategies = [_to_strategy(e, sample_rows=sample_rows) for e in kept[:_MAX_STRATEGIES]]
    unverified = [s for s in strategies if s.columns_verified is False]
    if unverified:
        _log.warning(
            "candidate %s/%s: %d strategy(ies) reference columns not found in sample rows",
            candidate.source,
            candidate.id,
            len(unverified),
        )
    return strategies

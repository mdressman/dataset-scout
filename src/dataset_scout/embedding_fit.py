"""Embedding-based label-intent fit assessment.

A dedicated pipeline stage (not a Probe) that scores each candidate's
semantic fit to the user's intent by comparing dense embeddings. Runs
between cheap probes and the strategy assessor: cheap signals filter
out noise, embedding fit floats genuinely-relevant candidates, then
the assessor pays the per-call cost on a focused shortlist.

Why a stage and not a Probe? Probes are stateless metadata-only by
contract; embedding fit needs row sampling AND a backend embedding
call. Bolting that onto the Probe protocol would distort it. The
pipeline owns this stage and writes results into the existing
``Scorecard.label_intent_fit: SubScore | None`` slot.

Backend: routed through :mod:`dataset_scout.embedder`. Default is local
sentence-transformers (set ``DATASET_SCOUT_EMBEDDING_BACKEND=sbert``,
needs the ``dataset-scout[local-embeddings]`` extra). Legacy AOAI
embeddings remain available via
``DATASET_SCOUT_EMBEDDING_BACKEND=aoai`` plus the existing
``AZURE_OPENAI_EMBEDDING_DEPLOYMENT`` env var. When no backend is
available the stage no-ops gracefully — the rest of the pipeline still
runs.

Caching: per-text-hash, persistent across runs in the ``embedding``
cache namespace. The intent embedding is cached too, so re-running
with the same brief on a fresh candidate set only pays for new
candidate embeddings. Cache keys include the embedder ``name`` +
``model`` so a backend switch (sbert ↔ aoai) can't serve stale
cross-backend vectors.

Determinism: candidate text is composed from a fixed-seed sample of
rows (``seed=42``, ``n=5``), serialised in a stable canonical form.
Same candidate x same revision -> same text -> same cache key -> same
score.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from typing import TYPE_CHECKING, Any

from dataset_scout.core import (
    Candidate,
    DecompositionDirection,
    Evidence,
    Intent,
    Scorecard,
    SubScore,
)
from dataset_scout.embedder import Embedder, build_embedder
from dataset_scout.errors import LLMError, SourceUnsupportedError

# Re-exported for back-compat with tests that monkeypatch these names
# at the embedding_fit module path. Production code goes through the
# Embedder protocol now and never invokes either function from here.
from dataset_scout.llm_client import import_litellm, make_token_provider  # noqa: F401

if TYPE_CHECKING:
    from dataset_scout.cache import Cache
    from dataset_scout.context import ScoutContext
    from dataset_scout.sources.base import Source


# Bumped when the prompt-construction policy or sampling shape changes
# in a way that would invalidate cached embeddings. v2 introduces
# embedder-name + embedder-model keying (was ctx.aoai_embedding_deployment)
# so cross-backend runs (sbert ↔ aoai) don't pollute each other.
EMBEDDING_FIT_VERSION = "2"

# How many sample rows to pull per candidate.
_SAMPLE_N = 5
_SAMPLE_SEED = 42

# Per-row text cap so a single huge field doesn't dominate the embedding.
_PER_ROW_PREVIEW = 240

# Total cap on the candidate text we send to the embedding endpoint —
# guards against runaway bills on enormous descriptions.
_MAX_CANDIDATE_CHARS = 4000

_log = logging.getLogger(__name__)


# ─── public entry point ────────────────────────────────────────────


def assess_label_intent_fit(
    scorecards: list[Scorecard],
    intent: Intent,
    *,
    ctx: ScoutContext,
    source_index: dict[str, Source] | None = None,
    directions: list[DecompositionDirection] | None = None,
    cache: Cache | None = None,
    timeout_s: float = 30.0,
    embedder: Embedder | None = None,
) -> int:
    """Fill in ``Scorecard.label_intent_fit`` for each scorecard, in place.

    Returns the number of scorecards updated. Returns 0 (and no-ops) if:
      - no scorecards were supplied, OR
      - ``embedder`` is None and :func:`build_embedder` returns None
        (no backend available — sbert not installed and no AOAI config).

    Tests can inject ``embedder=`` directly to bypass the factory and
    avoid mocking torch / litellm internals.

    Per-candidate failures (sample fetch, embedding call, parse) are
    logged and the candidate's ``label_intent_fit`` is left as None —
    the rest of the run is unaffected.
    """
    if not scorecards:
        return 0

    if embedder is None:
        embedder = build_embedder(ctx)
    if embedder is None:
        return 0

    intent_text = _compose_intent_text(intent, directions or [])
    intent_vec = _get_or_compute_embedding(
        intent_text,
        embedder=embedder,
        cache=cache,
    )
    if intent_vec is None:
        # Hard failure on the intent embedding: skip the whole stage
        # cleanly. Per-candidate failures still wouldn't be useful
        # without the intent vector.
        _log.warning("embedding-fit stage skipped: intent embedding failed")
        return 0

    updated = 0
    for sc in scorecards:
        cand = sc.candidate
        sample_rows = _fetch_sample_rows(source_index, cand)
        cand_text = _compose_candidate_text(cand, sample_rows)
        cand_vec = _get_or_compute_embedding(
            cand_text,
            embedder=embedder,
            cache=cache,
        )
        if cand_vec is None:
            sc.label_intent_fit = SubScore(
                value=None,
                status="low_confidence",
                evidence=[
                    Evidence(
                        kind="embedding_fit",
                        detail="embedding call failed for this candidate",
                    )
                ],
                probe_version=EMBEDDING_FIT_VERSION,
            )
            continue

        cosine = _cosine(intent_vec, cand_vec)
        # Cosine on common embedding models is in [-1, 1]; clamp to
        # [0, 1] for a friendlier signal in the report. Negative
        # cosines on these embeddings are rare and not informative.
        score = max(0.0, cosine)
        evidence_detail = (
            f"cosine={cosine:.3f} (clamped {score:.3f}); "
            f"sample_n={len(sample_rows)}; "
            f"backend={embedder.name}/{embedder.model}"
        )
        status = "ok" if sample_rows else "low_confidence"
        sc.label_intent_fit = SubScore(
            value=round(score, 4),
            status=status,
            evidence=[Evidence(kind="embedding_fit", detail=evidence_detail)],
            probe_version=EMBEDDING_FIT_VERSION,
        )
        updated += 1

    return updated


# ─── text composition ──────────────────────────────────────────────


def _compose_intent_text(
    intent: Intent,
    directions: list[DecompositionDirection],
) -> str:
    """Build the canonical intent embedding text.

    Includes the brief, detection target, threat families, and any
    decomposition direction summaries. Stable across runs.
    """
    parts: list[str] = [intent.raw_brief]
    if intent.detection_target:
        parts.append(f"Detection target: {intent.detection_target}")
    if intent.threat_families:
        parts.append("Threat families: " + ", ".join(intent.threat_families))
    if intent.deployment_context:
        parts.append(f"Deployment: {intent.deployment_context}")
    for d in directions:
        line = f"Direction {d.name}: {d.rationale}"
        if d.expected_finds:
            line += f" Looking for: {d.expected_finds}"
        parts.append(line)
    return "\n".join(parts)


def _compose_candidate_text(
    candidate: Candidate,
    sample_rows: list[dict[str, Any]],
) -> str:
    """Build the canonical candidate embedding text.

    Order is fixed: description, then a stable serialisation of sample
    rows. The serialisation is sorted-key JSON with a per-value preview
    cap, so two runs against the same dataset revision produce the same
    bytes (and the same cache key).
    """
    md = candidate.metadata
    parts: list[str] = []
    if md.description:
        parts.append(md.description)
    if md.task_categories:
        parts.append("Tasks: " + ", ".join(md.task_categories))
    if md.tags:
        # Cap to 20 tags to bound text size; sorted for stability.
        parts.append("Tags: " + ", ".join(sorted(md.tags)[:20]))
    for i, row in enumerate(sample_rows):
        rendered = _render_row_for_embedding(row)
        parts.append(f"Row {i + 1}: {rendered}")
    text = "\n".join(parts)
    if len(text) > _MAX_CANDIDATE_CHARS:
        text = text[:_MAX_CANDIDATE_CHARS] + "…"
    return text


def _render_row_for_embedding(row: dict[str, Any]) -> str:
    """Stable, capped key=value rendering of a single row."""
    pairs: list[str] = []
    for key in sorted(row):
        value = row[key]
        rendered = _stringify(value)
        if len(rendered) > _PER_ROW_PREVIEW:
            rendered = rendered[:_PER_ROW_PREVIEW] + "…"
        pairs.append(f"{key}={rendered}")
    return " | ".join(pairs)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, bytes):
        return f"<bytes len={len(value)}>"
    try:
        return json.dumps(value, ensure_ascii=False, default=repr, sort_keys=True)
    except Exception:
        return repr(value)


# ─── sample-row fetch ──────────────────────────────────────────────


def _fetch_sample_rows(
    source_index: dict[str, Source] | None,
    candidate: Candidate,
) -> list[dict[str, Any]]:
    """Best-effort deterministic sample fetch.

    Returns at most `_SAMPLE_N` rows. Returns `[]` if:
      - no source plugin available
      - source explicitly does not stream (`SourceUnsupportedError`)
      - any other failure (logged at WARNING)
    """
    if source_index is None:
        return []
    source = source_index.get(candidate.source)
    if source is None:
        return []
    if not candidate.streamable:
        # Honour the candidate's own declaration; cheaper than catching
        # the exception below.
        return []
    rows: list[dict[str, Any]] = []
    try:
        for row in source.stream_rows(
            candidate, config=None, split="train", take=_SAMPLE_N, seed=_SAMPLE_SEED
        ):
            rows.append(dict(row))
            if len(rows) >= _SAMPLE_N:
                break
    except SourceUnsupportedError:
        return []
    except Exception as exc:  # pragma: no cover - defensive
        _log.warning(
            "embedding-fit: row fetch failed for %s/%s: %s",
            candidate.source,
            candidate.id,
            exc,
        )
        return []
    return rows


# ─── embedding call (with cache) ───────────────────────────────────


def _embedding_cache_key(text: str, *, embedder_name: str, embedder_model: str) -> str:
    """Cache key includes backend identity so cross-backend lookups miss.

    A 384-dim sbert vector and a 1536-dim AOAI vector for the same
    text are not interchangeable. Keying on
    ``embedder.name + embedder.model`` ensures a backend switch
    cleanly invalidates without poisoning the existing cache.
    """
    canonical = f"{EMBEDDING_FIT_VERSION}\n{embedder_name}\n{embedder_model}\n{text}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _get_or_compute_embedding(
    text: str,
    *,
    embedder: Embedder,
    cache: Cache | None,
) -> list[float] | None:
    """Return an embedding vector for ``text``, hitting cache when possible.

    Returns None on call failure (logged at WARNING).
    """
    key: str | None = None
    if cache is not None:
        key = _embedding_cache_key(
            text, embedder_name=embedder.name, embedder_model=embedder.model
        )
        cached = cache.get_json("embedding", key)
        if isinstance(cached, list) and all(isinstance(v, (int, float)) for v in cached):
            return [float(v) for v in cached]

    try:
        vecs = embedder.embed([text])
    except LLMError as exc:
        _log.warning("embedding call failed: %s", exc)
        return None
    except Exception as exc:  # pragma: no cover - defensive
        _log.warning("embedding call failed: %s", exc)
        return None

    if not vecs:
        _log.warning("embedding response had unexpected shape (empty)")
        return None
    vec = vecs[0]

    if cache is not None and key is not None:
        cache.set_json("embedding", key, vec)
    return vec


# ─── math ──────────────────────────────────────────────────────────


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    num = 0.0
    da = 0.0
    db = 0.0
    for x, y in zip(a, b, strict=True):
        num += x * y
        da += x * x
        db += y * y
    if da == 0.0 or db == 0.0:
        return 0.0
    return num / (math.sqrt(da) * math.sqrt(db))


__all__ = ["EMBEDDING_FIT_VERSION", "assess_label_intent_fit"]

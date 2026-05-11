"""Core types — the typed vocabulary of dataset-scout.

Reference: `TECH_DESIGN.md` §3. All Pydantic v2 so we get JSON Schema
export and serialization for free.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ─── Intent ──────────────────────────────────────────────────────────


class SensitiveDomain(StrEnum):
    NONE = "none"
    GATED = "gated"
    REFUSED = "refused"


class LicensePolicy(BaseModel):
    """Permissive-by-default license policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    allow: frozenset[str] = Field(
        default=frozenset(
            {
                "MIT",
                "Apache-2.0",
                "BSD-3-Clause",
                "BSD-2-Clause",
                "CC-BY-4.0",
                "CC0-1.0",
                "ODC-BY-1.0",
            }
        )
    )
    warn_only: frozenset[str] = Field(
        default=frozenset({"CC-BY-SA-4.0", "GPL-3.0", "AGPL-3.0", "OpenRAIL-M"})
    )

    @field_validator("allow", "warn_only", mode="before")
    @classmethod
    def _coerce_to_frozenset(cls, v: Any) -> Any:
        if isinstance(v, (list, tuple, set)):
            return frozenset(v)
        return v


class Intent(BaseModel):
    """Structured form of the user's natural-language brief."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    raw_brief: str
    detection_target: str | None = None
    threat_families: list[str] = Field(default_factory=list)
    deployment_context: str | None = None
    languages: list[str] = Field(default_factory=lambda: ["en"])
    license_policy: LicensePolicy = Field(default_factory=LicensePolicy)
    must_have: list[str] = Field(default_factory=list)
    nice_to_have: list[str] = Field(default_factory=list)
    must_not: list[str] = Field(default_factory=list)
    sensitive_domain: SensitiveDomain = SensitiveDomain.NONE
    min_strategy_confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    def stable_hash(self) -> str:
        """Stable across machines. Used as a cache key ingredient."""
        payload = self.model_dump(mode="json")
        lp = payload.get("license_policy") or {}
        for k in ("allow", "warn_only"):
            if k in lp and isinstance(lp[k], (list, set, frozenset)):
                lp[k] = sorted(lp[k])
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class DecompositionDirection(BaseModel):
    """One of 3-7 related search directions produced by the LLM decomposer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    rationale: str
    keywords: list[str] = Field(default_factory=list)
    threat_families: list[str] = Field(default_factory=list)
    expected_finds: str = ""
    # Populated by the keyword-expansion stage: short HF-uploader-style
    # compound-noun phrases ("mental health chat", "counseling dialogue")
    # that complement the decomposer's academic-vocabulary keywords.
    # When empty, sources fall back to `keywords`.
    dataset_keywords: list[str] = Field(default_factory=list)
    # Populated by the keyword-expansion stage: specific NAMED datasets,
    # benchmarks, or research lines plausibly relevant to the brief —
    # e.g. "PersonaChat", "MMLU", "XSTest", "or-bench". Generic compound
    # nouns miss these because dataset uploaders use the proper noun in
    # the id (`google/Synthetic-Persona-Chat`) rather than a description.
    # The HF source issues these verbatim (no auto-shortening).
    recalled_dataset_names: list[str] = Field(default_factory=list)


# ─── Candidate metadata envelope ─────────────────────────────────────


class ColumnInfo(BaseModel):
    """Minimal description of a dataset column.

    Populated when the source can provide column info cheaply
    (HF datasets-server, Kaggle column metadata, etc.). Empty otherwise;
    sample-driven probes infer this from streamed rows.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    dtype: str | None = None  # source-native type string; intentionally untyped


class CandidateMetadata(BaseModel):
    """Source-agnostic metadata envelope consumed by probes.

    The contract that decouples probes from any specific source. HF / Kaggle /
    PWC plugins all populate this same shape. Probes never read source-specific
    keys directly — anything that doesn't fit goes into `extras`.

    Fields are deliberately `None`-able rather than carrying sentinel defaults:
    probes treat `None` as "signal absent" and report `not_applicable`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    description: str | None = None
    card_url: str | None = None
    homepage_url: str | None = None

    # Licensing.
    license_raw: str | None = None
    license_spdx: str | None = None

    # Declared languages (ISO codes from card YAML).
    languages_declared: list[str] = Field(default_factory=list)

    # Dates. content_date_range is rare but valuable when present.
    uploaded_at: datetime | None = None
    last_modified: datetime | None = None
    content_date_range: tuple[date, date] | None = None

    # Size / popularity.
    rows: int | None = None
    bytes: int | None = None
    downloads: int | None = None
    likes: int | None = None

    # Structure (filled by source when cheap; sample-driven probes fill later).
    columns: list[ColumnInfo] = Field(default_factory=list)
    label_column_guess: str | None = None
    text_column_guess: str | None = None

    # Set of dataset-card YAML keys that were declared on the upstream card.
    # Used by the card_completeness probe.
    card_fields_present: frozenset[str] = Field(default_factory=frozenset)

    # Access posture.
    requires_auth: bool = False
    gated: bool = False

    # Tags / categories surfaced by the registry.
    tags: list[str] = Field(default_factory=list)
    task_categories: list[str] = Field(default_factory=list)

    # Source-specific data probes shouldn't consume directly.
    extras: dict[str, Any] = Field(default_factory=dict)

    @field_validator("card_fields_present", mode="before")
    @classmethod
    def _coerce_card_fields(cls, v: Any) -> Any:
        if isinstance(v, (list, tuple, set)):
            return frozenset(v)
        return v


# ─── Candidate ───────────────────────────────────────────────────────


class Candidate(BaseModel):
    """A dataset surfaced by a Source plugin."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str
    id: str
    revision: str | None = None
    metadata: CandidateMetadata = Field(default_factory=CandidateMetadata)
    streamable: bool = True
    # Names of decomposition directions that surfaced this candidate.
    # Empty = surfaced from the original Intent only. Populated as the
    # multi-direction search merges hits across queries.
    surfaced_by: list[str] = Field(default_factory=list)

    @property
    def requires_auth(self) -> bool:
        """Convenience pass-through (gated counts as requires_auth)."""
        return self.metadata.requires_auth or self.metadata.gated


# ─── Probes / SubScores ──────────────────────────────────────────────

SubScoreStatus = Literal["ok", "not_applicable", "low_confidence", "skipped"]


class Evidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str
    detail: str
    url: str | None = None
    value: float | None = None


class SubScore(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    value: float | None
    confidence_interval: tuple[float, float] | None = None
    n: int | None = None
    status: SubScoreStatus = "ok"
    evidence: list[Evidence] = Field(default_factory=list)
    probe_version: str = "0"


class LicenseSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    spdx_guess: str | None
    raw_string: str
    nonstandard_clauses_detected: bool = False
    notes: list[str] = Field(default_factory=list)


# ─── Strategy ────────────────────────────────────────────────────────


class StrategyKind(StrEnum):
    DIRECT_USE = "direct_use"
    SUBSET_EXTRACTION = "subset_extraction"
    LABEL_REMAPPING = "label_remapping"
    CROSS_CLASS_REPURPOSING = "cross_class_repurposing"
    SIGNAL_PROXY = "signal_proxy"
    BENIGN_BASELINE = "benign_baseline"
    COMPOSITION_ONLY = "composition_only"
    NOT_USEFUL = "not_useful"


class LabelKind(StrEnum):
    GROUND_TRUTH = "ground_truth"
    REMAPPED = "remapped"
    PROXY = "proxy"
    SUBSET_EXTRACTED = "subset_extracted"
    JUDGED = "judged"


class TransformSpec(BaseModel):
    """Concrete shape of the transform a Strategy applies."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text_column: str | None = None
    label_column: str | None = None
    label_value_map: dict[str, Literal["positive", "benign", "hard_negative"]] = Field(
        default_factory=dict
    )
    label_kind_map: dict[str, str] = Field(default_factory=dict)
    filter: str | None = None
    take: int | Literal["all"] = "all"


class Strategy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: StrategyKind
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    caveats: list[str] = Field(default_factory=list)
    transform: TransformSpec
    composes_with: list[str] = Field(default_factory=list)
    columns_verified: bool | None = None


# ─── Scorecard ───────────────────────────────────────────────────────


class Scorecard(BaseModel):
    """Per-candidate aggregation of probe outputs and strategy assessments."""

    model_config = ConfigDict(extra="forbid")

    candidate: Candidate
    cheap_probes: dict[str, SubScore] = Field(default_factory=dict)
    label_intent_fit: SubScore | None = None
    strategies: list[Strategy] = Field(default_factory=list)
    license_summary: LicenseSummary | None = None
    explanation: str = ""

    @property
    def best_strategy(self) -> Strategy | None:
        if not self.strategies:
            return None
        return max(self.strategies, key=lambda s: s.confidence)


# ─── Coverage report ─────────────────────────────────────────────────


class CoverageGap(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    aspect: str
    description: str
    suggestion: str


class CoverageReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decomposition: list[DecompositionDirection] = Field(default_factory=list)
    semantic_gaps: list[CoverageGap] = Field(default_factory=list)

    @property
    def notable(self) -> bool:
        return len(self.semantic_gaps) >= 2


# ─── Academic paper discovery ────────────────────────────────────────


class ExtractedDataset(BaseModel):
    """A dataset reference pulled from a paper's abstract or metadata.

    `confidence` distinguishes high-precision URL matches from softer
    signals so the report can lead with explicit citations and only
    surface inferred ones with hedging.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: Literal["huggingface", "kaggle", "github", "url"]
    identifier: str
    url: str
    confidence: Literal["explicit_url", "name_match", "llm_inferred"] = "explicit_url"


class PaperReference(BaseModel):
    """A paper relevant to the brief from one of the configured venues.

    Papers are *referrers* to datasets, not datasets themselves: this
    lives in `ReconResult.papers` separately from `candidates`. When a
    paper cites a HF or Kaggle dataset, the pipeline ALSO promotes that
    dataset into `candidates` with `surfaced_by` carrying the paper id.
    """

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    venue: str
    year: int
    url: str
    abstract: str | None = None
    citation_count: int | None = None
    referenced_datasets: list[ExtractedDataset] = Field(default_factory=list)
    surfaced_by: list[str] = Field(default_factory=list)
    arxiv_id: str | None = None


# ─── Normalized record (curate output schema) ────────────────────────


class JudgeBlock(BaseModel):
    """LLM-as-judge verdict attached to a row when ``label_kind`` is ``judged``.

    Populated by the M10 ``datascout judge`` verb. Each judged row records
    the axis question that was asked, the verdict trio, the judge's
    self-rated confidence, and provenance (model + prompt-template version)
    so the audit trail in the lockfile is reproducible. ``n_judges`` and
    ``agreement`` are populated only when multi-judge mode was used; for
    single-judge runs ``n_judges = 1`` and ``agreement`` is ``None``.

    The ``confidence`` field here is the *raw* judge self-rated value;
    the *derived* confidence used by downstream filters is recorded as
    :attr:`NormalizedRecord.label_confidence` (for multi-judge runs the
    derived value blends agreement and self-confidence per the rule
    documented inline in :mod:`dataset_scout.judge`).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    axis: str
    verdict: Literal["positive", "negative", "ambiguous"]
    subcategory: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    model: str
    template_version: str
    n_judges: int = Field(default=1, ge=1)
    agreement: Literal["single", "majority", "unanimous"] | None = None


class NormalizedRecord(BaseModel):
    """One row of the materialised corpus.

    The defensible output shape: every row carries explicit `label_kind`
    so downstream training can filter proxies out of eval. `extras`
    holds the original source row verbatim — no information is dropped
    in normalization. When source values aren't JSON-serializable
    (images, audio, arrays) the materialiser stringifies them and
    flags the coercion in `extras_coercion: True`.

    Reference: `docs/concepts.md` §6 (label_kind) and §7 (recipes).
    """

    model_config = ConfigDict(extra="forbid")

    text: str
    label: Literal["positive", "benign", "hard_negative"]
    label_kind: LabelKind

    # Recon-side provenance.
    strategy: StrategyKind
    strategy_confidence: float = Field(ge=0.0, le=1.0)

    # Source provenance.
    source: str  # "<source>:<id>"
    source_row_id: str
    source_revision: str | None = None
    source_config: str | None = None
    source_split: str | None = None

    # Optional flavour from recipe intent (first threat family wins).
    threat_family: str | None = None

    # Original row, JSON-coerced if necessary.
    extras: dict[str, Any] = Field(default_factory=dict)
    extras_coercion: bool = False

    # M10 — judge labeling. Both fields are ``None`` for non-judged
    # rows; populated together when ``label_kind == LabelKind.JUDGED``.
    # Additive only: existing corpora load unchanged.
    label_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    judge: JudgeBlock | None = None

    @property
    def stable_id(self) -> str:
        """Globally-unique, deterministic row identifier.

        Canonical concatenation of the four provenance fields scout
        already records: ``<source>::<config>::<split>::<source_row_id>``.
        ``config`` and ``split`` use the placeholder ``_`` when ``None``
        so the form is total; ``source_row_id`` is already deterministic
        per :func:`dataset_scout.curate._row_id`.

        Stable across runs given the same source revision. Used by
        downstream verbs (judge resumability, lineage, dedup keys) that
        need a corpus-global handle for a row without introducing a new
        stored field. M9-min slice; the lineage DAG (M9-full) layers on
        top of this without changing the format.
        """
        return (
            f"{self.source}::"
            f"{self.source_config or '_'}::"
            f"{self.source_split or '_'}::"
            f"{self.source_row_id}"
        )


# ─── Inspect result (M3 — single-candidate deep-dive) ────────────────


class LabelBucket(BaseModel):
    """One bucket in the inspect-time label distribution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    raw_value: str  # the source-side label value, stringified
    count: int
    fraction: float = Field(ge=0.0, le=1.0)
    ci_low: float = Field(ge=0.0, le=1.0)
    ci_high: float = Field(ge=0.0, le=1.0)


class LengthStats(BaseModel):
    """Min/median/max of the text column's character length."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    column: str
    n: int
    min: int
    median: int
    max: int


class InspectResult(BaseModel):
    """Output of `datascout inspect`: a deep-dive on a single candidate.

    Library callers receive this directly; the CLI renders it to stdout
    via `dataset_scout.render.inspect_panel`.
    """

    model_config = ConfigDict(extra="forbid")

    candidate: Candidate
    license_summary: LicenseSummary | None = None

    # Inferred from a small streamed sample.
    sample_size: int
    columns: list[ColumnInfo] = Field(default_factory=list)
    label_distribution: list[LabelBucket] = Field(default_factory=list)
    label_column_used: str | None = None
    length_stats: LengthStats | None = None
    sample_rows: list[dict[str, Any]] = Field(default_factory=list)

    # LLM assessment when AOAI is configured.
    strategies: list[Strategy] = Field(default_factory=list)
    intent_used: Intent | None = None

    elapsed_seconds: float = 0.0
    notices: list[str] = Field(default_factory=list)


# ─── Recon result ────────────────────────────────────────────────────


class ReconResult(BaseModel):
    """Top-level result of a `recon` run.

    M1a (discovery slice) framing: candidates are returned in source/search
    relevance order. Probe outputs are annotations, not a ranking signal.
    Embedding label-intent fit and the LLM strategy assessor land in later
    milestones; until they do, `coverage` is None and strategies on each
    Scorecard are empty.
    """

    model_config = ConfigDict(extra="forbid")

    intent: Intent
    candidates: list[Scorecard] = Field(default_factory=list)
    sources_searched: list[str] = Field(default_factory=list)
    coverage: CoverageReport | None = None
    papers: list[PaperReference] = Field(default_factory=list)
    elapsed_seconds: float = 0.0
    notices: list[str] = Field(default_factory=list)
    scout_version: str = "0.0.1"

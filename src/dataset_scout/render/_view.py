"""Shared view-model for recon report renderers.

Single source of truth for the *decisions* both renderers make
(metadata-only vs full, lead with gaps vs candidates, etc.) AND the
*derived insight layer* both render: per-candidate verdicts,
strategy-kind groupings, and a recipe/curate preview that bridges
discovery → next action.

Per the rubber-duck pass, decoupling presentation framing from rendering
prevents Markdown and HTML reports from drifting as the product evolves.

This module exposes:
  - `ReconReportContext` — pre-computed flags + counts.
  - `CardVerdict` — at-a-glance "should I use this?" label per scorecard.
  - `StrategyGroup` — buckets of cards (direct fits / reframings / proxies / ...).
  - `RecipePreview` — what a `curate` run on this report would produce.

All HTML/Markdown formatting lives in the respective `*_report.py` modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from dataset_scout.core import ReconResult, Scorecard
from dataset_scout.pipeline import LLM_RUNTIME_HINT, METADATA_ONLY_NOTICE

# Substring sentinels lifted from the pipeline notice constants so that
# any future rewording of the notice text in pipeline.py is mirrored
# here automatically. The first sentence of each notice is unique
# enough to identify the mode; matching the entire string would be
# too brittle (notices can be lightly mutated by callers).
_METADATA_ONLY_SENTINEL = METADATA_ONLY_NOTICE.split(":", 1)[0]
_LLM_RUNTIME_SENTINEL = LLM_RUNTIME_HINT.split("(", 1)[0].strip()

# ─── Per-card verdict ───────────────────────────────────────────────


# Stable-keyed bucket name for grouping. Order matters in render: the
# tuple in DISPLAY_GROUPS_ORDER controls section order in the report.
GroupKey = Literal[
    "direct_fit",       # direct_use strategies
    "reframing",        # subset_extraction / label_remapping / cross_class_repurposing
    "signal_proxy",     # signal_proxy
    "benign_baseline",  # benign_baseline
    "not_useful",       # not_useful or no strategies
]


# Render order. "direct_fit" first because that's the strongest answer.
DISPLAY_GROUPS_ORDER: tuple[GroupKey, ...] = (
    "direct_fit",
    "reframing",
    "signal_proxy",
    "benign_baseline",
    "not_useful",
)


# Friendly section headers. Markdown + HTML both consume these.
GROUP_LABELS: dict[GroupKey, str] = {
    "direct_fit": "🎯 Direct fits",
    "reframing": "🔁 Reframings",
    "signal_proxy": "📡 Signal proxies",
    "benign_baseline": "🧊 Benign baselines",
    "not_useful": "❌ Not useful / unassessed",
}


GROUP_DESCRIPTIONS: dict[GroupKey, str] = {
    "direct_fit": (
        "Datasets whose labels and content map cleanly to the brief. "
        "Use as positive training examples; trust them in eval."
    ),
    "reframing": (
        "Datasets that need a label remap, a row-level filter, or a "
        "class flip to fit. Each carries a transform spec — read it."
    ),
    "signal_proxy": (
        "Adjacent-threat datasets useful as proxy positives during "
        "cold start. Train on; **exclude from eval**."
    ),
    "benign_baseline": (
        "No relevant positives, but useful as benign distribution "
        "background to lower false-positive rates."
    ),
    "not_useful": (
        "Returned by HF search but the assessor judged them irrelevant, "
        "or they weren't strategy-assessed. Skim and skip."
    ),
}


# Confidence-band labels for the verdict. Tuned against typical assessor
# output: 0.7+ is "strong", 0.5-0.7 is "moderate", <0.5 is "tentative".
def _confidence_band(confidence: float) -> str:
    if confidence >= 0.7:
        return "strong"
    if confidence >= 0.5:
        return "moderate"
    return "tentative"


# Map each strategy kind to the bucket it belongs in.
_KIND_TO_GROUP: dict[str, GroupKey] = {
    "direct_use": "direct_fit",
    "subset_extraction": "reframing",
    "label_remapping": "reframing",
    "cross_class_repurposing": "reframing",
    "signal_proxy": "signal_proxy",
    "benign_baseline": "benign_baseline",
    "not_useful": "not_useful",
    "composition_only": "not_useful",  # never emitted today, but covered
}


@dataclass(frozen=True)
class CardVerdict:
    """At-a-glance assessment of a single scorecard.

    Bubbles up the answer to "should I use this dataset?" so users
    don't have to read three nested strategy bullets to figure it out.
    """

    group: GroupKey
    headline: str        # e.g. "Direct fit (strong, 0.85)" or "Reframing (subset, 0.62)"
    one_liner: str       # First sentence of the best strategy's rationale, capped.
    use_as: str          # Practical "what to do with it" guidance.
    primary_kind: str | None  # The best strategy's kind value (for badge styling).
    confidence: float | None
    n_strategies: int


def _verdict_for(sc: Scorecard) -> CardVerdict:
    """Compute the at-a-glance verdict for a scorecard."""
    best = sc.best_strategy
    if best is None or not sc.strategies:
        return CardVerdict(
            group="not_useful",
            headline="Unassessed",
            one_liner="No strategy assessment ran for this candidate.",
            use_as="Inspect manually before deciding.",
            primary_kind=None,
            confidence=None,
            n_strategies=0,
        )

    kind_str = best.kind.value
    group = _KIND_TO_GROUP.get(kind_str, "not_useful")
    band = _confidence_band(best.confidence)

    headline_kind = {
        "direct_use": "Direct fit",
        "subset_extraction": "Reframing (subset)",
        "label_remapping": "Reframing (label remap)",
        "cross_class_repurposing": "Reframing (class flip)",
        "signal_proxy": "Signal proxy",
        "benign_baseline": "Benign baseline",
        "not_useful": "Not useful",
        "composition_only": "Composition-only",
    }.get(kind_str, kind_str)

    headline = f"{headline_kind} ({band}, {best.confidence:.2f})"

    use_as = {
        "direct_use": "Use as labeled positives.",
        "subset_extraction": "Filter to the relevant subset, then use.",
        "label_remapping": "Remap labels per the transform, then use.",
        "cross_class_repurposing": "Use as hard negatives.",
        "signal_proxy": "Train on; exclude from eval (proxy label).",
        "benign_baseline": "Use as benign distribution.",
        "not_useful": "Skip.",
        "composition_only": "Pair with another component (composition-only).",
    }.get(kind_str, "Review the strategy below.")

    # First sentence of the rationale, capped — the elevator pitch.
    rationale = best.rationale.strip()
    first = rationale.split(".", 1)[0].strip() + "." if "." in rationale else rationale
    if len(first) > 240:
        first = first[:237] + "…"

    return CardVerdict(
        group=group,
        headline=headline,
        one_liner=first,
        use_as=use_as,
        primary_kind=kind_str,
        confidence=best.confidence,
        n_strategies=len(sc.strategies),
    )


# ─── Strategy-kind grouping ────────────────────────────────────────


@dataclass(frozen=True)
class GroupedCard:
    """A scorecard plus its derived verdict. The render unit."""

    scorecard: Scorecard
    verdict: CardVerdict
    rank: int  # 1-indexed position within its group


@dataclass(frozen=True)
class StrategyGroup:
    """A bucket of cards sharing a primary-strategy kind."""

    key: GroupKey
    label: str
    description: str
    cards: tuple[GroupedCard, ...]

    @property
    def count(self) -> int:
        return len(self.cards)


# ─── Recipe / curate preview ───────────────────────────────────────


@dataclass(frozen=True)
class RecipePreviewComponent:
    """One line in the recipe-preview table."""

    candidate_id: str
    primary_kind: str
    confidence: float
    take: int | None  # None means "all"; integer is the auto-cap value
    label_kind: str | None  # ground_truth / proxy / remapped / etc.


@dataclass(frozen=True)
class RecipePreview:
    """Summary of what a `curate` run on this recon would produce."""

    n_components: int
    n_direct_fit: int
    n_reframing: int
    n_proxy: int
    n_benign: int
    components: tuple[RecipePreviewComponent, ...]
    estimated_rows: int  # Sum of `take` across components (cap at integer values)
    next_command: str

    @property
    def has_components(self) -> bool:
        return self.n_components > 0


def _build_recipe_preview(
    cards_by_group: dict[GroupKey, list[GroupedCard]],
    min_confidence: float = 0.5,
) -> RecipePreview:
    """Build the recipe preview by scanning cards above the confidence threshold.

    Mirrors recipe_draft.py logic: cards with best_strategy.confidence
    >= min_strategy_confidence land in the recipe; below are declined.
    """
    components: list[RecipePreviewComponent] = []
    total_rows = 0
    n_direct = n_reframe = n_proxy = n_benign = 0

    for group_key in DISPLAY_GROUPS_ORDER:
        if group_key == "not_useful":
            continue  # never lands in recipe
        for card in cards_by_group.get(group_key, []):
            v = card.verdict
            if v.confidence is None or v.confidence < min_confidence:
                continue
            sc = card.scorecard
            best = sc.best_strategy
            if best is None:
                continue
            transform = best.transform
            take = transform.take
            take_int: int | None = None if take == "all" else int(take)
            # Estimate: use `take` if capped, else assume 5000 (the
            # default auto-cap in recipe_draft.py).
            row_estimate = take_int if take_int is not None else 5000
            total_rows += row_estimate

            label_kind = None
            if transform.label_kind_map:
                values = list(transform.label_kind_map.values())
                # If all map to the same label_kind, surface it.
                unique = set(values)
                label_kind = next(iter(unique)) if len(unique) == 1 else "mixed"

            components.append(
                RecipePreviewComponent(
                    candidate_id=f"{sc.candidate.source}:{sc.candidate.id}",
                    primary_kind=v.primary_kind or "?",
                    confidence=v.confidence,
                    take=take_int,
                    label_kind=label_kind,
                )
            )
            if group_key == "direct_fit":
                n_direct += 1
            elif group_key == "reframing":
                n_reframe += 1
            elif group_key == "signal_proxy":
                n_proxy += 1
            elif group_key == "benign_baseline":
                n_benign += 1

    next_command = "datascout curate --from <out>/recipe.draft.yaml --out ./mycorpus"
    return RecipePreview(
        n_components=len(components),
        n_direct_fit=n_direct,
        n_reframing=n_reframe,
        n_proxy=n_proxy,
        n_benign=n_benign,
        components=tuple(components),
        estimated_rows=total_rows,
        next_command=next_command,
    )


# ─── ReconReportContext (extended) ─────────────────────────────────


@dataclass(frozen=True)
class ReconReportContext:
    """Pre-computed flags, counts, groupings, and recipe preview.

    Every "what mode are we in?", "what's the verdict?", and
    "what would curate produce?" question answered exactly once.
    """

    # Mode flags
    metadata_only: bool
    llm_runtime_error: bool
    has_strategies: bool
    has_decomposition: bool
    has_gaps: bool
    notable_gaps: bool
    sparse_coverage: bool
    no_direct_fits: bool
    # Counts
    n_candidates: int
    n_strategy_assessed: int
    n_directions: int
    n_papers: int
    n_paper_dataset_citations: int
    # Insight layer (NEW)
    groups: tuple[StrategyGroup, ...] = field(default_factory=tuple)
    recipe_preview: RecipePreview | None = None

    @classmethod
    def from_result(
        cls, result: ReconResult, *, min_strategy_confidence: float = 0.5
    ) -> ReconReportContext:
        metadata_only = any(_METADATA_ONLY_SENTINEL in n for n in result.notices)
        llm_runtime_error = any(_LLM_RUNTIME_SENTINEL in n for n in result.notices)
        has_strategies = any(sc.strategies for sc in result.candidates)
        has_decomposition = bool(result.coverage and result.coverage.decomposition)
        has_gaps = bool(result.coverage and result.coverage.semantic_gaps)
        notable_gaps = bool(result.coverage and len(result.coverage.semantic_gaps) >= 2)

        sparse_coverage = (
            has_decomposition
            and has_gaps
            and len(result.candidates) <= 5
            and not metadata_only
            and not llm_runtime_error
        )
        no_direct_fits = bool(
            has_strategies
            and not any(
                (sc.best_strategy is not None and sc.best_strategy.kind.value == "direct_use")
                for sc in result.candidates
            )
        )

        n_directions = len(result.coverage.decomposition) if result.coverage else 0
        n_strategy_assessed = sum(1 for sc in result.candidates if sc.strategies)
        n_paper_dataset_citations = sum(
            len(p.referenced_datasets) for p in result.papers
        )

        # Compute verdicts and group cards.
        cards_by_group: dict[GroupKey, list[GroupedCard]] = {
            k: [] for k in DISPLAY_GROUPS_ORDER
        }
        for sc in result.candidates:
            verdict = _verdict_for(sc)
            cards_by_group[verdict.group].append(GroupedCard(sc, verdict, rank=0))

        # Re-rank within each group: highest confidence first; cards
        # without confidence (verdict is None) sink to the bottom.
        groups: list[StrategyGroup] = []
        for key in DISPLAY_GROUPS_ORDER:
            cards = cards_by_group[key]
            cards.sort(
                key=lambda c: (c.verdict.confidence is None, -(c.verdict.confidence or 0.0))
            )
            ranked: list[GroupedCard] = []
            for i, c in enumerate(cards, start=1):
                ranked.append(GroupedCard(c.scorecard, c.verdict, rank=i))
            cards_by_group[key] = ranked
            groups.append(
                StrategyGroup(
                    key=key,
                    label=GROUP_LABELS[key],
                    description=GROUP_DESCRIPTIONS[key],
                    cards=tuple(ranked),
                )
            )

        recipe_preview = _build_recipe_preview(
            cards_by_group, min_confidence=min_strategy_confidence
        )

        return cls(
            metadata_only=metadata_only,
            llm_runtime_error=llm_runtime_error,
            has_strategies=has_strategies,
            has_decomposition=has_decomposition,
            has_gaps=has_gaps,
            notable_gaps=notable_gaps,
            sparse_coverage=sparse_coverage,
            no_direct_fits=no_direct_fits,
            n_candidates=len(result.candidates),
            n_strategy_assessed=n_strategy_assessed,
            n_directions=n_directions,
            n_papers=len(result.papers),
            n_paper_dataset_citations=n_paper_dataset_citations,
            groups=tuple(groups),
            recipe_preview=recipe_preview,
        )

    @property
    def show_gaps_lead(self) -> bool:
        """True when the report should lead with the sourcing-roadmap section."""
        return self.has_gaps and (self.notable_gaps or self.sparse_coverage)

    @property
    def show_papers(self) -> bool:
        """True when the report should render the academic-papers section."""
        return self.n_papers > 0

    @property
    def show_recipe_preview(self) -> bool:
        """True when the recipe-preview section should render.

        We only show the preview when there's at least one component
        that would actually land in the recipe; otherwise the section
        would mislead users into running curate on nothing.
        """
        return self.recipe_preview is not None and self.recipe_preview.has_components


__all__ = [
    "DISPLAY_GROUPS_ORDER",
    "GROUP_DESCRIPTIONS",
    "GROUP_LABELS",
    "CardVerdict",
    "GroupKey",
    "GroupedCard",
    "RecipePreview",
    "RecipePreviewComponent",
    "ReconReportContext",
    "StrategyGroup",
]

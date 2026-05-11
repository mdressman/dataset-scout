"""Curate orchestrator (M4b — audit-ready).

Recipe → materialised JSONL corpus + lockfile + manifest + report +
fingerprint + usage. M4b lands the leakage-aware splitter (MinHash +
LSH grouping) and the minimal filter expression DSL, flipping
`recipe.lock.yaml` `audit_readiness` from `preview` to `ready`.

Public API:

    from dataset_scout.curate import run_curate
    result = run_curate(recipe, out_dir, ctx=ctx)

The CLI verb is a thin wrapper.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml  # type: ignore[import-untyped]

from dataset_scout.context import ScoutContext
from dataset_scout.core import LabelKind, NormalizedRecord, StrategyKind
from dataset_scout.errors import DatasetScoutError
from dataset_scout.recipe import (
    DEFAULT_SPLITS,
    Recipe,
    RecipeComponent,
    normalize_split_proportions,
)

if TYPE_CHECKING:
    from dataset_scout.sources.base import Source


# Bumped when the curate output schema changes in a breaking way.
CURATE_VERSION = "1"

# Heuristic order for picking a row-id column when the source row has
# multiple plausible candidates. First match wins.
_ID_COLUMN_PREFERENCE: tuple[str, ...] = (
    "id",
    "uuid",
    "row_id",
    "idx",
    "index",
)


# Output filenames written under <out>/.
JSONL_FILENAMES = {
    "train": "train.jsonl",
    "val": "val.jsonl",
    "test": "test.jsonl",
}


# ─── public types ────────────────────────────────────────────────────


class CurateOverrides:
    """Effective-value record: what the recipe said vs what curate ran with.

    Both go into the lockfile so a reviewer reading the lock can tell
    whether a CLI flag deviated from the source-of-truth recipe.
    """

    __slots__ = ("min_conf_effective", "min_conf_recipe", "seed_effective", "seed_recipe")

    def __init__(
        self,
        *,
        seed_recipe: int,
        seed_effective: int,
        min_conf_recipe: float,
        min_conf_effective: float,
    ) -> None:
        self.seed_recipe = seed_recipe
        self.seed_effective = seed_effective
        self.min_conf_recipe = min_conf_recipe
        self.min_conf_effective = min_conf_effective

    @property
    def seed_overridden(self) -> bool:
        return self.seed_recipe != self.seed_effective

    @property
    def min_conf_overridden(self) -> bool:
        return self.min_conf_recipe != self.min_conf_effective


class CurateResult:
    """Summary of a curate run; mostly used by the CLI to print a tidy line."""

    def __init__(
        self,
        *,
        out_dir: Path,
        components_kept: int,
        components_skipped: int,
        components_failed: int,
        rows_per_split: dict[str, int],
        fingerprint: str,
        elapsed_seconds: float,
        failures: list[dict[str, Any]] | None = None,
        components_zero_row: int = 0,
    ) -> None:
        self.out_dir = out_dir
        self.components_kept = components_kept
        self.components_skipped = components_skipped
        self.components_failed = components_failed
        self.rows_per_split = rows_per_split
        self.fingerprint = fingerprint
        self.elapsed_seconds = elapsed_seconds
        self.failures = failures or []
        # Components that materialised (no upstream error) but produced zero
        # rows after row-to-record conversion — usually because the recipe's
        # text / label columns didn't match anything in the source. Counted
        # separately from `components_kept` (which is "non-failed") and
        # surfaced in the CLI exit panel so the user notices honestly.
        self.components_zero_row = components_zero_row

    @property
    def total_rows(self) -> int:
        return sum(self.rows_per_split.values())


# ─── recipe loading ──────────────────────────────────────────────────


def load_recipe(path: Path) -> Recipe:
    """Parse a recipe YAML file into a typed `Recipe`."""
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text) or {}
    return Recipe.model_validate(data)


# ─── filter handling (M4b: minimal expression DSL) ──────────────────


def _compile_component_filter(c: RecipeComponent) -> Callable[[dict[str, Any]], bool] | None:
    """Compile a component's filter expression to a callable, or None.

    Compile errors (disallowed syntax / unknown function) raise
    `DatasetScoutError` with the offending component id; the user
    must fix the recipe.
    """
    if not c.transform.filter:
        return None
    from dataset_scout.filter_dsl import FilterCompileError, compile_filter

    try:
        return compile_filter(c.transform.filter)
    except FilterCompileError as exc:
        raise DatasetScoutError(
            f"component {c.id!r} has an invalid filter expression: {exc}. "
            f"Offending expression: {c.transform.filter!r}"
        ) from exc


def _precompile_filters(
    components: list[RecipeComponent],
) -> dict[str, Any]:
    """Validate every component's filter at the top of curate."""
    return {c.id: _compile_component_filter(c) for c in components}


# ─── composition cross-references ────────────────────────────────────


def _validate_composition_references(components: list[RecipeComponent]) -> None:
    """Cheap referential integrity. Full portfolio semantics deferred."""
    ids = {c.id for c in components}
    for c in components:
        if c.id in c.composes_with:
            raise DatasetScoutError(f"Component {c.id} composes_with includes itself.")
        for partner in c.composes_with:
            if partner not in ids:
                raise DatasetScoutError(
                    f"Component {c.id} composes_with references unknown id '{partner}'."
                )
        if c.strategy == StrategyKind.COMPOSITION_ONLY:
            raise DatasetScoutError(
                f"Component {c.id} uses strategy 'composition_only', "
                "which is reserved for a future portfolio pass."
            )


# ─── filter components by min_strategy_confidence ────────────────────


def _filter_by_confidence(
    components: list[RecipeComponent], threshold: float
) -> tuple[list[RecipeComponent], list[RecipeComponent]]:
    kept: list[RecipeComponent] = []
    dropped: list[RecipeComponent] = []
    for c in components:
        if c.strategy == StrategyKind.NOT_USEFUL:
            dropped.append(c)
            continue
        if c.strategy_confidence < threshold:
            dropped.append(c)
            continue
        kept.append(c)
    return kept, dropped


# ─── row identity ────────────────────────────────────────────────────


def _row_id(row: dict[str, Any], fallback_index: int) -> tuple[str, str]:
    """Return (row_id, identity_method).

    Identity methods, in order of preference:
      - `column:<name>` — value of an upstream id column
      - `sha256:row` — sha256 of canonical-JSON of the row
      - `index` — last-resort fallback (when JSON canonicalisation fails)
    """
    for col in _ID_COLUMN_PREFERENCE:
        if col in row and row[col] is not None:
            return str(row[col]), f"column:{col}"
    try:
        canonical = json.dumps(_jsonable(row), sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return f"idx:{fallback_index}", "index"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16], "sha256:row"


# ─── multimodal coercion ─────────────────────────────────────────────


def _jsonable(value: Any) -> Any:
    """Coerce a value into something JSON can serialize.

    Returns the value unchanged when already JSON-friendly; stringifies
    bytes / arrays / nested feature objects otherwise. This is a pragmatic
    fallback so multimodal HF datasets don't crash the writer.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return repr(value)


def _coerce_extras(row: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Return (jsonable_row, was_coerced)."""
    coerced = False
    out: dict[str, Any] = {}
    for k, v in row.items():
        new_v = _jsonable(v)
        if new_v is not v and not isinstance(v, (dict, list, tuple)):
            coerced = True
        out[str(k)] = new_v
    return out, coerced


# ─── source dispatch (mirrors pipeline.py) ───────────────────────────


def _build_source_index(
    ctx: ScoutContext,
    *,
    sources_override: list[Source] | None = None,
) -> dict[str, Source]:
    """Index sources by `name`. Tests can pass `sources_override` to
    inject a FakeSource."""
    from dataset_scout.sources.factory import build_source_index

    return build_source_index(ctx, override=sources_override)


# ─── component materialisation ───────────────────────────────────────


def _classify_component_failure(c: RecipeComponent, exc: BaseException) -> dict[str, Any]:
    """Map a per-component exception to a structured failure record with
    a category and an actionable hint. Categories are stable strings so
    the lockfile + report stay machine-readable.

    The classifier is heuristic — we look at exception type names and
    message text, since HuggingFace `datasets` raises a fan-out of
    typed errors with similar surfaces. Unknown failures still get
    captured (under category="unknown") so nothing is silently dropped.
    """
    msg = str(exc) or exc.__class__.__name__
    type_name = exc.__class__.__name__
    lower = msg.lower()

    # Source explicitly does not support streaming for this candidate
    # (e.g. Kaggle, which is discovery-only). Distinct from "no rows" —
    # this is structural, not a transient blip.
    if "SourceUnsupportedError" in type_name:
        category = "unsupported_source"
        hint = (
            "This source does not support streaming rows (e.g. Kaggle is "
            "discovery-only). Remove the component, or materialise the "
            "data manually using the source's native tools and re-add as "
            "a local file in a future recipe shape."
        )
    # Auth-gated dataset (HF requires login).
    elif "gated" in lower or "must be authenticated" in lower or "401" in msg:
        category = "gated_dataset"
        hint = "Authenticate (`HF_TOKEN=...`) or remove this component from the recipe."
    # Multi-config dataset that wasn't pinned to one.
    elif "config name is missing" in lower or "pick one among the available configs" in lower:
        category = "missing_config"
        hint = (
            "Set `source_config: <config_name>` on this component (the HF "
            f"dataset {c.source_id!r} has multiple configs)."
        )
    # Split name doesn't exist on the dataset.
    elif "bad split" in lower or "available splits" in lower:
        category = "bad_split"
        hint = (
            "Set `source_split: <split_name>` on this component to one of "
            "the available splits, or remove the component."
        )
    elif (
        "no (supported) data files" in lower
        or "doesn't contain any data files" in lower
        or "couldn't find any data file" in lower
    ):
        category = "no_data_files"
        hint = "The upstream dataset has no parseable data files — remove this component."
    elif (
        "error tokenizing data" in lower
        or "parsererror" in type_name.lower()
        or "could not parse" in lower
    ):
        category = "parse_error"
        hint = "The upstream data file is malformed; remove this component."
    elif "datasetnotfounderror" in type_name.lower() or "404" in msg or "does not exist" in lower:
        category = "not_found"
        hint = "The dataset was deleted or renamed — remove this component."
    elif "connection" in lower or "timeout" in lower or "network" in lower:
        category = "network"
        hint = "Transient network error; rerun once connectivity is back."
    elif "429" in msg or "rate limit" in lower or "too many requests" in lower:
        category = "rate_limited"
        hint = (
            "Hit upstream rate limits. Set HF_TOKEN for higher quotas, "
            "or lower --max-concurrency on curate."
        )
    elif "references columns not found" in lower:
        category = "schema_mismatch"
        hint = (
            "The recipe's text_column or label_column doesn't match the actual "
            "dataset schema. Edit the recipe to use real column names."
        )
    else:
        category = "unknown"
        hint = "See the message; if it recurs, file an issue with the trace."

    # Cap message length so the lockfile stays readable.
    short_msg = msg if len(msg) <= 240 else msg[:237] + "..."

    return {
        "id": c.id,
        "source": c.source,
        "source_id": c.source_id,
        "category": category,
        "exception_type": type_name,
        "message": short_msg,
        "hint": hint,
    }


def _validate_component_schema(
    component: RecipeComponent,
    first_row: dict[str, Any],
) -> list[str]:
    """Check transform column names against actual row schema.

    Returns a list of missing column names. Empty list means all columns verified.
    """
    available = set(first_row.keys())
    missing: list[str] = []
    if component.transform.text_column and component.transform.text_column not in available:
        missing.append(component.transform.text_column)
    if component.transform.label_column and component.transform.label_column not in available:
        missing.append(component.transform.label_column)
    return missing


def _materialize_component(
    component: RecipeComponent,
    *,
    source: Source,
    threat_family: str | None,
    filter_fn: Any = None,
    max_rows_override: int | None = None,
) -> Iterator[NormalizedRecord]:
    """Stream rows for one component and yield NormalizedRecord per row.

    The `take` parameter on the recipe transform is honored. Rows whose
    label_value_map mapping is missing for the source-side label are
    silently dropped (the user can add to the map and re-run).

    `filter_fn`, when provided, is the compiled filter expression. Rows
    that fail the filter are skipped before the transform runs — this
    is critical for `subset_extraction`, where we materialize only the
    rows the strategy describes.

    `max_rows_override`, when set, lowers the recipe's `take` value
    (or replaces ``"all"``) so a user can do a fast first-pass curate
    without hand-editing the recipe. It never raises the recipe's
    cap — recipe is the source of truth.
    """
    candidate = _component_to_candidate(component)
    take = component.transform.take
    take_int = None if take == "all" else int(take)
    if max_rows_override is not None:
        take_int = max_rows_override if take_int is None else min(take_int, max_rows_override)
    rows = source.stream_rows(
        candidate,
        config=component.source_config,
        split=component.source_split,
        take=take_int,
    )
    schema_checked = False
    for i, row in enumerate(rows):
        if not schema_checked:
            missing = _validate_component_schema(component, row)
            if missing:
                raise DatasetScoutError(
                    f"Component {component.id!r} references columns not found in "
                    f"the dataset: {missing}. Available columns: {sorted(row.keys())}. "
                    f"Edit the recipe to fix text_column / label_column."
                )
            schema_checked = True
        if filter_fn is not None:
            try:
                if not filter_fn(row):
                    continue
            except Exception:
                continue
        rec = _row_to_record(component, row, i, threat_family)
        if rec is not None:
            yield rec


def _component_to_candidate(component: RecipeComponent) -> Any:
    """Build a minimal Candidate for the source's stream_rows call.

    We only need source/id/revision; the rest of the metadata envelope
    isn't read by HuggingFaceSource.stream_rows.
    """
    from dataset_scout.core import Candidate, CandidateMetadata

    return Candidate(
        source=component.source,
        id=component.source_id,
        revision=component.revision,
        metadata=CandidateMetadata(),
    )


def _materialize_all(
    components: list[RecipeComponent],
    *,
    source_index: dict[str, Source],
    threat_family: str | None,
    filter_fns: dict[str, Any],
    max_rows_override: int | None,
    max_concurrency: int,
) -> tuple[dict[str, list[NormalizedRecord]], dict[str, dict[str, Any]]]:
    """Materialise components in parallel via a thread pool.

    Per-component ``_materialize_component`` calls are I/O bound (HF
    `load_dataset(streaming=True)` setup is the dominant cost), so a
    small thread pool gives near-linear speedup on multi-component
    recipes without changing the per-row code path.

    Returns ``(records_by_id, failures_by_id)``. Both are dicts
    keyed by component id so the caller can reassemble in original
    recipe order — preserving the determinism contract on
    `all_records` ordering, MinHash cluster identity, and the
    corpus fingerprint.

    KeyboardInterrupt is handled explicitly: pending tasks are
    cancelled and in-flight workers are allowed to finish, so
    Ctrl-C is responsive even with a large queue.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Clamp workers to the meaningful range. 1 worker == sequential
    # (still goes through the executor for code-path uniformity).
    workers = max(1, min(max_concurrency, len(components)))

    records_by_id: dict[str, list[NormalizedRecord]] = {}
    failures_by_id: dict[str, dict[str, Any]] = {}

    def _one(c: RecipeComponent) -> tuple[str, list[NormalizedRecord] | dict[str, Any]]:
        try:
            recs = list(
                _materialize_component(
                    c,
                    source=source_index[c.source],
                    threat_family=threat_family,
                    filter_fn=filter_fns.get(c.id),
                    max_rows_override=max_rows_override,
                )
            )
            return c.id, recs
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            return c.id, _classify_component_failure(c, exc)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, c): c for c in components}
        try:
            for fut in as_completed(futures):
                cid, payload = fut.result()
                if isinstance(payload, list):
                    records_by_id[cid] = payload
                else:
                    failures_by_id[cid] = payload
        except KeyboardInterrupt:
            for f in futures:
                f.cancel()
            raise

    return records_by_id, failures_by_id


def _row_to_record(
    component: RecipeComponent,
    row: dict[str, Any],
    index: int,
    threat_family: str | None,
) -> NormalizedRecord | None:
    """Apply transform to one row. Returns None if the row should be skipped."""
    transform = component.transform

    text_col = transform.text_column
    if text_col is None or text_col not in row:
        return None
    text_val = row[text_col]
    if text_val is None:
        return None
    text = str(text_val)

    label_col = transform.label_column
    label_kind: LabelKind = LabelKind.GROUND_TRUTH
    if label_col is None:
        # No label column: derive from label_kind_map["all"] and
        # label_value_map["all"] when present, else default to benign + ground_truth.
        label_value = transform.label_value_map.get("all", "benign")
        label_kind = _resolve_label_kind(transform.label_kind_map.get("all"))
    else:
        if label_col not in row or row[label_col] is None:
            return None
        raw_label = str(row[label_col])
        if raw_label not in transform.label_value_map:
            return None  # row label not mapped — skip honestly
        label_value = transform.label_value_map[raw_label]
        label_kind = _resolve_label_kind(
            transform.label_kind_map.get(raw_label) or transform.label_kind_map.get("all")
        )

    extras, coerced = _coerce_extras(row)
    row_id, _ = _row_id(row, index)

    source_str = f"{component.source}:{component.source_id}"

    return NormalizedRecord(
        text=text,
        label=label_value,
        label_kind=label_kind,
        strategy=component.strategy,
        strategy_confidence=component.strategy_confidence,
        source=source_str,
        source_row_id=row_id,
        source_revision=component.revision,
        source_config=component.source_config,
        source_split=component.source_split,
        threat_family=threat_family,
        extras=extras,
        extras_coercion=coerced,
    )


def _resolve_label_kind(value: str | None) -> LabelKind:
    if value is None:
        return LabelKind.GROUND_TRUTH
    try:
        return LabelKind(value)
    except ValueError:
        return LabelKind.GROUND_TRUTH


# ─── splitter (M4b: leakage-aware via MinHash + LSH) ────────────────


def _split_records(
    records: list[NormalizedRecord],
    proportions: dict[str, float],
    *,
    seed: int,
    leakage_keys: list[str],
) -> tuple[dict[str, list[NormalizedRecord]], dict[str, Any]]:
    """Leakage-aware split using MinHash + LSH.

    Returns (splits, dedup_stats). `dedup_stats` carries the cluster
    counts and parameters used so they can land in `recipe.lock.yaml`.
    Near-duplicate rows always end up in the SAME split — eval signal
    can't leak into training via paraphrases.
    """
    from dataset_scout.dedup import leakage_aware_split

    return leakage_aware_split(
        records,
        proportions,
        seed=seed,
        leakage_keys=leakage_keys,
    )


# ─── output writers ──────────────────────────────────────────────────


def _write_jsonl(path: Path, records: list[NormalizedRecord]) -> str:
    """Write records as JSONL and return the sha256 of the file's bytes."""
    h = hashlib.sha256()
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            line = rec.model_dump_json() + "\n"
            f.write(line)
            h.update(line.encode("utf-8"))
    return h.hexdigest()


def _write_lockfile(
    path: Path,
    *,
    recipe: Recipe,
    kept: list[RecipeComponent],
    dropped: list[RecipeComponent],
    failures: list[dict[str, Any]],
    realized: dict[str, dict[str, Any]],
    splits: dict[str, list[NormalizedRecord]],
    dedup_stats: dict[str, Any],
    overrides: CurateOverrides,
    fingerprint: str,
    started_iso: str,
    elapsed_s: float,
) -> None:
    label_warnings = _label_distribution_warnings(splits)
    payload = {
        "recipe_version": recipe.recipe_version,
        "curate_version": CURATE_VERSION,
        "audit_readiness": "ready",
        "audit_readiness_notes": [
            (
                "Splits use MinHash + LSH grouping ("
                f"num_perm={dedup_stats.get('num_perm')}, "
                f"threshold={dedup_stats.get('threshold')}); "
                "near-duplicate rows are kept on the same side of the split."
            ),
            (
                "Filter expressions parsed via the dataset-scout DSL "
                "(ast-validated; whitelist of operators + functions)."
            ),
        ],
        "intent": recipe.intent.model_dump(mode="json"),
        "min_strategy_confidence": {
            "recipe": overrides.min_conf_recipe,
            "effective": overrides.min_conf_effective,
            "overridden_by_cli": overrides.min_conf_overridden,
        },
        "seed": {
            "recipe": overrides.seed_recipe,
            "effective": overrides.seed_effective,
            "overridden_by_cli": overrides.seed_overridden,
        },
        "splits": {
            "method": dedup_stats.get("method", "minhash_lsh"),
            "num_perm": dedup_stats.get("num_perm"),
            "threshold": dedup_stats.get("threshold"),
            "shingle_size": dedup_stats.get("shingle_size"),
            "dedup_version": dedup_stats.get("dedup_version"),
            "clusters_total": dedup_stats.get("clusters_total"),
            "clusters_singleton": dedup_stats.get("clusters_singleton"),
            "clusters_largest": dedup_stats.get("clusters_largest"),
            "rows_in_dup_clusters": dedup_stats.get("rows_in_dup_clusters"),
            "proportions_recipe": recipe.splits.model_dump(mode="json"),
            "realized": {name: len(rows) for name, rows in splits.items()},
        },
        "leakage_keys": list(recipe.leakage_keys),
        "components": [_component_lock_entry(c, realized.get(c.id, {})) for c in kept],
        "declined_components": [
            {
                "id": c.id,
                "source": c.source,
                "source_id": c.source_id,
                "strategy": c.strategy.value,
                "strategy_confidence": c.strategy_confidence,
                "reason": (
                    "below min_strategy_confidence"
                    if c.strategy_confidence < overrides.min_conf_effective
                    and c.strategy != StrategyKind.NOT_USEFUL
                    else c.strategy.value
                ),
            }
            for c in dropped
        ],
        "failed_components": list(failures),
        "label_distribution_warnings": label_warnings,
        "dedup_impact": {
            "rows_total_before_split": sum(len(rows) for rows in splits.values()),
            "rows_in_dup_clusters": dedup_stats.get("rows_in_dup_clusters", 0),
            "clusters_total": dedup_stats.get("clusters_total", 0),
            "clusters_non_singleton": (
                dedup_stats.get("clusters_total", 0)
                - dedup_stats.get("clusters_singleton", 0)
            ),
        },
        "fingerprint": fingerprint,
        "started_at": started_iso,
        "elapsed_seconds": round(elapsed_s, 3),
        "scout_version": _scout_version(),
    }
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _component_lock_entry(c: RecipeComponent, realized: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": c.id,
        "source": c.source,
        "source_id": c.source_id,
        "revision": c.revision,
        "source_config": c.source_config,
        "source_split": c.source_split,
        "strategy": c.strategy.value,
        "strategy_confidence": c.strategy_confidence,
        "rationale": c.rationale,
        "caveats": list(c.caveats),
        "transform": c.transform.model_dump(mode="json"),
        "composes_with": list(c.composes_with),
        "realized": realized,
    }


def _write_manifest(path: Path, lock_path: Path) -> None:
    """Machine-readable equivalent of the lockfile (just JSON of the YAML)."""
    payload = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def _write_report(
    path: Path,
    *,
    recipe: Recipe,
    splits: dict[str, list[NormalizedRecord]],
    realized: dict[str, dict[str, Any]],
    failures: list[dict[str, Any]],
    overrides: CurateOverrides,
    dedup_stats: dict[str, Any],
    fingerprint: str,
    elapsed_s: float,
) -> None:
    total = sum(len(v) for v in splits.values())
    label_counts = _aggregate_label_counts(splits)
    licenses = _aggregate_licenses(realized)

    lines: list[str] = [
        f"# Curated corpus — {recipe.intent.brief.strip().splitlines()[0][:80]}",
        "",
        "> ✅ **Audit-ready.**  ",
        "> Splits are leakage-aware: near-duplicate rows are clustered "
        f"via MinHash + LSH (num_perm={dedup_stats.get('num_perm')}, "
        f"threshold={dedup_stats.get('threshold')}) and assigned to a "
        "single split. Filter expressions are parsed via a whitelisted "
        "expression DSL. `recipe.lock.yaml` records every parameter and "
        "override.",
        "",
        "## Stats",
        "",
        f"- **{splits['train'].__len__():,} train** · "
        f"**{splits['val'].__len__():,} val** · "
        f"**{splits['test'].__len__():,} test** "
        f"({total:,} rows total)",
        f"- **Labels:** "
        f"{label_counts['positive']} positive · "
        f"{label_counts['benign']} benign · "
        f"{label_counts['hard_negative']} hard_negative",
        f"- **Label kinds:** "
        f"{label_counts['ground_truth']} ground_truth · "
        f"{label_counts['proxy']} proxy · "
        f"{label_counts['remapped']} remapped · "
        f"{label_counts['subset_extracted']} subset_extracted",
        f"- **Licenses:** {licenses or '(unknown)'}",
        f"- **Dedup clusters:** {dedup_stats.get('clusters_total', 0):,} "
        f"({dedup_stats.get('clusters_singleton', 0):,} singletons, "
        f"largest = {dedup_stats.get('clusters_largest', 0)}; "
        f"{dedup_stats.get('rows_in_dup_clusters', 0):,} rows in dup clusters)",
        f"- **Fingerprint:** `{fingerprint[:16]}…`",
        f"- **Wall-clock:** {elapsed_s:.2f}s",
        "",
    ]
    if overrides.min_conf_overridden or overrides.seed_overridden:
        lines += [
            "## CLI overrides applied",
            "",
        ]
        if overrides.min_conf_overridden:
            lines.append(
                f"- `--min-strategy-confidence` "
                f"({overrides.min_conf_recipe} → {overrides.min_conf_effective})"
            )
        if overrides.seed_overridden:
            lines.append(f"- `--seed` ({overrides.seed_recipe} → {overrides.seed_effective})")
        lines.append("")
    lines += [
        "## Components",
        "",
    ]
    for cid, info in realized.items():
        rows = info.get("rows_taken", 0)
        license_raw = info.get("license_raw") or "(unknown)"
        lines.append(f"- `{cid}` — {rows:,} rows · license `{license_raw}`")

    if failures:
        lines += [
            "",
            f"## Components skipped due to upstream errors ({len(failures)})",
            "",
            "These components were in the recipe but couldn't be "
            "materialized. The lockfile records each one under "
            "`failed_components` with category and hint.",
            "",
        ]
        for f in failures:
            lines.append(f"- `{f['id']}` — **{f['category']}** · {f['hint']}")

    label_warnings = _label_distribution_warnings(splits)
    if label_warnings:
        lines += [
            "",
            "## ⚠️ Label distribution warnings",
            "",
        ]
        for w in label_warnings:
            lines.append(f"- {w}")

    dup_rows = dedup_stats.get("rows_in_dup_clusters", 0)
    dup_clusters_non_singleton = (
        dedup_stats.get("clusters_total", 0) - dedup_stats.get("clusters_singleton", 0)
    )
    if dup_rows > 0:
        lines += [
            "",
            "## Dedup impact",
            "",
            f"- **{dup_rows:,}** rows fell into **{dup_clusters_non_singleton:,}** "
            f"near-duplicate clusters (kept together in the same split to prevent leakage)",
        ]
        total_rows = sum(len(v) for v in splits.values())
        if total_rows > 0:
            pct = dup_rows / total_rows * 100
            lines.append(f"- Near-duplicate rate: **{pct:.1f}%** of total rows")

    lines += [
        "",
        "---",
        "",
        "_License signals are an SPDX-best-effort guess. Always read the "
        "upstream card before redistributing. This is not legal advice._",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _aggregate_label_counts(
    splits: dict[str, list[NormalizedRecord]],
) -> dict[str, int]:
    out: dict[str, int] = {
        "positive": 0,
        "benign": 0,
        "hard_negative": 0,
        "ground_truth": 0,
        "proxy": 0,
        "remapped": 0,
        "subset_extracted": 0,
    }
    for recs in splits.values():
        for rec in recs:
            out[rec.label] += 1
            out[rec.label_kind.value] += 1
    return out


def _label_distribution_warnings(
    splits: dict[str, list[NormalizedRecord]],
) -> list[str]:
    """Generate warnings for skewed or suspicious label distributions."""
    warnings: list[str] = []
    label_counts = _aggregate_label_counts(splits)
    total = sum(label_counts.get(lbl, 0) for lbl in ("positive", "benign", "hard_negative"))
    if total == 0:
        return warnings

    proxy_count = label_counts.get("proxy", 0)
    ground_truth_count = label_counts.get("ground_truth", 0)
    kind_total = proxy_count + ground_truth_count + label_counts.get("remapped", 0) + label_counts.get("subset_extracted", 0)

    if kind_total > 0 and proxy_count / kind_total > 0.5:
        warnings.append(
            f"Over half the rows ({proxy_count}/{kind_total}) have label_kind=proxy. "
            f"Exclude proxies from evaluation splits."
        )

    if kind_total > 0 and ground_truth_count == 0:
        warnings.append(
            "No ground_truth rows in the corpus. All labels are derived "
            "(proxy, remapped, or subset_extracted)."
        )

    positive_count = label_counts.get("positive", 0)
    benign_count = label_counts.get("benign", 0)
    if total > 0 and positive_count == 0:
        warnings.append("No positive-labeled rows in the corpus.")
    if total > 0 and benign_count == 0:
        warnings.append("No benign-labeled rows in the corpus.")

    if total > 10:
        minority = min(positive_count, benign_count)
        majority = max(positive_count, benign_count)
        if majority > 0 and minority / majority < 0.05:
            warnings.append(
                f"Severe class imbalance: {minority} vs {majority} "
                f"(positive vs benign). Consider rebalancing."
            )

    return warnings


def _aggregate_licenses(realized: dict[str, dict[str, Any]]) -> str:
    from collections import Counter

    counter: Counter[str] = Counter()
    for info in realized.values():
        spdx = info.get("license_spdx") or info.get("license_raw") or "(unknown)"
        counter[spdx] += 1
    if not counter:
        return ""
    return ", ".join(f"{count} {name}" for name, count in counter.most_common())


def _write_fingerprint(path: Path, fingerprint: str) -> None:
    path.write_text(fingerprint + "\n", encoding="utf-8")


def _write_usage(path: Path, out_dir: Path) -> None:
    name = out_dir.name
    path.write_text(
        (
            "# Usage snippets\n"
            "\n"
            "```python\n"
            "# huggingface_hub datasets\n"
            "from datasets import load_dataset\n"
            f"ds = load_dataset('json', data_files={{'train': '{name}/train.jsonl', 'val': '{name}/val.jsonl', 'test': '{name}/test.jsonl'}})\n"
            "```\n"
            "\n"
            "```python\n"
            "# pandas\n"
            "import pandas as pd\n"
            f"train = pd.read_json('{name}/train.jsonl', lines=True)\n"
            "```\n"
            "\n"
            "```python\n"
            "# raw jsonl\n"
            "import json\n"
            f"rows = [json.loads(l) for l in open('{name}/train.jsonl')]\n"
            "```\n"
        ),
        encoding="utf-8",
    )


def _scout_version() -> str:
    from dataset_scout import __version__

    return __version__


def _compute_fingerprint(jsonl_hashes: list[str]) -> str:
    h = hashlib.sha256()
    for hh in jsonl_hashes:
        h.update(hh.encode("ascii"))
    return h.hexdigest()


# ─── orchestrator ────────────────────────────────────────────────────


def run_curate(
    recipe: Recipe,
    out_dir: Path,
    *,
    ctx: ScoutContext,
    sources_override: list[Source] | None = None,
    seed_override: int | None = None,
    min_strategy_confidence_override: float | None = None,
    max_rows_per_component: int | None = None,
    max_concurrency: int = 4,
) -> CurateResult:
    """Materialise a recipe into a corpus directory.

    Raises `DatasetScoutError` on validation problems (filter strings,
    composition references, missing components). Source-side errors
    propagate; M4b adds a more granular error policy.
    """
    import time

    started = time.monotonic()
    started_iso = datetime.now(UTC).isoformat()

    # ── effective values (recipe-authoritative, CLI as override) ──
    overrides = CurateOverrides(
        seed_recipe=recipe.seed,
        seed_effective=seed_override if seed_override is not None else recipe.seed,
        min_conf_recipe=recipe.min_strategy_confidence,
        min_conf_effective=(
            min_strategy_confidence_override
            if min_strategy_confidence_override is not None
            else recipe.min_strategy_confidence
        ),
    )

    # ── validation ──
    filter_fns = _precompile_filters(recipe.components)
    _validate_composition_references(recipe.components)
    kept, dropped = _filter_by_confidence(recipe.components, overrides.min_conf_effective)

    if not kept:
        raise DatasetScoutError(
            "No components remain after applying min_strategy_confidence "
            f"({overrides.min_conf_effective:.2f}). Lower the threshold "
            "or hand-edit the recipe."
        )

    # ── source dispatch ──
    source_index = _build_source_index(ctx, sources_override=sources_override)
    for c in kept:
        if c.source not in source_index:
            raise DatasetScoutError(
                f"Component {c.id} uses unknown source '{c.source}'. "
                f"Enabled sources: {list(source_index)}."
            )

    # ── materialize (parallel across components) ──
    threat_family = recipe.intent.threat_families[0] if recipe.intent.threat_families else None
    per_component, failures_by_id = _materialize_all(
        kept,
        source_index=source_index,
        threat_family=threat_family,
        filter_fns=filter_fns,
        max_rows_override=max_rows_per_component,
        max_concurrency=max_concurrency,
    )

    # Reassemble in original recipe (`kept`) order so split assignment,
    # the lockfile components list, and the corpus fingerprint are
    # deterministic regardless of which workers finished first.
    all_records: list[NormalizedRecord] = []
    realized: dict[str, dict[str, Any]] = {}
    materialised: list[RecipeComponent] = []
    failures: list[dict[str, Any]] = []
    for c in kept:
        if c.id in per_component:
            recs = per_component[c.id]
            materialised.append(c)
            all_records.extend(recs)
            realized[c.id] = {
                "rows_taken": len(recs),
                "label_kind_counts": _label_kind_counts(recs),
                "license_raw": _component_license_raw(recs),
                "license_spdx": _component_license_spdx(recs),
                "row_identity_method": _component_identity_method(recs),
                "filter_applied": c.transform.filter,
            }
        elif c.id in failures_by_id:
            failures.append(failures_by_id[c.id])

    if not materialised:
        # All components failed — emit a structured error so the CLI
        # can show the user exactly why each one didn't survive.
        bullets = "\n".join(f"  - {f['id']} [{f['category']}]: {f['hint']}" for f in failures)
        raise DatasetScoutError(
            "All components failed to materialize. No corpus written.\n" + bullets
        )

    # ── split (leakage-aware) ──
    proportions = (
        normalize_split_proportions(recipe.splits)
        if any((recipe.splits.train, recipe.splits.val, recipe.splits.test))
        else dict(DEFAULT_SPLITS)
    )
    splits, dedup_stats = _split_records(
        all_records,
        proportions,
        seed=overrides.seed_effective,
        leakage_keys=list(recipe.leakage_keys),
    )

    # ── write ──
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_hashes: list[str] = []
    for split_name, filename in JSONL_FILENAMES.items():
        h = _write_jsonl(out_dir / filename, splits[split_name])
        jsonl_hashes.append(h)
    fingerprint = _compute_fingerprint(jsonl_hashes)

    # Recipe is dumped via Pydantic; we don't preserve the original
    # YAML formatting verbatim today (the dump is canonical-equivalent).
    (out_dir / "recipe.yaml").write_text(
        yaml.safe_dump(recipe.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    elapsed_s = time.monotonic() - started
    lock_path = out_dir / "recipe.lock.yaml"
    _write_lockfile(
        lock_path,
        recipe=recipe,
        kept=materialised,
        dropped=dropped,
        failures=failures,
        realized=realized,
        splits=splits,
        dedup_stats=dedup_stats,
        overrides=overrides,
        fingerprint=fingerprint,
        started_iso=started_iso,
        elapsed_s=elapsed_s,
    )
    _write_manifest(out_dir / "manifest.json", lock_path)
    _write_report(
        out_dir / "report.md",
        recipe=recipe,
        splits=splits,
        realized=realized,
        failures=failures,
        overrides=overrides,
        dedup_stats=dedup_stats,
        fingerprint=fingerprint,
        elapsed_s=elapsed_s,
    )
    _write_fingerprint(out_dir / "fingerprint.txt", fingerprint)
    _write_usage(out_dir / "usage.md", out_dir)

    return CurateResult(
        out_dir=out_dir,
        components_kept=len(materialised),
        components_skipped=len(dropped),
        components_failed=len(failures),
        components_zero_row=sum(1 for c in materialised if not per_component.get(c.id)),
        rows_per_split={k: len(v) for k, v in splits.items()},
        fingerprint=fingerprint,
        elapsed_seconds=round(elapsed_s, 3),
        failures=failures,
    )


def _label_kind_counts(records: list[NormalizedRecord]) -> dict[str, int]:
    out: dict[str, int] = {}
    for rec in records:
        out[rec.label_kind.value] = out.get(rec.label_kind.value, 0) + 1
    return out


def _component_license_raw(records: list[NormalizedRecord]) -> str | None:
    # Curate doesn't yet thread license info from the source — placeholder
    # that the lockfile records "(unknown)". M1b's cache will carry license.
    return None


def _component_license_spdx(records: list[NormalizedRecord]) -> str | None:
    return None


def _component_identity_method(records: list[NormalizedRecord]) -> str | None:
    if not records:
        return None
    # All records in a component use the same row-identity method (the
    # method depends on the row's columns, which are the same per source).
    rec = records[0]
    # Re-derive method from the row id format.
    if rec.source_row_id.startswith("idx:"):
        return "index"
    if len(rec.source_row_id) == 16 and all(c in "0123456789abcdef" for c in rec.source_row_id):
        return "sha256:row"
    return "column"


__all__ = [
    "CurateOverrides",
    "CurateResult",
    "load_recipe",
    "run_curate",
]

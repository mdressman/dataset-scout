# Judged corpus shape

Stable JSONL contract for downstream consumers that ingest
scout-produced corpora. This page is the public surface; it changes
only with explicit deprecation notice.

## File format

Each line of a scout corpus is a JSON object conforming to
`dataset_scout.NormalizedRecord` (pydantic v2). Any consumer can
import the model directly via `pip install dataset-scout`, or parse
the JSON without taking a dep — the field set is documented here.

## Field reference

| Field | Type | Required | Notes |
|---|---|---|---|
| `text` | string | yes | Row text used for downstream training/eval. |
| `label` | `"positive" \| "benign" \| "hard_negative"` | yes | Three-class scout label. `hard_negative` is a recon-side category for explicitly difficult negatives surfaced by strategy assessment; the M10 judge never *produces* `hard_negative` (it produces `positive` or `benign` only — see "judge label mapping" below). Downstream consumers may treat `benign` and `hard_negative` identically at the axis level (both are `not positive`) or distinguish them for sampling purposes; both are valid. |
| `label_kind` | `"ground_truth" \| "remapped" \| "proxy" \| "subset_extracted" \| "judged"` | yes | Provenance of the label. `"judged"` indicates the LLM judge produced this label (M10). |
| `strategy` | enum | yes | Recon-time strategy that selected the source row. |
| `strategy_confidence` | float `0.0–1.0` | yes | Recon-time strategy confidence. |
| `source` | string | yes | `"<source>:<source_id>"`, e.g. `"huggingface:org/ds"`. |
| `source_row_id` | string | yes | Row identity within the source (column value, sha256 of canonical JSON, or `idx:N` fallback). |
| `source_revision` | string | optional | Source revision (HF revision, dataset version). |
| `source_config` | string | optional | Source config (e.g. HF subset name); `null` when not applicable. |
| `source_split` | string | optional | Source split (e.g. `"train"`); `null` when not applicable. |
| `threat_family` | string | optional | First threat family from the recipe intent. |
| `extras` | object | yes | The original source row, JSON-coerced. Never lossy. |
| `extras_coercion` | bool | yes | `true` if any extras values were stringified during coercion. |
| `label_confidence` | float `0.0–1.0` | optional (M10) | Derived row-level confidence. `null` for non-judged rows. |
| `judge` | object | optional (M10) | Present only when `label_kind == "judged"`. See below. |

## `judge` object (M10)

When `label_kind == "judged"`, the `judge` field is populated with:

| Field | Type | Notes |
|---|---|---|
| `axis` | string | The labeling question, e.g. `"prompt_attack"`. |
| `verdict` | `"positive" \| "negative" \| "ambiguous"` | Raw judge verdict. |
| `subcategory` | string | Short kebab-case classification, e.g. `"injection-system-prompt-extract"`. |
| `confidence` | float `0.0–1.0` | Judge's *raw self-rated* confidence. |
| `rationale` | string | One-sentence explanation. |
| `model` | string | Judge model identifier (litellm-style), e.g. `"azure/gpt-4o-2024-08"`, `"github_copilot/gpt-5-mini"`, `"openai/gpt-4o-mini"`. |
| `template_version` | string | Scout-internal prompt version (used for cache audit; not coordinated externally). |
| `n_judges` | int `≥ 1` | `1` for single-judge runs. |
| `agreement` | `"single" \| "majority" \| "unanimous"` | Aggregation rule. Always populated by the M10 judge: `"single"` when `n_judges == 1`, otherwise `"majority"` or `"unanimous"` per the run's configuration. The pydantic model declares `... \| None` for forward compatibility, but consumers can rely on a non-null value from any scout-produced corpus. |

## Judge label mapping (M10)

The M10 judge emits one of three verdicts: `"positive"`, `"negative"`,
or `"ambiguous"`. Mapped to scout's three-class `label` as follows:

| Judge verdict | Scout `label` | Notes |
|---|---|---|
| `"positive"` | `"positive"` | Promoted only if `confidence >= threshold`. |
| `"negative"` | `"benign"` | Promoted only if `confidence >= threshold`. The judge does **not** produce `"hard_negative"` — that's a recon-side category. |
| `"ambiguous"` | *not promoted* | Row keeps its pre-existing label. `judge` block still written for review. |

The *derived* row-level confidence (used by downstream filtering) is
the top-level `label_confidence` field, **not** `judge.confidence`. For
single-judge runs they are equal; for multi-judge runs `label_confidence`
blends agreement with self-confidence per the rule documented inline
in `dataset_scout.judge`.

## Stable row identity (M9-min)

Every record has a globally-unique deterministic identifier reachable
via `record.stable_id` (Python) or by concatenating four fields:

```
<source>::<config_or_underscore>::<split_or_underscore>::<source_row_id>
```

For example: `"huggingface:org/ds::default::train::abc123"`. Use this
when you need a single corpus-global row handle (e.g. checkpoint state,
lineage references, dedup keys across recipes).

## Promotion rule (M10)

Scout uses an *explicit gap* policy: a row receives `label_kind: judged`
**only** when the judge returned a clean `"positive"` or `"negative"`
verdict at or above the configured confidence threshold (default `0.8`).
Below the threshold or on `"ambiguous"`, the row keeps its original
label and `label_kind`, but the `judge` block is still populated so
reviewers can see why promotion was declined.

In practice this means consumers of judged corpora can filter to
high-confidence rows with:

```python
[r for r in records if r.label_kind == "judged"
                    and (r.label_confidence or 0) >= threshold]
```

without needing to inspect `judge.verdict` directly.

## What scout deliberately does *not* emit

Scope-clarifying non-features:

- Scout does not emit any framework-specific enrichment-axis envelope.
  Consumers should adapt scout's flat record format directly — the
  same as they would for any custom dataset.
- Scout does not coordinate prompt template versions, cache keys, or
  judge model defaults with any other tool. Each consumer is free to
  read what it needs from the fields above.
- Scout does not publish a "compatibility extras" emit mode. The
  format above is the only format scout writes.

## Stability guarantee

Fields documented here are stable across scout versions within a
major release. Additive changes (new optional fields) are always
permitted; existing fields are not renamed or repurposed without a
deprecation cycle.

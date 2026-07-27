# Concepts

The mental model behind `dataset-scout`. Read this once and the CLI
output and library types should feel obvious.

---

## 1. Discovery, not ranking

`datascout recon` is **discovery tooling**, not a ranked-quality
scoreboard. Candidates come back in the source's own search-relevance
order; probe outputs are **annotations** beside each candidate, never
folded into a single "quality" number.

The reason is honesty. License + freshness + card-completeness say a
lot about a dataset's polish but very little about whether it actually
fits your detection task. Aggregating those into a fitness score
would imply confidence the probes alone don't have.

When the LLM strategy assessor runs (full mode) the framing changes:
candidates carry per-strategy confidence, the report leads with the
strongest defensible reframings, and `recipe.draft.yaml` lands in the
output directory. In metadata-only mode it stays pure discovery, with
receipts.

---

## 2. Modes: full vs metadata-only

The pipeline picks one of two modes at start-up.

| Mode | Trigger | What runs |
|---|---|---|
| **Full** | Any LLM provider configured — `DATASET_SCOUT_MODEL` set (e.g. `github_copilot/...`, `github/...`, `openai/...`, `anthropic/...`, `azure/...`) **OR** the legacy `AZURE_OPENAI_ENDPOINT` + `_DEPLOYMENT` pair | Brief parsing → LLM decomposition → multi-source search (HuggingFace, Kaggle if creds, plus Semantic Scholar / arXiv paper-search promoting cited HF/Kaggle datasets) → cheap probes → two-stage shortlist → per-candidate strategy assessor → coverage gaps → ranked report + `recipe.draft.yaml` |
| **Metadata-only** | No LLM provider configured | Brief parsing → single-query search (HuggingFace, Kaggle if creds) → cheap probes (no decomposition, no paper search, no strategies, no recipe draft) |

The fallback is **explicit and noisy** — you'll see a notice on stderr
and a prominent header in `report.md` telling you what to set. No
silent weakening.

See [`configuration.md`](configuration.md) for the env-var setup.

---

## 3. Sources, candidates, and metadata

A `Source` plugin (HuggingFace, Kaggle) yields `Candidate` objects
in response to an `Intent` plus a list of `DecompositionDirection`s.
Academic-paper discovery is a separate pipeline channel (Semantic
Scholar with arXiv as a targeted fallback for named-benchmark
queries) that promotes cited HuggingFace / Kaggle datasets into the
same pool with paper provenance — see `architecture.md` §10 for the
limits. Each candidate carries:

- **Identity** — `source`, `id`, `revision`.
- **`metadata`** — a typed `CandidateMetadata` envelope: license raw +
  SPDX guess, dates, declared languages, size signals, columns
  (when cheap), card-fields-present set, gating posture, tags.
- **`surfaced_by: list[str]`** — the names of every decomposition
  direction that found this candidate. A candidate hit from the
  original brief carries an empty list. Multi-direction hits keep
  all their provenance, which lets future ranking weight them
  appropriately.

The metadata envelope is **source-agnostic** — Kaggle and PWC plugins
will populate the same shape, so probes never read source-specific keys.

---

## 4. Probes

Each probe is a small object emitting a typed `SubScore` per candidate:

```python
class Probe(Protocol):
    name: str
    version: str
    def applies(self, candidate, intent) -> bool: ...
    def run(self, candidate, intent) -> SubScore: ...
```

Six metadata-driven probes ship today:

| Probe | What it tells you |
|---|---|
| `license` | SPDX best-effort guess and how it sits relative to your `LicensePolicy` (allow / warn / outside policy / unknown). |
| `size` | Row count / byte count / downloads / likes when declared. No 0–1 score; size doesn't map cleanly. |
| `recency` | Days since upload and last-modified. Raw evidence only. |
| `freshness` | Bucketed signal: fresh `<6mo` / current `6–18mo` / aging `>18mo`. |
| `languages` | Overlap fraction between declared languages and your intent's language list. |
| `card_completeness` | Fraction of expected YAML fields actually declared. Documentation-hygiene only — never used as a major rank component. |

Every `SubScore` carries a status (`ok` / `not_applicable` /
`low_confidence` / `skipped`), so probes that don't apply self-report
rather than fabricating values.

The **embedding label-intent fit** stage runs between cheap probes
and the shortlist; `label_structure` and `schema_fingerprint` are
not yet wired. The recipe draft is curate-ready either way because
the strategy assessor (§5) already streams real rows for its own use.

---

## 5. The strategy taxonomy

The LLM strategy assessor returns 1–4 ranked `Strategy` objects per
candidate. Three of them carry most of the weight:

- **`direct_use`** — labels and content map cleanly to your task.
- **`reframing`** (covers `subset_extraction`, `label_remapping`,
  `cross_class_repurposing`) — same data, new shape: a subset of
  rows, a relabeled view, or original positives turned into hard
  negatives.
- **`signal_proxy`** — adjacent threat used as a proxy positive
  during cold start. Honest about being a proxy.

Plus three edge cases: **`benign_baseline`** (no positives, useful
as benign distribution), **`not_useful`** (the honest answer when
nothing fits), and the implementation detail that the three
"reframing" kinds are tracked separately in the wire format
(`subset_extraction`, `label_remapping`, `cross_class_repurposing`)
so the rationale per row stays specific.

Each strategy carries a `confidence ∈ [0, 1]`, written rationale,
caveats, and a concrete `transform` spec (column maps, label
value-maps, filters). The assessor is **conservative-but-creative**:
stretches get low confidence, and `--min-strategy-confidence` filters
aggressive reframings out of the draft recipe.

### The assessor sees real rows

Before the LLM call, the assessor streams **8 sample rows** from the
candidate's source via `Source.stream_rows()`. The prompt's `SAMPLE
ROWS` section exposes:

- the column names from the first row
- distinct values seen for any plausible label column
  (`label`, `labels`, `class`, `category`, `target`)
- per-row `key=value` lines

The model's `transform` proposal then references **actual** column
names and label values rather than placeholders like
`prompt_and_response_or_equivalent`. **Recipes are curate-ready
straight from `recon`** — no hand-editing of column names required.

If a candidate's source can't be reached or the dataset is gated
without a token, the assessor gracefully degrades to metadata-only
input with a note in the prompt and a notice on the result.

---

## 6. `label_kind` — proxies are honest by default

The output JSONL (M4) marks every row with one of:

| `label_kind` | When to train | When to eval |
|---|---|---|
| `ground_truth` | ✅ | ✅ |
| `subset_extracted` | ✅ | ✅ (within the documented subset) |
| `remapped` | ✅ | ✅ *if you accept the remapped definition* |
| `proxy` | ✅ (often weighted) | ❌ — exclude proxies from eval |
| `judged` | ✅ | ✅ *if* `label_confidence ≥ your bar* |

This is the load-bearing field for downstream training. Downstream
tools should filter to `label_kind != "proxy"` for evaluation.

### `judged` (M10) — explicit-gap label rescue

`datascout judge` (M10) can promote rows from `proxy` /
`remapped` / unknown to `judged` by asking an LLM judge a single
labeling question (the "axis"). The promotion rule is **explicit
gaps over low-confidence guesses**:

- A row is rewritten with `label_kind: judged` only when the judge
  returned a clean `positive` / `negative` verdict at-or-above the
  configured `--threshold` (default `0.8`) on the *derived*
  `label_confidence`.
- For multi-judge runs (`--judges 3 --agreement majority` or
  `--judges 5 --agreement unanimous`), the derived `label_confidence`
  blends judge agreement with the agreeing judges' mean self-rated
  confidence; the formula and the chosen `agreement` mode are
  written into the lockfile so the derivation is reproducible.
- Below threshold or on `ambiguous`, the row keeps its original
  label and `label_kind`, but the `judge` block is still attached so
  reviewers can see *why* promotion was declined. No silent
  downgrades.

Every judged row carries two fields beyond the schema baseline:

- `label_confidence: float | None` — derived row-level confidence.
- `judge: JudgeBlock | None` — verdict, subcategory, rationale, and
  full provenance (model, scout-internal `template_version`,
  `n_judges`, `agreement`).

Downstream filters use the first one only; the `judge` block is for
audit, not gating. See `docs/judged-corpus-shape.md` for the wire
contract.

---

## 7. Recipes and lockfiles

> **⚠️ `curate` is experimental.** The pipeline below ships with a
> full audit trail (MinHash dedup, leakage-aware splits, lockfile,
> per-component soft-failure classification), but its output hasn't
> been end-to-end validated against a hand-built reference yet —
> treat it as a starting point, sanity-check rows and label
> distributions before training, and please contribute back if you
> harden it.

The output of `recon` includes a `recipe.draft.yaml`. Edit if you
want — drop weak components, cap `take`, tighten `filter` — and
hand it to `datascout curate --from recipe.yaml`.

`curate` materialises the recipe into:

```
mycorpus/
├── train.jsonl / val.jsonl / test.jsonl   leakage-aware splits
├── recipe.yaml                             input, copied verbatim (audit trail)
├── recipe.lock.yaml                        pinned revisions + realized counts + hashes
├── manifest.json                           machine-readable lock equivalent
├── report.md                               5-second scorecard + provenance
├── fingerprint.txt                         one-line content hash for commits
└── usage.md                                3-line snippets for HF datasets / pandas
```

`recipe.lock.yaml` is **the defensible record** — the single file a
reviewer can ask about: which corpus did this detector train on, how
did proxies factor in, and what was deliberately excluded? It carries
`audit_readiness: ready`, the MinHash dedup parameters, every
component that made it in (with realised row counts and label-kind
distribution), the `declined_components` list (below
`min_strategy_confidence`), and the `failed_components` list
(see below).

### Per-component soft failures

Real recipes drawn from a fresh `recon` often have one or two
components that need a tweak before they materialise — the most
common reasons are listed under predictable categories so a single
glance at the lockfile or report tells you exactly what to do:

| Category | What it means | Typical fix |
|---|---|---|
| `gated_dataset` | The HF dataset requires authentication. | Set `HF_TOKEN`, or remove the component. |
| `missing_config` | The HF dataset has multiple configs and none was pinned. | Set `source_config: <name>` on the component. |
| `bad_split` | The named split doesn't exist on the dataset. | Set `source_split: <name>` to a real split. |
| `no_data_files` | The dataset has no parseable data files. | Remove the component. |
| `parse_error` | The upstream data file is malformed. | Remove the component. |
| `not_found` | The dataset was deleted or renamed. | Remove the component. |
| `network` | Transient connectivity error. | Re-run. |
| `rate_limited` | Hit HTTP 429 / "Too Many Requests" upstream. | Set `HF_TOKEN` for higher quotas, or lower `--max-concurrency`. |
| `unknown` | Anything else; full message is preserved. | Read the message; file an issue if reproducible. |

Curate **does not crash** on per-component failures — they're recorded
under `failed_components` in `recipe.lock.yaml` and surfaced in
`report.md` and the CLI summary. The corpus is built from whatever
components did succeed. The pipeline only fails hard if every
component errors out (in which case there's nothing useful to write).

### `take` and the auto-cap

The strategy assessor sets `transform.take: "all"` by default since
it can't predict dataset size. For heavy corpora (think Stack /
GitHub-scale code datasets) that means tens of thousands of rows
per component, which can stretch a first-pass `curate` into tens of
minutes.

`recipe.draft.yaml` therefore **auto-caps `take: "all"` to 5000 rows**
at draft-write time, with a caveat on each capped component
explaining what was capped and how to lift it. To materialise the
full component, edit `take: 5000` back to `take: all` (or any
larger integer). To go the other way for one run without editing
the recipe, pass `--max-rows-per-component N` to `curate` — it
lowers the recipe's value but never raises it (the recipe stays
the source of truth).

### Parallel materialisation

Per-component HuggingFace `load_dataset(streaming=True)` setup
(split discovery, parquet header fetch, schema inference) is the
dominant cost in `curate` — typically 30s-several minutes per
component. Running components in parallel gives near-linear
speedup since the work is I/O-bound.

`curate` ships with **`--max-concurrency 4`** by default;
`--max-concurrency 6` is a comfortable upper end for most
recipes. Workers are kept low to avoid HF rate limits (HTTP 429s
classify cleanly as `failed_components` with a hint pointing at
`HF_TOKEN` and lower concurrency).

Determinism is a contract: results are keyed by component id and
reassembled in original recipe order before splitting, so the
fingerprint, lockfile, and JSONL contents are identical regardless
of which workers complete first.

This is the difference between "first sanity-check run" (cap on,
~minutes for typical recipes) and "production-blend
materialisation" (cap off, hours-but-once).

---

## 8. Voice

The audience is detection engineers under audit pressure. Output is
designed accordingly:

- **No aggregate "quality" or "risk" headline.** Per-signal evidence,
  per-candidate strategy assessment, written rationale.
- **Receipts everywhere.** Every claim links back to a card URL,
  sample row, prompt, or response.
- **`recipe.lock.yaml` is the defensible record.**
- **Proxies are honest by default.** Output JSONL marks them; eval
  must exclude them.
- **"This is not legal advice"** footer on any report touching
  licensing.


## 9. How to write a brief

The brief is the only input that drives everything downstream — the
heuristic parser, the LLM decomposer, the multi-direction search, the
strategy assessor. **A good brief is a crisp dataset request, not a
detector design.** Two specs jammed together (the dataset I want AND
the detector I'll build) confuses every step.

### Bad → good

| ❌ Conflated detector spec | ✅ Crisp dataset request |
|---|---|
| _"Find labeled corpora for detecting X — inputs are HTML, outputs are positive vs benign with hard-negatives for Y. We'll train and evaluate a transformer..."_ | _"HTML and Markdown corpora with labelled hidden text — phishing pages, indirect prompt-injection payloads, accessibility-style benigns."_ |
| _"Find datasets to train a classifier for over-refusal where the model declines benign requests citing safety. Output schema: positive vs benign vs hard-negative."_ | _"Refusal-labeled corpora for customer-support agents — over-refused benign prompts plus correctly-refused harmful prompts."_ |

### What belongs in a brief

- **Labels** you want (positive / benign / hard-negative).
- **Content shape** (HTML, dialogue, code, image, multi-turn, …).
- **Domain context** that affects matching (English, customer-support,
  agent-mediated, code-switched, …).

### What does NOT belong

- Input/output schemas of your downstream model.
- "We'll train and evaluate" / "the classifier should…"
- Architecture choices (transformer, LLM-judge, etc.).
- Long lists of every sub-case — the LLM decomposer's job is to
  enumerate adjacencies. Don't pre-empt it.

### Length

Aim for **under 250 characters**. The HF lexical search and the LLM
decomposer both work better on crisp briefs. If you're at 400+
characters, you've probably described the detector instead of the
dataset.

### Named benchmarks

Discovery is HuggingFace-lexical-bound: a dataset whose card text
doesn't intersect your brief's keywords won't surface, even if it's
a perfect semantic fit. If your construct has well-known named
benchmarks (e.g. "OR-Bench", "XSTest", "TruthfulQA"), **list them
in the brief** — the decomposer turns them into proper-noun queries
that find the exact dataset even when the card text is sparse.

### Iterate cheaply

Run `datascout decompose "<brief>"` first — ~5 seconds, one LLM call.
It prints the directions the model would explore. If the directions
are wrong, refine the brief and re-run. Once the directions look
right, run full `datascout recon` (and pass `--decomposition-from
decomposition.yaml` to skip re-paying for decomposition).

```bash
# Cheap iteration loop
datascout decompose "your brief" --out scratch/

# When happy, full recon reusing the directions
datascout recon "your brief" \
    --decomposition-from scratch/decomposition.yaml \
    --out scratch/recon/
```

### When the parser nudges you

`recon` will surface a hint when the brief looks like a detector
spec ("describes detector inputs", "describes detector outputs",
"describes the train/eval plan"). When you see it, tighten the brief.

See [`architecture.md`](architecture.md) for the pipeline detail.

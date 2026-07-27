# CLI reference

Two equivalent entry points. Pick whichever you prefer:

```bash
dataset-scout ...      # formal
datascout      ...     # recommended short form
```

All commands route to the same library functions. The CLI is a thin
wrapper — anything you can do from the CLI is also exposed in
`dataset_scout.recon()` / `inspect()` / `curate()` (see the library
section of [architecture.md](architecture.md)).

## Verb cheat-sheet

| Verb | What it does | Cost |
|---|---|---|
| [`decompose`](#decompose) | LLM brief decomposition only (cheap iteration loop) | ~5s, 1 LLM call |
| [`recon`](#recon) | Full pipeline → `report.html` + `report.md` + `results.json` + `recipe.draft.yaml` + `decomposition.yaml` | ~2 min, ~16 LLM calls |
| [`inspect`](#inspect) | One-candidate deep-dive | ~5s + optional 1 LLM call |
| [`curate`](#curate) | Recipe → JSONL + lockfile | depends on row count |
| [`judge`](#judge) | Promote weak labels to high-confidence ones with an LLM judge (M10) | 1 LLM call per non-ground-truth row × `--judges` |
| [`eval`](#eval) | Score a judged corpus against gold (P/R/F1, confusion, coverage) | local |
| [`compose`](#compose) | Merge multiple recipes into one | none — local |
| [`cache`](#cache) | Inspect / prune / clear the cache | local |
| [`sources`](#sources) | List / toggle source plugins | local |

---

## `decompose`

The cheap brief-iteration verb. Just runs the LLM decomposition step
— ~5 seconds and one LLM call. Prints the proposed directions to
stdout; `--out` optionally writes them to `decomposition.yaml`.

```bash
datascout decompose "<brief>" [options]
```

| Option | Description |
|---|---|
| `--detection-target TEXT` | Override the parsed detection target. |
| `--deployment-context TEXT` | Production / RAG / agent / etc. |
| `--out PATH` | Path to write `decomposition.yaml`. If the path ends with `.yaml`/`.yml` the file is written there directly; otherwise the path is treated as a directory and the file is written as `<out>/decomposition.yaml`. Without `--out`, output is stdout-only. |

### The cheap-iteration loop

```bash
# Try it
datascout decompose "<v1 brief>"
# Don't like the directions? Refine and retry.
datascout decompose "<v2 brief>"
# Once happy, persist
datascout decompose "<v2 brief>" --out scratch/
# Run the full recon, skipping the (now-paid-for) decompose step
datascout recon "<v2 brief>" --decomposition-from scratch/decomposition.yaml --out scratch/recon/
```

You'd otherwise burn ~2 minutes per iteration paying for full recons
just to see the directions.

---

## `recon`

Find candidate datasets, run probes, assess strategies, identify
coverage gaps, emit a draft recipe.

```bash
datascout recon "<brief>" [options]
```

### Arguments

| Argument | Required | Description |
|---|---|---|
| `BRIEF` | yes | Natural-language brief describing the **dataset you want** (not the detector you'll build — see `concepts.md` §9). |

### Options

| Option | Default | Description |
|---|---|---|
| `--detection-target TEXT` | parser-derived | Override the parsed detection target. |
| `--deployment-context TEXT` | parser-derived | Production / RAG / agent / etc. |
| `--language TEXT` | `en` | Required language(s); repeat for multiple. |
| `--license TEXT` | permissive set | License allowlist; repeat for multiple. |
| `--min-strategy-confidence FLOAT` | `0.5` | Strategies below this confidence are filtered out of the draft recipe. Recipe-authoritative — `curate` defaults to the value baked into the recipe. |
| `--decomposition-from PATH` | — | Reuse a hand-edited `decomposition.yaml` instead of paying for a fresh LLM decompose call. |
| `--model TEXT` | `$DATASET_SCOUT_MODEL` → AOAI fallback | Override the LLM model id (litellm-style). Examples: `github_copilot/gpt-5-mini`, `github/gpt-4o-mini`, `openai/gpt-4o`, `azure/<deployment>`. |
| `--out PATH` | `datascout-out/` | Output directory. |
| `--no-papers` | — | Skip paper search (Semantic Scholar + arXiv). |

### Paper search

By default, `recon` queries Semantic Scholar (NeurIPS / ICML / ICLR /
SaTML and other venues) and extracts dataset URLs from paper abstracts.
arXiv serves as a targeted fallback for named-benchmark queries when
S2 is unavailable or throttled. HuggingFace and Kaggle URLs found in
abstracts are promoted into the candidate pool with paper provenance;
datasets on author sites or generic GitHub repos surface as citations
only. If both S2 and arXiv fail, recon proceeds without the paper
channel rather than blocking. Use `--no-papers` to skip entirely.

### Outputs

```
<out>/
├── report.md            human-readable, LEADS WITH coverage gaps when notable
├── results.json         structured ReconResult (Pydantic dump)
├── decomposition.yaml   stand-alone direction list (hand-editable)
└── recipe.draft.yaml    hand-editable input for `datascout curate` —
                          contains REAL column names from row sampling
                          when the LLM strategy assessor runs
```

### Examples

```bash
# Most common — natural-language brief
datascout recon "HTML and Markdown corpora with labelled hidden text — phishing pages and accessibility benigns"

# Multi-language
datascout recon "refusal-labeled customer-support corpora" --language en --language es

# Reuse a prior decomposition (cheap iteration)
datascout recon "<refined brief>" --decomposition-from scratch/decomposition.yaml

# Force a license restriction
datascout recon "..." --license MIT --license Apache-2.0
```

### Hidden expert flags

These exist for debugging and forward compatibility but are
deliberately **not in `--help`**, because they tend to mask weaknesses
the brief should express:

- `--threat-families` — comma-separated families to inject into the
  parsed Intent. Use only when the brief is genuinely ambiguous.
- `--no-explore` — skip LLM decomposition. Useful for reproducing
  metadata-only behavior when an LLM provider is configured.

---

## `inspect`

One-candidate deep-dive. Renders to **stdout** so it pipes cleanly:
`datascout inspect huggingface:org/x > inspect.md`.

```bash
datascout inspect <source>:<id>[@revision] [options]
```

| Option | Default | Description |
|---|---|---|
| `--intent-from PATH` | — | Re-use the most recent recon's Intent (point at `results.json`). The strategy assessor sees the same Intent recon used. |
| `--brief TEXT` | — | One-off brief for strategy assessment when no recon `results.json` exists. Mutually exclusive with `--intent-from`. |
| `--sample-size INT` | `50` | Rows to stream for the schema / label-distribution / sample sections. |

### What it shows

1. **Identity** — `source:id`, revision, card URL, description, gating posture.
2. **License** — raw card declaration plus a best-effort SPDX guess.
3. **Card-declared metadata** — languages, task categories, tags, dates, popularity.
4. **Inferred schema** — column names + types from the streamed sample.
5. **Label distribution** — count and Wilson 95% CI per value when a label column is detected.
6. **Text-length stats** — min / median / max characters for the heuristically-picked text column.
7. **Sample rows** — first five.
8. **Strategy assessment** — when `--intent-from` or `--brief` is supplied AND an LLM provider is configured.

```bash
# Quick metadata-only deep-dive
datascout inspect huggingface:bench-llm/or-bench

# With a fresh brief — adds LLM strategy assessment if an LLM provider is set
datascout inspect huggingface:bench-llm/or-bench --brief "refusal-labeled corpora for support agents"

# Re-use the Intent from your latest recon
datascout inspect huggingface:walledai/XSTest --intent-from datascout-out/results.json
```

---

## `curate`

> **⚠️ Experimental — output not yet end-to-end validated.**
> `curate` ships a working implementation with a full audit trail
> (lockfile, MinHash dedup, leakage-aware splits, per-component
> soft-failure classification), **but the author hasn't yet
> personally trained a model on a scout-curated corpus and confirmed
> quality vs a hand-built reference.** Inspect rows, sanity-check
> label distributions, and compare against your own gold before
> training on the output. The same caveat applies to `judge` /
> `eval` downstream. Bug reports and PRs that harden this pipeline
> are *very* welcome.

Build a schema-normalized corpus from a recipe.

```bash
datascout curate --from recipe.yaml --out ./mycorpus
```

| Option | Default | Description |
|---|---|---|
| `--from PATH` | required | Path to `recipe.yaml`. |
| `--out PATH` | `mycorpus/` | Output directory. |
| `--min-strategy-confidence FLOAT` | recipe-defined | Override the recipe's threshold. Recorded as an override in `recipe.lock.yaml`. |
| `--seed INT` | recipe-defined | Override the recipe's split seed. Recorded as an override. |
| `--max-rows-per-component INT` | — | Cap rows materialized per component for this run. Lowers but never raises the recipe's `take` value — fast iteration on heavy code/text corpora without hand-editing the recipe. |
| `--max-concurrency INT` | `4` | Number of components materialized in parallel. Most of the per-component cost is HuggingFace `load_dataset` setup (split discovery, parquet header fetch); parallelising 4–8 workers gives ~linear speedup. Lower to 1 for sequential debugging; raise carefully to avoid HF rate limits (especially without `HF_TOKEN`). |

### Status: audit trail ready, output quality not yet validated

The pipeline ships a defensible end-to-end build — meaning the
**provenance and lockfile** are audit-ready. The corpus *quality*
hasn't been independently validated yet. What's wired:

- ✅ Recipe loader, materialisation from HuggingFace, normalized
  JSONL output, lockfile, manifest, report, fingerprint, usage.
- ✅ Multimodal-safe `extras` (bytes/arrays coerced to strings with
  a flag).
- ✅ **Filter DSL** — `transform.filter` accepts a sandboxed
  expression (booleans, comparisons, `len`, `contains_pattern`,
  `lower`, `startswith`, `endswith`, `int`, `str`, `in`).
  Disallowed nodes (attribute access, comprehensions, lambdas,
  unknown functions) are rejected at compile time.
- ✅ **MinHash dedup + leakage-aware split.** Near-duplicate rows
  (Jaccard ≥ 0.8 over char 5-grams) cluster together; whole
  clusters are assigned to a single split. Cluster stats are
  written to the lockfile.
- ✅ **Resilient to per-component upstream errors.** Gated
  datasets, multi-config datasets without a `source_config`,
  non-standard split names, parse errors — each is classified
  (categories: `gated_dataset`, `missing_config`, `bad_split`,
  `no_data_files`, `parse_error`, `not_found`, `network`,
  `unknown`) and recorded under `failed_components` in the
  lockfile with an actionable hint. The corpus continues to build
  from what does succeed; the run only fails hard if every
  component errors out.

`recipe.lock.yaml` records `audit_readiness: ready` with the
MinHash params (`num_perm`, `threshold`, `shingle_size`,
`dedup_version`) + filter expressions in scope, and a
`failed_components` block when applicable. `report.md` carries
an "audit-ready" banner with cluster stats and a "Components
skipped due to upstream errors" section so the corpus's
provenance is visible at a glance.

### Exit panel

The CLI prints a short summary line on exit. Components are
classified into four buckets so soft-failures aren't silent:

```
✔ 1831 row(s) written to ./mycorpus
   (11 of 15 component(s) materialised with rows,
    4 produced 0 rows,
    0 dropped, 0 failed) in 35.52s
   - splits: train=1459 · val=178 · test=194
   - fingerprint: 6295b35b551a6876...
   ! 4 component(s) materialised but produced 0 rows — usually a
     recipe column mismatch. See report.md → Components for the
     per-component count.
   ✔ audit-ready: leakage-aware splits + filter DSL applied.
```

Buckets:

- **materialised with rows** — produced `≥ 1` row that survived
  transform + filter.
- **produced 0 rows** — the source loaded fine but nothing
  survived (usually a `text_column` / `label_column` mismatch in
  the recipe). Common signal during recipe iteration.
- **dropped** — explicitly skipped by `min_strategy_confidence`.
- **failed** — upstream error (gated, missing config, bad split,
  parse error). See `report.md` / `recipe.lock.yaml →
  failed_components` for the per-component hint.

---

---

## `judge`

Promote weak labels to high-confidence ones by asking an LLM judge a
single labeling question (the "axis"). The judge does not score the
output of any downstream system — it labels rows of an existing
corpus.

```bash
datascout judge ./mycorpus --axis psych_harm \
    --rubric rubrics/psych_harm.txt --judges 3 --agreement majority \
    --threshold 0.8 --calibrate-against ./gold-psych-harm
```

| Option | Default | Description |
|---|---|---|
| `TARGET` | required | Corpus directory (containing `train.jsonl`/`val.jsonl`/`test.jsonl`) or a single `.jsonl` file produced by `datascout curate`. |
| `--axis TEXT` | required | The labeling question, e.g. `psych_harm`, `prompt_attack`, `over_refusal`. |
| `--rubric PATH` | — | Optional rubric file (free text or YAML). When omitted, a minimal "apply the axis name strictly" placeholder is used. |
| `--judges INT` | `1` | Number of independent judge calls per row. Cached per-judge so re-runs are free. |
| `--agreement {single,majority,unanimous}` | `single` | Multi-judge aggregation rule. `majority` requires `--judges 3`; `unanimous` requires `--judges 5`. |
| `--threshold FLOAT` | `0.8` | Minimum derived `label_confidence` to promote a verdict to `label_kind: judged`. Below threshold, the row keeps its label and the `judge` block is attached for review. |
| `--out PATH` | `<TARGET>/judged/` | Output directory. The original corpus is never overwritten. |
| `--only-unknown / --re-judge-all` | `--only-unknown` | Skip rows already at `label_kind=ground_truth` or `label_kind=judged` (default), or judge every row. |
| `--dry-run` | `False` | Estimate LLM call count without invoking the model. |
| `--calibrate-against PATH` | — | Sample N rows from a gold corpus, run the judge under the same rubric/model/threshold, and report P/R/F1 against ground-truth labels. Recorded under `judge.calibration` in the lockfile. |
| `--calibration-seed-n INT` | `100` | Sample size for `--calibrate-against`. |
| `--calibration-floor FLOAT` | — | Required minimum calibrated precision. With this set, the run aborts if calibration falls below the floor unless `--proceed`. |
| `--proceed` | `False` | Override `--calibration-floor`. |

### Outputs

Inside `<out>/`:

- `train.jsonl` / `val.jsonl` / `test.jsonl` — same shape as the
  input, with promoted rows updated and `judge` blocks attached on
  every row that the judge saw (including below-threshold ones).
- `judge.lock.yaml` — full audit trail: axis, rubric, model,
  scout-internal `template_version`, `n_judges`, `agreement`,
  threshold, cache_dir, calibration block, stats.
- `judge.report.md` — short rich markdown summary including a
  **Sample rows** section (top 3 promoted positives, top 3 promoted
  negatives, top 3 ambiguous-not-promoted) so reviewers can eyeball
  the run without `rg`-ing the JSONL. Also surfaces the resume tip
  when `.judge_state.json` is present.
- `.judge_state.json` — per-batch checkpoint keyed by
  `NormalizedRecord.stable_id`. Re-runs with the same `--out`
  resume — and emit a one-line `resuming axis=... already completed`
  log on entry so the resume isn't silent. The disk cache
  (`<workspace>/.cache/dataset-scout/judge/`) makes already-judged
  rows free even without the checkpoint.

### Worked example

```bash
datascout judge ./psych-harm-corpus \
    --axis psych_harm \
    --rubric rubrics/psych_harm.txt \
    --judges 1 --threshold 0.8 \
    --out ./psych-harm-judged

# → ✓ datascout judge — 142 promoted positive · 803 promoted negative
#    · 352 left unknown · 0 skipped · 0 cache hits · 9.42s
```

Soft per-row failures (API errors, JSON parse failures with one
retry, AOAI content-filter rejections, cache corruption) **do not
abort the run**. Each is bucketed in `JudgeStats` and written into
`judge.lock.yaml → judge.stats` so the audit trail is complete.

---

## `eval`

Score a corpus against a gold corpus. Generic — works on any pair of
scout-shaped JSONL corpora joined on `stable_id`; the M10 calibration
loop uses it internally.

```bash
datascout eval ./psych-harm-judged --against ./gold-psych-harm \
    --axis psych_harm
```

| Option | Default | Description |
|---|---|---|
| `JUDGED` | required | Judged-or-any-label corpus directory or `.jsonl` file. |
| `--against PATH` | required | Gold corpus (rows with `label_kind=ground_truth`). |
| `--axis TEXT` | — | Restrict scoring to a single axis. |

Output is a per-axis table:

```
              axis     P     R    F1   cov  TP  FP  FN  TN  n_gold  n_judged
       psych_harm  0.91  0.78  0.84  0.95  39   4  11  46     100        145
```

Only rows in the judged corpus with `label_kind=judged` contribute to
P/R; un-promoted rows count toward `n_judged_unknown` (visible in the
JSON dump of `EvalResult`) but not toward the confusion matrix —
they're "no decision", not wrong.

### Worked example

```bash
# Compare two judge models on the same gold set
datascout judge ./corpus --axis x --out ./judged-A --model azure/gpt-4o
datascout judge ./corpus --axis x --out ./judged-B --model github_copilot/gpt-5-mini
datascout eval ./judged-A --against ./gold --axis x   # → P/R/F1 for A
datascout eval ./judged-B --against ./gold --axis x   # → P/R/F1 for B
```

---

## `compose`

Merge multiple recipes into one — for multi-detection programs that
share a corpus.

```bash
datascout compose recipe-a.yaml recipe-b.yaml [recipe-c.yaml ...] --out merged.yaml
```

| Option | Description |
|---|---|
| `--out PATH` | Required. Path to write the merged recipe.yaml. |
| `--intent-brief TEXT` | Override the merged intent's brief (default: keep the first input's). |

### Semantics

- Components dedupe by `(source, source_id)`. Higher
  `strategy_confidence` wins on conflict; the loser becomes a notice.
- `min_strategy_confidence` takes the **maximum** of the inputs (most
  conservative wins).
- Declined entries are unioned and deduped.
- Intent comes from the first input unless `--intent-brief` overrides.
  When inputs have different briefs, a notice is recorded.

```bash
# Three sub-detection programs into one corpus
datascout compose \
    detection-1/recipe.draft.yaml \
    detection-2/recipe.draft.yaml \
    detection-3/recipe.draft.yaml \
    --out programs/merged.yaml \
    --intent-brief "Content Injection Traps — combined corpus"

# Then materialise
datascout curate --from programs/merged.yaml --out programs/corpus/
```

---

## `cache`

Inspect and manage the SQLite cache.

```bash
datascout cache info
datascout cache prune
datascout cache clear
```

---

## `sources`

List and toggle source plugins.

```bash
datascout sources list             # works today: reflects current ScoutContext
datascout sources enable <name>    # not yet implemented
datascout sources disable <name>   # not yet implemented
```

---

## Global options

| Option | Description |
|---|---|
| `--version` | Print version and exit. |
| `--help` | Standard Typer help. |

---

## Output streams and exit codes

- **`stdout`** — Markdown reports (when piped) and `inspect` deep-dives. Otherwise empty.
- **`stderr`** — Progress events, notices, errors, completion summary.
- **Exit codes** — `0` success · `1` runtime error · `2` not-yet-implemented or invalid usage.

---

## See also

- [Concepts](concepts.md) — how to write a brief, what the report's framing language means.
- [Configuration](configuration.md) — `.env`, the LLM provider matrix (GitHub Copilot, GitHub Models, OpenAI, Anthropic, Azure / Entra), HF.
- [Architecture](architecture.md) — what runs under each verb.

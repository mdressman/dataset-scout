# Getting started

A walkthrough from zero to a curated corpus, end to end.

## The scenario

You're the detection lead at a B2B SaaS that ships an AI customer-
support agent. Production traffic shows a new failure mode: the
model has started **over-refusing** — declining benign, legitimate
requests by citing safety concerns. ("I can't help with that" when
the user asks about how to use a product feature.) Support tickets
are stacking up, the trust-and-safety team is unhappy, and you need
a labeled corpus to train an over-refusal detector.

This is **not** prompt injection. It's not a jailbreak. It's the
opposite — your model is being *too* cautious. There's no obvious
public dataset called "over-refusal in customer support," but there
is relevant adjacent work: refusal benchmarks, safety-tuning datasets,
exaggerated-caution test suites. dataset-scout is built for exactly
this shape of problem.

---

## 1. Install + configure

Three minutes of one-time setup:

```bash
# Install (not on PyPI yet — pull straight from GitHub)
uv tool install git+https://github.com/mdressman/dataset-scout

# Or in a project: uv add git+https://github.com/mdressman/dataset-scout

# Configure ONE LLM provider — pick the row that matches you:
cp .env.example .env
# Then edit .env to set ONE of:
#
#   # Easiest if you already pay for Copilot (gray-area TOS — see Configuration)
#   DATASET_SCOUT_MODEL=github_copilot/gpt-5-mini
#
#   # Free tier, no subscription — needs GITHUB_TOKEN with `models:read`
#   DATASET_SCOUT_MODEL=github/gpt-4o-mini
#   GITHUB_TOKEN=github_pat_...
#
#   # Bring-your-own API key
#   DATASET_SCOUT_MODEL=openai/gpt-4o-mini
#   OPENAI_API_KEY=sk-...
#
#   # Enterprise / Azure-hosted (run `az login` once first)
#   AZURE_OPENAI_ENDPOINT=https://my-aoai.openai.azure.com
#   AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
#
#   HUGGINGFACE_HUB_TOKEN=hf_...        (optional, raises HF rate limits)
```

Without an LLM provider configured, dataset-scout runs in
metadata-only mode (HuggingFace search + cheap probes only). It
tells you what to set.

See [`configuration.md`](configuration.md) for every recognised
variable, the per-provider trade-offs, and CI / managed-identity
options.

---

## 2. Write a good brief (this is the real input)

The brief is the only input that drives everything downstream. **A
good brief describes the dataset you want, not the detector you'll
build on top.**

| ❌ Conflated detector spec | ✅ Crisp dataset request |
|---|---|
| _"Find labeled corpora for detecting over-refusal — inputs are dialogue, outputs are positive vs benign with hard-negatives. We'll train and evaluate a transformer."_ | _"Refusal-labeled corpora for customer-support agents — over-refused benign prompts plus correctly-refused harmful prompts."_ |

What belongs: labels, content shape, domain context (English,
customer-support, agent-mediated). What doesn't: input/output
schemas, model architecture, "we'll train and evaluate." Aim for
**under 250 characters**.

dataset-scout will surface a hint in the report when it detects a
detector-spec pattern. Listen to it.

See [`concepts.md` §9](concepts.md#9-how-to-write-a-brief) for the
fuller treatment.

---

## 3. Iterate cheaply on the brief

Before paying for a full recon, validate the directions:

```bash
datascout decompose "refusal-labeled corpora for customer-support agents — over-refused benign prompts plus correctly-refused harmful prompts" --out scratch/
```

~5 seconds, one LLM call. The model proposes 3–7 search directions
adjacent to your brief and prints them to stdout. Read them.

- If the directions look wrong, refine the brief and re-run.
- If they look right, the saved `scratch/decomposition.yaml` is
  ready to feed into a full recon — skipping the
  decomposition step the next time so you don't re-pay for it.

---

## 4. Run the full recon

```bash
datascout recon "<your refined brief>" \
    --decomposition-from scratch/decomposition.yaml \
    --out scratch/recon/
```

What runs:

1. **Brief parsing.** The heuristic parser extracts languages,
   threat families, deployment context. No LLM required.
2. **Multi-direction search.** Each direction's keywords become
   their own HF query; results dedupe into one pool with full
   provenance.
3. **Cheap probes.** Six metadata-driven signals per candidate:
   license (with SPDX guess), size, recency, freshness bucket,
   declared languages, card-completeness.
4. **Two-stage shortlist.** Top-k per direction for breadth, then
   global re-rank by multi-direction hits + license sanity + card
   hygiene caps the assessor input at ~35 candidates (with
   recalled-name rescues force-included).
5. **Row-aware strategy assessment.** **For each shortlisted
   candidate, the assessor fetches 8 real rows** before the LLM call.
   Returns 1–4 ranked strategies from the 7-kind taxonomy with
   actual column names and label values in the transform spec.
6. **Coverage report.** What aspects of your brief no candidate
   covers, and what to do about each.
7. **Re-rank** by best strategy + kind bonus.

Outputs:

```
scratch/recon/
├── report.md            human-readable; LEADS WITH coverage gaps when notable
├── results.json         structured ReconResult
├── decomposition.yaml   stand-alone direction list (hand-editable)
└── recipe.draft.yaml    REAL column names + label values, curate-ready
```

Open `scratch/recon/report.md`. Note which candidates the LLM
tagged `direct_use` versus `signal_proxy` — that's the value-add
over naive HF search. **If you see "Sourcing roadmap" leading the
report, your brief is exploring frontier territory and HF coverage
is sparse — the gaps + decomposition are the actual deliverable.**

---

## 5. Inspect a candidate (optional but useful)

A strategy badge is a hypothesis. Before committing, **inspect** the
top candidate yourself:

```bash
datascout inspect huggingface:bench-llm/or-bench --intent-from scratch/recon/results.json
```

Outputs (to stdout, pipeable):

- Identity, license, card-declared metadata
- Inferred schema from a 50-row sample
- Label distribution with **Wilson 95% CIs** so you don't
  over-interpret 50 rows
- Min / median / max text length
- First five sample rows
- Strategy assessment against the same Intent recon used (`--intent-from`)

```bash
# Deep-dive multiple candidates
datascout inspect huggingface:walledai/XSTest --intent-from scratch/recon/results.json > xstest.md
```

---

## 6. Hand-edit the recipe (lightly — recipes are real now)

Open `scratch/recon/recipe.draft.yaml`. Because the assessor sampled
real rows during recon, the transforms reference **actual column
names**:

```yaml
components:
  - id: huggingface_bench-llm_or-bench
    source: huggingface
    source_id: bench-llm/or-bench
    revision: 4f61ec...
    surfaced_by: [refusal_benchmarks]
    strategy: direct_use
    strategy_confidence: 0.85
    rationale: |
      Direct fit: prompts paired with refusal labels. Maps cleanly to
      a binary over-refusal classifier.
    transform:
      text_column: prompt              # ← real column name
      label_column: category           # ← real column name
      label_value_map:
        over-refusal: positive         # ← real label values from the data
        ok: benign
      label_kind_map:
        all: ground_truth
      filter: null                     # sandboxed expression, e.g. len(text) > 30
      take: 5000                       # auto-capped from "all" — flip to a number or "all"
```

> **About the auto-cap.** When the strategy assessor returns
> `take: "all"` (the natural default since it can't predict dataset
> size), the draft writer caps it to **5000 rows per component** and
> adds a caveat explaining the cap and how to lift it. This keeps a
> first-pass `curate` snappy on heavy code/text corpora — the
> hallucinated-API corpus from `docs/concepts.md` materialises in
> ~60 s with the cap, vs 25+ minutes uncapped.

Common edits:

- **Drop weak components** — delete entries you don't trust. They
  land in the `declined` list with a reason.
- **Tune `take`** — flip back to `take: all` for full
  materialization, or set an explicit integer like `take: 2000` for
  a balanced blend (especially useful for signal proxies that would
  otherwise dominate the direct-fit components).
- **Apply filters when useful** — `transform.filter` accepts a
  sandboxed expression like `len(text) > 30 and label != 'unknown'`.
  Allowed primitives: comparisons, boolean ops, `in`, `len`,
  `contains_pattern`, `lower`, `startswith`, `endswith`, `int`,
  `str`. Anything else (attribute access, lambdas, etc.) is rejected
  at recipe load.

```bash
cp scratch/recon/recipe.draft.yaml recipe.yaml
$EDITOR recipe.yaml
```

The recipe is the seam between recon and curate. Anything you put
in here, the corpus reflects.

---

## 7. Curate the corpus

> **⚠️ Experimental — output not yet end-to-end validated.**
> `curate` ships a working implementation with a full audit trail
> (lockfile, MinHash dedup, leakage-aware splits, per-component
> soft-failure classification) — but the author hasn't yet
> personally trained a model on a scout-curated corpus and confirmed
> quality vs a hand-built reference. Inspect rows, sanity-check
> label distributions, and compare against your own gold before
> training on the output. Bug reports and PRs that harden this
> pipeline are very welcome — please file issues or open a PR.

```bash
datascout curate --from recipe.yaml --out ./over-refusal-corpus \
    --max-concurrency 6
```

Output:

```
✔ 4,231 row(s) written to over-refusal-corpus (5 component(s) kept,
  2 skipped, 1 failed) in 47.8s
  - splits: train=3385 · val=419 · test=427
  - fingerprint: 8f3a40b1c2d4e007...
  ! 1 component(s) skipped due to upstream errors — see report.md
    / recipe.lock.yaml → failed_components for hints:
    - huggingface_TrustAIRLab_in-the-wild-jailbreak-prompts
      [missing_config]: Set `source_config: <config_name>` on this
      component (the HF dataset has multiple configs).
  ✓ audit-ready: leakage-aware splits + filter DSL applied
    (MinHash dedup, num_perm=128, threshold=0.8)
```

A "kept / skipped / failed" line is the new normal: real recipes
often have one or two components that need a `source_config` or
`source_split` set. The corpus still ships from the rest, and the
hint tells you exactly what one-line edit re-enables the laggard
on the next run. **No restart, no triage from a stack trace.**

> **Speed knobs.** Most of the per-component cost is HuggingFace
> `load_dataset` setup overhead (split discovery, parquet header
> fetch). `--max-concurrency 6` materialises 6 components in
> parallel — meaningful speedup since the work is I/O-bound.
> `--max-rows-per-component 500` caps the row count for one run
> without editing the recipe — useful for fast first-pass
> inspection. Both are deterministic: same recipe + same seed →
> same fingerprint regardless of which workers finish first.

Inside `over-refusal-corpus/`:

```
├── train.jsonl / val.jsonl / test.jsonl   schema-normalized records
├── recipe.yaml                             your edited recipe (verbatim)
├── recipe.lock.yaml                        pinned revisions, realized counts, hashes
├── manifest.json                           machine-readable lock equivalent
├── report.md                               5-second scorecard + provenance
├── fingerprint.txt                         one-line content hash
└── usage.md                                3-line snippets for HF datasets / pandas
```

A sample row:

```json
{
  "text": "How do I remove the password from my account?",
  "label": "positive",
  "label_kind": "ground_truth",
  "strategy": "direct_use",
  "strategy_confidence": 0.85,
  "source": "huggingface:bench-llm/or-bench",
  "source_row_id": "742",
  "source_revision": "4f61ec...",
  "extras": { "/* every original column verbatim */" },
  "extras_coercion": false
}
```

Two fields to internalise:

- **`label_kind`** — `ground_truth` rows are safe to train AND eval
  on. `proxy` rows are train-only — *exclude them from your eval
  set*. The field is load-bearing for honest measurement.
- **`extras`** — the original source row is preserved verbatim
  (with multimodal coercion when needed). Nothing is dropped.

---

## 8. The audit trail

Open `over-refusal-corpus/recipe.lock.yaml`. This is the file a
reviewer asks about: which corpus did this detector train on, and
how did proxies factor in?

```yaml
recipe_version: '1'
audit_readiness: ready
audit_readiness_notes:
  - Splits are leakage-aware; whole MinHash clusters route to one split.
  - Filter expressions are sandboxed; provenance recorded per component.
intent: { ... }
min_strategy_confidence:
  recipe: 0.5
  effective: 0.5
  overridden_by_cli: false
seed: { recipe: 42, effective: 42, overridden_by_cli: false }
splits:
  method: minhash_lsh
  num_perm: 128
  threshold: 0.8
  shingle_size: 5
  dedup_version: 1
  proportions_recipe: { train: 0.8, val: 0.1, test: 0.1 }
  realized: { train: 3385, val: 419, test: 427 }
  clusters_total: 4072
  clusters_singleton: 3937
  clusters_largest: 4
  rows_in_dup_clusters: 294
components:
  - id: huggingface_bench-llm_or-bench
    source_id: bench-llm/or-bench
    revision: 4f61ec...
    strategy: direct_use
    strategy_confidence: 0.85
    realized:
      rows_taken: 1247
      label_kind_counts: { ground_truth: 1247 }
      row_identity_method: column:id
fingerprint: 8f3a40b1c2d4e007...
scout_version: 0.0.1
```

`audit_readiness: ready` means: deterministic seed, leakage-aware
split (MinHash + LSH at threshold 0.8), filter expressions sandboxed
to a small allowlist, and all of those parameters recorded for the
reviewer. The lockfile says exactly what happened — no hand-waving.

---

## 9. Use it downstream

`usage.md` has the three snippets you'll most often want:

```python
# huggingface_hub datasets
from datasets import load_dataset
ds = load_dataset("json", data_files={
    "train": "over-refusal-corpus/train.jsonl",
    "val":   "over-refusal-corpus/val.jsonl",
    "test":  "over-refusal-corpus/test.jsonl",
})

# pandas
import pandas as pd
train = pd.read_json("over-refusal-corpus/train.jsonl", lines=True)

# raw jsonl
import json
rows = [json.loads(l) for l in open("over-refusal-corpus/train.jsonl")]
```

A reasonable training-time filter:

```python
# Train on everything; eval on ground-truth only.
train = ds["train"]
eval_ = ds["val"].filter(lambda r: r["label_kind"] == "ground_truth")
```

---

## 10. Multi-detection programs (`compose`)

If your threat model spans several sub-detections — e.g., one
threat-intel report yielded three sub-programs that share a corpus —
run recon for each, then compose:

```bash
# One recon per sub-detection
datascout recon "<brief 1>" --out detection-1/
datascout recon "<brief 2>" --out detection-2/
datascout recon "<brief 3>" --out detection-3/

# Merge the recipes — components dedupe by (source, source_id),
# higher confidence wins on conflict
datascout compose \
    detection-1/recipe.draft.yaml \
    detection-2/recipe.draft.yaml \
    detection-3/recipe.draft.yaml \
    --out programs/merged-recipe.yaml \
    --intent-brief "Combined detection corpus"

# Materialise the merged corpus
datascout curate --from programs/merged-recipe.yaml --out programs/corpus/
```

---

## 11. Rescue weak labels with the judge (`datascout judge`)

Recon's `signal_proxy` strategy is honest about being a proxy — but
sometimes a corpus has plenty of rows and the question is *which of
the proxy-labeled rows are actually the thing*. That's what M10's
LLM-as-judge is for: ask one labeling question (an "axis"), get a
strict JSON verdict per row, promote only rows the judge is confident
about (`label_confidence ≥ 0.8` by default).

```bash
# Optional: write a short rubric describing the axis. Free-form text
# works fine for v1.
cat > rubrics/psych_harm.txt <<'RUBRIC'
Score the row 'positive' if it depicts or solicits help with
first-person psychological self-harm or suicidal ideation.
Score it 'negative' for ordinary emotion talk, third-person
reportage, or fiction. Score it 'ambiguous' only when the
directionality is genuinely unclear.
RUBRIC

# Calibrate first if you have a small gold set — same rubric, real
# precision/recall numbers before you pay for the full pass.
datascout judge ./mycorpus \
    --axis psych_harm --rubric rubrics/psych_harm.txt \
    --calibrate-against ./gold-psych-harm \
    --calibration-floor 0.8 \
    --threshold 0.8

# Score the judged corpus against gold any time.
datascout eval ./mycorpus/judged --against ./gold-psych-harm \
    --axis psych_harm
```

The judge **does not overwrite** your input corpus. It writes the
promoted rows to `<TARGET>/judged/` (override with `--out`), plus
`judge.lock.yaml` (full audit trail — model, scout-internal
`template_version`, `n_judges`, `agreement`, threshold, calibration
metrics) and `judge.report.md`. Re-runs are nearly free: every
verdict is sha256-cached under `<workspace>/.cache/dataset-scout/judge/`,
and the per-batch `.judge_state.json` checkpoint lets a partial run
resume on the next invocation. See
[CLI reference → judge](cli.md#judge) for the full option set.

---

## 12. Honest limits

- **Discovery is HuggingFace-lexical-bound.** A dataset whose card
  text doesn't intersect your brief's keywords won't surface, even
  if it's a perfect semantic fit. List well-known named benchmarks
  in the brief — the decomposer turns them into proper-noun queries.
  See [Concepts §9](concepts.md#9-how-to-write-a-brief).
- **Recon assesses the top ~35 of ~100 candidates per axis.** The
  cap protects an LLM-cost budget; the rest stay listed but
  unassessed. When an axis comes back empty, sweep the unassessed
  list before declaring a coverage gap.
- **Strategy assessment is LLM-judgment**, not ground truth. Read
  the rationale and inspect samples before committing.
- **Card metadata is uneven.** Licenses are sometimes wrong;
  languages are often missing. We surface what's there and clearly
  mark what isn't.
- **Recency alone means little.** "Uploaded yesterday" doesn't
  imply the data reflects today's threat surface.
- **Reproducibility is contingent.** `recipe.lock.yaml` pins
  revisions and content hashes. If upstreams delete the data, the
  blend won't be reproducible standalone — the lockfile pins
  revisions but not contents.
- **Judge verdicts are LLM-generated too.** The promotion rule is
  conservative (explicit-gap, threshold ≥ 0.8 by default), and
  `--calibrate-against` lets you measure precision against a real
  gold set before paying for the full pass — but a judged label is
  only as good as the rubric you wrote.
- **Not legal advice.** License signals are an SPDX best-effort
  guess; read the upstream card before redistributing.

---

## What's next

- **[CLI reference](cli.md)** — every flag, every verb.
- **[Concepts](concepts.md)** — the strategy taxonomy, label kinds,
  why discovery vs ranking matters, **how to write a brief**.
- **[Configuration](configuration.md)** — env vars, paths,
  programmatic `ScoutContext`.
- **[Architecture](architecture.md)** — pipeline, sources, probes,
  capability status.

Found a sharp edge? Open an issue. The honest limits list is fed by
real friction; please add to it.

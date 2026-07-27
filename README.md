<h1 align="center">dataset-scout</h1>

<p align="center">
  <em>Your dataset-discovery teammate.</em><br>
  Tell it what you need. Get back a recon report you can ship to your team — with receipts — in minutes.
</p>

<p align="center">
  <a href="docs/getting-started.md">Getting started</a> ·
  <a href="docs/concepts.md">Concepts</a> ·
  <a href="docs/cli.md">CLI</a> ·
  <a href="docs/architecture.md">Architecture</a>
</p>

---

ML practitioners spend hours hand-searching HuggingFace for datasets
that fit a problem — then second-guessing whether the creative
reframings they have in mind actually hold up. `dataset-scout` runs
that loop for you and hands back a **recon report**: every candidate
ranked by strategy fit, every claim tied back to a card, a column,
and a sample row.

---

## What you get back: the recon report

`datascout recon "your brief"` produces a self-contained
**`report.html`** (plus a Markdown twin for PRs and audit trails)
that summarizes everything the pipeline learned about candidate
datasets for your problem.

> 📄 **[See an example →](https://htmlpreview.github.io/?https://github.com/mdressman/dataset-scout/blob/main/docs/example-report.html)** —
> a real recon for a *"Claiming to be Capable of Relationship
> Development"* psych-risk-factor brief. Open it in your browser to
> see the actual output shape. (Or clone the repo and open
> `docs/example-report.html` directly.)

What's in the report:

- 🎯 **Candidate cards grouped by strategy kind** — direct fits,
  reframings, signal proxies, benign baselines — with rationale,
  **real column names**, sample-row evidence, license + freshness
  badges, and per-strategy confidence.
- 📋 **At-a-glance scoreboard** — verdict mix in 5 seconds.
- 🧭 **Sourcing roadmap / coverage gaps** — leads the report when
  your brief is in frontier territory and HF coverage is sparse
  (sparse coverage is a deliverable, not a failure).
- 🔁 **Decomposition** — the LLM's expansion of your brief into
  search directions, collapsed but inspectable.
- 📚 **Related papers** from NeurIPS / ICML / ICLR / SaTML / arXiv
  — with the HF/Kaggle datasets they cite already promoted into the
  candidate pool with paper provenance.
- 🧾 **Recipe preview** — the bridge to an optional downstream
  corpus build (see *Bridge to a corpus*, below).

The HTML is self-contained: embedded CSS, no JavaScript, no external
assets. Open it offline, attach it to a ticket, paste it into a doc.
The Markdown twin renders from the same view-model so the two can't
drift — pick whichever channel fits the audience.

---

## The loop

```bash
# Pick ONE LLM provider — see Configuration for the full matrix.
echo "DATASET_SCOUT_MODEL=github_copilot/gpt-5-mini" > .env  # uses your GitHub Copilot subscription
# OR: echo "DATASET_SCOUT_MODEL=github/gpt-4o-mini" > .env   # free GitHub Models tier (needs GITHUB_TOKEN)
# OR: az login && cat <<EOF > .env                           # Azure OpenAI / Entra
# AZURE_OPENAI_ENDPOINT=https://my-aoai.openai.azure.com
# AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
# EOF

# 1. Iterate on the brief cheaply (~5s, one LLM call)
datascout decompose "your brief here" --out scratch/

# 2. When the directions look right, run the full recon (~2 min)
datascout recon "your brief here" \
    --decomposition-from scratch/decomposition.yaml \
    --out scratch/recon/
# → scratch/recon/report.html      ← open this
# → scratch/recon/report.md        ← share this in PRs / Slack
# → scratch/recon/results.json     ← machine-readable
# → scratch/recon/recipe.draft.yaml + decomposition.yaml
```

> **No Azure account?** `DATASET_SCOUT_MODEL=github_copilot/gpt-5-mini`
> reuses the GitHub Copilot subscription you already have. First call
> opens an OAuth device-code flow; thereafter the token is cached in
> `~/.config/litellm/github_copilot/`. See
> [Configuration](docs/configuration.md) for the trade-offs and the
> full provider list.

> **Tip:** `decompose` is the cheap brief-iteration loop. Use it to
> refine your brief before paying for the full ~2-minute recon.

---

## Why scout

Three things you don't get from a tab full of HuggingFace search
results:

1. **Reframings, not just matches.** Each candidate card in the
   report carries ranked strategies — *direct fit*, *subset
   extraction*, *signal proxy*, *benign baseline* — with rationale,
   caveats, and a transform spec. On novel briefs the typical shape
   is zero direct fits plus a defensible stack of reframings, and
   that's the deliverable.

2. **Receipts everywhere.** Strategy assessment reads **real rows**
   from each candidate before writing the rationale, so the report
   references actual column names and label values (no
   `prompt_or_response_equivalent` placeholders). Every claim ties
   back to a card, a column, and a sample.

3. **Coverage gaps are first-class output.** When the data isn't
   there, the report leads with what's missing and where to look
   next — sourcing roadmap, not failure mode.

Discovery spans HuggingFace, Kaggle, and academic papers from
NeurIPS / ICML / ICLR / SaTML and other venues — HF/Kaggle URLs
found in paper abstracts are promoted into the candidate pool with
paper provenance.

---

## Bridge to a corpus *(experimental)*

Every recon also writes a `recipe.draft.yaml` with the same real
column names the report references. If you want to materialize a
JSONL corpus from it:

```bash
datascout curate --from scratch/recon/recipe.draft.yaml --out ./mycorpus
```

> **⚠️ `curate` is experimental — output not yet end-to-end validated.**
> The pipeline ships a working implementation with a full audit
> trail (lockfile, MinHash dedup, leakage-aware splits, per-component
> soft-failure classification), but the author hasn't yet trained a
> model on a scout-curated corpus and confirmed quality vs a
> hand-built reference. **The recon report is the value-prop today.**
> Treat curate output as a starting point, sanity-check rows and
> label distributions, and please file issues or PRs if you harden
> the pipeline. Same caveat applies to the optional `judge` / `eval`
> flow downstream.

---

## Who this is for

Any practitioner who needs to **assemble a labeled corpus from public
data** — quickly, defensibly, and without playing column-name
whack-a-mole. Examples:

- **Detection engineers** building a prompt-injection / over-refusal /
  unsafe-output classifier. *"Find me direct fits where they exist
  and creative reframings of related work so the corpus is robust."*
- **Eval engineers** assembling a labeled reference set for a
  retrospective grade of a deployed system.
- **ML researchers** doing fast triage across HuggingFace for a new
  problem area — *"what's even out there?"*
- **Data scientists** stitching multiple narrow corpora into one
  cohesive blend that preserves class balance.

Sweet spot: AI-security and detection work, where reframings of
adjacent data are how you survive frontier-territory briefs. But the
loop generalizes.

---

## Documentation

- **[Getting started](docs/getting-started.md)** — install, configure,
  run the loop end-to-end, with the over-refusal detection scenario
  as the working example.
- **[Concepts](docs/concepts.md)** — the mental model: **how to write
  a good brief**, the strategy taxonomy, label kinds, modes, recipes
  and lockfiles.
- **[CLI reference](docs/cli.md)** — every verb, every flag, with
  worked examples.
- **[Architecture](docs/architecture.md)** — pipeline diagram,
  module layout, source plugin contract, milestone status.
- **[Configuration](docs/configuration.md)** — `ScoutContext`, the
  multi-provider LLM matrix (GitHub Copilot, GitHub Models, OpenAI,
  Anthropic, Azure OpenAI / Entra), HuggingFace tokens, every
  recognised env var.

---

## Install

Two equivalent CLI entry points: `dataset-scout` (formal) and
`datascout` (recommended short form).

```bash
# Install as a uv tool, straight from GitHub (no clone, no PyPI):
uv tool install git+https://github.com/mdressman/dataset-scout
datascout --help

# Or developer install:
git clone https://github.com/mdressman/dataset-scout
cd dataset-scout
uv sync
uv run datascout --help
```

Python 3.11+. Not on PyPI yet — `uv add dataset-scout` will work
once published; until then use the `git+` URL above. Heavy deps
(litellm, azure-identity, datasets) load lazily — metadata-only
runs cost no LLM-import overhead.

---

## Advanced features

`recon` is the headline verb — it produces the report. `decompose`
is the cheap brief-iteration helper, and `curate` is the
*experimental* downstream step. The rest of the surface is for
specific needs:

- **`datascout inspect <source>:<id>`** — single-candidate deep-dive:
  schema from a streamed sample, label distribution with Wilson 95% CIs,
  text-length stats, license, and a strategy assessment against your
  current Intent. Pipes cleanly to a file.
- **`datascout compose r1.yaml r2.yaml ...`** — merge multiple
  recipes into one. For multi-detection programs that share a corpus.
- **`datascout judge ./mycorpus --axis X`** — opt-in LLM-as-judge label
  rescue for proxy / weakly-labeled rows. Multi-judge agreement
  (single / majority-of-3 / unanimous-of-5), explicit-gap promotion at
  a configurable threshold, sha256-keyed disk cache, per-batch resumable,
  and `--calibrate-against` reports P/R/F1 vs gold *before* the full
  pass with an optional precision floor that aborts a too-low run.
  Strictly audited; never the default.
- **`datascout eval ./judged --against ./gold`** — score any judged
  corpus against any gold corpus.
- **`datascout render <out_dir>`** — re-emit `report.html` / `report.md`
  from an existing `results.json`. No recon, no API calls — useful for
  iterating on report styling. `--html-only` / `--md-only` to skip one.

See [`docs/cli.md`](docs/cli.md) for the full surface.

---

## Honest limits

- **`curate` output is not yet end-to-end validated.** The
  recon → curate path ships a working implementation with a full
  audit trail (lockfile, MinHash dedup, leakage-aware splits,
  per-component soft-failure classification), but the author hasn't
  yet trained a model on a scout-curated corpus and confirmed
  quality against a hand-built reference. Treat output as a starting
  point: inspect rows, sanity-check label distributions, and compare
  to your own gold before relying on it. The same caveat applies to
  the opt-in `judge` / `eval` flow downstream of `curate`. Bug
  reports and PRs that harden this pipeline are very welcome.
- **Strategy assessment is LLM judgment, not ground truth.** Inspect
  samples before committing.
- **Discovery is HuggingFace-lexical-bound.** A dataset whose card
  text doesn't intersect the brief's keywords (or a decomposition
  direction's recalled names) won't surface, even if it's a
  semantically perfect fit. If your construct has well-known named
  benchmarks, list them in the brief — the decomposer turns them
  into proper-noun queries.
- **Recon assesses the top ~35 of ~100 candidates per axis.** That
  cap protects an LLM-cost budget; the rest stay listed but
  unassessed. When an axis comes back empty, sweep the unassessed
  list before declaring a coverage gap.
- **Paper-only datasets (no HF/Kaggle home) won't auto-promote to
  candidates.** Scout extracts dataset references from paper
  abstracts but only promotes ones with a HuggingFace or Kaggle URL.
  Datasets hosted on author sites or generic GitHub repos surface
  as paper citations but stop there — verify those manually.
- **Paper search is rate-limited.** Semantic Scholar throttles under
  parallel runs; arXiv falls back as a targeted second source for
  named-benchmark queries. If both fail, recon proceeds without the
  paper channel rather than blocking.
- **Card metadata is uneven.** License fields are sometimes missing
  or wrong; `language:` is often absent. We surface what's there and
  mark what isn't — no fabrication.
- **Recency alone means little.** "Uploaded yesterday" doesn't imply
  the data reflects today's threat surface.
- **Reproducibility is contingent.** `recipe.lock.yaml` pins revisions
  and content hashes; if upstreams delete the data, only an archive
  (a future feature) makes the blend reproducible standalone.
- **Judge verdicts are LLM-generated too.** Conservative promotion
  rules, calibration mode, and precision floors mitigate but don't
  eliminate the risk. A judged label is only as good as the rubric
  you wrote.
- **Not legal advice.** License signals are an SPDX best-effort guess;
  read the upstream card before redistributing.

---

## License

MIT. See [LICENSE](LICENSE).

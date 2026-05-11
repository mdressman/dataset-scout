# Requirements

## Functional requirements

### R1: Brief-driven discovery
- Accept a natural-language brief (target: under 250 chars) describing the desired dataset
- Parse the brief to extract intent, domain, content shape, and label expectations
- Detect and warn when the brief describes a detector spec instead of a dataset request
- **Stakeholders:** All primary users

### R2: LLM decomposition
- Expand the brief into multiple search directions (reframings, adjacencies, named benchmarks)
- Support cheap iteration via `decompose` command (~5s, single LLM call)
- Allow reuse of decomposition output in full recon via `--decomposition-from`
- **Stakeholders:** Detection engineers, ML researchers

### R3: Multi-source search
- Search HuggingFace (primary), Kaggle (when credentials available), and academic papers (Semantic Scholar, arXiv fallback)
- Promote HF/Kaggle datasets cited in paper abstracts into the candidate pool with paper provenance
- Deduplicate candidates surfaced by multiple directions while preserving provenance
- **Stakeholders:** All primary users

### R4: Strategy assessment with real evidence
- Stream real sample rows from each candidate before LLM assessment
- Produce per-candidate ranked strategies (direct_use, reframing variants, signal_proxy, benign_baseline, not_useful)
- Each strategy carries confidence, rationale, caveats, and a concrete transform spec with actual column names
- **Stakeholders:** Detection engineers, reviewers

### R5: Self-contained recon report
- Generate HTML report (embedded CSS, no JS, no external assets) and Markdown twin from same view model
- Include: candidate cards grouped by strategy, scoreboard, coverage gaps/sourcing roadmap, decomposition, related papers, recipe preview
- Offline-capable, attachable to tickets
- **Stakeholders:** All users, reviewers

### R6: Draft recipe output
- Write `recipe.draft.yaml` with real column names from assessment
- Auto-cap `take: all` to 5000 rows with documented caveats
- **Stakeholders:** Detection engineers, data scientists

### R7: Corpus materialization (experimental)
- `curate` command materializes recipe into train/val/test JSONL splits
- MinHash dedup, leakage-aware splits, per-component soft-failure classification
- Lockfile (`recipe.lock.yaml`) as defensible audit record
- Parallel materialization with deterministic output regardless of worker completion order
- **Stakeholders:** Detection engineers, compliance

### R8: Label honesty
- Every row carries `label_kind` (ground_truth, subset_extracted, remapped, proxy, judged)
- Proxies are marked by default, eval must exclude them
- **Stakeholders:** Eval engineers, compliance

### R9: Judge and eval (experimental)
- Optional LLM-as-judge label rescue with multi-judge agreement, explicit-gap promotion, calibration mode
- Eval scoring against gold corpus
- **Stakeholders:** Detection engineers, eval engineers

## Non-functional requirements

### NF1: Performance
- Full recon completes in ~2 minutes for typical briefs
- Decompose completes in ~5 seconds
- Curate with auto-cap completes in minutes for typical recipes

### NF2: Graceful degradation
- Without AOAI credentials: explicit fallback to metadata-only mode with noisy warning
- Per-component failures in curate don't crash the pipeline
- Paper search failures don't block recon

### NF3: Reproducibility
- Lockfile pins revisions and content hashes
- Parallel curate produces identical output regardless of worker order

### NF4: Extensibility
- Source plugins via entry-point group (`dataset_scout.sources`)
- Probe protocol for adding new metadata-driven checks
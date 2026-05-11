# Solution

## Approach

dataset-scout is a CLI pipeline that automates dataset discovery, reframing, and corpus assembly. The solution decomposes into a staged pipeline where each stage can run independently or chain together.

## Pipeline architecture

### Stage 1: Brief parsing and decomposition
- Heuristic parser extracts intent (domain, content shape, label expectations) from a natural-language brief
- LLM decomposer expands the brief into 5-15 search directions: reframings, adjacencies, named benchmarks turned into proper-noun queries
- `decompose` command exposes this as a cheap iteration loop (~5s, one LLM call)

### Stage 2: Multi-source search
- Each decomposition direction fires parallel searches across source plugins (HuggingFace primary, Kaggle when credentials available)
- Academic paper search (Semantic Scholar primary, arXiv fallback for named-benchmark queries) finds papers citing relevant datasets
- HF/Kaggle URLs extracted from paper abstracts are promoted into the candidate pool with paper provenance
- Candidates are deduplicated across directions while preserving `surfaced_by` provenance

### Stage 3: Cheap probes
- Six metadata-driven probes run on each candidate: license, size, recency, freshness, languages, card_completeness
- Each probe produces a typed SubScore with explicit status (ok, not_applicable, low_confidence, skipped)
- No aggregate score, probes are annotations for human judgment

### Stage 4: Shortlisting
- Two-stage shortlist reduces ~100 candidates per axis to ~35 for LLM assessment
- Embedding-based label-intent fit runs between cheap probes and shortlist
- Protects LLM cost budget while keeping the most promising candidates

### Stage 5: Strategy assessment
- Streams 8 real sample rows from each shortlisted candidate via Source.stream_rows()
- LLM produces 1-4 ranked strategies per candidate from the taxonomy: direct_use, subset_extraction, label_remapping, cross_class_repurposing, signal_proxy, benign_baseline, not_useful
- Each strategy carries confidence, rationale, caveats, and a transform spec with actual column names and label values
- Graceful degradation to metadata-only when source is unreachable or gated

### Stage 6: Report generation
- Single view model feeds both HTML and Markdown renderers (can't drift)
- HTML is self-contained: embedded CSS, no JS, no external assets
- Includes: candidate cards grouped by strategy kind, scoreboard, coverage gaps/sourcing roadmap, decomposition, related papers, recipe preview
- Coverage gaps are first-class output, not failure mode

### Stage 7: Recipe and curate (experimental)
- `recipe.draft.yaml` written with real column names from assessment, auto-capped at 5000 rows per component
- `curate` materializes recipe into train/val/test JSONL with MinHash dedup, leakage-aware splits
- Parallel materialization (default 4 workers) with deterministic output
- Per-component soft-failure classification (gated_dataset, missing_config, bad_split, etc.)
- `recipe.lock.yaml` as defensible audit record

### Stage 8: Judge and eval (experimental)
- Optional LLM-as-judge label rescue for proxy/remapped/unknown rows
- Multi-judge agreement modes (single, majority-of-3, unanimous-of-5)
- Calibration mode with precision floor before full pass
- Eval scoring against gold corpus

## Key design decisions

1. **No aggregate quality score.** Per-signal evidence and per-candidate strategy assessment with rationale. Avoids false confidence.
2. **Proxies are honest by default.** `label_kind` field on every row, downstream eval must exclude proxies.
3. **Receipts everywhere.** Every claim ties to a card, column, sample row. No placeholder column names.
4. **Coverage gaps are deliverables.** Sparse coverage on frontier briefs is the expected case, reported as sourcing roadmap.
5. **Graceful degradation over hard failure.** Missing AOAI falls back to metadata-only mode. Per-component curate failures don't crash. Paper search failures don't block recon.
6. **Source-agnostic metadata envelope.** CandidateMetadata is the same shape regardless of source, so probes and assessment don't need source-specific code.

## Technology choices

- Python 3.11+, Pydantic for data models, Typer for CLI, structlog for logging
- Azure OpenAI via litellm + azure-identity (Entra auth)
- HuggingFace Hub API for search and streaming
- datasketch for MinHash dedup
- Heavy deps load lazily to keep metadata-only runs fast
- Source plugins via entry-point group for extensibility
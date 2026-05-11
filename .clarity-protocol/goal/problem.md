# Problem Statement

## What is dataset-scout?

ML practitioners spend hours manually searching HuggingFace, Kaggle, and academic papers for datasets that fit a problem, then second-guessing whether creative reframings of adjacent data actually hold up. There is no automated tool that runs this discovery-and-reframing loop and hands back a defensible recon report with receipts (real column names, sample rows, license signals, strategy rationale).

## The core problem

Assembling a labeled corpus from public data for AI detection and safety work is slow, error-prone, and hard to audit. The specific pain points:

1. **Discovery is lexical-bound and manual.** HuggingFace search only finds datasets whose card text intersects your keywords. Adjacent datasets that could be reframed (subset extraction, label remapping, signal proxy) are invisible.
2. **Reframings are ad-hoc.** Practitioners informally reason about whether a toxicity dataset could serve as a proxy for over-refusal detection, but there's no structured assessment with confidence levels and caveats.
3. **No receipts.** Claims about dataset fitness aren't tied back to actual column names, label values, and sample rows. Placeholder column names like `prompt_or_response_equivalent` are common.
4. **Coverage gaps are invisible.** When the data doesn't exist for a frontier brief, there's no structured output showing what's missing and where to look next.
5. **Audit trail is absent.** Detection engineers under compliance pressure need a defensible record of which corpus a detector trained on, how proxies factored in, and what was excluded.

## Who experiences this problem?

- Detection engineers building prompt-injection, over-refusal, or unsafe-output classifiers
- Eval engineers assembling labeled reference sets
- ML researchers doing fast triage across HuggingFace for new problem areas
- Data scientists stitching multiple narrow corpora into one cohesive blend

Sweet spot: AI-security and detection work, where reframings of adjacent data are how you survive frontier-territory briefs.

## What does success look like?

- A practitioner writes a brief (under 250 chars), runs `datascout recon`, and gets a self-contained HTML report with ranked candidates, strategy assessments, real column names, sample evidence, license badges, coverage gaps, and a draft recipe in under 3 minutes.
- The report is shippable to a team without further editing.
- The optional `curate` path materializes a JSONL corpus with MinHash dedup, leakage-aware splits, and a lockfile that serves as the defensible audit record.

## Current state

The project is at v0.0.1 (Pre-Alpha). The core recon pipeline is functional: brief parsing, LLM decomposition, multi-source search (HuggingFace, Kaggle, Semantic Scholar/arXiv paper search), cheap probes, two-stage shortlist, per-candidate strategy assessment with real row streaming, coverage gap analysis, HTML+Markdown report generation, and draft recipe output. The `curate` path is experimental and not yet end-to-end validated. MIT licensed, Python 3.11+, uses Azure OpenAI via Entra auth.
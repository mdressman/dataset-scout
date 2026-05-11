# Security Policy

## Supported versions

`dataset-scout` is pre-1.0 (Development Status :: 2 - Pre-Alpha). Only
the latest release on `main` receives security fixes. There are no
LTS branches.

| Version | Supported          |
| ------- | ------------------ |
| `main` (latest) | ✅          |
| < latest        | ❌          |

## Reporting a vulnerability

**Please do not file public issues for security problems.**

Use GitHub's private vulnerability reporting:

1. Open <https://github.com/mdressman/dataset-scout/security/advisories/new>
2. Describe the issue, impact, and a reproduction (a brief + command
   line is usually enough — see `CONTRIBUTING.md` for the bug-report
   shape).
3. We'll acknowledge within 5 business days and aim to triage within
   10. Coordinated disclosure timelines are negotiable.

If GitHub advisories aren't available to you, open a minimal public
issue asking for a contact channel — **do not include the
vulnerability details** in that issue.

## Scope

In scope:

- Code in `src/dataset_scout/` shipped via PyPI/GitHub releases.
- The CLI entry points (`datascout`, `dataset-scout`).
- CI workflows in `.github/workflows/` (supply-chain risks,
  privilege escalation, secret exposure).

Out of scope (file a regular issue instead):

- Bugs in third-party datasets, HuggingFace/Kaggle APIs, Semantic
  Scholar, arXiv, or LLM providers. We're a client, not a host.
- LLM strategy-assessor output quality, hallucinated reframings, or
  inconsistent recommendations — that's a correctness issue, not a
  vulnerability.
- Denial-of-service from passing pathologically large briefs or
  recipes (we'll add safeguards but won't treat reports as
  embargoed).

## What counts as a vulnerability

- Arbitrary code execution from a crafted brief, recipe, or dataset
  card.
- Path traversal / unsafe filesystem writes outside the configured
  output directory.
- Credential disclosure (AOAI tokens, HF tokens, Kaggle keys) in
  logs, error messages, or the recon report.
- Supply-chain risks in pinned dependencies (we welcome reports even
  if upstream hasn't issued a CVE yet).

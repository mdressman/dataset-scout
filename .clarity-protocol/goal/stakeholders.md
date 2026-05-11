# Stakeholders

## Primary users

### Detection engineers
- **Need:** Quickly find and assemble labeled corpora for AI safety classifiers (prompt injection, over-refusal, unsafe output)
- **Concern:** Audit trail, defensibility, proxy honesty. Need to show reviewers exactly which data trained a detector and why
- **Context:** Under compliance pressure, frontier-territory briefs where direct-fit datasets rarely exist

### Eval engineers
- **Need:** Assemble labeled reference sets for retrospective grading of deployed systems
- **Concern:** Label quality, ground-truth vs proxy distinction, reproducibility

### ML researchers
- **Need:** Fast triage of what public datasets exist for a new problem area
- **Concern:** Coverage completeness, not missing semantically relevant datasets due to lexical search limits

### Data scientists
- **Need:** Stitch multiple narrow corpora into one cohesive blend preserving class balance
- **Concern:** Deduplication, leakage-aware splits, consistent schemas across sources

## Secondary stakeholders

### Team leads / reviewers
- **Need:** Review recon reports to approve dataset choices without re-doing the search
- **Concern:** Report readability, receipts tied to evidence, clear coverage gap communication

### Compliance / audit
- **Need:** Defensible record of training data provenance
- **Concern:** License compliance, lockfile completeness, proxy labeling transparency

## The project maintainer
- Open-source MIT project, solo maintainer currently
- Prioritizing recon report quality over curate pipeline validation
- Seeking contributors to harden the curate path
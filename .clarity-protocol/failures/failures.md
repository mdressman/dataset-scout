# Failure Analysis

## FM1: Report contains hallucinated evidence
**Raw failures:** F1 (hallucinated column names), F9 (decomposition leads astray)

**Chain:** Bad decomposition directions → irrelevant candidates surfaced → assessor fed unfamiliar data → hallucinated column names/label values in transform spec → recipe.draft.yaml has invalid transforms → curate fails or produces garbage rows → user trusts report without verifying

**Intervention points:**
1. Decomposition validation (user checks directions via `decompose` command)
2. Real row streaming grounds assessment in actual data
3. Curate-time validation of column names against actual schema
4. Report carries "verify sample rows" guidance

**Severity:** High — undermines the core "receipts everywhere" value proposition
**Likelihood:** Medium — real row streaming significantly mitigates but doesn't eliminate

---

## FM2: Incomplete discovery creates false confidence
**Raw failures:** F2 (lexical search gap), F5 (rate limiting)

**Chain:** Lexical search misses relevant datasets → rate limits prevent full candidate fetch → report shows only partial results → user treats report as comprehensive → misses critical datasets for their task

**Intervention points:**
1. Document lexical-bound limitation prominently in report
2. Named benchmark hints in brief help find exact datasets
3. Rate limit failures surfaced in report with "retry" guidance
4. Coverage gap section explicitly flags what categories of data weren't found

**Severity:** Medium — mitigated by coverage gap being first-class output
**Likelihood:** High for niche/frontier briefs

---

## FM3: Reproducibility breaks on upstream changes
**Raw failures:** F3 (stale/deleted datasets)

**Chain:** Upstream dataset deleted or modified → lockfile pins a revision that no longer exists → curate can't rebuild corpus → audit trail points to data that's gone

**Intervention points:**
1. Lockfile captures content hashes for detection
2. Future: archive feature to snapshot data locally
3. Document that reproducibility is contingent on upstream availability

**Severity:** Medium — affects long-term reproducibility
**Likelihood:** Medium — datasets do get deleted, especially controversial ones

---

## FM4: Cost and rate limit degradation
**Raw failures:** F4 (cost runaway), F5 (rate limiting)

**Chain:** Broad brief → many decomposition directions → many candidates per direction → expensive LLM assessment calls + HF rate limits → either high cost or degraded results from dropped candidates

**Intervention points:**
1. ~35 candidate cap per axis
2. Two-stage shortlist reduces LLM calls
3. `--max-concurrency` controls HF request rate
4. Missing: hard budget stop, cost estimation before run

**Severity:** Medium — cost is real money, degradation is silent
**Likelihood:** Medium — depends on brief breadth

---

## FM5: Label quality erosion in downstream pipeline
**Raw failures:** F7 (judge mislabeling), F10 (dedup removes distinct rows), F6 (wrong license)

**Chain:** Judge promotes incorrect labels → dedup removes rows that look similar but differ meaningfully → license guess is wrong → user trains on mislabeled, over-deduped, potentially non-compliant data

**Intervention points:**
1. `label_kind` field makes proxy/judged status explicit
2. Calibration mode with precision floor before full judge pass
3. Multi-judge agreement modes
4. Dedup parameters in lockfile for reproducibility
5. "Not legal advice" disclaimer on license signals

**Severity:** High — directly affects model quality and compliance
**Likelihood:** Low-Medium — multiple mitigations in place, but no published baselines
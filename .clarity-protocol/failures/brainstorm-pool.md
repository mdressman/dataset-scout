# Failure Brainstorming Pool

## F1: LLM hallucinated column names in strategy assessment
- **Description:** The strategy assessor could hallucinate column names or label values that don't exist in the actual dataset, despite being fed 8 real sample rows. This would produce transform specs that fail at curate time or mislead reviewers.
- **Pre-existing:** Yes (mitigated by streaming real rows, but not eliminated)

## F2: Lexical search misses semantically relevant datasets
- **Description:** HuggingFace search is lexical-bound. A dataset whose card text doesn't overlap the brief's keywords or decomposition directions won't surface, even if it's a perfect semantic fit. Users may trust the report as comprehensive when it's actually incomplete.
- **Pre-existing:** Yes (documented as known limit)

## F3: Stale or deleted upstream datasets break reproducibility
- **Description:** recipe.lock.yaml pins revisions and hashes, but if upstreams delete the data, the corpus can't be rebuilt. No archive feature exists yet.
- **Pre-existing:** Yes (documented as known limit)

## F4: LLM cost runaway on broad briefs
- **Description:** A very broad brief could decompose into many directions, each surfacing many candidates, leading to expensive LLM assessment calls. The ~35 candidate cap per axis helps but a brief with many axes could still be costly.
- **Pre-existing:** Partially (cap exists but no budget hard-stop)

## F5: Rate limiting from HuggingFace or Semantic Scholar
- **Description:** Parallel searches and row streaming can trigger rate limits (HTTP 429). This degrades recon quality silently if not all candidates or rows are fetched.
- **Pre-existing:** Yes (soft-failure classification exists for curate, less clear for recon)

## F6: Incorrect license SPDX guesses
- **Description:** License fields on HF cards are uneven. The SPDX guess could be wrong, leading users to believe a dataset is usable when it's not.
- **Pre-existing:** Yes (documented with "not legal advice" disclaimer)

## F7: Judge promotes incorrect labels
- **Description:** The LLM judge could systematically mislabel rows, and without a calibration baseline, users may trust judged labels at face value. The precision floor helps but doesn't guarantee correctness.
- **Pre-existing:** Yes (mitigated by calibration mode, multi-judge agreement, but no published baseline)

## F8: Metadata-only mode silently delivers low-value results
- **Description:** Without AOAI credentials, the pipeline falls back to metadata-only mode. If the warning is missed (e.g., in automated pipelines), users get a report without strategy assessment and may not realize the quality difference.
- **Pre-existing:** Partially (noisy warning exists but could be missed in automation)

## F9: Decomposition directions lead assessment astray
- **Description:** If the LLM decomposer produces poor or off-topic directions, the entire downstream search and assessment is built on a bad foundation. The cheap `decompose` iteration loop mitigates this but requires the user to check.
- **Pre-existing:** Partially (iteration loop exists)

## F10: MinHash dedup removes near-duplicates that are actually distinct
- **Description:** Aggressive dedup thresholds could remove rows that are superficially similar but carry meaningfully different labels or context, reducing corpus quality.
- **Pre-existing:** Possible (parameters are in lockfile but threshold tuning guidance is unclear)
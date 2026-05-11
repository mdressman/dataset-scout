# Failure Management Plans

## FM1: Report contains hallucinated evidence

**Risk:** High severity, Medium likelihood
**Goal:** Ensure every column name, label value, and transform spec in the report and recipe is verifiable against actual dataset content.

### Prevention

1. **Schema validation at curate time.** Before materializing any recipe component, load the first batch of rows and verify that every column name in the transform spec exists in the actual schema. Fail the component with a clear `schema_mismatch` error category if not. This catches hallucinated columns before they produce garbage rows.

2. **Column verification flag in assessment.** After the assessor produces a transform spec, cross-check the referenced column names against the columns observed in the streamed sample rows. Attach a `columns_verified: true/false` flag to each strategy in the report. Unverified strategies get a visible warning badge.

3. **Decomposition direction review.** The `decompose` cheap iteration loop already exists. Add a brief quality check: if decomposition directions contain terms that don't appear in any HF search results, flag them as potentially off-target before proceeding to full recon.

### Detection

1. **Curate-time schema mismatch errors.** Already partially exists via soft-failure classification. Extend to specifically detect column-name mismatches vs other errors.

2. **Report-level verification summary.** Add a small section to the report footer showing how many strategies had verified vs unverified column references.

### Response

1. Schema mismatch at curate time: component fails with `schema_mismatch` category, user sees which columns were expected vs actual, and can edit the recipe.
2. Unverified columns in report: visible badge warns the reviewer, strategy confidence is capped at a lower ceiling.

### Residual risk

Label *values* (not just column names) could still be hallucinated. Verifying that specific label values exist in the streamed sample is harder because 8 rows may not cover all labels. This is accepted as residual risk, mitigated by the "verify sample rows" guidance in the report.

---

## FM5: Label quality erosion in downstream pipeline

**Risk:** High severity, Low-Medium likelihood
**Goal:** Ensure users can trust label_kind assignments and judge verdicts, with clear signals when confidence is low.

### Prevention

1. **Published calibration baselines.** Run the judge pipeline on 3-5 representative labeling tasks with known gold labels. Publish precision/recall/F1 numbers in documentation so users know what to expect. This is currently open question OQ5.

2. **Mandatory calibration before first full judge run.** Require `--calibrate-against <gold>` on first use for a new axis. The calibration report shows P/R/F1 and the precision floor mechanism aborts if quality is too low. Document this as the recommended workflow, not just an option.

3. **Dedup threshold guidance.** Document the MinHash dedup threshold's effect on different data types. Provide recommended thresholds for text classification (more aggressive), dialogue data (less aggressive), and code (least aggressive). Expose the threshold as a recipe-level parameter.

4. **License signal confidence.** Add a `license_confidence` field (high/medium/low) based on whether the SPDX guess came from a structured field, card text extraction, or inference. Surface this in the report alongside the license badge.

### Detection

1. **Label distribution monitoring in curate output.** After materialization, report the label distribution per label_kind. Unexpected distributions (e.g., 99% proxy, 1% ground_truth) get a warning in the lockfile and report.

2. **Judge agreement metrics.** For multi-judge runs, report inter-judge agreement (Cohen's kappa or equivalent) in the lockfile. Low agreement signals unreliable verdicts.

3. **Dedup impact report.** Report how many rows were removed by dedup and from which components, so users can assess whether dedup was too aggressive.

### Response

1. Low calibration scores: precision floor aborts the judge run with a clear message. User adjusts the rubric or axis definition.
2. Skewed label distribution: warning in report, user reviews and adjusts recipe component weights or filters.
3. Low inter-judge agreement: warning in lockfile, user increases judge count or tightens the rubric.
4. High dedup removal rate: warning with affected components listed, user can adjust threshold or inspect removed rows.

### Residual risk

Even with calibration, the judge's performance on the user's specific data may differ from the calibration set. The precision floor mitigates but doesn't eliminate this. Accepted as residual risk, with the guidance that judged labels should always be spot-checked before training.
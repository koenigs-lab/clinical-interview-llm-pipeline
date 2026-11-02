# Confirmatory analysis plan template

Fill in a copy before the held-out test is opened and pass it to `cilp freeze --analysis-plan`. The freeze stores the file's hash and rejects later changes.

## Study and data

- Task, instrument and version, intended population and use, roles, consent and access approval:
- Participant unit, and inclusion and exclusion rules for sites, languages, interviewers, raters, and recording settings:
- Reference-rating procedure, rater independence, adjudication, and rules for insufficient evidence and missing labels:
- Participant split, external cohorts, and checks for duplicates and site or batch leakage:
- Transcript and privacy approvals, residual-risk assessment, retention and deletion plan:

## Frozen model conditions

- Base checkpoint with its revision or hash, and the model and dependency licenses:
- Prompt and rubric version, input mode, evidence mode and section declarations, quantization, context limit, training objective (`token_ce` or `label_rps`), and decision rule (default: lower median of the label-token distribution):
- Adapter-off comparator and the selected configuration (adapter strategy, learning rate, epochs, and seed, one for all targets) with its jobs and the training and validation data hashes:
- Further comparators, compute budget, and repeated-inference conditions:

## Outcomes and progression criteria

- Primary metric (default: `mean_target_mae_itt`, the macro-averaged intention-to-predict MAE with the worst-case error for invalid or missing predictions), sensitivity analysis (default: constant-baseline fallback), and secondary metrics on valid outputs:
- Minimum valid-output rate for selection, exact agreement over all eligible cases, facet-specific QWK and linear kappa, and severe-error rate (two or more category steps):
- Participant-clustered intervals, the primary paired contrast (e.g. adapter on vs. off on `mean_target_mae_itt`), secondary target-level contrasts with their multiplicity handling (e.g. Holm), and the number of configurations compared during selection:
- Within-person (profile) agreement across facets:
- Subgroup variables with minimum reportable cell sizes, and separate analyses of transcription and rating errors:
- Privacy checks for transcripts, prompts, outputs, and logs, and memorization probes:
- Calibration of the label probabilities (Brier score, ranked probability score, top-label ECE):
- Review procedure for severe errors, over- and underestimation, insufficient evidence, and automation bias:
- Numerical go/no-go thresholds, decision authority, remediation, and conditions for renewed validation:

## Approval

- Protocol version and date, study lead, clinical reviewer, privacy reviewer, and evaluation lead:
- Statement that the above was fixed before the held-out test was accessed:

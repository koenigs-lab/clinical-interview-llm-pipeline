# From the manuscript's framework stages to this repository

The manuscript describes a methodological framework, and this repository implements it with a fictional follow-along example. The [workflow guide](workflow.md) lists the commands in order.

| Framework stage | Commands | Private handoff and human decision |
| --- | --- | --- |
| **A.1 Interview acquisition** | `inventory-check`, `task-check`; [task guide](adapting-a-task.md) | Controlled source index, consent and eligibility decision, instrument version, pseudonymous ID, expected speakers, reference-rating plan. The re-identification key stays with the data steward. |
| **A.2 Transcription, diarization, correction, de-identification** | `transcribe` or `import-tsv`; `diarize` or `import-rttm`; `align-speakers`; `speaker-map`; `correction-summary`; `transcript-approve`; `privacy-propose` (local LLM search over batches of turns, optional `--patterns`); `privacy-apply`; `privacy-approve` | ASR, interval, mapped, corrected, and de-identified versions stay separate. The transcript reviewer checks audio, speakers, and section markers; the privacy reviewer decides every proposed identifier in the mapping file, adds missed ones, and checks residual risk, including combinations across turns. Hash-bound approvals gate dataset construction. |
| **A.3 Segmentation, labeling, dataset construction** | `label-agreement`; `split [--stratify]`; `build` | Independent ratings stay separate from adjudicated labels. Participants are split before examples exist, and labels are hash-bound to the split. Evidence comes from declared interview sections with explicit boundary rules. The manifest records counts, exclusions, section coverage, hashes, and duplicate flags. |
| **B.1 Prompting and optional retrieval** | task file, rubrics, `build` | Versioned score-only prompt and evidence selection. Each prompt holds the instructions, the rubric, and the selected evidence. |
| **B.2 Optional parameter-efficient adaptation** | `plan`, `train` | Versioned plan over adapter granularity (one adapter for all targets, per group, or per target), learning rate, epochs, and seeds, with a fixed training objective (`token_ce` or the ordinal `label_rps`); GPU safety wrapper; loss diagnostics and too-long exclusions by target and label. |
| **B.3 Model comparison and selection** | `baseline`, `predict [--base-only]`, `evaluate`, `compare`, `merge-predictions` | Label-token scoring with a lower-median decision rule; constant lower-median baseline; same-checkpoint adapter-off/on contrast; primary metric `mean_target_mae_itt`; paired contrasts with participant-clustered intervals; descriptive calibration. Selection on validation data only. |
| **Freeze gate** | `freeze --selected-condition adapted\|base_only` with a private [analysis plan](analysis-plan-template.md) | Binds task, plan, split (and thus labels), the selected jobs, which come from one configuration and cover every target, the selected condition, and the analysis plan's hash. Test build, prediction, evaluation, and comparison require it. Held-out data is recognised from the dataset itself. |
| **C.1 Bounded output generation and human review** | test `build`/`predict`/`evaluate`; `review-sheet --freeze`; `review-summary`; `assisted-ratings` | One provisional ordinal score per target from the frozen selection, reviewed by a trained person. Predictions, reference ratings, and assisted ratings stay separate. |

## Manuscript statements the repository supports

| Manuscript element | Repository basis |
| --- | --- |
| Participant assignment precedes example construction, and reference ratings are frozen before prediction review | `split` hashes inventory and labels; `build` refuses changed inputs; `freeze` binds the split |
| Score-only output, one integer per target | `predict` writes the lower median of the label-token distribution |
| Same-checkpoint adapter-off/on contrast | `predict --base-only`; `compare` |
| Constant lower-median baseline | `baseline` |
| Primary metric with invalid outputs counted | `evaluate` → `selection_metrics.mean_target_mae_itt` |
| Over-length inputs counted before any GPU job | `plan` → `too_long` |
| Target-specific QWK, MAE, severe errors, exact agreement; participant-clustered intervals with identical resamples | `evaluate`, `compare` |
| Human–human agreement on the same scale | `label-agreement` (two raters; same kappa weights, severe disagreement) |
| Sealed test set | `freeze` gate on test `build`, `predict`, `evaluate`, `compare`, and `review-sheet` |

## Possible extensions

Users can extend the pipeline for their own studies, for example with retrieval, summaries, evidence-linked output, few-shot or reasoning prompts, further calibration methods, agreement statistics for more than two raters, grouped cross-validation, chunking of long interviews, participant-level identifier detection, or tools for prospective assisted-versus-unassisted studies.

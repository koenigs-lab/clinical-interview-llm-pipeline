# Workflow

In the examples, `/secure/...` stands for the protected storage that holds all study data, from audio and transcripts to predictions and adapters. Every command writes new files and refuses to overwrite existing ones. The stages follow the manuscript.

## A.1 Interview acquisition

- **Inventory** `participant_id,source_path,eligible`: research pseudonyms and private source paths. The re-identification key stays with the data steward. `eligible=yes` records an eligibility decision.
- **Labels** `participant_id,target_id,label`, the independently established reference ratings. Optional `reference_rater_id,reference_status` record provenance. If present, every row needs both.

```bash
cilp task-check --task /secure/task/task.toml
cilp inventory-check --inventory /secure/cohort/inventory.csv
```

[Adapting a task](adapting-a-task.md) describes targets, sections, prompts, and adapters.

## A.2 Transcription, diarization, correction, and de-identification

### Transcription and speakers

```bash
cilp transcribe --audio /secure/audio/p001.wav --local-model /secure/models/asr --output /secure/turns/p001_asr.jsonl --participant p001 --session s1 --language de --gpu 0 --watch-disk-path /secure --safety-log-dir /secure/safety
cilp diarize --audio /secure/audio/p001.wav --local-pipeline /secure/models/pyannote-local --output /secure/turns/p001_intervals.jsonl --gpu 0 --watch-disk-path /secure --safety-log-dir /secure/safety
# or import intervals from another diarizer:
cilp import-rttm --input /secure/diarization/p001.rttm --recording-id p001 --output /secure/turns/p001_intervals.jsonl
cilp align-speakers --transcript /secure/turns/p001_asr.jsonl --intervals /secure/turns/p001_intervals.jsonl --output /secure/turns/p001_aligned.jsonl
cilp speaker-map --transcript /secure/turns/p001_aligned.jsonl --mapping /secure/review/p001_speakers.csv --output /secure/turns/p001_mapped.jsonl
```

- `transcribe` uses a local faster-whisper model with word timestamps and leaves every speaker `unknown`. `diarize` runs a complete local pyannote 4 pipeline with Hugging Face Hub access disabled. Both load their models from local paths.
- Both need an explicit device. `--gpu N` runs on one GPU inside the safety wrapper (`--dry-run` shows the wrapped call), and `--cpu` runs, more slowly, on a host without a GPU. `transcribe` also takes `--compute-type`, `--cpu-threads`, and `--beam-size`. The settings are recorded in the output's `.meta.json`, because compute type and device can change transcripts slightly.
- `align-speakers` gives a word a speaker only if that speaker alone overlaps it and covers more than half of it. Coverage counts a collar of 0.25 s around the speaker's intervals (`--collar`), because ASR word times and speaker boundaries rarely agree exactly. Words overlapping two speakers stay `unknown`. The mapping CSV (`diarized_speaker,role`, e.g. `SPEAKER_00,interviewer`) is a reviewed proposal, and individual turns may still need correction.
- For an already corrected transcript in TSV form (`start,end,speaker,segment`), use `cilp import-tsv` for the initial and the corrected version.

The fictional [ASR turn](../examples/synthetic/asr_turns.jsonl), [RTTM file](../examples/synthetic/diarization.rttm), and [role map](../examples/synthetic/speaker-map.csv) exercise `import-rttm`, `align-speakers`, and `speaker-map` without audio or models.

### Correction and transcript approval

Correct a **separate** copy against the audio: negations, qualifiers, temporal references, missing or hallucinated speech, speaker errors, and clinically consequential omissions. Insert the section markers your task uses ([evidence selection](adapting-a-task.md#evidence-selection)). Keep `participant_id`, `session_id`, and `turn_id` when editing a turn, and document any split or merged turns before import.

```bash
cilp correction-summary --before /secure/turns/p001_mapped.jsonl --after /secure/turns/p001_corrected.jsonl
cilp transcript-approve --before /secure/turns/p001_mapped.jsonl --corrected /secure/turns/p001_corrected.jsonl --audio /secure/audio/p001.wav --approval /secure/review/p001_transcript_approval.json --reviewer reviewer-id --expected-speaker participant --expected-speaker interviewer --source-kind audio --attest-reviewed
```

The correction summary counts changed turns, speakers, and tokens as an audit aid. The approval binds the corrected file, its source, and the audio by hash and rejects unknown or unexpected speakers. Combine approved session files into one JSONL only with their IDs intact, and approve the combined file.

### De-identification

A local language model proposes identifiers, and a person decides about each of them.

```bash
cilp privacy-propose --transcript /secure/turns/all_corrected.jsonl --transcript-approval /secure/review/all_transcript_approval.json --model LOCAL_MODEL_ID --endpoint http://127.0.0.1:2929/v1 --mapping /secure/privacy/mapping.csv
```

- The model reads batches of consecutive turns from one transcript, as many as fit `--context-tokens` (default 4096) and at most `--rows-per-batch` (default 20), and returns the identifiers in each turn: persons, e-mail addresses, phone numbers, dates, addresses, and IDs. `--indirect` adds organizations, locations, occupations, and rare events. A batch is smaller when the answer for all its turns would not fit, and an answer cut off at the length limit is asked again in two halves. `--patterns` also adds deterministic matches for e-mail addresses, phone numbers, and URLs.
- Use a large instruction-tuned model. We recommend Llama 3 70B Instruct, served by llama.cpp or another OpenAI-compatible server on the same machine at a loopback `/v1` endpoint. `--model` must name a model the server reports at `/v1/models`, and the mapping's metadata records it.
- The mapping file has one row per distinct text and transcript: `type`, `original`, a typed `token` such as `PERSON_0001`, the number of `occurrences` in the transcript (0 means the model's text does not occur verbatim), and an empty `anonymize` column. It is bound to the transcript approval.
- Review the mapping **in place**. Set `anonymize` to 1 (replace) or 0 (keep) in every row. Proposed rows stay in the file. Edit a token where a description preserves rating-relevant meaning better (for example `a small town`), and add rows for identifiers the search missed, including inflected or shortened forms.

```bash
cilp privacy-apply --transcript /secure/turns/all_corrected.jsonl --mapping /secure/privacy/mapping.csv --output /secure/privacy/all_reviewed.jsonl
cilp privacy-approve --transcript /secure/privacy/all_reviewed.jsonl --approval /secure/privacy/approval.json --reviewer reviewer-id --attest-reviewed --require-distinct-reviewer
```

`privacy-apply` replaces every occurrence of each chosen text in its transcript, longest texts first, also inside longer words, so rows that occur unexpectedly often deserve a second look. Approve only after checking every output turn, including identifying combinations of details that lie further apart than one batch. An empty mapping needs the same full review. `--require-distinct-reviewer` refuses approval by the person who approved the transcript. The approval binds the reviewed file to its upstream approvals, so editing either invalidates the dataset gate.

## A.3 Segmentation, labeling, and dataset construction

Keep independent ratings (`participant_id,target_id,rater_id,label`) separate from the adjudicated reference labels. For exactly two raters:

```bash
cilp label-agreement --task /secure/task/task.toml --ratings /secure/ratings/independent.csv --output /secure/ratings/agreement.json
```

It reports exact agreement, severe disagreement (two or more steps), and quadratic- and linear-weighted kappa per target, with the weights used for model evaluation. To compare model and human agreement fairly, also evaluate the model against each single rater. Freeze the reference labels before anyone sees model outputs.

```bash
cilp split --task /secure/task/task.toml --inventory /secure/cohort/inventory.csv --labels /secure/cohort/labels.csv --output /secure/splits/split-42.json --seed 42 --stratify
cilp build --task /secure/task/task.toml --transcript /secure/privacy/all_reviewed.jsonl --approval /secure/privacy/approval.json --inventory /secure/cohort/inventory.csv --labels /secure/cohort/labels.csv --split /secure/splits/split-42.json --output-dir /secure/datasets/task-v1
```

- Participants are split before examples exist, and the split file hashes inventory and labels. `--stratify` orders participants by mean reference severity and deals them to the splits in proportion. Without it, the split is seeded random.
- `build` writes `train.jsonl`, `validation.jsonl`, and `manifest.json`. The test split is built only after the freeze. Each example records its split, its source turns, and reference provenance. Later commands read the split from the data and the manifest, so held-out data is recognised from the dataset itself.
- The manifest records hashes, denominators (participants, interviews, ratings, examples), score frequencies, exclusions, a [section report](adapting-a-task.md#evidence-selection), and an exact and five-word-shingle duplicate screen across participants and splits. The screen flags candidates for review.

## B.1 Prompting and optional retrieval

The task file and rubric files form the versioned prompt. Choose the input mode (`participant_only` or `dialogue`) and evidence mode before comparing models. If they are themselves research questions, compare them as separate task versions on validation data. Inspect rendered examples. Keep the instruction to answer with one permitted integer: scoring reads the probabilities at the answer position.

## B.2 Optional parameter-efficient adaptation

```bash
cilp plan --task /secure/task/task.toml --experiments /secure/task/experiments.toml --train /secure/datasets/task-v1/train.jsonl --validation /secure/datasets/task-v1/validation.jsonl --model-path /secure/models/local-chat-model --output-root /secure/runs --output /secure/plans/comparison.json
cilp train --plan /secure/plans/comparison.json --job JOB_NAME --gpu 0 --watch-disk-path /secure --safety-log-dir /secure/safety --dry-run
cilp train --plan /secure/plans/comparison.json --job JOB_NAME --gpu 0 --watch-disk-path /secure --safety-log-dir /secure/safety
```

- The experiment file varies adapter strategy, learning rate, epochs, and seed. LoRA settings, maximum length, quantization, and the training objective are fixed per file ([adapting a task](adapting-a-task.md#adapters-experiments-and-training-objective)). The plan records task, rubric, and data hashes and refuses changed inputs. Record how many configurations you compare.
- `plan` loads the model's tokenizer on the CPU and records in the plan how many training and validation examples exceed `max_length`, by target and label. Training skips such examples, and prediction marks them `too_long`, which counts as the worst possible error. Raise `max_length` or narrow the evidence until no validation example is too long. On a host without the model, `--skip-length-check` leaves the count out.
- `train` runs inside the safety wrapper, which checks GPU and CPU temperature, memory, disk, startup load, and power caps and logs to `--safety-log-dir`. Run the smallest job first. Concurrent jobs need disjoint GPU indices and separate log directories. Check the wrapper's UUID mapping.
- Each run's `result.json` records loss per epoch as a diagnostic (selection uses `evaluate`) and counts examples skipped for exceeding `max_length` by target and label.

## B.3 Model comparison and selection

```bash
cilp baseline --train /secure/datasets/task-v1/train.jsonl --data /secure/datasets/task-v1/validation.jsonl --output /secure/eval/baseline.csv
cilp predict --plan /secure/plans/comparison.json --job JOB_NAME --data /secure/datasets/task-v1/validation.jsonl --base-only --output /secure/eval/JOB_NAME_base.csv --gpu 0 --watch-disk-path /secure --safety-log-dir /secure/safety
cilp predict --plan /secure/plans/comparison.json --job JOB_NAME --data /secure/datasets/task-v1/validation.jsonl --output /secure/eval/JOB_NAME.csv --gpu 0 --watch-disk-path /secure --safety-log-dir /secure/safety
cilp evaluate --task /secure/task/task.toml --truth /secure/datasets/task-v1/validation.jsonl --predictions /secure/eval/JOB_NAME.csv --output /secure/eval/JOB_NAME.json
cilp compare --task /secure/task/task.toml --truth /secure/datasets/task-v1/validation.jsonl --left /secure/eval/JOB_NAME_base.csv --right /secure/eval/JOB_NAME.csv --output /secure/eval/JOB_NAME_paired.json
```

**Label-token scoring.** At the answer position, `predict` reads the model's probabilities for the permitted label tokens, renormalises them, and writes their lower median as `prediction` (the median minimises expected absolute error, matching the MAE-based metric and the lower-median baseline). Further columns: `p_<label>`, `mode`, `expected_value`, `status`, and `label_mass`, the probability placed on any permitted token before renormalisation. Prompts longer than `max_length` get `status=too_long` and no prediction. The `.meta.json` records the job, plan and data hashes, split, label token IDs, and whether the adapter was used. `--base-only` scores with the same checkpoint and settings without the adapter: the adapter-off/on contrast.

**Metrics.** `evaluate` reports per target and pooled:

- the primary metric `mean_target_mae_itt`: per-target MAE in which every invalid or missing prediction receives the worst possible error for its reference score, averaged over targets
- `mean_target_mae` on valid outputs only, `null` unless every target has valid predictions
- valid-output rate, exact agreement over all eligible cases, and among valid outputs MAE, quadratic- and linear-weighted kappa, and the severe-error rate (two or more steps, with within-one agreement as its complement)
- participant-clustered bootstrap intervals for the primary metric and each target's MAE and QWK, with counts of resamples in which a statistic was undefined
- secondary calibration of the exported probabilities (Brier score, ranked probability score, top-label ECE), descriptive only

Kappa is `null` when undefined. Pooled kappa is secondary because it mixes between-target differences into agreement. `--fallback-predictions baseline.csv` adds a sensitivity analysis that scores invalid cases with the baseline instead of the worst case. `compare` reports the paired ITT contrast (primary) and the contrast on cases valid in both files, with identical participant resamples. Negative differences favour the right-hand file.

A by-group or per-target job predicts only its targets. Combine them with `cilp merge-predictions --input A.csv --input B.csv --output ALL.csv`, which keeps the probability columns and refuses overlaps, mixed conditions, or files from different plans or data. Select one configuration (adapter strategy, learning rate, epochs, and seed) on the prespecified primary metric over all targets, computed from the merged predictions of its groups, and the valid-output threshold, and write down the rationale.

## Freeze gate and C.1 human review

Fill in the [analysis-plan template](analysis-plan-template.md) before test access, then freeze the selection:

```bash
cilp freeze --task /secure/task/task.toml --plan /secure/plans/comparison.json --split /secure/splits/split-42.json --job JOB_NAME --analysis-plan /secure/protocol/confirmatory_analysis.md --selected-condition adapted --rationale 'Lowest validation mean_target_mae_itt with valid-output rate >= 0.95' --output /secure/eval/freeze.json
cilp build --task /secure/task/task.toml --transcript /secure/privacy/all_reviewed.jsonl --approval /secure/privacy/approval.json --inventory /secure/cohort/inventory.csv --labels /secure/cohort/labels.csv --split /secure/splits/split-42.json --output-dir /secure/datasets/task-v1-frozen --include-test --freeze /secure/eval/freeze.json
cilp predict --plan /secure/plans/comparison.json --job JOB_NAME --data /secure/datasets/task-v1-frozen/test.jsonl --freeze /secure/eval/freeze.json --output /secure/eval/test_predictions.csv --gpu 0 --watch-disk-path /secure --safety-log-dir /secure/safety
cilp evaluate --task /secure/task/task.toml --truth /secure/datasets/task-v1-frozen/test.jsonl --predictions /secure/eval/test_predictions.csv --freeze /secure/eval/freeze.json --output /secure/eval/test_metrics.json
cilp review-sheet --predictions /secure/eval/test_predictions.csv --freeze /secure/eval/freeze.json --shuffle-seed 7 --output /secure/review/blind_sheet.csv
```

- The freeze binds task, plan, split, selected jobs, the analysis plan's hash, and the selected condition: `adapted`, or `base_only` when the prompted model won or no adaptation was done. The selected jobs must come from one configuration and cover every target exactly once, and `--split` must be the split behind the plan's data.
- Held-out data is recognised from the dataset itself. `predict` on test data runs only for frozen jobs, and `evaluate` and `compare` on test data accept only predictions from those jobs on that exact file, plus the constant baseline. The adapter-off comparator may be predicted on test (`--base-only`) because the confirmatory contrast needs both conditions. `evaluate` labels each report `selected` or `comparator`.
- `review-sheet` requires the freeze and accepts only predictions of the selected condition from the frozen jobs. The sheet hides reference scores and probabilities. Reviewers fill `disposition` (`accept`, `revise`, `reject`), `revised_value`, `reviewer_id`, `correction_type`, `review_time_seconds`, and `review_note`.

```bash
cilp review-summary --task /secure/task/task.toml --completed /secure/review/completed.csv --output /secure/review/summary.json
cilp assisted-ratings --task /secure/task/task.toml --completed /secure/review/completed.csv --output /secure/review/assisted_ratings.csv
```

Both apply the same validation. Assisted scores are exported separately from model predictions and reference labels, and rejected proposals have none.

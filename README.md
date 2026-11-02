# Clinical Interview LLM Pipeline

Research software for rating clinical interviews with local language models. It turns locally held interview audio or transcripts into reviewed, participant-separated datasets, compares prompted and LoRA-adapted models on ordinal ratings, and hands the frozen model's outputs to human reviewers. The construct, rating scale, rubrics, and interview sections are defined in a task file, so the code assumes no particular instrument.

The repository contains the code and fictional examples. The [workflow guide](docs/workflow.md) follows the seven stages of the accompanying manuscript.

## Limitations

- Alpha research software without clinical validation, and not a diagnostic product. The synthetic tests and the fictional example show that the commands work, not that ratings are valid.
- The local language model only proposes identifiers. De-identification rests on a complete human review. The model reads batches of consecutive turns, so identifying combinations of details that lie further apart are left to the reviewer.
- Labels must be contiguous integers that are single tokens for the model's tokenizer. Prompts longer than the configured context are not chunked. They count as invalid predictions.
- The `parent` field of a target is metadata only. Element, domain, or total scores are not computed.
- The freeze before test access is enforced by the commands but cannot stop anyone from opening files directly.

## Install and test

Python 3.11 or newer. The core commands use only the standard library. Install the optional dependencies for the steps you run.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests
python -m pip install -e '.[asr]'       # local faster-whisper transcription
python -m pip install -e '.[diarize]'   # local pyannote diarization
python -m pip install -e '.[train]'     # LoRA training and prediction (CUDA)
```

## Hardware

Data preparation, review, dataset construction, and evaluation run on an ordinary CPU. Transcription and diarization run on a GPU, or on the CPU of a host without one. Training and prediction need an NVIDIA GPU with enough memory for the chosen model, and the identifier search needs a local model server, which for large models also means GPUs. Depending on the server, its set-up, and its capacity, GPU jobs run one after another or side by side on separate GPUs. Each GPU job uses one GPU and runs inside the bundled safety wrapper, which checks temperature, memory, disk space, and power limits.

## Fictional example

Six invented participants, three invented targets, a three-point scale. Generated files go to the ignored `data/` directory.

```bash
mkdir -p data/demo
cilp task-check --task examples/synthetic/task.toml
# Attests the review of the fictional turns.
cilp transcript-approve --before examples/synthetic/turns.jsonl --corrected examples/synthetic/turns.jsonl --approval data/demo/transcript_approval.json --reviewer example-reviewer --expected-speaker participant --expected-speaker interviewer --source-kind synthetic --attest-reviewed
# The example uses the pattern pass. Real transcripts use the local LLM search (--model, see the workflow guide).
cilp privacy-propose --transcript examples/synthetic/turns.jsonl --transcript-approval data/demo/transcript_approval.json --patterns --mapping data/demo/mapping.csv
cilp privacy-apply --transcript examples/synthetic/turns.jsonl --mapping data/demo/mapping.csv --output data/demo/approved.jsonl
cilp privacy-approve --transcript data/demo/approved.jsonl --approval data/demo/approval.json --reviewer example-privacy-reviewer --attest-reviewed --require-distinct-reviewer
cilp split --task examples/synthetic/task.toml --inventory examples/synthetic/inventory.csv --labels examples/synthetic/labels.csv --output data/demo/split.json
cilp build --task examples/synthetic/task.toml --transcript data/demo/approved.jsonl --approval data/demo/approval.json --inventory examples/synthetic/inventory.csv --labels examples/synthetic/labels.csv --split data/demo/split.json --output-dir data/demo/dataset
cilp baseline --train data/demo/dataset/train.jsonl --data data/demo/dataset/validation.jsonl --output data/demo/baseline.csv
cilp evaluate --task examples/synthetic/task.toml --truth data/demo/dataset/validation.jsonl --predictions data/demo/baseline.csv --output data/demo/baseline_metrics.json
```

With a local chat model, `cilp plan` expands the example experiment file into 24 jobs (three adapter arrangements × two learning rates × two epoch counts). It also counts, with the model's tokenizer, the examples longer than `max_length`:

```bash
cilp plan --task examples/synthetic/task.toml --experiments examples/synthetic/experiments.toml --train data/demo/dataset/train.jsonl --validation data/demo/dataset/validation.jsonl --model-path /secure/models/local-chat-model --output-root /secure/clinical-runs --output data/demo/plan.json
```

Training, prediction, model selection, the freeze, and human review are described in the [workflow guide](docs/workflow.md).

## Documentation

- [Workflow](docs/workflow.md): every command, stage by stage.
- [Adapting a task](docs/adapting-a-task.md): targets, scales, interview sections, prompts, adapters, and training objectives. [`examples/lpfs-skeleton`](examples/lpfs-skeleton/README.md) is a structure-only starting point for the Level of Personality Functioning Scale.
- [Paper to repository](docs/paper-to-repo.md): which manuscript statements the code supports.
- [Analysis plan template](docs/analysis-plan-template.md): what to fix in writing before the held-out test is opened.

## Layout

- `src/clinical_interview_pipeline/`: file formats, transcription and diarization adapters, review and privacy gates, splitting, training, evaluation, and the `cilp` command.
- `src/clinical_interview_pipeline/resources/safe_gpu_run.sh`: GPU watchdog and resource checks.
- `examples/`: the fictional task and an LPFS task skeleton.
- `tests/`: automated tests that check the commands on fictional data.

## Citation

If you use the pipeline, please cite the accompanying article and the archived release. [`CITATION.cff`](CITATION.cff) holds the citation metadata, which GitHub shows under *Cite this repository*.

## License

The code is available under the [PolyForm Noncommercial License 1.0.0](LICENSE.md). It may be used, changed, and shared for any noncommercial purpose, including use by universities, public research institutions, and public health organizations, however they are funded. Copies must include the license and its required notice. Commercial use needs a separate license from Lukas Königs and André Kerber, Freie Universität Berlin. For inquiries, contact Lukas Königs (lukas.koenigs@fu-berlin.de) or André Kerber (andre.kerber@fu-berlin.de).

Data, base models, and clinical instruments are subject to their own permissions.

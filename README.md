# Clinical Interview LLM Pipeline

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24011034.svg)](https://doi.org/10.5281/zenodo.24011034)

Research software for rating clinical interviews with local language models. It turns locally held interview audio or transcripts into reviewed, participant-separated datasets, compares prompted and LoRA-adapted models on ordinal ratings, and hands the frozen model's outputs to human reviewers. The construct, rating scale, rubrics, and interview sections are defined in a task file, so the code assumes no particular instrument.

The repository contains the code and fictional examples. The [workflow guide](docs/workflow.md) follows the seven stages of the accompanying manuscript.

## Limitations

- Alpha research software without clinical validation, and not a diagnostic product. The synthetic tests and the fictional example show that the commands work, not that ratings are valid.
- The local language model only proposes identifiers. De-identification rests on a complete human review. The model reads batches of consecutive turns, so identifying combinations of details that lie further apart are left to the reviewer.
- Labels must be contiguous integers that are single tokens for the model's tokenizer. Prompts longer than the configured context are not chunked. They count as invalid predictions.
- The `parent` field of a target is metadata only. Element, domain, or total scores are not computed.
- The freeze before test access is enforced by the commands but cannot stop anyone from opening files directly.

## Install

1. Get the code. On GitHub, choose *Code → Download ZIP* and unzip it, or run `git clone https://github.com/koenigs-lab/clinical-interview-llm-pipeline.git`.
2. Open a terminal in the folder. On macOS, open Terminal, type `cd ` followed by a space, drag the folder onto the window, and press Return.
3. Run the installer:

```bash
bash install.sh
```

The installer uses Python 3.11 to 3.13 if one is installed and otherwise downloads Python 3.12 into the folder. It creates the environment `.venv`, installs the pipeline, runs the tests, and ends with `cilp --help`. Add the optional parts you need after `install.sh`, or `all` for every part:

```bash
bash install.sh asr       # local faster-whisper transcription
bash install.sh diarize   # local pyannote diarization, which also needs FFmpeg
bash install.sh train     # LoRA training and prediction on Linux with an NVIDIA GPU
```

In every new terminal, activate the environment from the folder with `source .venv/bin/activate`. On Windows, the same steps run in WSL (Windows Subsystem for Linux).

A manual set-up needs Python 3.11 or newer. The core commands use only the standard library.

```bash
python3.12 -m venv .venv                       # any Python 3.11 or newer
source .venv/bin/activate
python -m pip install -e .                     # or -e '.[asr,diarize,train]'
python -m unittest discover -s tests
```

## Hardware

Data preparation, review, dataset construction, and evaluation run on an ordinary CPU. Transcription and diarization run on a GPU, or on the CPU of a host without one. Training and prediction need an NVIDIA GPU with enough memory for the chosen model, and the identifier search needs a local model server, which for large models also means GPUs. Each GPU job uses one GPU and runs inside the bundled safety wrapper, which checks temperature, memory, disk space, and power limits. On a host with several GPUs, jobs run side by side, one per GPU ([B.2](docs/workflow.md#b2-optional-parameter-efficient-adaptation)).

## Models

All models are read from local storage, and each step expects its own format:

| Step | Option | Model |
| --- | --- | --- |
| `transcribe` | `--local-model` | Folder of a faster-whisper model in CTranslate2 format, e.g. `Systran/faster-whisper-large-v3` |
| `diarize` | `--local-pipeline` | Folder of a pyannote 4 pipeline with its `config.yaml`, e.g. `pyannote/speaker-diarization-community-1` |
| `privacy-propose` | `--model`, `--endpoint` | Name of a chat model on a local OpenAI-compatible server, e.g. a GGUF file in llama.cpp's `llama-server`, or a model in Ollama (`--endpoint http://127.0.0.1:11434/v1`) |
| `plan`, `train`, `predict` | `--model-path` | Folder of an instruction-tuned chat model in Hugging Face Transformers format: `config.json`, `*.safetensors` weights, and tokenizer files with a chat template, e.g. `meta-llama/Meta-Llama-3-8B-Instruct` |

A GGUF file serves only the identifier search, through llama.cpp or Ollama. Training and prediction need the Hugging Face folder. With `quantization = "4bit"` in the experiment file, they quantize its weights to 4 bit while loading them onto the GPU. A snapshot folder in the Hugging Face cache works as well. The `hf` command, installed with any optional part, downloads the folders once:

```bash
hf download Systran/faster-whisper-large-v3 --local-dir /secure/models/asr
hf download pyannote/speaker-diarization-community-1 --local-dir /secure/models/pyannote-local
hf download meta-llama/Meta-Llama-3-8B-Instruct --exclude 'original/*' --local-dir /secure/models/local-chat-model
```

Gated models, such as the pyannote pipeline and Llama, need a Hugging Face account that has accepted their conditions, and `hf auth login` once.

## Fictional example

Six invented participants, three invented targets, a three-point scale. Generated files go to the ignored `data/` directory.

```bash
source .venv/bin/activate   # once per terminal
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

- `install.sh`: set-up of Python, the environment, and the pipeline.
- `src/clinical_interview_pipeline/`: file formats, transcription and diarization adapters, review and privacy gates, splitting, training, evaluation, and the `cilp` command.
- `src/clinical_interview_pipeline/resources/safe_gpu_run.sh`: GPU watchdog and resource checks.
- `examples/`: the fictional task and an LPFS task skeleton.
- `tests/`: automated tests that check the commands on fictional data.

## Citation

If you use the pipeline, please cite the accompanying article and the archived release. [`CITATION.cff`](CITATION.cff) holds the citation metadata, which GitHub shows under *Cite this repository*. The DOI badge leads to the newest release on Zenodo, and each release has its own DOI on its Zenodo page.

## License

The code is available under the [PolyForm Noncommercial License 1.0.0](LICENSE.md). It may be used, changed, and shared for any noncommercial purpose, including use by universities, public research institutions, and public health organizations, however they are funded. Copies must include the license and its required notice. Commercial use needs a separate license from Lukas Königs and André Kerber, Freie Universität Berlin. For inquiries, contact Lukas Königs (lukas.koenigs@fu-berlin.de) or André Kerber (andre.kerber@fu-berlin.de).

Data, base models, and clinical instruments are subject to their own permissions.

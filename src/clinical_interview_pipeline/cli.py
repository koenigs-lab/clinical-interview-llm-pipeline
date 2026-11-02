"""Command-line handoffs for a private, configurable interview workflow."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

from . import __version__
from .datasets import build_examples, constant_baseline, dataset_split, make_split
from .diarization import align_speakers, diarize_audio, import_rttm
from .evaluation import compare_predictions, evaluate
from .experiments import load_job, make_plan
from .gates import (check_held_out_predictions, check_review_predictions, freeze_role,
                    freeze_selection, held_out_gate, verified_freeze)
from .label_agreement import label_agreement
from .privacy import apply_reviews, approve, propose
from .privacy_llm import DEFAULT_CATEGORIES, DEFAULT_ENDPOINT, INDIRECT_CATEGORIES
from .records import read_inventory, sha256_file
from .reviews import assisted_ratings, make_review_sheet, merge_predictions, summarize_review
from .task import load_task
from .training import local_tokenizer, predict_job, train_job
from .transcripts import (approve_transcript, correction_summary, import_tsv,
                          map_speakers, transcribe_audio)


COMMAND_HELP = {
    "task-check": "A.1: validate a task's targets, sections, scale, prompts, and rubrics",
    "inventory-check": "A.1: count eligible pseudonymous participants",
    "import-tsv": "A.2: import a timed, speaker-labelled transcript TSV",
    "transcribe": "A.2: transcribe audio with a local faster-whisper model (--gpu N or --cpu)",
    "diarize": "A.2: run an offline local pyannote pipeline (--gpu N or --cpu)",
    "import-rttm": "A.2: import speaker intervals from an RTTM file",
    "align-speakers": "A.2: conservatively align ASR words to speaker intervals",
    "speaker-map": "A.2: apply a reviewed speaker-to-role mapping",
    "correction-summary": "A.2: compare initial and corrected transcript turns",
    "transcript-approve": "A.2: attest review of a corrected transcript and bind source hashes",
    "privacy-propose": "A.2: local LLM identifier search writing the reviewable mapping file",
    "privacy-apply": "A.2: replace every chosen text with its token, in a new file",
    "privacy-approve": "A.2: attest final privacy and meaning review",
    "label-agreement": "A.3: assess two independent ordinal raters before adjudication",
    "split": "A.3: assign pseudonymous participants to data splits",
    "build": "A.3/B.1: build source-linked examples from approved transcripts",
    "baseline": "B.3: predict the training-set lower median per target",
    "evaluate": "B.3: primary ITT metric, target-wise ordinal scores, calibration, invalid outputs",
    "compare": "B.3: paired ITT and common-valid MAE contrasts of two model outputs",
    "plan": "B.2: expand adapter strategies, rates, epochs, and seeds into jobs",
    "train": "B.2: run one QLoRA job through the GPU safety wrapper",
    "predict": "B.3: label-token scoring with or without the adapter (GPU wrapper)",
    "freeze": "Gate B.3 -> C.1: bind selected jobs, split, and confirmatory analysis plan",
    "merge-predictions": "C.1: combine non-overlapping target predictions",
    "review-sheet": "C.1: make a reference-blind human review sheet of the frozen selection",
    "review-summary": "C.1: summarize accept/revise/reject decisions",
    "assisted-ratings": "C.1: export reviewed scores separately from predictions",
}


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _run_tag(path: Path, suffix: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", Path(path).stem) + "_" + suffix


def _run_gpu_command(args, run_tag: str, internal_args: list[str]) -> str:
    if not re.fullmatch(r"[0-9]+", args.gpu):
        raise ValueError("--gpu must be a single numeric GPU index; each job uses one GPU")
    if args.watch_disk_path is None or args.safety_log_dir is None:
        raise ValueError("GPU runs need --watch-disk-path and --safety-log-dir")
    if not args.watch_disk_path.is_dir():
        raise FileNotFoundError(args.watch_disk_path)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_tag):
        raise ValueError("Unsafe GPU run tag")
    wrapper = files("clinical_interview_pipeline.resources").joinpath("safe_gpu_run.sh")
    if not args.dry_run:
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": args.gpu,
            "WATCH_DISK_PATH": str(args.watch_disk_path),
            "RUN_BASE_DIR": str(args.safety_log_dir),
            "RUN_TAG": run_tag,
            "CILP_SAFE_GPU_WRAPPER": "1",
        })
        subprocess.run(
            ["bash", str(wrapper), "--", sys.executable, "-m",
             "clinical_interview_pipeline.cli", *internal_args],
            env=env, check=True,
        )
    return str(wrapper)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cilp", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name, *required, **optional):
        item = commands.add_parser(name, help=COMMAND_HELP.get(name))
        for field in required:
            item.add_argument("--" + field.replace("_", "-"), required=True, type=_path)
        for field, options in optional.items():
            item.add_argument("--" + field.replace("_", "-"), **options)
        return item

    def gpu_options(item, required: bool):
        item.add_argument("--gpu", required=required,
                          help="One numeric GPU index. The job runs inside the safety wrapper.")
        item.add_argument("--watch-disk-path", required=required, type=_path)
        item.add_argument("--safety-log-dir", required=required, type=_path)
        item.add_argument("--dry-run", action="store_true")

    def device_options(item):
        # Transcription and diarization ask for a device instead of silently using the CPU.
        device = item.add_mutually_exclusive_group(required=True)
        device.add_argument("--gpu",
                            help="One numeric GPU index. The job runs inside the safety wrapper.")
        device.add_argument("--cpu", action="store_true",
                            help="Run on the CPU, e.g. on a host without a GPU")
        item.add_argument("--watch-disk-path", type=_path)
        item.add_argument("--safety-log-dir", type=_path)
        item.add_argument("--dry-run", action="store_true")

    def asr_options(item):
        item.add_argument("--language")
        item.add_argument("--compute-type", default="auto",
                          help="CTranslate2 compute type, e.g. auto, int8, float16")
        item.add_argument("--cpu-threads", type=int, default=0,
                          help="CPU threads for CPU runs (0 = library default)")
        item.add_argument("--beam-size", type=int, default=5)

    c = command("task-check", "task")
    c.add_argument("--allow-placeholders", action="store_true",
                   help="Accept placeholder rubrics, e.g. to inspect a task skeleton")
    command("inventory-check", "inventory")
    c = command("import-tsv", "input", "output", participant={"required": True}, session={"required": True})
    for field, default in (("text-column", "segment"), ("speaker-column", "speaker"),
                           ("start-column", "start"), ("end-column", "end")):
        c.add_argument("--" + field, default=default)
    c = command("transcribe", "audio", "local_model", "output",
                participant={"required": True}, session={"required": True})
    asr_options(c)
    device_options(c)
    c = command("_transcribe-job", "audio", "local_model", "output",
                participant={"required": True}, session={"required": True})
    asr_options(c)
    c = command("diarize", "audio", "local_pipeline", "output")
    device_options(c)
    command("_diarize-job", "audio", "local_pipeline", "output")
    c = command("import-rttm", "input", "output")
    c.add_argument("--recording-id")
    c = command("align-speakers", "transcript", "intervals", "output")
    c.add_argument("--collar", type=float, default=0.25,
                   help="Seconds around a speaker interval that still count as covered (default 0.25)")
    command("speaker-map", "transcript", "mapping", "output")
    command("correction-summary", "before", "after")
    c = command("transcript-approve", "before", "corrected", "approval",
                reviewer={"required": True})
    c.add_argument("--expected-speaker", action="append", required=True)
    c.add_argument("--source-kind", choices=["audio", "transcript", "synthetic"],
                   default="audio")
    c.add_argument("--audio", type=_path)
    c.add_argument("--attest-reviewed", action="store_true")
    c = command("privacy-propose", "transcript", "transcript_approval", "mapping")
    c.add_argument("--model", help="Model name served by the local OpenAI-compatible endpoint")
    c.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    c.add_argument("--indirect", action="store_true",
                   help="Also search organizations, locations, occupations, and rare events")
    c.add_argument("--context-tokens", type=int, default=4096,
                   help="Context window used to size batches of consecutive turns")
    c.add_argument("--rows-per-batch", type=int, default=20)
    c.add_argument("--patterns", action="store_true",
                   help="Also add pattern matches for e-mail addresses, phone numbers, and URLs")
    command("privacy-apply", "transcript", "mapping", "output")
    c = command("privacy-approve", "transcript", "approval", reviewer={"required": True})
    c.add_argument("--attest-reviewed", action="store_true")
    c.add_argument("--require-distinct-reviewer", action="store_true",
                   help="Refuse approval if the transcript reviewer approves privacy too")
    c = command("split", "task", "inventory", "labels", "output")
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--train-fraction", type=float, default=0.7)
    c.add_argument("--validation-fraction", type=float, default=0.15)
    c.add_argument("--stratify", action="store_true",
                   help="Spread participant mean reference severity evenly across splits")
    command("label-agreement", "task", "ratings", "output")
    c = command("build", "task", "transcript", "approval", "inventory", "labels", "split", "output_dir")
    c.add_argument("--include-test", action="store_true")
    c.add_argument("--freeze", type=_path)
    command("baseline", "train", "data", "output")
    c = command("evaluate", "task", "truth", "predictions", "output")
    c.add_argument("--freeze", type=_path, help="Required for held-out test data")
    c.add_argument("--bootstrap-draws", type=int, default=1000)
    c.add_argument("--fallback-predictions", type=_path,
                   help="Sensitivity analysis: score invalid outputs with these predictions")
    c = command("compare", "task", "truth", "left", "right", "output")
    c.add_argument("--freeze", type=_path, help="Required for held-out test data")
    c.add_argument("--bootstrap-draws", type=int, default=1000)
    c = command("plan", "task", "experiments", "train", "validation", "model_path", "output_root", "output")
    c.add_argument("--skip-length-check", action="store_true",
                   help="Skip counting over-length examples, e.g. on a host without the model")
    c = command("train", "plan", job={"required": True})
    gpu_options(c, required=True)
    command("_train-job", "plan", job={"required": True})
    c = command("predict", "plan", "data", "output", job={"required": True})
    c.add_argument("--freeze", type=_path, help="Required for held-out test data")
    gpu_options(c, required=True)
    c.add_argument("--base-only", action="store_true",
                   help="Predict with the same base checkpoint and settings, without the adapter")
    c = command("_predict-job", "plan", "data", "output", job={"required": True})
    c.add_argument("--freeze", type=_path)
    c.add_argument("--base-only", action="store_true")
    c = command("freeze", "task", "plan", "split", "output", rationale={"required": True})
    c.add_argument("--job", action="append", required=True)
    c.add_argument("--analysis-plan", required=True, type=_path)
    c.add_argument("--selected-condition", required=True, choices=["adapted", "base_only"],
                   help="Which condition was selected on validation: the trained adapters, "
                        "or the same base model without them (prompted baseline)")
    c = command("merge-predictions", "output")
    c.add_argument("--input", action="append", required=True, type=_path)
    c = command("review-sheet", "predictions", "output", freeze={"required": True, "type": _path})
    c.add_argument("--shuffle-seed", type=int,
                   help="Randomise row order to reduce order effects in review")
    command("review-summary", "task", "completed", "output")
    command("assisted-ratings", "task", "completed", "output")
    return parser


def main(argv: list[str] | None = None) -> int:
    # Files and folders the commands create are accessible only to their owner.
    os.umask(0o077)
    args = _parser().parse_args(argv)
    c = args.command
    if c == "task-check":
        task = load_task(args.task, allow_placeholders=args.allow_placeholders)
        result = {"task_id": task.id, "targets": len(task.targets), "labels": task.labels,
                  "evidence_mode": task.evidence_mode, "sections": len(task.sections)}
    elif c == "inventory-check":
        result = {"eligible_participants": len(read_inventory(args.inventory))}
    elif c == "import-tsv":
        result = {"turns": import_tsv(args.input, args.output, args.participant, args.session,
                                      args.text_column, args.speaker_column,
                                      args.start_column, args.end_column)}
    elif c in {"transcribe", "_transcribe-job"}:
        options = (args.language, args.compute_type, args.cpu_threads, args.beam_size)
        if c == "_transcribe-job":
            result = {"turns": transcribe_audio(args.audio, args.local_model, args.output,
                                                 args.participant, args.session, options[0],
                                                 "cuda", *options[1:])}
        elif args.cpu:
            if args.dry_run:
                raise ValueError("--dry-run applies to GPU runs, not to --cpu")
            result = {"turns": transcribe_audio(args.audio, args.local_model, args.output,
                                                 args.participant, args.session, options[0],
                                                 "cpu", *options[1:])}
        else:
            if args.output.exists():
                raise FileExistsError(args.output)
            internal = ["_transcribe-job", "--audio", str(args.audio),
                        "--local-model", str(args.local_model), "--output", str(args.output),
                        "--participant", args.participant, "--session", args.session,
                        "--compute-type", args.compute_type,
                        "--cpu-threads", str(args.cpu_threads),
                        "--beam-size", str(args.beam_size)]
            if args.language:
                internal.extend(["--language", args.language])
            wrapper = _run_gpu_command(args, _run_tag(args.output, "asr"), internal)
            result = {"output": str(args.output), "gpu": args.gpu, "wrapper": wrapper,
                      "will_execute": not args.dry_run}
    elif c in {"diarize", "_diarize-job"}:
        if c == "_diarize-job":
            result = {"intervals": diarize_audio(args.audio, args.local_pipeline, args.output,
                                                 "cuda")}
        elif args.cpu:
            if args.dry_run:
                raise ValueError("--dry-run applies to GPU runs, not to --cpu")
            result = {"intervals": diarize_audio(args.audio, args.local_pipeline, args.output)}
        else:
            if args.output.exists():
                raise FileExistsError(args.output)
            internal = ["_diarize-job", "--audio", str(args.audio),
                        "--local-pipeline", str(args.local_pipeline), "--output", str(args.output)]
            wrapper = _run_gpu_command(args, _run_tag(args.output, "diarize"), internal)
            result = {"output": str(args.output), "gpu": args.gpu, "wrapper": wrapper,
                      "will_execute": not args.dry_run}
    elif c == "import-rttm":
        result = {"intervals": import_rttm(args.input, args.output, args.recording_id)}
    elif c == "align-speakers":
        result = align_speakers(args.transcript, args.intervals, args.output, args.collar)
    elif c == "speaker-map":
        result = {"turns": map_speakers(args.transcript, args.mapping, args.output)}
    elif c == "correction-summary":
        result = correction_summary(args.before, args.after)
    elif c == "transcript-approve":
        receipt = approve_transcript(args.before, args.corrected, args.approval,
                                     args.reviewer, args.expected_speaker,
                                     args.attest_reviewed, args.source_kind, args.audio)
        result = {"status": receipt["status"], "corrections": receipt["corrections"]}
    elif c == "privacy-propose":
        if args.indirect and not args.model:
            raise ValueError("--indirect applies to the LLM search; add --model")
        categories = DEFAULT_CATEGORIES + (INDIRECT_CATEGORIES if args.indirect else ())
        result = propose(args.transcript, args.mapping, args.transcript_approval, args.model,
                         args.endpoint, args.patterns, categories, args.context_tokens,
                         args.rows_per_batch)
    elif c == "privacy-apply":
        receipt = apply_reviews(args.transcript, args.mapping, args.output)
        result = {"status": receipt["status"], "replacements": receipt["n_replacements"]}
    elif c == "privacy-approve":
        approve(args.transcript, args.reviewer, args.approval, args.attest_reviewed,
                args.require_distinct_reviewer)
        result = {"status": "approved", "approval": str(args.approval)}
    elif c == "split":
        result = {name: len(ids) for name, ids in make_split(
            args.inventory, args.labels, load_task(args.task), args.output,
            args.seed, args.train_fraction, args.validation_fraction, args.stratify
        ).items()}
    elif c == "label-agreement":
        result = label_agreement(load_task(args.task), args.ratings, args.output)
    elif c == "build":
        if args.include_test:
            if not args.freeze:
                raise ValueError("Building test data requires --freeze")
            verified_freeze(args.freeze, args.task, sha256_file(args.split))
        result = build_examples(load_task(args.task), args.transcript, args.approval,
                                args.inventory, args.labels, args.split, args.output_dir,
                                args.include_test)
        result = {name: value["examples"] for name, value in result["splits"].items()}
    elif c == "baseline":
        result = {"predictions": constant_baseline(args.train, args.data, args.output)}
    elif c == "evaluate":
        check_held_out_predictions(args.freeze, args.task, args.truth,
                                   [p for p in (args.predictions, args.fallback_predictions) if p])
        report = evaluate(load_task(args.task), args.truth, args.predictions,
                          args.output, dataset_split(args.truth)["split"], args.bootstrap_draws,
                          fallback_path=args.fallback_predictions,
                          annotations=freeze_role(args.freeze, args.predictions))
        result = {"selection_metrics": report["selection_metrics"],
                  "participants": report["n_participants"]}
    elif c == "compare":
        check_held_out_predictions(args.freeze, args.task, args.truth, [args.left, args.right])
        report = compare_predictions(load_task(args.task), args.truth, args.left,
                                     args.right, args.output, dataset_split(args.truth)["split"],
                                     args.bootstrap_draws)
        result = {key: report[key] for key in (
            "eligible_cases", "common_valid_cases",
            "mean_target_mae_itt_difference_right_minus_left",
            "mean_target_mae_difference_right_minus_left")}
    elif c == "plan":
        task = load_task(args.task)
        tokenizer = None if args.skip_length_check else local_tokenizer(args.model_path)
        plan = make_plan(task, args.experiments, args.train, args.validation, args.model_path,
                         args.output_root, args.output, tokenizer)
        result = {"jobs": len(plan["jobs"]), "plan": str(args.output)}
        if plan["too_long"] is not None:
            result["too_long"] = {split: plan["too_long"][split]["n"]
                                  for split in ("train", "validation")}
            if result["too_long"]["validation"]:
                print(f"Warning: {result['too_long']['validation']} validation examples exceed "
                      f"max_length ({plan['too_long']['max_length']} tokens). Prediction cannot "
                      "score them, so they count with the worst possible error. Raise max_length "
                      "or narrow the evidence.", file=sys.stderr)
    elif c == "train":
        job = load_job(args.plan, args.job)
        if Path(job["run_dir"]).exists():
            raise FileExistsError(job["run_dir"])
        wrapper = _run_gpu_command(
            args, args.job, ["_train-job", "--plan", str(args.plan), "--job", args.job]
        )
        if args.dry_run:
            result = {"job": args.job, "gpu": args.gpu, "wrapper": wrapper,
                      "run_dir": job["run_dir"], "will_execute": False}
        else:
            result = {"job": args.job, "run_dir": job["run_dir"]}
    elif c == "_train-job":
        report = train_job(load_job(args.plan, args.job))
        result = {"train_examples": report["train_examples"],
                  "validation_examples": report["validation_examples"]}
    elif c in {"predict", "_predict-job"}:
        job = load_job(args.plan, args.job)
        held_out_gate(args.freeze, Path(job["task_path"]), args.data, args.plan, [args.job])
        if c == "predict":
            if args.output.exists():
                raise FileExistsError(args.output)
            internal = ["_predict-job", "--plan", str(args.plan), "--job", args.job,
                        "--data", str(args.data), "--output", str(args.output)]
            if args.freeze:
                internal.extend(["--freeze", str(args.freeze)])
            if args.base_only:
                internal.append("--base-only")
            wrapper = _run_gpu_command(
                args, args.job + ("_predict_base" if args.base_only else "_predict"), internal
            )
            result = {"job": args.job, "output": str(args.output), "gpu": args.gpu,
                      "wrapper": wrapper, "base_only": args.base_only,
                      "will_execute": not args.dry_run}
        else:
            provenance = {"plan_sha256": sha256_file(args.plan),
                          "split": dataset_split(args.data)["split"]}
            result = {"predictions": predict_job(job, args.data, args.output,
                                                 use_adapter=not args.base_only,
                                                 provenance=provenance)}
    elif c == "freeze":
        record = freeze_selection(args.task, args.plan, args.split, args.job, args.analysis_plan,
                                  args.selected_condition, args.rationale, args.output)
        result = {"jobs": len(record["jobs"]), "selected_condition": record["selected_condition"],
                  "freeze": str(args.output)}
    elif c == "merge-predictions":
        result = {"predictions": merge_predictions(args.input, args.output)}
    elif c == "review-sheet":
        condition = check_review_predictions(args.predictions, args.freeze)
        result = {"rows": make_review_sheet(args.predictions, args.output, args.shuffle_seed),
                  "condition": condition}
    elif c == "review-summary":
        result = summarize_review(load_task(args.task), args.completed, args.output)
    elif c == "assisted-ratings":
        result = {"assisted_ratings": assisted_ratings(load_task(args.task),
                                                        args.completed, args.output)}
    else:
        raise AssertionError(c)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

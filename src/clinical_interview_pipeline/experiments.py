"""Versioned comparison plans. Targets and adapter groups are independent."""

from __future__ import annotations

import itertools
import json
import re
import tomllib
from dataclasses import replace
from pathlib import Path
from typing import Any

from .datasets import dataset_split
from .records import read_jsonl, sha256_file, write_json
from .task import ADAPTER_STRATEGIES, Task, adapter_groups, load_task
from .training import OBJECTIVES, too_long_report


def _run_token(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(value))


def make_plan(
    task: Task,
    experiment_path: Path,
    train_path: Path,
    validation_path: Path,
    model_path: Path,
    output_root: Path,
    plan_path: Path,
    tokenizer=None,
) -> dict[str, Any]:
    """Jobs for every adapter strategy, learning rate, epoch count, and seed.

    With the model's tokenizer, the plan also records the examples longer than
    max_length, by target and label.
    """
    with experiment_path.open("rb") as stream:
        config = tomllib.load(stream)["experiments"]
    if not config.get("learning_rates") or not config.get("epochs"):
        raise ValueError("Supply at least one learning rate and epoch count")
    if any(int(value) != value or int(value) < 1 for value in config["epochs"]):
        raise ValueError("Epoch counts must be positive integers")
    if any(float(value) <= 0 for value in config["learning_rates"]):
        raise ValueError("Learning rates must be positive")
    objective = str(config.get("objective", "token_ce"))
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {', '.join(OBJECTIVES)}")
    if int(config.get("max_length", 4096)) < 32:
        raise ValueError("max_length must be at least 32 tokens")
    if int(config.get("batch_size", 1)) < 1 or int(config.get("gradient_accumulation", 16)) < 1:
        raise ValueError("Batch size and gradient accumulation must be positive")
    strategies = config.get("adapter_strategies", [task.adapter_strategy])
    for strategy in strategies:
        if strategy not in ADAPTER_STRATEGIES:
            raise ValueError(f"Adapter strategies must be among {', '.join(ADAPTER_STRATEGIES)}")
    if (dataset_split(train_path)["split"] != "train"
            or dataset_split(validation_path)["split"] != "validation"):
        raise ValueError("Plans use the train and validation files written by cilp build")
    train_rows = read_jsonl(train_path)
    validation_rows = read_jsonl(validation_path)
    available = {row["target_id"] for row in train_rows}
    jobs: list[dict[str, Any]] = []
    for strategy in strategies:
        groups = adapter_groups(replace(task, adapter_strategy=strategy))
        for group_name, target_ids in groups.items():
            if not set(target_ids) <= available:
                raise ValueError(f"Training examples missing for group {group_name}")
            for rate, epochs, seed in itertools.product(
                config["learning_rates"], config["epochs"], config.get("seeds", [42])
            ):
                name = "_".join((
                    _run_token(task.id),
                    _run_token(strategy),
                    _run_token(group_name),
                    f"lr{_run_token(rate)}",
                    f"ep{int(epochs)}",
                    f"seed{int(seed)}",
                    *([objective] if objective != "token_ce" else []),
                ))
                jobs.append({
                    "name": name,
                    "task_path": str(task.path),
                    "train_path": str(train_path.resolve()),
                    "validation_path": str(validation_path.resolve()),
                    "model_path": str(model_path.resolve()),
                    "run_dir": str((output_root / name).resolve()),
                    "adapter_strategy": strategy,
                    "adapter_group": group_name,
                    "target_ids": target_ids,
                    "learning_rate": float(rate),
                    "epochs": int(epochs),
                    "seed": int(seed),
                    "objective": objective,
                    "max_length": int(config.get("max_length", 4096)),
                    "batch_size": int(config.get("batch_size", 1)),
                    "gradient_accumulation": int(config.get("gradient_accumulation", 16)),
                    "lora_r": int(config.get("lora_r", 16)),
                    "lora_alpha": int(config.get("lora_alpha", 32)),
                    "lora_dropout": float(config.get("lora_dropout", 0.05)),
                    "target_modules": list(config.get(
                        "target_modules",
                        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    )),
                    "quantization": str(config.get("quantization", "4bit")),
                })
    names = [job["name"] for job in jobs]
    if not jobs or len(names) != len(set(names)):
        raise ValueError("Experiment plan is empty or run names collide")
    plan = {
        "task_id": task.id,
        "task_sha256": sha256_file(task.path),
        "rubric_sha256": {
            target.id: sha256_file(target.rubric) for target in task.targets
        },
        "experiment_sha256": sha256_file(experiment_path),
        "train_sha256": sha256_file(train_path),
        "validation_sha256": sha256_file(validation_path),
        "jobs": jobs,
        "too_long": (too_long_report(jobs[0], task.labels, tokenizer)
                     if tokenizer is not None else None),
    }
    write_json(plan_path, plan)
    return plan


def load_job(plan_path: Path, job_name: str) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan["train_sha256"] != sha256_file(Path(plan["jobs"][0]["train_path"])):
        raise ValueError("Training data changed since experiment planning")
    if plan["validation_sha256"] != sha256_file(Path(plan["jobs"][0]["validation_path"])):
        raise ValueError("Validation data changed since experiment planning")
    found = [job for job in plan["jobs"] if job["name"] == job_name]
    if len(found) != 1:
        raise ValueError(f"Unknown job: {job_name}")
    job = found[0]
    if plan["task_sha256"] != sha256_file(Path(job["task_path"])):
        raise ValueError("Task configuration changed since experiment planning")
    task = load_task(Path(job["task_path"]))
    if plan["rubric_sha256"] != {
        target.id: sha256_file(target.rubric) for target in task.targets
    }:
        raise ValueError("Rubrics changed since experiment planning")
    return job

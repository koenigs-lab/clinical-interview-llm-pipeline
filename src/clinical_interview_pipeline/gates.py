"""The freeze before the held-out test, and the checks that enforce it."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .datasets import dataset_split
from .experiments import load_job
from .records import sha256_file, write_json
from .reviews import prediction_condition, read_meta
from .task import load_task


def freeze_selection(task_path: Path, plan_path: Path, split_path: Path, job_names: list[str],
                     analysis_plan: Path, selected_condition: str, rationale: str,
                     output_path: Path) -> dict:
    """Record the validation-based selection before the held-out test split is opened.

    The selected jobs come from one configuration and cover every target once.
    'adapted' requires their trained adapters, and 'base_only' selects the same
    base model without them.
    """
    if not rationale.strip():
        raise ValueError("Freeze rationale cannot be blank")
    if not analysis_plan.is_file() or not analysis_plan.read_text(encoding="utf-8").strip():
        raise ValueError("Supply a completed confirmatory analysis plan")
    if len(job_names) != len(set(job_names)):
        raise ValueError("Freeze selects the same job twice")
    jobs = [load_job(plan_path, name) for name in job_names]
    if dataset_split(Path(jobs[0]["train_path"]))["split_sha256"] != sha256_file(split_path):
        raise ValueError("--split is not the participant split behind the plan's data")
    selected_targets = [target for job in jobs for target in job["target_ids"]]
    if len(selected_targets) != len(set(selected_targets)):
        raise ValueError("Frozen jobs contain overlapping targets")
    if set(selected_targets) != set(load_task(task_path).target_map):
        raise ValueError("Frozen jobs must cover every task target")
    # One configuration for all targets, chosen on the macro-averaged primary metric.
    configurations = {(job["adapter_strategy"], job["learning_rate"], job["epochs"], job["seed"])
                      for job in jobs}
    if len(configurations) != 1:
        raise ValueError("Frozen jobs must come from one configuration (adapter strategy, "
                         "learning rate, epochs, and seed)")
    strategy, rate, epochs, seed = configurations.pop()
    if selected_condition == "adapted":
        missing = [job["name"] for job in jobs if not (Path(job["run_dir"]) / "adapter").is_dir()]
        if missing:
            raise ValueError(f"Selected condition 'adapted' but no trained adapter for: {missing}")
    record = {
        "status": "frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_sha256": sha256_file(task_path),
        "plan_sha256": sha256_file(plan_path),
        "split_sha256": sha256_file(split_path),
        "jobs": job_names,
        "selected_configuration": {"adapter_strategy": strategy, "learning_rate": rate,
                                   "epochs": epochs, "seed": seed},
        # Both conditions may be predicted on test, because the adapter contrast needs both.
        # Only the selected condition goes to human review (C.1).
        "selected_condition": selected_condition,
        "comparator_condition": "base_only" if selected_condition == "adapted" else None,
        "selection_rationale": rationale,
        "analysis_plan_path": str(analysis_plan),
        "analysis_plan_sha256": sha256_file(analysis_plan),
        "package_version": __version__,
    }
    write_json(output_path, record)
    return record


def verified_freeze(path: Path, task_path: Path, split_sha256: str | None = None,
                    plan_path: Path | None = None, jobs: list[str] | None = None) -> dict:
    """The freeze record, checked against the task, analysis plan, split, plan, and jobs."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "frozen" or value.get("task_sha256") != sha256_file(task_path):
        raise ValueError("Test freeze does not match this task")
    if not value.get("analysis_plan_path") or not value.get("analysis_plan_sha256"):
        raise ValueError("Test freeze needs a confirmatory analysis plan")
    analysis_path = Path(value["analysis_plan_path"])
    if not analysis_path.is_file() or value["analysis_plan_sha256"] != sha256_file(analysis_path):
        raise ValueError("Confirmatory analysis plan changed after freeze")
    if split_sha256 is not None and value.get("split_sha256") != split_sha256:
        raise ValueError("Participant split differs from the frozen split")
    if plan_path is not None and value.get("plan_sha256") != sha256_file(plan_path):
        raise ValueError("Experiment plan differs from the frozen plan")
    if jobs is not None and not set(jobs) <= set(value.get("jobs", [])):
        raise ValueError("Job was not selected at the freeze")
    return value


def held_out_gate(freeze_path: Path | None, task_path: Path, data_path: Path,
                  plan_path: Path | None = None, jobs: list[str] | None = None) -> dict | None:
    """The verified freeze when data_path holds held-out test participants, else None.

    The split is read from the dataset and its manifest.
    """
    split = dataset_split(data_path)
    if split["split"] != "test":
        return None
    if freeze_path is None:
        raise ValueError("Held-out test data requires --freeze")
    return verified_freeze(freeze_path, task_path, split["split_sha256"], plan_path, jobs)


def check_held_out_predictions(freeze_path: Path | None, task_path: Path, truth_path: Path,
                               prediction_paths: list[Path]) -> None:
    """On test data, accept only predictions of the frozen jobs, or the constant baseline,
    made on this exact file."""
    freeze = held_out_gate(freeze_path, task_path, truth_path)
    if freeze is None:
        return
    for path in prediction_paths:
        meta = read_meta(path)
        if meta is None:
            raise ValueError(f"{path} has no .meta.json; test predictions need provenance")
        if meta.get("data_sha256") != sha256_file(truth_path):
            raise ValueError(f"{path} was not predicted on this test file")
        if prediction_condition(path) == "constant_baseline":
            continue
        if (meta.get("plan_sha256") != freeze.get("plan_sha256")
                or not set(meta.get("jobs", [])) <= set(freeze.get("jobs", []))):
            raise ValueError(f"{path} does not come from the frozen plan and jobs")


def freeze_role(freeze_path: Path | None, prediction_path: Path) -> dict:
    """A prediction file's condition and, given a freeze, whether it is the selected one."""
    condition = prediction_condition(prediction_path)
    role = {"prediction_condition": condition}
    if freeze_path is not None:
        selected = json.loads(freeze_path.read_text(encoding="utf-8")).get("selected_condition")
        role.update({"frozen_selected_condition": selected,
                     "role": "selected" if condition == selected else "comparator"})
    return role


def check_review_predictions(prediction_path: Path, freeze_path: Path) -> str:
    """Only predictions of the frozen selection go to human review. Returns their condition."""
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen" or "selected_condition" not in frozen:
        raise ValueError("Freeze record has no selected condition")
    condition = prediction_condition(prediction_path)
    if condition != frozen["selected_condition"]:
        raise ValueError(
            f"Predictions are '{condition}' but the frozen selection is "
            f"'{frozen['selected_condition']}'; only the selected condition goes to review"
        )
    meta = read_meta(prediction_path) or {}
    if (meta.get("plan_sha256") != frozen.get("plan_sha256")
            or not set(meta.get("jobs", [])) <= set(frozen.get("jobs", []))):
        raise ValueError("Review sheets are made only from the frozen plan's selected jobs")
    return condition

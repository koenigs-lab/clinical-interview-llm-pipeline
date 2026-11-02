"""Blind review sheets and explicit human dispositions for model outputs."""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

from .records import read_csv, sidecar, write_csv, write_json
from .task import Task

REVIEW_COLUMNS = ("participant_id", "target_id", "prediction", "disposition", "revised_value",
                  "reviewer_id", "correction_type", "review_time_seconds", "review_note")


def read_meta(path: Path) -> dict | None:
    """The .meta.json sidecar written next to a prediction file, if any."""
    meta = sidecar(path)
    return json.loads(meta.read_text(encoding="utf-8")) if meta.is_file() else None


def prediction_condition(path: Path) -> str:
    """'adapted', 'base_only', 'constant_baseline', or 'unknown' from the sidecar."""
    meta = read_meta(path)
    if meta is None:
        return "unknown"
    if "condition" in meta:
        return str(meta["condition"])
    if "adapter_used" not in meta:
        return "unknown"
    return "adapted" if meta["adapter_used"] else "base_only"


def merge_predictions(inputs: list[Path], output: Path) -> int:
    """Combine non-overlapping prediction files, keeping probability columns."""
    if not inputs:
        raise ValueError("Supply at least one prediction file")
    merged: dict[tuple[str, str], dict] = {}
    columns: list[str] | None = None
    for path in inputs:
        with Path(path).open(newline="", encoding="utf-8-sig") as stream:
            header = next(csv.reader(stream), [])
        if columns is None:
            columns = header
        elif header != columns:
            raise ValueError("Prediction files to merge must have identical columns")
        for row in read_csv(path, {"participant_id", "target_id", "prediction"}):
            key = (row["participant_id"], row["target_id"])
            if key in merged:
                raise ValueError(f"Duplicate prediction for {key[0]} and {key[1]}")
            merged[key] = row
    conditions = {prediction_condition(path) for path in inputs}
    if len(conditions) > 1:
        raise ValueError(f"Cannot merge predictions from different conditions: {sorted(conditions)}")
    metas = [read_meta(path) for path in inputs]
    shared = {}
    if all(meta is not None for meta in metas):
        for field in ("plan_sha256", "data_sha256", "split"):
            values = {meta.get(field) for meta in metas}
            if len(values) > 1:
                raise ValueError(f"Prediction files to merge differ in {field}")
            shared[field] = values.pop()
    write_csv(output, list(columns or []), (merged[key] for key in sorted(merged)))
    if all(meta is not None for meta in metas):
        write_json(sidecar(output), {
            "merged_from": [str(path) for path in inputs],
            **shared,
            "adapter_used": metas[0].get("adapter_used"),
            "decision_rule": metas[0].get("decision_rule"),
            "jobs": [job for meta in metas for job in meta.get("jobs", [])],
        })
    return len(merged)


def make_review_sheet(prediction_path: Path, output_path: Path,
                      shuffle_seed: int | None = None) -> int:
    """Reference-blind sheet showing only the point prediction (no probabilities).

    The status column, when present, tells the reviewer why a prediction is empty.
    """
    rows = read_csv(
        prediction_path, {"participant_id", "target_id", "prediction"}
    )
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(rows)
    status = ["status"] if rows and "status" in rows[0] else []
    write_csv(
        output_path,
        ["participant_id", "target_id", "prediction", *status, *REVIEW_COLUMNS[3:]],
        ({**row, **{column: "" for column in REVIEW_COLUMNS[3:]}} for row in rows),
    )
    return len(rows)


def _reviewed_rows(task: Task, completed_path: Path) -> list[dict]:
    """Validated rows of a completed review sheet, with one rule set for every consumer."""
    rows = read_csv(completed_path, set(REVIEW_COLUMNS))
    valid = {str(label) for label in task.labels}
    seen: set[tuple[str, str]] = set()
    output = []
    for row in rows:
        key = (row["participant_id"].strip(), row["target_id"].strip())
        if not all(key) or key in seen or key[1] not in task.target_map:
            raise ValueError("Duplicate or invalid participant-target review")
        seen.add(key)
        if not row["reviewer_id"].strip():
            raise ValueError("Every review needs a reviewer id")
        try:
            seconds = float(row["review_time_seconds"])
        except ValueError as error:
            raise ValueError("Every review needs a numeric review time") from error
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Review time cannot be negative")
        prediction = row["prediction"].strip()
        decision = row["disposition"].strip()
        if decision == "accept":
            if prediction not in valid:
                raise ValueError("Only a valid model prediction can be accepted")
            score = prediction
        elif decision == "revise":
            score = row["revised_value"].strip()
            if score not in valid:
                raise ValueError("Revisions need a revised value on the task scale")
            if not row["correction_type"].strip():
                raise ValueError("Revisions need a correction type")
        elif decision == "reject":
            score = ""
            if not row["review_note"].strip():
                raise ValueError("Rejections need a review note")
        else:
            raise ValueError("Disposition must be accept, revise, or reject")
        output.append({"participant_id": key[0], "target_id": key[1],
                       "assisted_label": score, "reviewer_id": row["reviewer_id"].strip(),
                       "disposition": decision,
                       "correction_type": row["correction_type"].strip(),
                       "review_time_seconds": seconds,
                       "review_note": row["review_note"].strip()})
    return output


def summarize_review(task: Task, completed_path: Path, output_path: Path) -> dict:
    rows = _reviewed_rows(task, completed_path)
    report = {
        "n_reviewed": len(rows),
        "dispositions": dict(sorted(Counter(row["disposition"] for row in rows).items())),
        "dispositions_by_target": {
            target.id: dict(sorted(Counter(row["disposition"] for row in rows
                                           if row["target_id"] == target.id).items()))
            for target in task.targets
        },
        "correction_types": dict(sorted(Counter(row["correction_type"] for row in rows
                                                if row["correction_type"]).items())),
    }
    write_json(output_path, report)
    return report


def assisted_ratings(task: Task, completed_path: Path, output_path: Path) -> int:
    """Export reviewed ratings separately from predictions and reference labels."""
    rows = _reviewed_rows(task, completed_path)
    write_csv(output_path, ["participant_id", "target_id", "assisted_label",
                            "reviewer_id", "disposition", "correction_type",
                            "review_time_seconds", "review_note"], rows)
    return len(rows)

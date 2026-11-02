"""Validated file formats shared by the pipeline steps."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def is_safe_id(value: str) -> bool:
    """Pseudonyms and target ids: letters, digits, dots, underscores, and hyphens."""
    return bool(SAFE_ID.fullmatch(value))


def sidecar(path: str | Path, kind: str = "meta") -> Path:
    """The provenance file written next to an output: <file>.meta.json or <file>.receipt.json."""
    return Path(f"{path}.{kind}.json")


def write_json(path: str | Path, value: dict) -> None:
    """Write a new JSON file. Like every pipeline output, it fails if the file already exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if line.strip():
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError(f"{path}:{number} is not an object")
                rows.append(item)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_csv(path: str | Path, required: set[str]) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            missing = required - set(reader.fieldnames or [])
            raise ValueError(f"CSV missing columns: {sorted(missing)}")
        return [dict(row) for row in reader]


def write_csv(path: str | Path, columns: list[str], rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_turns(path: str | Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    keys: set[tuple[str, str, str]] = set()
    for row in rows:
        for field in ("participant_id", "session_id", "turn_id", "speaker", "text"):
            if field not in row or not isinstance(row[field], str):
                raise ValueError(f"Transcript row needs string {field}")
        key = (row["participant_id"], row["session_id"], row["turn_id"])
        if key in keys:
            raise ValueError("Duplicate participant/session/turn key")
        keys.add(key)
        if (not is_safe_id(row["participant_id"])
                or not row["session_id"].strip() or not row["turn_id"].strip()
                or not row["speaker"].strip()):
            raise ValueError("Invalid participant, session, turn, or speaker id")
        if row.get("start") is not None and row.get("end") is not None:
            start, end = float(row["start"]), float(row["end"])
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
                raise ValueError("Invalid turn timestamps")
        elif (row.get("start") is None) != (row.get("end") is None):
            raise ValueError("Turn timestamps need both start and end")
    if not rows:
        raise ValueError("Transcript file is empty")
    return rows


def read_labels(path: str | Path, permitted: set[int]) -> dict[tuple[str, str], int]:
    rows = read_csv(path, {"participant_id", "target_id", "label"})
    labels: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["participant_id"].strip(), row["target_id"].strip())
        if not all(key):
            raise ValueError("Empty participant or target in labels")
        if key in labels:
            raise ValueError("Duplicate participant-target label")
        label = int(row["label"])
        if label not in permitted:
            raise ValueError(f"Label outside declared scale for target {key[1]}")
        labels[key] = label
    if not labels:
        raise ValueError("No labels found")
    return labels


def read_label_provenance(path: str | Path) -> dict[tuple[str, str], dict[str, str]]:
    rows = read_csv(path, {"participant_id", "target_id", "label"})
    if not rows:
        return {}
    fields = {"reference_rater_id", "reference_status"}
    present = fields & set(rows[0])
    if present and present != fields:
        raise ValueError("Reference provenance needs both rater and status columns")
    if not present:
        return {}
    provenance = {}
    for row in rows:
        key = (row["participant_id"].strip(), row["target_id"].strip())
        if not row["reference_rater_id"].strip() or not row["reference_status"].strip():
            raise ValueError("Every reference rating needs rater and status provenance")
        provenance[key] = {
            "reference_rater_id": row["reference_rater_id"].strip(),
            "reference_status": row["reference_status"].strip(),
        }
    return provenance


def read_inventory(path: str | Path) -> set[str]:
    rows = read_csv(path, {"participant_id", "source_path", "eligible"})
    eligible: set[str] = set()
    seen: set[str] = set()
    for row in rows:
        participant_id = row["participant_id"].strip()
        if (not is_safe_id(participant_id)
                or participant_id in seen):
            raise ValueError("Inventory participant ids must be unique and nonempty")
        seen.add(participant_id)
        if row["eligible"].strip().lower() not in {"yes", "no"}:
            raise ValueError("Inventory eligible must be yes or no")
        if row["eligible"].strip().lower() == "yes":
            if not row["source_path"].strip():
                raise ValueError("Eligible inventory row has no source path")
            eligible.add(participant_id)
    return eligible

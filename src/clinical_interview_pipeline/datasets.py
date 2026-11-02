"""Participant-separated examples from approved transcripts and task rubrics."""

from __future__ import annotations

import json
import hashlib
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .privacy import verify_approval
from .records import (
    read_inventory,
    read_jsonl,
    read_label_provenance,
    read_labels,
    read_turns,
    sha256_file,
    sidecar,
    write_csv,
    write_json,
    write_jsonl,
)
from .task import Target, Task


def _duplicate_audit(items: list[tuple[str, str, str, str]]) -> dict[str, Any]:
    """Flag exact and high-overlap cross-participant evidence for review."""
    normalized = [" ".join(text.lower().split()) for _, _, _, text in items]
    shingles = [set(zip(*(words[i:] for i in range(5))))
                if len(words := text.split()) >= 5 else {tuple(words)}
                for text in normalized]
    exact: dict[str, list[int]] = defaultdict(list)
    for index, text in enumerate(normalized):
        exact[hashlib.sha256(text.encode("utf-8")).hexdigest()].append(index)
    exact_pairs = set()
    for members in exact.values():
        for left in members:
            for right in members:
                if left < right and items[left][0] != items[right][0]:
                    exact_pairs.add((left, right))
    inverted: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, tokens in enumerate(shingles):
        for token in tokens:
            inverted[token].append(index)
    candidate_counts: Counter[tuple[int, int]] = Counter()
    for members in inverted.values():
        if len(members) > 12:
            continue  # common phrasing contributes little to near-duplicate detection
        for position, left in enumerate(members):
            for right in members[position + 1:]:
                if items[left][0] != items[right][0] and items[left][1] == items[right][1]:
                    candidate_counts[(left, right)] += 1
    near_pairs = [
        (left, right) for (left, right), shared in candidate_counts.items()
        if shared >= 3 and (left, right) not in exact_pairs
        and len(shingles[left] & shingles[right]) / len(shingles[left] | shingles[right]) >= 0.8
    ]
    cross_split_exact = sum(items[left][2] != items[right][2] for left, right in exact_pairs)
    cross_split_near = sum(items[left][2] != items[right][2] for left, right in near_pairs)
    flagged = sorted(("exact", left, right) for left, right in exact_pairs)
    flagged += sorted(("near", left, right) for left, right in near_pairs)
    return {
        "exact_cross_participant_pairs": len(exact_pairs),
        "near_cross_participant_pairs": len(near_pairs),
        "exact_cross_split_pairs": cross_split_exact,
        "near_cross_split_pairs": cross_split_near,
        "flagged_pairs": [
            {"kind": kind,
             "left": dict(zip(("participant_id", "target_id", "split"), items[left][:3])),
             "right": dict(zip(("participant_id", "target_id", "split"), items[right][:3]))}
            for kind, left, right in flagged[:100]
        ],
        "flagged_pairs_truncated": max(0, len(flagged) - 100),
        "note": "Heuristic five-word-shingle screen only. Investigate flagged pairs and other derived products privately.",
    }


def _stratified_assignment(
    participants: list[str],
    labels: dict[tuple[str, str], int],
    sizes: dict[str, int],
    randomizer: random.Random,
) -> dict[str, list[str]]:
    """Spread overall severity evenly across splits.

    Participants are ordered by mean reference label (ties broken at random by
    the seed) and dealt to splits so that each split's running count tracks
    its target share. Uses reference labels only.
    """
    scores: dict[str, list[int]] = defaultdict(list)
    for (participant, _), value in labels.items():
        scores[participant].append(value)
    keys = {participant: randomizer.random() for participant in participants}
    ordered = sorted(participants,
                     key=lambda p: (sum(scores[p]) / len(scores[p]), keys[p]))
    total = len(ordered)
    assigned = {name: [] for name in sizes}
    for position, participant in enumerate(ordered, start=1):
        open_splits = [name for name in sizes if len(assigned[name]) < sizes[name]]
        name = max(open_splits, key=lambda item: (
            position * sizes[item] / total - len(assigned[item]), -list(sizes).index(item)))
        assigned[name].append(participant)
    return {name: sorted(members) for name, members in assigned.items()}


def make_split(
    inventory_path: Path,
    labels_path: Path,
    task: Task,
    output_path: Path,
    seed: int = 42,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    stratify: bool = False,
) -> dict[str, list[str]]:
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1):
        raise ValueError("Split fractions must be strictly between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions leave no test participants")
    eligible = read_inventory(inventory_path)
    labels = read_labels(labels_path, set(task.labels))
    if set(target_id for _, target_id in labels) - set(task.target_map):
        raise ValueError("Labels contain unknown targets")
    participants = sorted(eligible & {participant for participant, _ in labels})
    if len(participants) < 3:
        raise ValueError("At least three eligible labelled participants are needed")
    n_train = max(1, int(len(participants) * train_fraction))
    n_validation = max(1, int(len(participants) * validation_fraction))
    if n_train + n_validation >= len(participants):
        raise ValueError("Not enough participants for all three splits")
    sizes = {"train": n_train, "validation": n_validation,
             "test": len(participants) - n_train - n_validation}
    randomizer = random.Random(seed)
    if stratify:
        splits = _stratified_assignment(participants, labels, sizes, randomizer)
    else:
        randomizer.shuffle(participants)
        splits = {
            "train": sorted(participants[:n_train]),
            "validation": sorted(participants[n_train:n_train + n_validation]),
            "test": sorted(participants[n_train + n_validation:]),
        }
    payload = {
        "seed": seed,
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "stratification": "participant_mean_label" if stratify else None,
        "eligible_inventory_sha256": sha256_file(inventory_path),
        "labels_sha256": sha256_file(labels_path),
        "splits": splits,
    }
    write_json(output_path, payload)
    return splits


def load_split(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    splits = payload["splits"]
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("Expected train, validation, and test splits")
    all_ids = [value for members in splits.values() for value in members]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Participant overlap between splits")
    if any(not members for members in splits.values()):
        raise ValueError("Every split needs at least one participant")
    return splits


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    open_ended: bool


def _markers(task: Task, rows: list[dict], pattern: str) -> list[int]:
    """Turns matching a section marker, searched only in non-participant turns."""
    compiled = re.compile(pattern)
    return [index for index, row in enumerate(rows)
            if row["speaker"] != task.participant_speaker and compiled.search(row["text"])]


def _section_spans(task: Task, rows: list[dict]) -> dict[str, _Span | None]:
    """Spans of the declared sections found in one session.

    A section ends at its own end marker if it declares one, otherwise at the
    next section start or the end of the session. A found section maps to None
    when that end is ambiguous: its declared end marker is absent, or a
    declared section missing from this session could have been absorbed into
    it. Sections are declared in interview order, and sections not found in the
    session are absent from the result.
    """
    session = rows[0].get("session_id", "?") if rows else "?"
    starts: dict[str, int] = {}
    for section in task.sections:
        found = _markers(task, rows, section.start_pattern)
        if len(found) > 1:
            raise ValueError(
                f"Start marker of section {section.id} matches {len(found)} turns in "
                f"session {session}; markers must be unique"
            )
        if found:
            starts[section.id] = found[0]
    if len(set(starts.values())) != len(starts):
        raise ValueError(f"Two sections start at the same turn in session {session}")
    declared = [section.id for section in task.sections]
    position = {section_id: index for index, section_id in enumerate(declared)}
    by_turn = sorted(starts, key=starts.__getitem__)
    missing = [section_id for section_id in declared if section_id not in starts]
    in_order = by_turn == sorted(by_turn, key=position.__getitem__)
    spans: dict[str, _Span | None] = {}
    for index, section_id in enumerate(by_turn):
        start = starts[section_id]
        following = by_turn[index + 1] if index + 1 < len(by_turn) else None
        limit = starts[following] if following else len(rows)
        section = task.section_map[section_id]
        if section.end_pattern is not None:
            ends = [turn for turn in _markers(task, rows, section.end_pattern)
                    if start < turn <= limit]
            spans[section_id] = _Span(start, ends[0], False) if ends else None
            continue
        upper = position[following] if following else len(declared)
        absorbed = (any(position[section_id] < position[other] < upper for other in missing)
                    if in_order else bool(missing))
        spans[section_id] = None if absorbed else _Span(start, limit, following is None)
    return spans


def _select_evidence(task: Task, target: Target, sessions: list[list[dict]],
                     spans: list[dict[str, _Span | None]]) -> dict[str, Any]:
    """Evidence turns per session for one target, with its section coverage."""
    if task.evidence_mode == "full_interview":
        return {"parts": sessions, "status": None}
    if task.evidence_mode == "tagged":
        return {"parts": [[row for row in rows if target.id in row.get("target_ids", [])]
                          for rows in sessions], "status": None}
    parts: list[list[dict]] = []
    found: set[str] = set()
    ambiguous: set[str] = set()
    open_ended = False
    for rows, session_spans in zip(sessions, spans):
        usable = []
        for section_id in target.sections:
            if section_id not in session_spans:
                continue
            found.add(section_id)
            span = session_spans[section_id]
            if span is None:
                ambiguous.add(section_id)
            else:
                usable.append(span)
                open_ended = open_ended or span.open_ended
        parts.append([row for span in sorted(usable, key=lambda item: item.start)
                      for row in rows[span.start:span.end]])
    if not found:
        status = "missing_section_anchor"
    elif not any(parts):
        status = "ambiguous_section_boundary"
    else:
        status = None
    return {"parts": parts, "status": status,
            "incomplete": bool(ambiguous or set(target.sections) - found),
            "open_ended": open_ended}


def _render_evidence(task: Task, parts: list[list[dict]]) -> str:
    """Evidence text, with sessions headed [Session k] when more than one contributes."""
    blocks = []
    for rows in parts:
        if any(row["speaker"] == "unknown" for row in rows):
            raise ValueError("Unknown speaker: correct or import diarization first")
        if task.input_mode == "participant_only":
            lines = [row["text"].strip() for row in rows
                     if row["speaker"] == task.participant_speaker and row["text"].strip()]
        else:
            lines = [f"{row['speaker']}: {row['text'].strip()}"
                     for row in rows if row["text"].strip()]
        if lines:
            blocks.append(lines)
    if len(blocks) == 1:
        return "\n".join(blocks[0])
    return "\n\n".join(f"[Session {number}]\n" + "\n".join(lines)
                        for number, lines in enumerate(blocks, start=1))


def build_examples(
    task: Task,
    transcript_path: Path,
    approval_path: Path,
    inventory_path: Path,
    labels_path: Path,
    split_path: Path,
    output_dir: Path,
    include_test: bool = False,
) -> dict[str, Any]:
    verify_approval(transcript_path, approval_path)
    if output_dir.exists():
        raise FileExistsError(f"Dataset output exists: {output_dir}")
    turns = read_turns(transcript_path)
    eligible = read_inventory(inventory_path)
    labels = read_labels(labels_path, set(task.labels))
    splits = load_split(split_path)
    label_provenance = read_label_provenance(labels_path)
    split_record = json.loads(split_path.read_text(encoding="utf-8"))
    if (split_record.get("eligible_inventory_sha256") != sha256_file(inventory_path)
            or split_record.get("labels_sha256") != sha256_file(labels_path)):
        raise ValueError("Inventory or labels changed after participant splitting")
    if any(participant not in eligible for members in splits.values() for participant in members):
        raise ValueError("Split contains an ineligible participant")
    if any(row["participant_id"] not in eligible for row in turns):
        raise ValueError("Approved transcript contains a participant outside eligible inventory")
    if set(target_id for _, target_id in labels) - set(task.target_map):
        raise ValueError("Labels contain unknown targets")
    by_participant: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in turns:
        by_participant[row["participant_id"]][row["session_id"]].append(row)
    output_dir.mkdir(parents=True)
    summary: dict[str, Any] = {
        "task_id": task.id,
        "task_version": task.version,
        "task_sha256": sha256_file(task.path),
        "rubric_sha256": {
            target.id: sha256_file(target.rubric) for target in task.targets
        },
        "transcript_sha256": sha256_file(transcript_path),
        "approval_sha256": sha256_file(approval_path),
        "labels_sha256": sha256_file(labels_path),
        "split_path": str(split_path.resolve()),
        "split_sha256": sha256_file(split_path),
        "splits": {},
    }
    chosen_splits = ("train", "validation", "test") if include_test else ("train", "validation")
    audit_items: list[tuple[str, str, str, str]] = []
    for split_name in chosen_splits:
        examples: list[dict[str, Any]] = []
        exclusions: Counter[str] = Counter()
        section_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for participant in splits[split_name]:
            sessions = list(by_participant.get(participant, {}).values())
            if not sessions:
                exclusions["missing_transcript"] += len(task.targets)
                continue
            spans = ([_section_spans(task, rows) for rows in sessions]
                     if task.evidence_mode == "section" else [])
            for target in task.targets:
                key = (participant, target.id)
                if key not in labels:
                    exclusions["missing_label"] += 1
                    continue
                selection = _select_evidence(task, target, sessions, spans)
                if selection["status"]:
                    exclusions[selection["status"]] += 1
                    section_counts[selection["status"]][target.id] += 1
                    continue
                if selection.get("incomplete"):
                    section_counts["incomplete_section_coverage"][target.id] += 1
                if selection.get("open_ended"):
                    section_counts["open_ended_section"][target.id] += 1
                evidence = _render_evidence(task, selection["parts"])
                if not evidence:
                    exclusions["missing_evidence"] += 1
                    continue
                audit_items.append((participant, target.id, split_name, evidence))
                system, user = task.render_prompts(target, task.rubric_text(target.id), evidence)
                examples.append({
                    "id": f"{participant}:{target.id}",
                    "split": split_name,
                    "participant_id": participant,
                    "target_id": target.id,
                    "label": labels[key],
                    "reference_provenance": label_provenance.get(key),
                    "source_turns": [
                        {**{field: row[field] for field in
                            ("session_id", "turn_id", "speaker", "start", "end")
                            if field in row},
                         "in_prompt": (task.input_mode == "dialogue"
                                       or row["speaker"] == task.participant_speaker)}
                        for part in selection["parts"] for row in part
                    ],
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": str(labels[key])},
                    ],
                })
        path = output_dir / f"{split_name}.jsonl"
        write_jsonl(path, examples)
        summary["splits"][split_name] = {
            "participants": len(splits[split_name]),
            "participants_with_examples": len({row["participant_id"] for row in examples}),
            "interviews_with_examples": len({
                (row["participant_id"], source["session_id"])
                for row in examples for source in row["source_turns"]
            }),
            "reference_ratings_used": len(examples),
            "reference_ratings_with_provenance": sum(
                row["reference_provenance"] is not None for row in examples
            ),
            "examples": len(examples),
            "examples_by_target": dict(sorted(Counter(
                row["target_id"] for row in examples
            ).items())),
            "labels": dict(sorted(Counter(
                str(row["label"]) for row in examples
            ).items())),
            "labels_by_target": {
                target.id: dict(sorted(Counter(
                    str(row["label"]) for row in examples
                    if row["target_id"] == target.id
                ).items()))
                for target in task.targets
            },
            "exclusions": dict(sorted(exclusions.items())),
        }
        if task.evidence_mode == "section":
            summary["splits"][split_name]["section_report"] = {
                name: dict(sorted(section_counts[name].items()))
                for name in ("missing_section_anchor", "ambiguous_section_boundary",
                             "incomplete_section_coverage", "open_ended_section")
            }
    summary["duplicate_audit"] = _duplicate_audit(audit_items)
    write_json(output_dir / "manifest.json", summary)
    return summary


def dataset_split(data_path: Path) -> dict[str, str]:
    """Split of a dataset file written by build, verified against its manifest and split file.

    Commands derive the split from the data, so held-out participants stay in
    the test split.
    """
    data_path = Path(data_path)
    rows = read_jsonl(data_path)
    names = {row.get("split") for row in rows}
    if not rows or len(names) != 1 or None in names:
        raise ValueError(f"{data_path} is not a nonempty dataset file written by cilp build")
    name = str(names.pop())
    manifest_path = data_path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Dataset manifest missing next to {data_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("split_path") or not manifest.get("split_sha256"):
        raise ValueError("Dataset manifest does not record its participant split; rebuild it")
    split_path = Path(manifest["split_path"])
    if not split_path.is_file() or sha256_file(split_path) != manifest["split_sha256"]:
        raise ValueError("Participant split changed after dataset construction")
    members = set(load_split(split_path).get(name, []))
    if any(row["participant_id"] not in members for row in rows):
        raise ValueError(f"{data_path} contains participants outside the {name} split")
    return {"split": name, "split_sha256": manifest["split_sha256"]}


def constant_baseline(train_path: Path, data_path: Path, output_path: Path) -> int:
    """Lower median of each target's training labels, written for every row of data_path."""
    if dataset_split(train_path)["split"] != "train":
        raise ValueError("The baseline must be computed from the training split")
    split = dataset_split(data_path)["split"]
    train = read_jsonl(train_path)
    data = read_jsonl(data_path)
    by_target: dict[str, list[int]] = defaultdict(list)
    for row in train:
        by_target[row["target_id"]].append(int(row["label"]))
    median = {
        target: sorted(values)[(len(values) - 1) // 2]
        for target, values in by_target.items()
    }
    missing = {row["target_id"] for row in data} - set(median)
    if missing:
        raise ValueError(f"No training labels for targets: {sorted(missing)}")
    write_csv(
        output_path,
        ["participant_id", "target_id", "prediction"],
        (
            {
                "participant_id": row["participant_id"],
                "target_id": row["target_id"],
                "prediction": median[row["target_id"]],
            }
            for row in data
        ),
    )
    write_json(sidecar(output_path), {"condition": "constant_baseline",
                                      "decision_rule": "lower_median_of_training_labels",
                                      "train_sha256": sha256_file(train_path),
                                      "data_sha256": sha256_file(data_path),
                                      "split": split})
    return len(data)

"""Agreement of two independent ordinal raters before adjudication."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .evaluation import linear_weighted_kappa, quadratic_weighted_kappa
from .records import read_csv, sha256_file, write_json
from .task import Task


def label_agreement(task: Task, ratings_path: Path, output_path: Path) -> dict:
    rows = read_csv(ratings_path, {"participant_id", "target_id", "rater_id", "label"})
    raters = sorted({row["rater_id"].strip() for row in rows})
    if len(raters) != 2 or not all(raters):
        raise ValueError("This command needs exactly two independent raters; use an ordinal multi-rater method otherwise")
    ratings: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for row in rows:
        participant, target, rater = (row[field].strip()
                                      for field in ("participant_id", "target_id", "rater_id"))
        if not participant or target not in task.target_map or rater not in raters:
            raise ValueError("Invalid participant, target, or rater in independent ratings")
        score = int(row["label"])
        if score not in task.labels:
            raise ValueError("Independent rating outside task scale")
        key = (participant, target)
        if rater in ratings[key]:
            raise ValueError("Duplicate independent rating")
        ratings[key][rater] = score
    by_target = {}
    for target in task.targets:
        pairs = [
            (value[raters[0]], value[raters[1]])
            for (participant, target_id), value in ratings.items()
            if target_id == target.id and len(value) == 2
        ]
        n = len(pairs)
        by_target[target.id] = {
            "paired_participants": n,
            "partially_rated_participants": sum(
                target_id == target.id and len(value) == 1
                for (_, target_id), value in ratings.items()
            ),
            "exact_agreement": sum(first == second for first, second in pairs) / n if n else None,
            # Same definition as the model's severe_error_rate: at least two steps apart.
            "severe_disagreement_rate": sum(abs(first - second) >= 2
                                            for first, second in pairs) / n if n else None,
            # Same weights as model evaluation, so human-human and model-reference
            # agreement are reported on one scale. None means kappa is undefined.
            "quadratic_weighted_kappa": quadratic_weighted_kappa(pairs, task.labels),
            "linear_weighted_kappa": linear_weighted_kappa(pairs, task.labels),
        }
    report = {"task_id": task.id, "ratings_sha256": sha256_file(ratings_path),
              "rater_ids": raters, "by_target": by_target,
              "note": ("Computed before adjudication. Keep final reference labels separate. "
                       "Compare with model agreement against single raters, not only "
                       "against adjudicated labels, which are more reliable than any one rater.")}
    write_json(output_path, report)
    return report

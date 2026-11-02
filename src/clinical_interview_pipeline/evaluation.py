"""Ordinal evaluation, with invalid outputs and participants kept explicit.

Primary selection metric (manuscript Stage B.3): the macro-average over
targets of an intention-to-predict MAE in which every invalid or missing
prediction receives the worst possible error for its reference score. Metrics
conditional on valid outputs are reported as secondary. Kappa is ``None`` when
undefined (0/0), which happens only when every reference score and every
prediction fall in one and the same category.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .records import read_csv, read_jsonl, sha256_file, write_json
from .task import Task

Row = tuple[str, str, int, "int | None"]
PRIMARY_METRIC = "mean_target_mae_itt"


def weighted_kappa(pairs: list[tuple[int, int]], labels: tuple[int, ...],
                   power: int = 2) -> float | None:
    """Cohen's weighted kappa with |i-j|**power disagreement weights.

    power=2 gives quadratic, power=1 linear weights. Returns None when the
    expected disagreement is zero, i.e. when kappa is 0/0 and undefined.
    """
    if not pairs:
        return None
    index = {label: offset for offset, label in enumerate(labels)}
    n = len(labels)
    observed = [[0] * n for _ in range(n)]
    true_counts = [0] * n
    pred_counts = [0] * n
    for true, predicted in pairs:
        i, j = index[true], index[predicted]
        observed[i][j] += 1
        true_counts[i] += 1
        pred_counts[j] += 1
    denominator = max(1, (n - 1) ** power)
    weight = [[abs(i - j) ** power / denominator for j in range(n)] for i in range(n)]
    observed_error = sum(weight[i][j] * observed[i][j] for i in range(n) for j in range(n))
    expected_error = sum(
        weight[i][j] * true_counts[i] * pred_counts[j] / len(pairs)
        for i in range(n) for j in range(n)
    )
    if expected_error == 0:
        return None
    return 1 - observed_error / expected_error


def quadratic_weighted_kappa(pairs: list[tuple[int, int]], labels: tuple[int, ...]) -> float | None:
    return weighted_kappa(pairs, labels, power=2)


def linear_weighted_kappa(pairs: list[tuple[int, int]], labels: tuple[int, ...]) -> float | None:
    return weighted_kappa(pairs, labels, power=1)


def worst_case_error(true: int, labels: tuple[int, ...]) -> int:
    return max(true - labels[0], labels[-1] - true)


def ranked_probability_score(probabilities: list[float], target_index: int) -> float:
    """Squared distance between predicted and observed cumulative distributions.

    Normalised by the number of category boundaries, so 0 is a perfect forecast
    and 1 puts all probability on the category furthest from the reference.
    The label_rps training objective computes the same quantity.
    """
    boundaries = len(probabilities) - 1
    cumulative = 0.0
    total = 0.0
    for index in range(boundaries):
        cumulative += probabilities[index]
        total += (cumulative - (1.0 if target_index <= index else 0.0)) ** 2
    return total / boundaries


def _calibration(cases: list[tuple[int, list[float]]], labels: tuple[int, ...],
                 bins: int = 10) -> dict[str, Any]:
    """Brier score, ranked probability score, and top-label ECE on scored cases."""
    if not cases:
        return {"n": 0, "brier": None, "rps": None, "ece_top_label": None}
    index = {label: offset for offset, label in enumerate(labels)}
    brier = rps = 0.0
    totals = [[0, 0.0, 0.0] for _ in range(bins)]   # cases, confidence, correct
    for true, probabilities in cases:
        target = index[true]
        brier += sum((p - (1.0 if k == target else 0.0)) ** 2
                     for k, p in enumerate(probabilities))
        rps += ranked_probability_score(probabilities, target)
        confidence = max(probabilities)
        mode = probabilities.index(confidence)
        cell = totals[min(int(confidence * bins), bins - 1)]
        cell[0] += 1
        cell[1] += confidence
        cell[2] += mode == target
    n = len(cases)
    ece = sum(abs(correct - confidence) for count, confidence, correct in totals if count) / n
    return {"n": n, "brier": brier / n, "rps": rps / n, "ece_top_label": ece}


def _metrics(rows: list[Row], labels: tuple[int, ...],
             fallback: dict[tuple[str, str], int] | None = None) -> dict[str, Any]:
    usable = [(true, pred) for _, _, true, pred in rows if pred is not None]
    total = len(rows)
    output: dict[str, Any] = {
        "eligible": total,
        "valid": len(usable),
        "invalid": total - len(usable),
        "valid_output_rate": len(usable) / total if total else None,
        "exact_all_eligible": sum(true == pred for true, pred in usable) / total if total else None,
        "mae_itt_worst_case": (sum(
            abs(true - pred) if pred is not None else worst_case_error(true, labels)
            for _, _, true, pred in rows
        ) / total) if total else None,
        "score_distribution": dict(sorted(Counter(str(true) for _, _, true, _ in rows).items())),
    }
    if fallback is not None:
        errors = []
        for participant, target, true, pred in rows:
            value = pred if pred is not None else fallback.get((participant, target))
            errors.append(abs(true - value) if value is not None
                          else worst_case_error(true, labels))
        output["mae_itt_fallback"] = sum(errors) / total if total else None
    if not usable:
        output.update({"accuracy": None, "mae": None, "qwk": None, "linear_kappa": None,
                       "severe_error_rate": None})
        return output
    output.update({
        "accuracy": sum(true == pred for true, pred in usable) / len(usable),
        "mae": sum(abs(true - pred) for true, pred in usable) / len(usable),
        "qwk": quadratic_weighted_kappa(usable, labels),
        "linear_kappa": linear_weighted_kappa(usable, labels),
        # Within-one agreement is 1 - severe_error_rate (severe = at least two steps).
        "severe_error_rate": sum(abs(true - pred) >= 2 for true, pred in usable) / len(usable),
        "predicted_distribution": dict(sorted(Counter(str(pred) for _, pred in usable).items())),
    })
    return output


def _macro(values: list[float | None]) -> float | None:
    return sum(values) / len(values) if values and all(v is not None for v in values) else None


def _summary(rows: list[Row], task: Task,
             fallback: dict[tuple[str, str], int] | None = None) -> dict[str, float | None]:
    """Scalar statistics used for point estimates and bootstrap intervals."""
    by_target = {target.id: _metrics([row for row in rows if row[1] == target.id],
                                     task.labels, fallback)
                 for target in task.targets}
    pooled = _metrics(rows, task.labels)
    values: dict[str, float | None] = {
        PRIMARY_METRIC: _macro([entry["mae_itt_worst_case"] for entry in by_target.values()]),
        "mean_target_mae": _macro([entry["mae"] for entry in by_target.values()]),
        "exact_all_eligible": pooled["exact_all_eligible"],
        "valid_output_rate": pooled["valid_output_rate"],
    }
    if fallback is not None:
        values["mean_target_mae_itt_fallback"] = _macro(
            [entry["mae_itt_fallback"] for entry in by_target.values()])
    for target_id, entry in by_target.items():
        for key in ("mae", "qwk", "mae_itt_worst_case"):
            values[f"target:{target_id}:{key}"] = entry[key]
    return values


def _percentile(values: list[float], probability: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = int(position)
    fraction = position - lower
    if lower + 1 == len(values):
        return values[lower]
    return values[lower] * (1 - fraction) + values[lower + 1] * fraction


def _resamples(rows: list[tuple], draws: int, seed: int):
    """Participant-level cluster resamples, identical for identical truth and seed."""
    by_participant: dict[str, list[tuple]] = defaultdict(list)
    for row in rows:
        by_participant[row[0]].append(row)
    participants = sorted(by_participant)
    if len(participants) < 2 or draws <= 0:
        return
    randomizer = random.Random(seed)
    for _ in range(draws):
        yield [row for _ in participants
               for row in by_participant[randomizer.choice(participants)]]


def _intervals(samples: dict[str, list[float]], undefined: Counter,
               draws: int) -> dict[str, Any]:
    return {
        key: {"interval": [_percentile(values, 0.025), _percentile(values, 0.975)]
              if values else None,
              "undefined_resamples": undefined[key]}
        for key, values in samples.items()
    } | {key: {"interval": None, "undefined_resamples": draws}
         for key in undefined if key not in samples}


def _read_predictions(path: Path, task: Task,
                      permitted_keys: set[tuple[str, str]]) -> dict[tuple[str, str], int | None]:
    output: dict[tuple[str, str], int | None] = {}
    valid = {str(label) for label in task.labels}
    for row in read_csv(path, {"participant_id", "target_id", "prediction"}):
        key = (row["participant_id"], row["target_id"])
        if key in output:
            raise ValueError("Duplicate participant-target prediction")
        if key not in permitted_keys:
            raise ValueError("Predictions contain participants or targets outside the evaluation split")
        text = row["prediction"].strip()
        output[key] = int(text) if text in valid else None
    return output


def _read_probabilities(path: Path, task: Task,
                        permitted_keys: set[tuple[str, str]]) -> dict[tuple[str, str], list[float]]:
    """Exported label probabilities of scored rows, or nothing when the file has none."""
    columns = [f"p_{label}" for label in task.labels]
    rows = read_csv(path, {"participant_id", "target_id", "prediction"})
    if not rows or not set(columns) <= set(rows[0]):
        return {}
    output: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        key = (row["participant_id"], row["target_id"])
        if key not in permitted_keys or row.get("status", "scored") != "scored":
            continue
        try:
            values = [float(row[column]) for column in columns]
        except ValueError:
            continue
        total = sum(values)
        if total > 0 and all(math.isfinite(value) and value >= 0 for value in values):
            output[key] = [value / total for value in values]
    return output


def _read_truth(path: Path, task: Task) -> list[dict]:
    truth = read_jsonl(path)
    if not truth:
        raise ValueError("Evaluation truth is empty")
    keys = [(row["participant_id"], row["target_id"]) for row in truth]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate participant-target truth rows")
    if any(target_id not in task.target_map for _, target_id in keys):
        raise ValueError("Truth contains a target outside the task")
    if any(int(row["label"]) not in task.labels for row in truth):
        raise ValueError("Truth contains a label outside the task scale")
    return truth


def evaluate(
    task: Task,
    truth_path: Path,
    prediction_path: Path,
    output_path: Path,
    split_name: str,
    bootstrap_draws: int = 1000,
    seed: int = 42,
    fallback_path: Path | None = None,
    annotations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    truth = _read_truth(truth_path, task)
    keys = {(row["participant_id"], row["target_id"]) for row in truth}
    predicted = _read_predictions(prediction_path, task, keys)
    fallback = None
    if fallback_path is not None:
        fallback = {key: value for key, value in
                    _read_predictions(fallback_path, task, keys).items() if value is not None}
    rows: list[Row] = [
        (row["participant_id"], row["target_id"], int(row["label"]),
         predicted.get((row["participant_id"], row["target_id"])))
        for row in truth
    ]
    by_target = {
        target.id: _metrics([row for row in rows if row[1] == target.id], task.labels, fallback)
        for target in task.targets
    }
    overall = _metrics(rows, task.labels, fallback)
    # Pooling targets into one confusion matrix mixes between-target differences
    # into agreement, so the facet-specific kappas in by_target are the reported ones.
    overall["qwk_pooled_secondary"] = overall.pop("qwk")
    overall["linear_kappa_pooled_secondary"] = overall.pop("linear_kappa")
    probabilities = _read_probabilities(prediction_path, task, keys)
    calibration = None
    if probabilities:
        cases = {target.id: [(row[2], probabilities[(row[0], row[1])]) for row in rows
                             if row[1] == target.id and (row[0], row[1]) in probabilities]
                 for target in task.targets}
        calibration = {
            "by_target": {target_id: _calibration(items, task.labels)
                          for target_id, items in cases.items()},
            "pooled": _calibration([item for items in cases.values() for item in items],
                                   task.labels),
            "ece_bins": 10,
            "note": ("Scored cases only. Brier sums over categories, rps is normalised by the "
                     "number of category boundaries, and ece_top_label bins the mode's probability."),
        }
    point = _summary(rows, task, fallback)
    samples: dict[str, list[float]] = defaultdict(list)
    undefined: Counter = Counter()
    for sample in _resamples(rows, bootstrap_draws, seed):
        for key, value in _summary(sample, task, fallback).items():
            if value is None:
                undefined[key] += 1
            else:
                samples[key].append(value)
    evaluable = sum(entry["mae"] is not None for entry in by_target.values())
    report = {
        "task_id": task.id,
        "split": split_name,
        "n_participants": len({row[0] for row in rows}),
        "truth_sha256": sha256_file(truth_path),
        "prediction_sha256": sha256_file(prediction_path),
        "fallback_prediction_sha256": sha256_file(fallback_path) if fallback_path else None,
        "overall": overall,
        "by_target": by_target,
        "selection_metrics": {
            "primary_metric": PRIMARY_METRIC,
            PRIMARY_METRIC: point[PRIMARY_METRIC],
            "mean_target_mae": point["mean_target_mae"],
            "mean_target_mae_itt_fallback": point.get("mean_target_mae_itt_fallback"),
            "targets_with_valid_predictions": evaluable,
            "all_targets_evaluable": evaluable == len(task.targets),
            "valid_output_rate": overall["valid_output_rate"],
            "note": ("mean_target_mae (valid outputs only) is None unless every target has "
                     "valid predictions. Select on the primary metric."),
        },
        "calibration_secondary": calibration,
        "participant_bootstrap_95_percent_intervals": _intervals(
            samples, undefined, bootstrap_draws),
        "bootstrap_draws": bootstrap_draws,
        "seed": seed,
        **(annotations or {}),
    }
    write_json(output_path, report)
    return report


def compare_predictions(
    task: Task, truth_path: Path, left_path: Path, right_path: Path,
    output_path: Path, split_name: str, bootstrap_draws: int = 1000, seed: int = 42,
) -> dict[str, Any]:
    """Paired MAE contrasts, where negative differences favour the right-hand model.

    Primary: macro-averaged intention-to-predict (worst-case) MAE over all
    eligible cases. Secondary: MAE on cases valid in both outputs.
    """
    truth_rows = _read_truth(truth_path, task)
    truth = {(row["participant_id"], row["target_id"]): int(row["label"]) for row in truth_rows}
    left = _read_predictions(left_path, task, set(truth))
    right = _read_predictions(right_path, task, set(truth))
    paired = [(key[0], key[1], score, left.get(key), right.get(key))
              for key, score in truth.items()]

    def error(score: int, prediction: int | None) -> int:
        return abs(score - prediction) if prediction is not None else worst_case_error(score, task.labels)

    def itt_macro(rows: list[tuple]) -> float | None:
        values = []
        for target in task.targets:
            selected = [row for row in rows if row[1] == target.id]
            values.append(sum(error(s, r) - error(s, l) for _, _, s, l, r in selected)
                          / len(selected) if selected else None)
        return _macro(values)

    def common(rows: list[tuple]) -> list[tuple]:
        return [row for row in rows if row[3] is not None and row[4] is not None]

    def valid_contrast(rows: list[tuple]) -> float | None:
        rows = common(rows)
        return (sum(abs(s - r) - abs(s - l) for _, _, s, l, r in rows) / len(rows)) if rows else None

    def valid_macro(rows: list[tuple]) -> float | None:
        return _macro([valid_contrast([row for row in rows if row[1] == target.id])
                       for target in task.targets])

    statistics = {
        "mean_target_mae_itt_difference_right_minus_left": itt_macro,
        "mean_target_mae_difference_right_minus_left": valid_macro,
        "mae_difference_right_minus_left": valid_contrast,
    }
    samples: dict[str, list[float]] = defaultdict(list)
    undefined: Counter = Counter()
    for sample in _resamples(paired, bootstrap_draws, seed):
        for key, function in statistics.items():
            value = function(sample)
            if value is None:
                undefined[key] += 1
            else:
                samples[key].append(value)
    common_rows = common(paired)
    report = {
        "task_id": task.id, "split": split_name,
        "truth_sha256": sha256_file(truth_path),
        "left_prediction_sha256": sha256_file(left_path),
        "right_prediction_sha256": sha256_file(right_path),
        "primary_contrast": "mean_target_mae_itt_difference_right_minus_left",
        "eligible_cases": len(truth),
        "left_valid_cases": sum(value is not None for value in left.values()),
        "right_valid_cases": sum(value is not None for value in right.values()),
        "common_valid_cases": len(common_rows),
        "common_valid_participants": len({row[0] for row in common_rows}),
        **{key: function(paired) for key, function in statistics.items()},
        "by_target": {
            target.id: {
                "common_valid_cases": len(common([r for r in paired if r[1] == target.id])),
                "mae_difference_right_minus_left":
                    valid_contrast([r for r in paired if r[1] == target.id]),
            }
            for target in task.targets
        },
        "participant_bootstrap_95_percent_intervals": _intervals(
            samples, undefined, bootstrap_draws),
        "bootstrap_draws": bootstrap_draws,
        "seed": seed,
        "note": "Negative differences favour the right-hand model.",
    }
    write_json(output_path, report)
    return report

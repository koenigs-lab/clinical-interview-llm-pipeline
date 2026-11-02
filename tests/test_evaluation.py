"""Ordinal metrics, invalid outputs, calibration, and human-human agreement."""

from __future__ import annotations

import json
import unittest

from clinical_interview_pipeline.evaluation import (compare_predictions, evaluate,
                                                    linear_weighted_kappa,
                                                    quadratic_weighted_kappa,
                                                    ranked_probability_score)
from clinical_interview_pipeline.label_agreement import label_agreement

from support import TARGETS, WorkspaceTest


class EvaluationTest(WorkspaceTest):
    def _truth_and_models(self):
        truth = [{"participant_id": p, "target_id": t, "label": label}
                 for p in "abcd" for t, label in zip(TARGETS, (1, 2, 0))]
        truth_path = self.root / "truth.jsonl"
        truth_path.write_text("\n".join(json.dumps(r) for r in truth), encoding="utf-8")

        def write(name, rule):
            path = self.root / name
            path.write_text("participant_id,target_id,prediction\n" + "\n".join(
                f"{r['participant_id']},{r['target_id']},{rule(r)}" for r in truth),
                encoding="utf-8")
            return path
        answers_all = write("A.csv", lambda r: r["label"] if r["target_id"] != "social.support"
                            else 1)
        gives_up = write("B.csv", lambda r: r["label"] if r["target_id"] != "social.support"
                         else "")
        fallback = write("fallback.csv", lambda r: 0)
        return truth_path, answers_all, gives_up, fallback

    def test_kappa_is_undefined_only_for_one_shared_category(self):
        labels = (0, 1, 2, 3, 4)
        self.assertIsNone(quadratic_weighted_kappa([(2, 2)] * 5, labels))
        self.assertEqual(quadratic_weighted_kappa([(1, 3)] * 5, labels), 0.0)
        self.assertEqual(quadratic_weighted_kappa([(0, 2), (2, 2), (4, 2)], labels), 0.0)
        self.assertEqual(quadratic_weighted_kappa([(0, 0), (4, 4)], labels), 1.0)
        pairs = [(0, 0), (1, 2), (2, 1), (4, 3), (3, 3)]
        self.assertNotEqual(quadratic_weighted_kappa(pairs, labels),
                            linear_weighted_kappa(pairs, labels))

    def test_primary_metric_penalises_giving_up_on_a_target(self):
        truth, answers_all, gives_up, _ = self._truth_and_models()
        a = evaluate(self.task, truth, answers_all, self.root / "a.json", "validation", 50)
        b = evaluate(self.task, truth, gives_up, self.root / "b.json", "validation", 50)
        self.assertAlmostEqual(a["selection_metrics"]["mean_target_mae_itt"], 1 / 3)
        self.assertAlmostEqual(b["selection_metrics"]["mean_target_mae_itt"], 2 / 3)
        self.assertIsNone(b["selection_metrics"]["mean_target_mae"])
        self.assertFalse(b["selection_metrics"]["all_targets_evaluable"])
        intervals = a["participant_bootstrap_95_percent_intervals"]
        self.assertIn("mean_target_mae_itt", intervals)
        self.assertIn("target:routine.sleep:qwk", intervals)
        self.assertIn("qwk_pooled_secondary", a["overall"])
        sleep = a["by_target"]["routine.sleep"]
        self.assertIn("severe_error_rate", sleep)
        self.assertNotIn("adjacent_or_exact", sleep)
        self.assertIsNone(a["calibration_secondary"])

    def test_fallback_sensitivity_and_paired_itt_contrast(self):
        truth, answers_all, gives_up, fallback = self._truth_and_models()
        b = evaluate(self.task, truth, gives_up, self.root / "b.json", "validation", 20,
                     fallback_path=fallback)
        self.assertEqual(b["selection_metrics"]["mean_target_mae_itt_fallback"], 0.0)
        paired = compare_predictions(self.task, truth, answers_all, gives_up,
                                     self.root / "paired.json", "validation", 50)
        self.assertAlmostEqual(paired["mean_target_mae_itt_difference_right_minus_left"], 1 / 3)
        self.assertEqual(paired["common_valid_cases"], 8)
        self.assertIsNone(paired["mean_target_mae_difference_right_minus_left"])

    def test_ranked_probability_score(self):
        self.assertEqual(ranked_probability_score([0.0, 0.0, 1.0], 2), 0.0)
        self.assertEqual(ranked_probability_score([1.0, 0.0, 0.0], 2), 1.0)
        self.assertAlmostEqual(ranked_probability_score([0.5, 0.5, 0.0], 2), 0.625)
        # Probability one step away costs less than two steps away.
        self.assertLess(ranked_probability_score([0.0, 1.0, 0.0], 2),
                        ranked_probability_score([1.0, 0.0, 0.0], 2))

    def test_calibration_from_exported_label_probabilities(self):
        truth = self.root / "truth.jsonl"
        truth.write_text("\n".join(json.dumps(row) for row in (
            {"participant_id": "a", "target_id": "routine.sleep", "label": 0},
            {"participant_id": "b", "target_id": "routine.sleep", "label": 2},
            {"participant_id": "c", "target_id": "routine.sleep", "label": 1})),
            encoding="utf-8")
        predictions = self.root / "scored.csv"
        predictions.write_text(
            "participant_id,target_id,prediction,status,p_0,p_1,p_2\n"
            "a,routine.sleep,0,scored,1.0,0.0,0.0\n"
            "b,routine.sleep,0,scored,0.5,0.5,0.0\n"
            "c,routine.sleep,,too_long,,,\n", encoding="utf-8")
        report = evaluate(self.task, truth, predictions, self.root / "m.json", "validation", 10)
        sleep = report["calibration_secondary"]["by_target"]["routine.sleep"]
        self.assertEqual(sleep["n"], 2)
        self.assertAlmostEqual(sleep["brier"], 0.75)       # (0 + 1.5) / 2
        self.assertAlmostEqual(sleep["rps"], 0.3125)       # (0 + 0.625) / 2
        self.assertAlmostEqual(sleep["ece_top_label"], 0.25)
        self.assertEqual(report["calibration_secondary"]["pooled"]["n"], 2)

    def test_independent_label_agreement(self):
        ratings = self.root / "independent.csv"
        ratings.write_text(
            "participant_id,target_id,rater_id,label\n"
            "fiction-1,routine.sleep,r1,0\n"
            "fiction-1,routine.sleep,r2,2\n"
            "fiction-2,routine.sleep,r1,2\n"
            "fiction-2,routine.sleep,r2,2\n", encoding="utf-8")
        report = label_agreement(self.task, ratings, self.root / "agreement.json")
        sleep = report["by_target"]["routine.sleep"]
        self.assertEqual(sleep["paired_participants"], 2)
        self.assertEqual(sleep["exact_agreement"], 0.5)
        self.assertEqual(sleep["severe_disagreement_rate"], 0.5)
        self.assertIsNone(report["by_target"]["social.support"]["quadratic_weighted_kappa"])


if __name__ == "__main__":
    unittest.main()

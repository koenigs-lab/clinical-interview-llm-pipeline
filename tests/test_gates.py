"""Freeze gate, held-out test access, merging, and human review (C.1)."""

from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path

from clinical_interview_pipeline.cli import main
from clinical_interview_pipeline.datasets import constant_baseline, make_split
from clinical_interview_pipeline.experiments import make_plan
from clinical_interview_pipeline.records import read_jsonl, sha256_file
from clinical_interview_pipeline.reviews import (assisted_ratings, make_review_sheet,
                                                 merge_predictions, summarize_review)

from support import EXAMPLE, built_dataset, WorkspaceTest


class GateTest(WorkspaceTest):
    def _planned(self):
        dataset, split = built_dataset(self.root, self.task)
        plan_path = self.root / "plan.json"
        plan = make_plan(self.task, EXAMPLE / "experiments.toml", dataset / "train.jsonl",
                         dataset / "validation.jsonl", self.root / "model", self.root / "runs",
                         plan_path)
        shared = next(job for job in plan["jobs"] if job["adapter_strategy"] == "shared")
        others = [job for job in plan["jobs"] if job["adapter_strategy"] == "shared"
                  and job["name"] != shared["name"]]
        analysis = self.root / "analysis.md"
        analysis.write_text("Primary: mean_target_mae_itt, adapter on vs off.\n", encoding="utf-8")
        return {"dataset": dataset, "split": split, "plan": plan_path, "job": shared,
                "other": others[0], "analysis": analysis}

    def _freeze(self, setup, condition="base_only", name="freeze.json", split=None,
                jobs=None):
        arguments = ["freeze", "--task", str(self.task.path), "--plan", str(setup["plan"]),
                     "--split", str(split or setup["split"]),
                     "--analysis-plan", str(setup["analysis"]),
                     "--selected-condition", condition, "--rationale", "fictional",
                     "--output", str(self.root / name)]
        for job in jobs or [setup["job"]["name"]]:
            arguments += ["--job", job]
        return main(arguments)

    def _test_build(self, setup, freeze="freeze.json"):
        output = self.root / "frozen-build"
        main(["build", "--task", str(self.task.path),
              "--transcript", str(self.root / "approved.jsonl"),
              "--approval", str(self.root / "approval.json"),
              "--inventory", str(EXAMPLE / "inventory.csv"),
              "--labels", str(EXAMPLE / "labels.csv"), "--split", str(setup["split"]),
              "--output-dir", str(output), "--include-test", "--freeze", str(self.root / freeze)])
        return output / "test.jsonl"

    def _predictions(self, name, data, meta):
        path = self.root / name
        rows = read_jsonl(data)
        path.write_text("participant_id,target_id,prediction\n" + "".join(
            f"{row['participant_id']},{row['target_id']},1\n" for row in rows), encoding="utf-8")
        if meta is not None:
            Path(str(path) + ".meta.json").write_text(json.dumps(
                {"data_sha256": sha256_file(data), "decision_rule": "ordinal_median", **meta}),
                encoding="utf-8")
        return path

    def test_test_split_needs_freeze(self):
        setup = self._planned()
        with self.assertRaises(ValueError):
            main(["build", "--task", str(self.task.path),
                  "--transcript", str(self.root / "approved.jsonl"),
                  "--approval", str(self.root / "approval.json"),
                  "--inventory", str(EXAMPLE / "inventory.csv"),
                  "--labels", str(EXAMPLE / "labels.csv"), "--split", str(setup["split"]),
                  "--output-dir", str(self.root / "test-build"), "--include-test"])

    def test_freeze_binds_complete_selection_split_and_analysis_plan(self):
        setup = self._planned()
        partial = json.loads(setup["plan"].read_text(encoding="utf-8"))["jobs"]
        by_group = next(job["name"] for job in partial if job["adapter_strategy"] == "by_group")
        with self.assertRaises(ValueError):
            self._freeze(setup, name="partial.json", jobs=[by_group])
        other_split = self.root / "other-split.json"
        make_split(EXAMPLE / "inventory.csv", EXAMPLE / "labels.csv", self.task, other_split,
                   seed=7)
        with self.assertRaises(ValueError):
            self._freeze(setup, name="foreign.json", split=other_split)
        self.assertEqual(self._freeze(setup), 0)
        self.assertTrue(self._test_build(setup).is_file())
        setup["analysis"].write_text("Changed after freeze.\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            self._test_build(setup)

    def test_freeze_records_selected_condition_and_requires_adapter(self):
        setup = self._planned()
        with self.assertRaises(ValueError):
            self._freeze(setup, "adapted", "f1.json")
        (Path(setup["job"]["run_dir"]) / "adapter").mkdir(parents=True)
        self.assertEqual(self._freeze(setup, "adapted", "f2.json"), 0)
        record = json.loads((self.root / "f2.json").read_text(encoding="utf-8"))
        self.assertEqual(record["selected_condition"], "adapted")
        self.assertEqual(record["comparator_condition"], "base_only")

    def test_freeze_selects_one_configuration_for_all_targets(self):
        setup = self._planned()
        jobs = json.loads(setup["plan"].read_text(encoding="utf-8"))["jobs"]
        by_group = [job for job in jobs if job["adapter_strategy"] == "by_group"]
        groups = sorted({job["adapter_group"] for job in by_group})
        first = next(job for job in by_group if job["adapter_group"] == groups[0])
        settings = (first["learning_rate"], first["epochs"], first["seed"])
        others = [job for job in by_group if job["adapter_group"] == groups[1]]
        mixed = next(job for job in others
                     if (job["learning_rate"], job["epochs"], job["seed"]) != settings)
        same = next(job for job in others
                    if (job["learning_rate"], job["epochs"], job["seed"]) == settings)
        with self.assertRaises(ValueError):
            self._freeze(setup, name="mixed.json", jobs=[first["name"], mixed["name"]])
        self.assertEqual(self._freeze(setup, name="one.json",
                                      jobs=[first["name"], same["name"]]), 0)
        record = json.loads((self.root / "one.json").read_text(encoding="utf-8"))
        self.assertEqual(record["selected_configuration"]["adapter_strategy"], "by_group")

    def test_held_out_data_is_recognised_without_a_split_label(self):
        setup = self._planned()
        self._freeze(setup)
        test = self._test_build(setup)
        gpu = ["--gpu", "0", "--watch-disk-path", str(self.root),
               "--safety-log-dir", str(self.root / "safety"), "--dry-run"]

        def predict(job, *extra):
            return main(["predict", "--plan", str(setup["plan"]), "--job", job,
                         "--data", str(test), "--output", str(self.root / "p.csv"),
                         *gpu, *extra])
        with self.assertRaises(ValueError):
            predict(setup["job"]["name"])                       # no freeze
        with self.assertRaises(ValueError):
            predict(setup["other"]["name"], "--freeze", str(self.root / "freeze.json"))
        self.assertEqual(predict(setup["job"]["name"], "--freeze",
                                 str(self.root / "freeze.json")), 0)

    def test_test_evaluation_accepts_only_frozen_predictions(self):
        setup = self._planned()
        self._freeze(setup)
        test = self._test_build(setup)
        plan_sha = sha256_file(setup["plan"])

        def evaluate(predictions, name, freeze=True):
            return main(["evaluate", "--task", str(self.task.path), "--truth", str(test),
                         "--predictions", str(predictions), "--output", str(self.root / name),
                         "--bootstrap-draws", "10",
                         *(["--freeze", str(self.root / "freeze.json")] if freeze else [])])
        selected = self._predictions("selected.csv", test, {
            "adapter_used": False, "jobs": [setup["job"]["name"]], "plan_sha256": plan_sha})
        with self.assertRaises(ValueError):
            evaluate(selected, "no-freeze.json", freeze=False)
        self.assertEqual(evaluate(selected, "selected.json"), 0)
        report = json.loads((self.root / "selected.json").read_text(encoding="utf-8"))
        self.assertEqual((report["split"], report["role"]), ("test", "selected"))
        other = self._predictions("other.csv", test, {
            "adapter_used": False, "jobs": [setup["other"]["name"]], "plan_sha256": plan_sha})
        with self.assertRaises(ValueError):
            evaluate(other, "other.json")
        with self.assertRaises(ValueError):
            evaluate(self._predictions("bare.csv", test, None), "bare.json")
        baseline = self.root / "baseline.csv"
        constant_baseline(setup["dataset"] / "train.jsonl", test, baseline)
        self.assertEqual(evaluate(baseline, "baseline.json"), 0)
        report = json.loads((self.root / "baseline.json").read_text(encoding="utf-8"))
        self.assertEqual(report["role"], "comparator")

    def test_only_the_frozen_selection_goes_to_review(self):
        setup = self._planned()
        (Path(setup["job"]["run_dir"]) / "adapter").mkdir(parents=True)
        self._freeze(setup, "adapted")
        plan_sha = sha256_file(setup["plan"])
        data = setup["dataset"] / "validation.jsonl"
        base = self._predictions("base.csv", data, {
            "adapter_used": False, "jobs": [setup["job"]["name"]], "plan_sha256": plan_sha})
        adapted = self._predictions("adapted.csv", data, {
            "adapter_used": True, "jobs": [setup["job"]["name"]], "plan_sha256": plan_sha})
        stray = self._predictions("stray.csv", data, {
            "adapter_used": True, "jobs": [setup["other"]["name"]], "plan_sha256": plan_sha})

        def sheet(predictions, name, *freeze):
            return main(["review-sheet", "--predictions", str(predictions),
                         "--output", str(self.root / name), *freeze])
        freeze = ("--freeze", str(self.root / "freeze.json"))
        with self.assertRaises(SystemExit):
            sheet(adapted, "s0.csv")
        with self.assertRaises(ValueError):
            sheet(base, "s1.csv", *freeze)
        with self.assertRaises(ValueError):
            sheet(stray, "s2.csv", *freeze)
        self.assertEqual(sheet(adapted, "s3.csv", *freeze), 0)

    def test_merge_keeps_probabilities_and_provenance(self):
        header = "participant_id,target_id,prediction,status,label_mass,mode,expected_value,p_0,p_1,p_2\n"
        meta = {"adapter_used": True, "plan_sha256": "plan", "data_sha256": "data",
                "split": "validation", "decision_rule": "ordinal_median"}
        files = []
        for name, row, job in (("g1.csv", "p1,routine.sleep,1,scored,0.9,1,1.0,0.1,0.8,0.1", "a"),
                               ("g2.csv", "p1,social.support,0,scored,0.9,0,0.2,0.8,0.2,0.0", "b")):
            path = self.root / name
            path.write_text(header + row + "\n", encoding="utf-8")
            Path(str(path) + ".meta.json").write_text(json.dumps({**meta, "jobs": [job]}),
                                                      encoding="utf-8")
            files.append(path)
        merged = self.root / "merged.csv"
        self.assertEqual(merge_predictions(files, merged), 2)
        self.assertIn("p_2", merged.read_text(encoding="utf-8").splitlines()[0])
        merged_meta = json.loads(Path(str(merged) + ".meta.json").read_text(encoding="utf-8"))
        self.assertEqual((merged_meta["jobs"], merged_meta["data_sha256"]), (["a", "b"], "data"))
        Path(str(files[1]) + ".meta.json").write_text(json.dumps(
            {**meta, "jobs": ["b"], "data_sha256": "other"}), encoding="utf-8")
        with self.assertRaises(ValueError):
            merge_predictions(files, self.root / "m2.csv")
        Path(str(files[1]) + ".meta.json").write_text(json.dumps(
            {**meta, "jobs": ["b"], "adapter_used": False}), encoding="utf-8")
        with self.assertRaises(ValueError):
            merge_predictions(files, self.root / "m3.csv")
        sheet = self.root / "sheet.csv"
        make_review_sheet(merged, sheet, shuffle_seed=1)
        content = sheet.read_text(encoding="utf-8")
        self.assertNotIn("p_0", content)
        self.assertIn("status", content.splitlines()[0])

    def test_review_rules_are_shared_by_summary_and_export(self):
        sheet = self.root / "completed.csv"
        columns = ["participant_id", "target_id", "prediction", "disposition", "revised_value",
                   "reviewer_id", "correction_type", "review_time_seconds", "review_note"]

        def write(rows):
            with sheet.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
        good = [
            {"participant_id": "p1", "target_id": "routine.sleep", "prediction": "0",
             "disposition": "accept", "revised_value": "", "reviewer_id": "r1",
             "correction_type": "", "review_time_seconds": "12", "review_note": ""},
            {"participant_id": "p1", "target_id": "social.support", "prediction": "0",
             "disposition": "revise", "revised_value": "2", "reviewer_id": "r1",
             "correction_type": "missed_evidence", "review_time_seconds": "30",
             "review_note": ""},
        ]
        write(good)
        summary = summarize_review(self.task, sheet, self.root / "summary.json")
        self.assertEqual(summary["dispositions"], {"accept": 1, "revise": 1})
        self.assertEqual(assisted_ratings(self.task, sheet, self.root / "assisted.csv"), 2)
        self.assertIn("p1,social.support,2,r1", (self.root / "assisted.csv").read_text())
        write([{**good[1], "revised_value": "7"}])
        with self.assertRaises(ValueError):
            summarize_review(self.task, sheet, self.root / "summary2.json")
        with self.assertRaises(ValueError):
            assisted_ratings(self.task, sheet, self.root / "assisted2.csv")


if __name__ == "__main__":
    unittest.main()

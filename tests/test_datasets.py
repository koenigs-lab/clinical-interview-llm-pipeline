"""Participant splits, evidence selection, section boundaries, and dataset files."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from clinical_interview_pipeline.datasets import (_duplicate_audit, _render_evidence,
                                                  _section_spans, _select_evidence,
                                                  build_examples, constant_baseline,
                                                  dataset_split, make_split)
from clinical_interview_pipeline.records import read_jsonl, write_jsonl
from clinical_interview_pipeline.task import load_task

from support import (EXAMPLE, TARGETS, approved_transcript, built_dataset, section,
                     target, turn, write_task, WorkspaceTest)


class DatasetTest(WorkspaceTest):
    def _task(self, body: str, evidence_mode: str = "section"):
        folder = self.root / f"task{len(list(self.root.glob('task*')))}"
        folder.mkdir()
        return load_task(write_task(folder, body, evidence_mode))

    def _spans(self, task, rows):
        return {key: (value.start, value.end, value.open_ended) if value else None
                for key, value in _section_spans(task, rows).items()}

    def test_split_build_and_examples(self):
        dataset, split = built_dataset(self.root, self.task)
        train = read_jsonl(dataset / "train.jsonl")
        validation = read_jsonl(dataset / "validation.jsonl")
        self.assertEqual((len(train), len(validation)), (12, 3))
        self.assertEqual({row["split"] for row in train}, {"train"})
        self.assertEqual(train[0]["reference_provenance"]["reference_rater_id"],
                         "fictional-panel")
        self.assertTrue(any(not item["in_prompt"] for item in train[0]["source_turns"]))
        self.assertTrue({row["participant_id"] for row in train}.isdisjoint(
            {row["participant_id"] for row in validation}))
        manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["split_path"], str(split.resolve()))
        self.assertIn("duplicate_audit", manifest)
        report = manifest["splits"]["train"]["section_report"]
        # The last section has no closing marker in the fictional interviews.
        self.assertEqual(report["open_ended_section"], {"social.support": 4})
        self.assertFalse(report["ambiguous_section_boundary"])

    def test_baseline_uses_training_labels_and_records_provenance(self):
        dataset, _ = built_dataset(self.root, self.task)
        predictions = self.root / "baseline.csv"
        self.assertEqual(constant_baseline(dataset / "train.jsonl",
                                           dataset / "validation.jsonl", predictions), 3)
        meta = json.loads(Path(str(predictions) + ".meta.json").read_text(encoding="utf-8"))
        self.assertEqual((meta["condition"], meta["split"]), ("constant_baseline", "validation"))
        with self.assertRaises(ValueError):
            constant_baseline(dataset / "validation.jsonl", dataset / "validation.jsonl",
                              self.root / "wrong.csv")

    def test_split_is_read_from_the_data_not_a_label(self):
        dataset, _ = built_dataset(self.root, self.task)
        self.assertEqual(dataset_split(dataset / "validation.jsonl")["split"], "validation")
        rows = read_jsonl(dataset / "validation.jsonl")
        relabelled = dataset / "relabelled.jsonl"
        write_jsonl(relabelled, [{**row, "split": "train"} for row in rows])
        with self.assertRaises(ValueError):
            dataset_split(relabelled)

    def test_duplicate_audit_flags_cross_split_reuse(self):
        text = "A distinctive fictional account about a very unusual lunar observatory job"
        report = _duplicate_audit([
            ("fiction-1", "routine.sleep", "train", text),
            ("fiction-2", "routine.sleep", "validation", text),
        ])
        self.assertEqual(report["exact_cross_split_pairs"], 1)
        self.assertEqual(report["flagged_pairs"][0]["left"]["participant_id"], "fiction-1")

    def test_stratified_split_balances_severity(self):
        inventory = self.root / "inventory.csv"
        labels = self.root / "labels.csv"
        people = [f"p{i:02d}" for i in range(40)]
        inventory.write_text("participant_id,source_path,eligible\n" + "".join(
            f"{p},/secure/{p}.wav,yes\n" for p in people), encoding="utf-8")
        labels.write_text("participant_id,target_id,label\n" + "".join(
            f"{p},{t},{min(2, i // 14)}\n" for i, p in enumerate(people) for t in TARGETS),
            encoding="utf-8")
        severity = {p: min(2, i // 14) for i, p in enumerate(people)}
        splits = make_split(inventory, labels, self.task, self.root / "s1.json",
                            seed=3, stratify=True)
        again = make_split(inventory, labels, self.task, self.root / "s2.json",
                           seed=3, stratify=True)
        self.assertEqual(splits, again)
        self.assertEqual([len(splits[k]) for k in ("train", "validation", "test")], [28, 6, 6])
        means = {k: sum(severity[p] for p in v) / len(v) for k, v in splits.items()}
        self.assertLess(max(means.values()) - min(means.values()), 0.25)

    def test_section_markers_ignore_participant_turns(self):
        rows = [turn(1, "interviewer", "SECTION: SLEEP"), turn(2, "participant", "fine"),
                turn(3, "participant", "SECTION: PLANNING"),
                turn(4, "interviewer", "SECTION: PLANNING"), turn(5, "participant", "plans"),
                turn(6, "interviewer", "SECTION: SUPPORT"), turn(7, "participant", "help")]
        self.assertEqual(self._spans(self.task, rows), {
            "routine.sleep": (0, 3, False), "routine.planning": (3, 5, False),
            "social.support": (5, 7, True)})
        with self.assertRaises(ValueError):
            _section_spans(self.task, rows + [turn(8, "interviewer", "SECTION: SLEEP")])

    def test_missing_marker_does_not_merge_neighbouring_sections(self):
        rows = [turn(1, "interviewer", "SECTION: SLEEP"), turn(2, "participant", "sleep"),
                turn(3, "interviewer", "How do you plan your day?"),
                turn(4, "participant", "planning"),
                turn(5, "interviewer", "SECTION: SUPPORT"), turn(6, "participant", "support")]
        # Without the planning marker, sleep could have absorbed planning answers.
        self.assertEqual(self._spans(self.task, rows),
                         {"routine.sleep": None, "social.support": (4, 6, True)})
        turns = read_jsonl(EXAMPLE / "turns.jsonl")
        edited = self.root / "turns.jsonl"
        write_jsonl(edited, [row for row in turns if not (
            row["participant_id"] == "p01" and row["text"] == "SECTION: SUPPORT")])
        approved, approval = approved_transcript(self.root, edited)
        split = self.root / "split.json"
        make_split(EXAMPLE / "inventory.csv", EXAMPLE / "labels.csv", self.task, split)
        summary = build_examples(self.task, approved, approval, EXAMPLE / "inventory.csv",
                                 EXAMPLE / "labels.csv", split, self.root / "ds",
                                 include_test=True)
        part = next(part for part in summary["splits"].values()
                    if part["exclusions"].get("missing_section_anchor"))
        self.assertEqual(part["exclusions"]["missing_section_anchor"], 1)
        self.assertEqual(part["section_report"]["ambiguous_section_boundary"],
                         {"routine.planning": 1})

    def test_facets_can_share_sections_and_draw_on_several(self):
        task = self._task(section("a", "^A$") + section("b", "^B$")
                          + section("closing", "^END$") + target("t1", ["a"])
                          + target("t2", ["a"]) + target("t3", ["a", "b"]))
        rows = [turn(1, "interviewer", "A"), turn(2, "participant", "first"),
                turn(3, "interviewer", "B"), turn(4, "participant", "second"),
                turn(5, "interviewer", "END"), turn(6, "participant", "goodbye")]
        spans = [_section_spans(task, rows)]
        texts = {}
        for item in task.targets:
            selection = _select_evidence(task, item, [rows], spans)
            texts[item.id] = _render_evidence(task, selection["parts"])
            self.assertFalse(selection["open_ended"])
        self.assertEqual(texts, {"t1": "first", "t2": "first", "t3": "first\nsecond"})

    def test_end_markers_close_sections_and_missing_ends_are_ambiguous(self):
        task = self._task(section("a", "^A$", "^A END$") + section("b", "^B$")
                          + target("t1", ["a"]) + target("t2", ["b"]))
        rows = [turn(1, "interviewer", "A"), turn(2, "participant", "inside"),
                turn(3, "interviewer", "A END"), turn(4, "participant", "small talk"),
                turn(5, "interviewer", "B"), turn(6, "participant", "b answer")]
        self.assertEqual(self._spans(task, rows), {"a": (0, 2, False), "b": (4, 6, True)})
        without_end = rows[:2] + rows[3:]
        self.assertEqual(self._spans(task, without_end)["a"], None)

    def test_order_and_multiple_sessions(self):
        task = self._task("".join(section(name, f"^{name.upper()}$") for name in "abcd")
                          + "".join(target(f"t{name}", [name]) for name in "abcd"))
        first = [turn(1, "interviewer", "A"), turn(2, "participant", "a1"),
                 turn(3, "interviewer", "B"), turn(4, "participant", "b1")]
        second = [turn(1, "interviewer", "C", "s2"), turn(2, "participant", "c2", "s2"),
                  turn(3, "interviewer", "D", "s2"), turn(4, "participant", "d2", "s2")]
        # b may run into c or d, which are missing from session 1; a is closed by b.
        self.assertEqual(self._spans(task, first), {"a": (0, 2, False), "b": None})
        self.assertEqual(self._spans(task, second), {"c": (0, 2, False), "d": (2, 4, True)})
        swapped = [first[2], first[3], first[0], first[1]]
        self.assertEqual(self._spans(task, swapped), {"a": None, "b": None})
        complete = swapped + [turn(5, "interviewer", "C"), turn(6, "interviewer", "D")]
        self.assertEqual(self._spans(task, complete)["b"], (0, 2, False))

    def test_sessions_are_labelled_when_several_contribute(self):
        task = self._task(target("t1"), "full_interview")
        one = [turn(1, "participant", "alpha")]
        two = [turn(1, "participant", "beta", "s2")]
        self.assertEqual(_render_evidence(task, [one]), "alpha")
        self.assertEqual(_render_evidence(task, [one, two]),
                         "[Session 1]\nalpha\n\n[Session 2]\nbeta")
        self.assertEqual(_render_evidence(task, [one, [turn(1, "interviewer", "q", "s2")]]),
                         "alpha")


if __name__ == "__main__":
    unittest.main()

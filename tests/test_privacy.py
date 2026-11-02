"""Identifier search, the reviewed mapping, and hash-bound privacy approval."""

from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from clinical_interview_pipeline.privacy import (MAPPING_COLUMNS, apply_reviews, approve, propose,
                                                 verify_approval)
from clinical_interview_pipeline.privacy_llm import (DEFAULT_CATEGORIES, INDIRECT_CATEGORIES,
                                                     _json_rows, batches, loopback_endpoint)
from clinical_interview_pipeline.records import write_jsonl
from clinical_interview_pipeline.transcripts import approve_transcript

from support import EXAMPLE, WorkspaceTest


def _fake_model(flags: dict[int, list[str]], requests: list):
    """Stand-in for the local server: flags persons by row and answers in a fenced block."""
    def chat(address, model, messages):
        requests.append(messages[1]["content"])
        payload = json.loads(messages[1]["content"].split("INPUT:", 1)[1])
        rows = [{"row_id": index, "persons": flags.get(index, []),
                 **{name: [] for name in DEFAULT_CATEGORIES if name != "persons"}}
                for index, _ in payload]
        return "```json\n" + json.dumps({"rows": rows})[:-1] + ",}\n```", "stop"
    return chat


class PrivacyTest(WorkspaceTest):
    def _source(self, texts: list[str]):
        source = self.root / "source.jsonl"
        write_jsonl(source, [{"participant_id": "fiction-1", "session_id": "s1",
                              "turn_id": f"t{number}", "speaker": "participant", "text": text}
                             for number, text in enumerate(texts, start=1)])
        approval = self.root / "transcript-approval.json"
        approve_transcript(source, source, approval, "reviewer-1", ["participant"], True,
                           source_kind="synthetic")
        return source, approval

    def _rows(self, mapping: Path) -> list[dict]:
        with mapping.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def _review(self, mapping: Path, decisions: dict[str, str], added=(), drop=()):
        rows = [{**row, "anonymize": decisions.get(row["original"], row["anonymize"])}
                for row in self._rows(mapping) if row["original"] not in drop]
        with mapping.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=MAPPING_COLUMNS)
            writer.writeheader()
            writer.writerows([*rows, *added])

    def test_a_detector_is_required_and_patterns_are_opt_in(self):
        source, transcript_approval = self._source(["Write to test@example.invalid."])
        with self.assertRaises(ValueError):
            propose(source, self.root / "none.csv", transcript_approval)
        mapping = self.root / "mapping.csv"
        propose(source, mapping, transcript_approval, patterns=True)
        self.assertEqual([(row["type"], row["original"], row["token"], row["occurrences"],
                           row["anonymize"]) for row in self._rows(mapping)],
                         [("EMAIL", "test@example.invalid", "EMAIL_0001", "1", "")])
        with self.assertRaises(ValueError):            # every row needs a decision
            apply_reviews(source, mapping, self.root / "undecided.jsonl")
        self._review(mapping, {"test@example.invalid": "1"})
        output = self.root / "safe.jsonl"
        apply_reviews(source, mapping, output)
        self.assertIn("Write to EMAIL_0001.", output.read_text(encoding="utf-8"))
        approval = self.root / "approval.json"
        approve(output, "reviewer-2", approval, attested=True)
        verify_approval(output, approval)
        output.write_text(output.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            verify_approval(output, approval)

    def test_llm_search_reads_batches_and_replaces_every_occurrence(self):
        source, transcript_approval = self._source([
            "My sister Anna Berg called.", "She lives in Kiel.",
            "Anna Berg visits often; Anna is kind."])
        requests: list = []
        mapping = self.root / "mapping.csv"
        with patch("clinical_interview_pipeline.privacy_llm._chat",
                   side_effect=_fake_model({0: ["Anna Berg"]}, requests)), \
                patch("clinical_interview_pipeline.privacy.served_models",
                      return_value=["local-model"]):
            result = propose(source, mapping, transcript_approval, model="local-model",
                             rows_per_batch=2)
        meta = json.loads(Path(str(mapping) + ".meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["detectors"]["llm"]["served_models"], ["local-model"])
        self.assertEqual(result["batches"], 2)
        self.assertIn('[0,"My sister Anna Berg called."],[1,"She lives in Kiel."]', requests[0])
        rows = self._rows(mapping)
        self.assertEqual([(row["token"], row["occurrences"]) for row in rows],
                         [("PERSON_0001", "2")])
        # The reviewer accepts the proposal and adds a short form the search missed.
        self._review(mapping, {"Anna Berg": "1"}, added=[{
            "participant_id": "fiction-1", "session_id": "s1", "type": "PERSON",
            "original": "Anna", "token": "PERSON_0002", "occurrences": "", "anonymize": "1"}])
        output = self.root / "safe.jsonl"
        receipt = apply_reviews(source, mapping, output)
        text = output.read_text(encoding="utf-8")
        self.assertIn("My sister PERSON_0001 called.", text)
        self.assertIn("PERSON_0001 visits often; PERSON_0002 is kind.", text)
        self.assertEqual((receipt["n_added_by_reviewer"], receipt["n_replacements"]), (1, 3))

    def test_reviewers_decide_proposals_instead_of_deleting_them(self):
        source, transcript_approval = self._source(["Call 030 1234 5678 tomorrow."])
        mapping = self.root / "mapping.csv"
        propose(source, mapping, transcript_approval, patterns=True)
        self._review(mapping, {}, drop={"030 1234 5678"})
        with self.assertRaises(ValueError):
            apply_reviews(source, mapping, self.root / "dropped.jsonl")

    def test_unreadable_model_output_stops_without_logging_text(self):
        source, transcript_approval = self._source(["Anna Berg called."])
        with patch("clinical_interview_pipeline.privacy_llm._chat", return_value=("not json", "stop")), \
                patch("clinical_interview_pipeline.privacy.served_models", return_value=["local-model"]):
            with self.assertRaises(ValueError) as caught:
                propose(source, self.root / "m.csv", transcript_approval, model="local-model")
        self.assertNotIn("Anna", str(caught.exception))
        with self.assertRaises(ValueError):
            propose(source, self.root / "remote.csv", transcript_approval, model="local-model",
                    endpoint="https://remote.example/v1")

    def test_cut_off_answers_are_split_and_row_lists_accepted(self):
        source, transcript_approval = self._source(["Anna Berg called.", "Fine.", "Thanks."])

        def chat(address, model, messages):
            payload = json.loads(messages[1]["content"].split("INPUT:", 1)[1])
            if len(payload) > 1:
                return '[{"row_id": 0, "persons": [', "length"
            index = payload[0][0]
            rows = [{"row_id": index, "persons": ["Anna Berg"] if index == 0 else []}]
            return json.dumps(rows), "stop"      # a bare list of rows, not {"rows": ...}
        mapping = self.root / "mapping.csv"
        with patch("clinical_interview_pipeline.privacy_llm._chat", side_effect=chat), \
                patch("clinical_interview_pipeline.privacy.served_models", return_value=None):
            result = propose(source, mapping, transcript_approval, model="local-model")
        self.assertEqual((result["batches"], result["batches_split"]), (1, 2))
        self.assertEqual([row["original"] for row in self._rows(mapping)], ["Anna Berg"])
        self.assertEqual(_json_rows('Here: [{"row_id": 1}]'), {"rows": [{"row_id": 1}]})

    def test_batches_follow_the_row_limit_and_context_budget(self):
        self.assertEqual(batches(["short"] * 5, DEFAULT_CATEGORIES, 4096, 2),
                         [[0, 1], [2, 3], [4]])
        # With all ten categories, an answer for more than 10 turns would not fit.
        groups = batches(["short"] * 25, DEFAULT_CATEGORIES + INDIRECT_CATEGORIES, 8192, 20)
        self.assertEqual([len(group) for group in groups], [10, 10, 5])
        self.assertEqual(batches(["a", "x" * 20000, "b"], DEFAULT_CATEGORIES, 4096, 20),
                         [[0], [1], [2]])
        with self.assertRaises(ValueError):
            batches(["a"], DEFAULT_CATEGORIES, 1000, 20)

    def test_model_name_must_match_the_served_model(self):
        source, transcript_approval = self._source(["Anna Berg called."])
        with patch("clinical_interview_pipeline.privacy.served_models",
                   return_value=["Meta-Llama-3-70B-Instruct-Q4_K_M.gguf"]):
            with self.assertRaises(ValueError) as caught:
                propose(source, self.root / "m.csv", transcript_approval, model="llama3-70b")
        self.assertIn("Meta-Llama-3-70B-Instruct-Q4_K_M.gguf", str(caught.exception))
        self.assertFalse((self.root / "m.csv").exists())

    def test_detector_endpoint_must_be_loopback(self):
        self.assertEqual(loopback_endpoint("http://[::1]/v1"), ("::1", 80))
        for endpoint in ("http://10.0.0.5:2929/v1", "https://127.0.0.1/v1",
                         "http://127.0.0.1:2929/v2", "http://user:pw@127.0.0.1/v1"):
            with self.assertRaises(ValueError):
                loopback_endpoint(endpoint)

    def test_distinct_reviewer_option(self):
        transcript = EXAMPLE / "turns.jsonl"
        t_approval = self.root / "t.json"
        approve_transcript(transcript, transcript, t_approval, "same",
                           ["participant", "interviewer"], True, source_kind="synthetic")
        mapping = self.root / "mapping.csv"
        propose(transcript, mapping, t_approval, patterns=True)
        reviewed = self.root / "reviewed.jsonl"
        apply_reviews(transcript, mapping, reviewed)
        with self.assertRaises(ValueError):
            approve(reviewed, "same", self.root / "a1.json", True, require_distinct_reviewer=True)
        approve(reviewed, "same", self.root / "a2.json", True)
        record = json.loads((self.root / "a2.json").read_text(encoding="utf-8"))
        self.assertFalse(record["distinct_reviewers"])


if __name__ == "__main__":
    unittest.main()

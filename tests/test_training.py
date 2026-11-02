"""Label-token scoring, experiment plans, and wrapped GPU entry points (mocked)."""

from __future__ import annotations

import csv
import json
import math
import os
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from clinical_interview_pipeline import training
from clinical_interview_pipeline.cli import main
from clinical_interview_pipeline.experiments import load_job, make_plan

from support import EXAMPLE, TARGETS, built_dataset, WorkspaceTest


class _FakeTensor:
    def __init__(self, values):
        self.values = list(values)

    def __getitem__(self, index):
        return _FakeTensor(self.values[i] for i in index)

    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self.values)


class _FakeTorch(types.SimpleNamespace):
    long = "long"

    @staticmethod
    def tensor(value, device=None):
        return value

    @staticmethod
    def ones(shape, device=None, dtype=None):
        return shape

    @staticmethod
    def no_grad():
        class Context:
            def __enter__(self):
                return None

            def __exit__(self, *args):
                return False
        return Context()

    @staticmethod
    def log_softmax(tensor, dim=-1):
        peak = max(tensor.values)
        total = sum(math.exp(v - peak) for v in tensor.values)
        return _FakeTensor(v - peak - math.log(total) for v in tensor.values)


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [10 + int(text)] if text.isdigit() and len(text) == 1 else [1, 2]

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return list(range(len(messages[-1]["content"])))


class _FakeModel:
    """Vocabulary of 16 tokens; label tokens 10, 11, 12 for labels 0, 1, 2."""

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask, logits_to_keep=None):
        assert logits_to_keep == 1
        vocabulary = [0.0] * 16
        vocabulary[10], vocabulary[11], vocabulary[12] = 1.0, 2.0, 0.5
        return types.SimpleNamespace(logits={(0, -1): _FakeTensor(vocabulary)})


class TrainingTest(WorkspaceTest):
    def test_label_distribution_and_lower_median(self):
        dist = training.label_distribution([math.log(0.1), math.log(0.3), math.log(0.1)],
                                           (0, 1, 2))
        self.assertAlmostEqual(dist["label_mass"], 0.5)
        self.assertAlmostEqual(dist["probabilities"][1], 0.6)
        self.assertEqual(training.ordinal_median({0: 0.5, 1: 0.0, 2: 0.5}), 0)
        self.assertEqual(training.ordinal_median({0: 0.2, 1: 0.2, 2: 0.6}), 2)
        record = training.score_record([math.log(0.45), math.log(0.1), math.log(0.45)],
                                       (0, 1, 2))
        self.assertEqual(record["prediction"], 1)   # median, not the tied modes 0/2
        self.assertEqual(record["mode"], 0)
        self.assertAlmostEqual(record["expected_value"], 1.0)

    def test_label_tokens_must_be_single_and_distinct(self):
        self.assertEqual(training.label_token_ids(_FakeTokenizer(), (0, 1, 2)), [10, 11, 12])
        with self.assertRaises(ValueError):
            training.label_token_ids(_FakeTokenizer(), (0, 10))

    def test_chat_template_accepts_batch_encoding(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

        row = {"messages": [{"role": "system", "content": "synthetic"},
                            {"role": "user", "content": "synthetic"},
                            {"role": "assistant", "content": "0"}]}
        self.assertEqual(training._prompt(Tokenizer(), row), [1, 2, 3])

    def test_logit_request_follows_the_forward_signature(self):
        class Current:
            def forward(self, input_ids, logits_to_keep=0):
                pass

        class Older:
            def forward(self, input_ids, num_logits_to_keep=0):
                pass

        class Opaque:
            def forward(self, *args, **kwargs):
                pass

        class Wrapper:   # PEFT passes extra keywords through to the base model
            def get_base_model(self):
                return Current()

            def forward(self, **kwargs):
                pass

        self.assertEqual(training._logit_kwargs(Current()), {"logits_to_keep": 1})
        self.assertEqual(training._logit_kwargs(Older()), {"num_logits_to_keep": 1})
        self.assertEqual(training._logit_kwargs(Opaque()), {})
        self.assertEqual(training._logit_kwargs(Wrapper()), {"logits_to_keep": 1})

    def test_predict_job_scores_labels_and_marks_too_long(self):
        data = self.root / "validation.jsonl"
        rows = []
        for participant, content in (("p1", "short"), ("p2", "x" * 200)):
            rows.append({"participant_id": participant, "target_id": "routine.sleep",
                         "label": 1, "messages": [
                             {"role": "system", "content": "s"},
                             {"role": "user", "content": content},
                             {"role": "assistant", "content": "1"}]})
        data.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        job = {"name": "fake", "run_dir": str(self.root / "run"), "task_path": str(self.task.path),
               "model_path": str(self.root), "quantization": "4bit", "max_length": 50,
               "target_ids": list(TARGETS)}
        fake_torch = _FakeTorch()
        with patch.dict(os.environ, {"CILP_SAFE_GPU_WRAPPER": "1"}), \
                patch.object(training, "_dependencies",
                             return_value=types.SimpleNamespace(torch=fake_torch, PeftModel=None)), \
                patch.object(training, "_load_model",
                             return_value=(_FakeTokenizer(), _FakeModel())):
            output = self.root / "pred.csv"
            self.assertEqual(training.predict_job(job, data, output, use_adapter=False,
                                                  provenance={"split": "validation",
                                                              "plan_sha256": "abc"}), 2)
        with output.open(encoding="utf-8") as stream:
            result = list(csv.DictReader(stream))
        self.assertEqual(result[0]["status"], "scored")
        self.assertEqual(result[0]["prediction"], "1")
        self.assertAlmostEqual(sum(float(result[0][f"p_{i}"]) for i in range(3)), 1.0, places=5)
        self.assertLess(float(result[0]["label_mass"]), 1.0)
        self.assertEqual(result[1]["status"], "too_long")
        self.assertEqual(result[1]["prediction"], "")
        meta = json.loads(Path(str(output) + ".meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["decision_rule"], "ordinal_median")
        self.assertFalse(meta["adapter_used"])
        self.assertEqual(meta["n_too_long"], 1)
        self.assertEqual((meta["jobs"], meta["split"], meta["plan_sha256"]),
                         (["fake"], "validation", "abc"))

    def test_training_skip_report_by_target_and_label(self):
        report = training._skip_report([{"target_id": "a", "label": 2},
                                        {"target_id": "a", "label": 2},
                                        {"target_id": "b", "label": 0}])
        self.assertEqual(report["n"], 3)
        self.assertEqual(report["by_target_and_label"]["a"]["2"], 2)

    def test_training_items_for_each_objective(self):
        rows = [{"label": 2, "target_id": "routine.sleep", "messages": [
            {"role": "system", "content": "s"}, {"role": "user", "content": "abc"},
            {"role": "assistant", "content": "2"}]}]
        tokenizer = _FakeTokenizer()
        tokenizer.eos_token_id = 99
        (row, ids, target), = training._encode(tokenizer, rows, 50, "token_ce", (0, 1, 2))[0]
        self.assertEqual((ids, target), ([0, 1, 2, 12, 99], [-100, -100, -100, 12, 99]))
        (row, ids, target), = training._encode(tokenizer, rows, 50, "label_rps", (0, 1, 2))[0]
        self.assertEqual((ids, target), ([0, 1, 2], 2))
        self.assertEqual(training._encode(tokenizer, rows, 4, "token_ce", (0, 1, 2))[0], [])

    def _plan(self, experiments: Path = EXAMPLE / "experiments.toml", name: str = "plan.json"):
        if not (self.root / "dataset").exists():
            built_dataset(self.root, self.task)
        dataset = self.root / "dataset"
        return make_plan(self.task, experiments, dataset / "train.jsonl",
                         dataset / "validation.jsonl", self.root / "offline-model",
                         self.root / "runs", self.root / name)

    def test_experiment_matrix_and_objective(self):
        plan = self._plan()
        self.assertEqual(len(plan["jobs"]), 24)
        job = plan["jobs"][0]
        self.assertEqual((job["epochs"], job["objective"]), (1, "token_ce"))
        self.assertEqual(load_job(self.root / "plan.json", job["name"])["name"], job["name"])
        text = (EXAMPLE / "experiments.toml").read_text(encoding="utf-8")
        ordinal = self.root / "ordinal.toml"
        ordinal.write_text(text.replace('"token_ce"', '"label_rps"'), encoding="utf-8")
        renamed = self._plan(ordinal, "ordinal.json")
        self.assertTrue(renamed["jobs"][0]["name"].endswith("_label_rps"))
        unknown = self.root / "unknown.toml"
        unknown.write_text(text.replace('"token_ce"', '"mse"'), encoding="utf-8")
        with self.assertRaises(ValueError):
            self._plan(unknown, "unknown.json")

    def test_plan_counts_examples_longer_than_max_length(self):
        built_dataset(self.root, self.task)
        dataset = self.root / "dataset"
        short = self.root / "short.toml"
        text = (EXAMPLE / "experiments.toml").read_text(encoding="utf-8")
        short.write_text(text.replace("max_length = 2048", "max_length = 60"), encoding="utf-8")
        tokenizer = _FakeTokenizer()
        tokenizer.eos_token_id = 99
        plan = make_plan(self.task, short, dataset / "train.jsonl", dataset / "validation.jsonl",
                         self.root / "model", self.root / "runs", self.root / "plan.json",
                         tokenizer)
        prompt = lambda row: len(row["messages"][1]["content"])     # the fake tokenizer's length
        validation = [row for row in training.read_jsonl(dataset / "validation.jsonl")]
        train = [row for row in training.read_jsonl(dataset / "train.jsonl")]
        report = plan["too_long"]
        self.assertEqual(report["max_length"], 60)
        self.assertEqual(report["validation"]["n"], sum(prompt(row) > 60 for row in validation))
        self.assertEqual(report["train"]["n"], sum(prompt(row) + 2 > 60 for row in train))
        self.assertGreater(report["validation"]["n"], 0)
        self.assertIsNone(self._plan(name="unchecked.json")["too_long"])   # no tokenizer

    def test_plan_command_needs_the_model_unless_the_length_check_is_skipped(self):
        built_dataset(self.root, self.task)
        dataset = self.root / "dataset"
        arguments = ["plan", "--task", str(self.task.path),
                     "--experiments", str(EXAMPLE / "experiments.toml"),
                     "--train", str(dataset / "train.jsonl"),
                     "--validation", str(dataset / "validation.jsonl"),
                     "--model-path", str(self.root / "missing-model"),
                     "--output-root", str(self.root / "runs")]
        with self.assertRaises((RuntimeError, FileNotFoundError)):
            main(arguments + ["--output", str(self.root / "checked.json")])
        self.assertFalse((self.root / "checked.json").exists())
        self.assertEqual(main(arguments + ["--output", str(self.root / "plan.json"),
                                           "--skip-length-check"]), 0)

    def test_plan_uses_the_training_and_validation_files(self):
        built_dataset(self.root, self.task)
        dataset = self.root / "dataset"
        with self.assertRaises(ValueError):
            make_plan(self.task, EXAMPLE / "experiments.toml", dataset / "validation.jsonl",
                      dataset / "validation.jsonl", self.root / "model", self.root / "runs",
                      self.root / "plan.json")

    def test_training_and_prediction_run_only_inside_the_wrapper(self):
        plan = self._plan()
        name = plan["jobs"][0]["name"]
        dataset = self.root / "dataset"
        gpu = ["--gpu", "0", "--watch-disk-path", str(self.root),
               "--safety-log-dir", str(self.root / "safety"), "--dry-run"]
        plan_path = str(self.root / "plan.json")
        self.assertEqual(main(["train", "--plan", plan_path, "--job", name, *gpu]), 0)
        for extra in ([], ["--base-only"]):
            self.assertEqual(main(["predict", "--plan", plan_path, "--job", name,
                                   "--data", str(dataset / "validation.jsonl"),
                                   "--output", str(self.root / "predictions.csv"),
                                   *gpu, *extra]), 0)
        with self.assertRaises(RuntimeError):
            main(["_train-job", "--plan", plan_path, "--job", name])
        with self.assertRaises(RuntimeError):
            main(["_predict-job", "--plan", plan_path, "--job", name,
                  "--data", str(dataset / "validation.jsonl"),
                  "--output", str(self.root / "predictions.csv")])


if __name__ == "__main__":
    unittest.main()

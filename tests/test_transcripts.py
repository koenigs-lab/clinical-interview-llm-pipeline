"""ASR and diarization adapters (mocked), speaker alignment, and transcript approval."""

from __future__ import annotations

import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from clinical_interview_pipeline.cli import main
from clinical_interview_pipeline.diarization import align_speakers, diarize_audio, import_rttm
from clinical_interview_pipeline.records import read_jsonl, write_jsonl
from clinical_interview_pipeline.transcripts import (approve_transcript, correction_summary,
                                                     map_speakers, transcribe_audio,
                                                     verify_transcript_approval)

from support import WorkspaceTest


class TranscriptTest(WorkspaceTest):
    def setUp(self):
        super().setUp()
        self.audio = self.root / "fictional.wav"
        self.audio.write_bytes(b"fictional audio fixture")

    def _fake_whisper(self, calls: dict):
        word = types.SimpleNamespace(start=0.1, end=0.8, word=" Hello")
        segment = types.SimpleNamespace(start=0.0, end=1.0, text="Hello", words=[word])

        class WhisperModel:
            def __init__(self, path, **kwargs):
                calls["model"] = kwargs

            def transcribe(self, path, **kwargs):
                calls["transcribe"] = kwargs
                return [segment], None

        fake = types.ModuleType("faster_whisper")
        fake.WhisperModel = WhisperModel
        return fake

    def _transcribe(self, device: str, calls: dict, env: dict):
        model_dir = self.root / "asr-model"
        model_dir.mkdir(exist_ok=True)
        output = self.root / f"asr-{device}.jsonl"
        with patch.dict(sys.modules, {"faster_whisper": self._fake_whisper(calls)}), \
                patch.dict(os.environ, env), \
                patch("clinical_interview_pipeline.transcripts.version", return_value="1.test"):
            count = transcribe_audio(self.audio, model_dir, output, "fiction-1", "s1",
                                     language="de", device=device, compute_type="int8",
                                     cpu_threads=4, beam_size=3)
        return count, output

    def test_cpu_asr_keeps_word_times_and_records_settings(self):
        calls: dict = {}
        count, output = self._transcribe("cpu", calls, {})
        self.assertEqual(count, 1)
        self.assertEqual(read_jsonl(output)[0]["words"][0]["text"], " Hello")
        self.assertEqual(calls["model"], {"device": "cpu", "compute_type": "int8",
                                          "cpu_threads": 4, "local_files_only": True})
        self.assertTrue(calls["transcribe"]["word_timestamps"])
        self.assertEqual(calls["transcribe"]["beam_size"], 3)
        meta = json.loads(Path(str(output) + ".meta.json").read_text(encoding="utf-8"))
        self.assertEqual((meta["device"], meta["compute_type"], meta["beam_size"]),
                         ("cpu", "int8", 3))

    def test_gpu_asr_needs_the_safety_wrapper(self):
        with self.assertRaises(RuntimeError):
            transcribe_audio(self.audio, self.root, self.root / "x.jsonl", "fiction-1", "s1",
                             device="cuda")
        calls: dict = {}
        _, output = self._transcribe("cuda", calls,
                                     {"CILP_SAFE_GPU_WRAPPER": "1", "CUDA_VISIBLE_DEVICES": "1"})
        self.assertEqual(calls["model"]["device"], "cuda")
        meta = json.loads(Path(str(output) + ".meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["cuda_visible_devices"], "1")

    def test_gpu_transcription_and_diarization_commands_use_the_wrapper(self):
        (self.root / "asr-model").mkdir()
        (self.root / "pyannote").mkdir()
        gpu = ["--gpu", "0", "--watch-disk-path", str(self.root),
               "--safety-log-dir", str(self.root / "safety"), "--dry-run"]
        transcribe = ["transcribe", "--audio", str(self.audio),
                      "--local-model", str(self.root / "asr-model"),
                      "--output", str(self.root / "asr.jsonl"),
                      "--participant", "fiction-1", "--session", "s1"]
        diarize = ["diarize", "--audio", str(self.audio),
                   "--local-pipeline", str(self.root / "pyannote"),
                   "--output", str(self.root / "intervals.jsonl")]
        self.assertEqual(main(transcribe + gpu), 0)
        self.assertEqual(main(diarize + gpu), 0)
        with self.assertRaises(SystemExit):
            main(transcribe)                           # no device chosen
        with self.assertRaises(ValueError):
            main(transcribe + ["--cpu", "--dry-run"])
        with self.assertRaises(ValueError):
            main(transcribe + ["--gpu", "0"])          # no safety log or disk path
        with self.assertRaises(RuntimeError):
            main(["_transcribe-job"] + transcribe[1:])
        with self.assertRaises(RuntimeError):
            main(["_diarize-job"] + diarize[1:])

    def _fake_pyannote(self, devices: list):
        turn = types.SimpleNamespace(start=0.0, end=1.0)

        class Annotation:
            def itertracks(self, yield_label=False):
                return [(turn, None, "S0")]

        class Pipeline:
            @staticmethod
            def from_pretrained(path):
                return Pipeline()

            def to(self, device):
                devices.append(device)

            def __call__(self, path):
                return types.SimpleNamespace(exclusive_speaker_diarization=Annotation())

        pyannote = types.ModuleType("pyannote")
        pyannote_audio = types.ModuleType("pyannote.audio")
        pyannote_audio.Pipeline = Pipeline
        torch = types.ModuleType("torch")
        torch.device = lambda name: name
        return {"pyannote": pyannote, "pyannote.audio": pyannote_audio, "torch": torch}

    def test_local_diarization_runs_offline_on_the_chosen_device(self):
        pipeline_dir = self.root / "pyannote-local"
        pipeline_dir.mkdir()
        (pipeline_dir / "config.yaml").write_text("fictional: true\n", encoding="utf-8")
        devices: list = []
        with patch.dict(os.environ, {}, clear=False), \
                patch.dict(sys.modules, self._fake_pyannote(devices)), \
                patch("clinical_interview_pipeline.diarization.version", return_value="4.test"):
            self.assertEqual(diarize_audio(self.audio, pipeline_dir, self.root / "a.jsonl"), 1)
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            with self.assertRaises(RuntimeError):
                diarize_audio(self.audio, pipeline_dir, self.root / "b.jsonl", "cuda")
            with patch.dict(os.environ, {"CILP_SAFE_GPU_WRAPPER": "1"}):
                diarize_audio(self.audio, pipeline_dir, self.root / "c.jsonl", "cuda")
        self.assertEqual(devices, ["cpu", "cuda"])
        self.assertEqual(read_jsonl(self.root / "a.jsonl")[0]["speaker"], "S0")

    def test_diarization_alignment_role_mapping_and_approval(self):
        source = self.root / "asr.jsonl"
        write_jsonl(source, [{"participant_id": "fiction-1", "session_id": "s1",
                              "turn_id": "asr-1", "speaker": "unknown", "text": "Hello there",
                              "start": 0.0, "end": 2.0,
                              "words": [{"start": 0.1, "end": 0.8, "text": " Hello"},
                                        {"start": 1.1, "end": 1.8, "text": " there"}]}])
        rttm = self.root / "speakers.rttm"
        rttm.write_text("SPEAKER rec1 1 0.0 1.0 <NA> <NA> S0 <NA> <NA>\n"
                        "SPEAKER rec1 1 1.0 1.0 <NA> <NA> S1 <NA> <NA>\n", encoding="utf-8")
        intervals = self.root / "intervals.jsonl"
        self.assertEqual(import_rttm(rttm, intervals), 2)
        aligned = self.root / "aligned.jsonl"
        self.assertEqual(align_speakers(source, intervals, aligned)["unknown_speaker_turns"], 0)
        self.assertEqual([row["speaker"] for row in read_jsonl(aligned)], ["S0", "S1"])
        mapping = self.root / "speaker-map.csv"
        mapping.write_text("diarized_speaker,role\nS0,interviewer\nS1,participant\n",
                           encoding="utf-8")
        mapped = self.root / "mapped.jsonl"
        self.assertEqual(map_speakers(aligned, mapping, mapped), 2)
        approval = self.root / "transcript-approval.json"
        with self.assertRaises(ValueError):
            approve_transcript(aligned, mapped, approval, "reviewer",
                               ["participant", "interviewer"], False)
        approve_transcript(aligned, mapped, approval, "reviewer",
                           ["participant", "interviewer"], True, source_kind="synthetic")
        verify_transcript_approval(mapped, approval)
        with self.assertRaises(ValueError):
            verify_transcript_approval(aligned, approval)

    def test_ambiguous_overlap_needs_manual_speaker_review(self):
        source = self.root / "asr.jsonl"
        write_jsonl(source, [{"participant_id": "fiction-1", "session_id": "s1",
                              "turn_id": "asr-1", "speaker": "unknown", "text": "Perhaps",
                              "start": 0.0, "end": 1.0}])
        intervals = self.root / "intervals.jsonl"
        write_jsonl(intervals, [{"start": 0.0, "end": 1.0, "speaker": "S0"},
                                {"start": 0.2, "end": 0.8, "speaker": "S1"}])
        aligned = self.root / "aligned.jsonl"
        self.assertEqual(align_speakers(source, intervals, aligned)["unknown_speaker_turns"], 1)
        with self.assertRaises(ValueError):
            approve_transcript(aligned, aligned, self.root / "approval.json", "reviewer",
                               ["participant", "interviewer"], True, source_kind="synthetic")

    def test_same_speaker_overlapping_intervals_do_not_double_count_coverage(self):
        source = self.root / "asr.jsonl"
        write_jsonl(source, [{"participant_id": "fiction-1", "session_id": "s1",
                              "turn_id": "asr-1", "speaker": "unknown", "text": "Perhaps",
                              "start": 0.0, "end": 2.0}])
        intervals = self.root / "intervals.jsonl"
        write_jsonl(intervals, [{"start": 0.0, "end": 0.6, "speaker": "S0"},
                                {"start": 0.2, "end": 0.8, "speaker": "S0"}])
        report = align_speakers(source, intervals, self.root / "aligned.jsonl", collar=0.0)
        self.assertEqual(report["unknown_speaker_turns"], 1)

    def test_collar_absorbs_early_word_times(self):
        # ASR places the first word slightly before the speaker's interval begins.
        source = self.root / "asr.jsonl"
        write_jsonl(source, [{"participant_id": "fiction-1", "session_id": "s1",
                              "turn_id": "asr-1", "speaker": "unknown", "text": "Thank you",
                              "start": 0.2, "end": 1.2,
                              "words": [{"start": 0.2, "end": 0.7, "text": " Thank"},
                                        {"start": 0.7, "end": 1.2, "text": " you"}]}])
        intervals = self.root / "intervals.jsonl"
        write_jsonl(intervals, [{"start": 0.5, "end": 1.2, "speaker": "S0"},
                                {"start": 1.5, "end": 3.0, "speaker": "S1"}])
        strict = align_speakers(source, intervals, self.root / "strict.jsonl", collar=0.0)
        tolerant = align_speakers(source, intervals, self.root / "tolerant.jsonl")
        self.assertEqual((strict["unknown_speaker_turns"], tolerant["unknown_speaker_turns"]), (1, 0))
        self.assertEqual([row["text"] for row in read_jsonl(self.root / "tolerant.jsonl")],
                         ["Thank you"])

    def test_audio_approval_binds_the_source_recording(self):
        transcript = self.root / "turns.jsonl"
        write_jsonl(transcript, [{"participant_id": "fiction-1", "session_id": "s1",
                                  "turn_id": "t1", "speaker": "participant", "text": "Fiction."}])
        approval = self.root / "approval.json"
        approve_transcript(transcript, transcript, approval, "reviewer",
                           ["participant"], True, audio_path=self.audio)
        verify_transcript_approval(transcript, approval)
        self.audio.write_bytes(b"changed fictional audio fixture")
        with self.assertRaises(ValueError):
            verify_transcript_approval(transcript, approval)

    def test_correction_summary_counts_token_edits(self):
        before, after = self.root / "before.jsonl", self.root / "after.jsonl"
        row = {"participant_id": "fiction-1", "session_id": "s1", "turn_id": "t1",
               "speaker": "participant"}
        write_jsonl(before, [{**row, "text": "I am not tired today"},
                             {**row, "turn_id": "t2", "text": "Fine"}])
        write_jsonl(after, [{**row, "text": "I am tired today ."},
                            {**row, "turn_id": "t2", "text": "Fine"}])
        summary = correction_summary(before, after)
        self.assertEqual((summary["deleted_tokens"], summary["inserted_tokens"]), (1, 1))
        self.assertEqual(summary["token_edit_operations"], 2)
        self.assertEqual(summary["changed_text_turns"], 1)


if __name__ == "__main__":
    unittest.main()

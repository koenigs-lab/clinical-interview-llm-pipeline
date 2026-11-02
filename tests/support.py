"""Shared fixtures for the tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from clinical_interview_pipeline.datasets import build_examples, make_split
from clinical_interview_pipeline.privacy import apply_reviews, approve, propose
from clinical_interview_pipeline.task import load_task
from clinical_interview_pipeline.transcripts import approve_transcript

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "synthetic"
LPFS = ROOT / "examples" / "lpfs-skeleton"
TARGETS = ("routine.sleep", "routine.planning", "social.support")


class WorkspaceTest(unittest.TestCase):
    """Each test gets a fresh temporary directory as self.root and the fictional task."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.task = load_task(EXAMPLE / "task.toml")


def approved_transcript(root: Path, transcript: Path = EXAMPLE / "turns.jsonl",
                        speakers: tuple[str, ...] = ("participant", "interviewer")):
    """Pass a fictional transcript through transcript review and privacy approval."""
    transcript_approval = root / "transcript-approval.json"
    approve_transcript(transcript, transcript, transcript_approval, "reviewer-1",
                       list(speakers), True, source_kind="synthetic")
    mapping = root / "mapping.csv"
    propose(transcript, mapping, transcript_approval, patterns=True)
    output = root / "approved.jsonl"
    apply_reviews(transcript, mapping, output)
    approval = root / "approval.json"
    approve(output, "reviewer-2", approval, attested=True)
    return output, approval


def built_dataset(root: Path, task, transcript: Path = EXAMPLE / "turns.jsonl",
                  include_test: bool = False):
    """Approved transcript, seeded participant split, and train/validation examples."""
    approved, approval = approved_transcript(root, transcript)
    split = root / "split.json"
    make_split(EXAMPLE / "inventory.csv", EXAMPLE / "labels.csv", task, split, seed=42)
    dataset = root / "dataset"
    build_examples(task, approved, approval, EXAMPLE / "inventory.csv",
                   EXAMPLE / "labels.csv", split, dataset, include_test)
    return dataset, split


def write_task(root: Path, body: str, evidence_mode: str = "section",
               system: str = "Fictional system prompt.") -> Path:
    """A small fictional task; body holds its [[sections]] and [[targets]] tables."""
    (root / "rubric.txt").write_text("Fictional rubric.", encoding="utf-8")
    path = root / "task.toml"
    path.write_text(
        "[task]\n"
        'id = "fictional"\n'
        'construct = "Fictional construct"\n'
        'version = "1"\n'
        "labels = [0, 1, 2]\n"
        f'evidence_mode = "{evidence_mode}"\n\n'
        "[prompt]\n"
        f"system = '{system}'\n"
        'user = "{rubric}\\n{evidence}"\n\n'
        + body, encoding="utf-8")
    return path


def target(target_id: str, sections: list[str] | None = None, extra: str = "") -> str:
    listed = f"sections = {sections!r}\n".replace("'", '"') if sections is not None else ""
    return (f'[[targets]]\nid = "{target_id}"\nname = "{target_id}"\n'
            f'rubric = "rubric.txt"\n{listed}{extra}\n')


def section(section_id: str, start: str, end: str | None = None) -> str:
    closing = f'end_pattern = "{end}"\n' if end else ""
    return f'[[sections]]\nid = "{section_id}"\nstart_pattern = "{start}"\n{closing}\n'


def turn(index: int, speaker: str, text: str, session: str = "s1",
         participant: str = "p01") -> dict:
    return {"participant_id": participant, "session_id": session,
            "turn_id": f"{session}-t{index:02d}", "speaker": speaker, "text": text}

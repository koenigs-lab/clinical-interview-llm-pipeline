"""Offline speaker intervals and conservative ASR-to-speaker reconciliation."""

from __future__ import annotations

import math
import os
import sys
from importlib.metadata import version
from pathlib import Path

from .records import read_jsonl, read_turns, sha256_file, sidecar, write_json, write_jsonl


def diarize_audio(audio_path: Path, local_pipeline: Path, output_path: Path,
                  device: str = "cpu") -> int:
    """Run a complete local pyannote 4 pipeline without Hub access.

    device="cuda" runs on the single GPU exposed by the safety wrapper
    (cilp diarize --gpu).
    """
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and os.environ.get("CILP_SAFE_GPU_WRAPPER") != "1":
        raise RuntimeError("Run GPU diarization through cilp diarize --gpu (safety wrapper)")
    if "huggingface_hub" in sys.modules and os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError("Set HF_HUB_OFFLINE=1 before importing Hugging Face libraries")
    if not audio_path.is_file() or not (local_pipeline / "config.yaml").is_file():
        raise FileNotFoundError("Audio and a complete local pyannote pipeline are required")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as error:
        raise RuntimeError("Install the optional diarization dependencies first") from error
    pipeline = Pipeline.from_pretrained(str(local_pipeline))
    if pipeline is None:
        raise RuntimeError("Local diarization pipeline did not load")
    pipeline.to(torch.device(device))
    result = pipeline(str(audio_path))
    annotation = getattr(result, "exclusive_speaker_diarization", None)
    if annotation is None:
        annotation = getattr(result, "speaker_diarization", result)
    intervals = [
        {"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)}
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    intervals.sort(key=lambda row: (row["start"], row["end"], row["speaker"]))
    _validate_intervals(intervals)
    write_jsonl(output_path, intervals)
    write_json(sidecar(output_path), {
        "audio_sha256": sha256_file(audio_path),
        "local_pipeline_config_sha256": sha256_file(local_pipeline / "config.yaml"),
        "pyannote_audio_version": version("pyannote.audio"),
        "device": device,
        "exclusive_intervals": getattr(result, "exclusive_speaker_diarization", None) is not None,
        "status": "automatic_unreviewed",
    })
    return len(intervals)


def import_rttm(rttm_path: Path, output_path: Path, recording_id: str | None = None) -> int:
    """Import standard SPEAKER RTTM rows from another diarizer."""
    intervals = []
    found_ids: set[str] = set()
    with rttm_path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 10 or fields[0] != "SPEAKER":
                raise ValueError(f"Invalid SPEAKER RTTM row {number}")
            found_ids.add(fields[1])
            if recording_id is not None and fields[1] != recording_id:
                continue
            start, duration = float(fields[3]), float(fields[4])
            intervals.append({"start": start, "end": start + duration, "speaker": fields[7]})
    if recording_id is None and len(found_ids) != 1:
        raise ValueError("RTTM contains multiple recordings; choose --recording-id")
    if recording_id is not None and recording_id not in found_ids:
        raise ValueError("Recording id not found in RTTM")
    intervals.sort(key=lambda row: (row["start"], row["end"], row["speaker"]))
    _validate_intervals(intervals)
    write_jsonl(output_path, intervals)
    write_json(sidecar(output_path), {"rttm_sha256": sha256_file(rttm_path),
                                  "recording_id": recording_id or next(iter(found_ids)),
                                  "status": "automatic_unreviewed"})
    return len(intervals)


def _validate_intervals(intervals: list[dict]) -> None:
    if not intervals:
        raise ValueError("No diarization speaker intervals")
    for row in intervals:
        start, end = float(row["start"]), float(row["end"])
        if (not isinstance(row.get("speaker"), str) or not row["speaker"].strip()
                or not math.isfinite(start) or not math.isfinite(end)
                or start < 0 or end <= start):
            raise ValueError("Invalid diarization interval")


def _speaker(start: float, end: float, intervals: list[dict], collar: float = 0.0) -> str:
    """The one speaker whose intervals overlap the word and cover more than half of it.

    Words overlapping two speakers, or no speaker, stay unknown. Coverage counts a
    collar around that speaker's intervals, because ASR word times and speaker
    boundaries rarely agree to the fraction of a second.
    """
    duration = end - start
    if duration <= 0:
        return "unknown"
    overlaps = [row for row in intervals
                if min(end, float(row["end"])) > max(start, float(row["start"]))]
    names = {row["speaker"] for row in overlaps}
    if len(names) != 1:
        return "unknown"
    speaker = next(iter(names))
    spans = sorted((max(start, float(row["start"]) - collar), min(end, float(row["end"]) + collar))
                   for row in intervals if row["speaker"] == speaker
                   and min(end, float(row["end"]) + collar) > max(start, float(row["start"]) - collar))
    covered = 0.0
    current_start, current_end = spans[0]
    for next_start, next_end in spans[1:]:
        if next_start <= current_end:
            current_end = max(current_end, next_end)
        else:
            covered += current_end - current_start
            current_start, current_end = next_start, next_end
    covered += current_end - current_start
    return speaker if covered > duration / 2 else "unknown"


def align_speakers(transcript_path: Path, intervals_path: Path, output_path: Path,
                   collar: float = 0.25) -> dict:
    """Speaker turns from word times where available. Ambiguous or overlapping speech stays unknown."""
    if collar < 0:
        raise ValueError("collar must not be negative")
    transcript = read_turns(transcript_path)
    sessions = {(row["participant_id"], row["session_id"]) for row in transcript}
    if len(sessions) != 1:
        raise ValueError("Align one recording/session at a time")
    intervals = read_jsonl(intervals_path)
    _validate_intervals(intervals)
    output = []
    for segment in transcript:
        if "start" not in segment or "end" not in segment:
            raise ValueError("ASR turns need timestamps for speaker alignment")
        words = segment.get("words") or [{"start": segment["start"],
                                          "end": segment["end"], "text": segment["text"]}]
        groups: list[dict] = []
        for word in words:
            start, end = float(word["start"]), float(word["end"])
            if start < float(segment["start"]) - 0.1 or end > float(segment["end"]) + 0.1:
                raise ValueError("Word timestamps fall outside the ASR segment")
            speaker = _speaker(start, end, intervals, collar)
            if groups and groups[-1]["speaker"] == speaker:
                groups[-1]["end"] = end
                groups[-1]["text"] += str(word["text"])
            else:
                groups.append({"start": start, "end": end, "speaker": speaker,
                               "text": str(word["text"])})
        for group in groups:
            if group["text"].strip():
                output.append({
                    "participant_id": segment["participant_id"],
                    "session_id": segment["session_id"],
                    "turn_id": f"turn-{len(output) + 1:06d}",
                    "speaker": group["speaker"],
                    "text": group["text"].strip(),
                    "start": group["start"],
                    "end": group["end"],
                    "asr_turn_id": segment["turn_id"],
                })
    if not output:
        raise ValueError("No aligned turns")
    write_jsonl(output_path, output)
    report = {"turns": len(output),
              "unknown_speaker_turns": sum(row["speaker"] == "unknown" for row in output),
              "asr_sha256": sha256_file(transcript_path),
              "diarization_sha256": sha256_file(intervals_path),
              "collar_seconds": collar,
              "status": "automatic_unreviewed"}
    write_json(sidecar(output_path), report)
    return report

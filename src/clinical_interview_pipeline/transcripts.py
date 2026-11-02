"""Local ASR, speaker handoffs, and audio-checked transcript approval."""

from __future__ import annotations

import csv
import difflib
import json
import os
import re
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from .records import read_csv, read_turns, sha256_file, sidecar, write_json, write_jsonl


def transcribe_audio(
    audio_path: Path,
    model_directory: Path,
    output_path: Path,
    participant_id: str,
    session_id: str,
    language: str | None = None,
    device: str = "cpu",
    compute_type: str = "auto",
    cpu_threads: int = 0,
    beam_size: int = 5,
) -> int:
    """Transcribe with a locally installed faster-whisper model directory.

    device="cuda" runs on the single GPU exposed by the safety wrapper
    (cilp transcribe --gpu). cpu_threads=0 keeps the library default. Speaker
    labels remain unknown, so diarize or import reviewed speaker labels next.
    """
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and os.environ.get("CILP_SAFE_GPU_WRAPPER") != "1":
        raise RuntimeError("Run GPU transcription through cilp transcribe --gpu (safety wrapper)")
    if beam_size < 1 or cpu_threads < 0:
        raise ValueError("beam_size must be positive and cpu_threads non-negative")
    if not audio_path.is_file() or not model_directory.is_dir():
        raise FileNotFoundError("Audio and local ASR model directory are required")
    try:
        from faster_whisper import WhisperModel
    except ImportError as error:
        raise RuntimeError("Install the optional asr dependencies first") from error

    model = WhisperModel(str(model_directory), device=device, compute_type=compute_type,
                         cpu_threads=cpu_threads, local_files_only=True)
    segments, _ = model.transcribe(
        str(audio_path), language=language, beam_size=beam_size,
        vad_filter=True, word_timestamps=True
    )
    rows = [
        {
            "participant_id": participant_id,
            "session_id": session_id,
            "turn_id": f"asr-{index:06d}",
            "speaker": "unknown",
            "text": segment.text.strip(),
            "start": float(segment.start),
            "end": float(segment.end),
            "words": [
                {"start": float(word.start), "end": float(word.end), "text": word.word}
                for word in (segment.words or [])
                if word.start is not None and word.end is not None and word.word.strip()
            ],
        }
        for index, segment in enumerate(segments, start=1)
        if segment.text.strip()
    ]
    if not rows:
        raise ValueError("ASR returned no nonempty segments")
    write_jsonl(output_path, rows)
    write_json(sidecar(output_path), {
        "audio_sha256": sha256_file(audio_path),
        "local_model_path": str(model_directory.resolve()),
        "faster_whisper_version": version("faster-whisper"),
        "language": language, "device": device, "compute_type": compute_type,
        "cpu_threads": cpu_threads, "beam_size": beam_size,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES") if device == "cuda" else None,
        "status": "automatic_unreviewed",
    })
    return len(rows)


def import_tsv(
    input_path: Path,
    output_path: Path,
    participant_id: str,
    session_id: str,
    text_column: str = "segment",
    speaker_column: str = "speaker",
    start_column: str = "start",
    end_column: str = "end",
) -> int:
    with input_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {text_column, speaker_column}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"TSV needs columns {sorted(required)}")
        rows = []
        for index, item in enumerate(reader, start=1):
            text = (item.get(text_column) or "").strip()
            if not text:
                continue
            row = {
                "participant_id": participant_id,
                "session_id": session_id,
                "turn_id": f"turn-{index:06d}",
                "speaker": (item.get(speaker_column) or "unknown").strip(),
                "text": text,
            }
            if item.get(start_column):
                row["start"] = float(item[start_column])
            if item.get(end_column):
                row["end"] = float(item[end_column])
            rows.append(row)
    if not rows:
        raise ValueError("No nonempty transcript turns")
    write_jsonl(output_path, rows)
    write_json(sidecar(output_path), {
        "source_tsv_sha256": sha256_file(input_path),
        "columns": {"text": text_column, "speaker": speaker_column,
                    "start": start_column, "end": end_column},
        "status": "imported_unreviewed",
    })
    return len(rows)


def correction_summary(before_path: Path, after_path: Path) -> dict[str, int | float]:
    """Count turns, speakers, and tokens changed between two versions, matched by turn ID."""
    before = read_turns(before_path)
    after = read_turns(after_path)
    before_by_key = {
        (row["participant_id"], row["session_id"], row["turn_id"]): row
        for row in before
    }
    after_by_key = {
        (row["participant_id"], row["session_id"], row["turn_id"]): row
        for row in after
    }
    if before_by_key.keys() != after_by_key.keys():
        raise ValueError("Correction changed turn identities; supply an explicit mapping")
    changed = 0
    changed_speakers = 0
    tokens_before = 0
    tokens_after = 0
    inserted = deleted = replaced = unchanged = 0
    for key, first in before_by_key.items():
        second = after_by_key[key]
        old = re.findall(r"\w+|\S", first["text"])
        new = re.findall(r"\w+|\S", second["text"])
        changed += first["text"] != second["text"]
        changed_speakers += first["speaker"] != second["speaker"]
        tokens_before += len(old)
        tokens_after += len(new)
        matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                unchanged += i2 - i1
            elif tag == "insert":
                inserted += j2 - j1
            elif tag == "delete":
                deleted += i2 - i1
            elif tag == "replace":
                replaced += max(i2 - i1, j2 - j1)
    return {
        "turns": len(before),
        "changed_text_turns": changed,
        "changed_speaker_turns": changed_speakers,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "token_edit_operations": inserted + deleted + replaced,
        "inserted_tokens": inserted,
        "deleted_tokens": deleted,
        "replaced_tokens": replaced,
        "unchanged_tokens": unchanged,
    }


def map_speakers(transcript_path: Path, mapping_path: Path, output_path: Path) -> int:
    """Apply reviewed recording-specific speaker-to-role mappings."""
    mapping: dict[str, str] = {}
    for row in read_csv(mapping_path, {"diarized_speaker", "role"}):
        source, role = row["diarized_speaker"].strip(), row["role"].strip()
        if (not source or source == "unknown" or not role or role == "unknown"
                or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", role)):
            raise ValueError("Speaker mapping needs nonempty speaker ids and role names")
        if source in mapping:
            raise ValueError(f"Duplicate mapping for {source}")
        mapping[source] = role
    if not mapping:
        raise ValueError("Speaker mapping is empty")
    turns = read_turns(transcript_path)
    used = {row["speaker"] for row in turns if row["speaker"] != "unknown"}
    if used - set(mapping):
        raise ValueError(f"Unmapped diarized speakers: {sorted(used - set(mapping))}")
    output = [{**row, "speaker": mapping.get(row["speaker"], "unknown")}
              for row in turns]
    write_jsonl(output_path, output)
    write_json(sidecar(output_path), {
        "input_sha256": sha256_file(transcript_path),
        "mapping_sha256": sha256_file(mapping_path),
        "status": "role_mapped_unreviewed",
    })
    return len(output)


def approve_transcript(
    before_path: Path, corrected_path: Path, output_path: Path,
    reviewer: str, expected_speakers: list[str], attested: bool,
    source_kind: str = "audio",
    audio_path: Path | None = None,
) -> dict:
    """Record a human source review, distinguishing audio from text or fiction."""
    if not attested or not reviewer.strip():
        raise ValueError("Reviewer and explicit source-review attestation are required")
    if source_kind not in {"audio", "transcript", "synthetic"}:
        raise ValueError("source_kind must be audio, transcript, or synthetic")
    if source_kind == "audio" and (audio_path is None or not audio_path.is_file()):
        raise FileNotFoundError("Audio approval requires the reviewed source recording")
    if source_kind != "audio" and audio_path is not None:
        raise ValueError("Only audio-source approval may include an audio file")
    if not expected_speakers or any(not role.strip() for role in expected_speakers):
        raise ValueError("Declare the expected speaker roles")
    summary = correction_summary(before_path, corrected_path)
    rows = read_turns(corrected_path)
    speakers = {row["speaker"] for row in rows}
    if "unknown" in speakers or speakers - set(expected_speakers):
        raise ValueError(f"Unreviewed or unexpected speakers: {sorted(speakers - set(expected_speakers))}")
    if any(not row["text"].strip() for row in rows):
        raise ValueError("Approved turns cannot be empty")
    receipt = {
        "status": "transcript_approved",
        "reviewer": reviewer.strip(),
        "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_kind": source_kind,
        "source_review_attested": True,
        "audio_review_attested": source_kind == "audio",
        "audio_path": str(audio_path.resolve()) if audio_path else None,
        "audio_sha256": sha256_file(audio_path) if audio_path else None,
        "source_path": str(before_path.resolve()),
        "source_sha256": sha256_file(before_path),
        "corrected_sha256": sha256_file(corrected_path),
        "expected_speakers": sorted(set(expected_speakers)),
        "corrections": summary,
    }
    write_json(output_path, receipt)
    return receipt


def verify_transcript_approval(transcript_path: Path, approval_path: Path) -> None:
    receipt = json.loads(approval_path.read_text(encoding="utf-8"))
    if (receipt.get("status") != "transcript_approved"
            or receipt.get("source_review_attested") is not True
            or receipt.get("source_kind") not in {"audio", "transcript", "synthetic"}
            or receipt.get("corrected_sha256") != sha256_file(transcript_path)):
        raise ValueError("Corrected transcript does not match its source-review approval")
    if sha256_file(Path(receipt["source_path"])) != receipt["source_sha256"]:
        raise ValueError("Source transcript changed after review")
    if receipt["source_kind"] == "audio":
        if (receipt.get("audio_review_attested") is not True
                or sha256_file(Path(receipt["audio_path"])) != receipt["audio_sha256"]):
            raise ValueError("Reviewed audio changed after transcript approval")
    rows = read_turns(transcript_path)
    if any(row["speaker"] not in receipt.get("expected_speakers", []) for row in rows):
        raise ValueError("Approved transcript contains an unreviewed speaker role")

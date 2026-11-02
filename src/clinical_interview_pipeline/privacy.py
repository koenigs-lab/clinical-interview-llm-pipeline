"""De-identification: proposed identifiers, human decisions, sign-off.

The local-LLM search (privacy_llm) and the optional pattern pass propose
identifier strings. Each distinct string of a transcript becomes one row of a
mapping file with a typed token (PERSON_0001, ...). A reviewer decides every
row (anonymize 1 or 0), may edit tokens, and may add rows the search missed.
Chosen strings are then replaced at every occurrence in that transcript.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .privacy_llm import (CATEGORIES, DEFAULT_CATEGORIES, DEFAULT_ENDPOINT, loopback_endpoint,
                          search_identifiers, served_models)
from .records import read_csv, read_turns, sha256_file, sidecar, write_csv, write_json, write_jsonl
from .transcripts import verify_transcript_approval


# Optional deterministic pass. The LLM search is the primary detector.
PATTERNS = {
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "PHONE": re.compile(r"(?<!\w)(?:\+\d{1,3}[ .-]?)?(?:\(?\d{2,5}\)?[ .-]?){2,4}\d{2,5}(?!\w)"),
    "URL": re.compile(r"\bhttps?://[^\s]+"),
}
MAPPING_COLUMNS = ("participant_id", "session_id", "type", "original", "token",
                   "occurrences", "anonymize")


def _entry_hash(participant: str, session: str, original: str) -> str:
    """Identity of a mapping row without storing the identifier itself."""
    return hashlib.sha256("|".join((participant, session, original)).encode("utf-8")).hexdigest()


def _transcripts(turns: list[dict]) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in turns:
        grouped[(row["participant_id"], row["session_id"])].append(row)
    return grouped


def propose(
    transcript_path: Path,
    mapping_path: Path,
    transcript_approval_path: Path,
    model: str | None = None,
    endpoint: str = DEFAULT_ENDPOINT,
    patterns: bool = False,
    categories: tuple[str, ...] = DEFAULT_CATEGORIES,
    context_tokens: int = 4096,
    rows_per_batch: int = 20,
) -> dict[str, Any]:
    """Write the reviewable mapping: one row per distinct proposed string per transcript."""
    if not model and not patterns:
        raise ValueError("Choose the local LLM search (--model), the pattern pass "
                         "(--patterns), or both")
    if set(categories) - set(CATEGORIES):
        raise ValueError(f"Unknown identifier categories: {sorted(set(categories) - set(CATEGORIES))}")
    # Validate the endpoint before reading any transcript content, and record the model
    # the server reports, because the server decides which model answers.
    address = loopback_endpoint(endpoint) if model else None
    served = served_models(address) if model else None
    if served is not None and model not in served:
        raise ValueError(f"--model names no model this server serves; it reports: {served}")
    verify_transcript_approval(transcript_path, transcript_approval_path)
    turns = read_turns(transcript_path)
    transcripts = _transcripts(turns)
    proposals: list[tuple[tuple[str, str], str, str]] = []
    counts: dict[str, int] = {}
    if model:
        found, counts = search_identifiers(transcripts, model, address, categories,
                                           context_tokens, rows_per_batch)
        proposals += [(key, CATEGORIES[category], value) for key, category, value in found]
    if patterns:
        proposals += [(key, kind, match.group())
                      for key, rows in transcripts.items() for row in rows
                      for kind, pattern in PATTERNS.items()
                      for match in pattern.finditer(row["text"])]
    entries = []
    seen: set[tuple[str, str, str]] = set()
    numbers: Counter[tuple[str, str, str]] = Counter()
    for (participant, session), kind, original in proposals:
        if (participant, session, original) in seen:
            continue
        seen.add((participant, session, original))
        numbers[(participant, session, kind)] += 1
        entries.append({
            "participant_id": participant, "session_id": session, "type": kind,
            "original": original,
            "token": f"{kind}_{numbers[(participant, session, kind)]:04d}",
            "occurrences": sum(row["text"].count(original)
                               for row in transcripts[(participant, session)]),
            "anonymize": "",
        })
    entries.sort(key=lambda row: (row["participant_id"], row["session_id"], row["type"],
                                  row["original"]))
    write_csv(mapping_path, list(MAPPING_COLUMNS), entries)
    write_json(sidecar(mapping_path), {
        "transcript_sha256": sha256_file(transcript_path),
        "transcript_approval_path": str(Path(transcript_approval_path).resolve()),
        "transcript_approval_sha256": sha256_file(transcript_approval_path),
        "detectors": {
            "llm": ({"model": model, "served_models": served, "endpoint": endpoint,
                     "categories": list(categories), "context_tokens": context_tokens,
                     "rows_per_batch": rows_per_batch} if model else None),
            "patterns": patterns,
        },
        **counts,
        "n_turns": len(turns),
        "n_entries": len(entries),
        "entries_not_found_in_text": sum(row["occurrences"] == 0 for row in entries),
        "proposed_entries": sorted(_entry_hash(row["participant_id"], row["session_id"],
                                               row["original"]) for row in entries),
        "status": "proposed_not_approved",
    })
    return {"turns": len(turns), "entries": len(entries), **counts}


def _replace_all(text: str, mapping: dict[str, str]) -> tuple[str, int]:
    """One pass over the text, in which longer originals win where they overlap."""
    pattern = re.compile("|".join(re.escape(original)
                                  for original in sorted(mapping, key=len, reverse=True)))
    return pattern.subn(lambda match: mapping[match.group()], text)


def apply_reviews(transcript_path: Path, mapping_path: Path, output_path: Path) -> dict[str, Any]:
    transcript_path = Path(transcript_path)
    mapping_path = Path(mapping_path)
    meta = json.loads(sidecar(mapping_path).read_text(encoding="utf-8"))
    if not meta.get("transcript_approval_path") or not meta.get("transcript_approval_sha256"):
        raise ValueError("Mapping metadata does not bind a transcript approval; "
                         "rerun privacy-propose")
    if meta["transcript_sha256"] != sha256_file(transcript_path):
        raise ValueError("Transcript changed after identifiers were proposed")
    transcript_approval_path = Path(meta["transcript_approval_path"])
    if sha256_file(transcript_approval_path) != meta["transcript_approval_sha256"]:
        raise ValueError("Transcript approval changed after identifiers were proposed")
    verify_transcript_approval(transcript_path, transcript_approval_path)
    turns = read_turns(transcript_path)
    transcripts = _transcripts(turns)
    chosen: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    decided: set[str] = set()
    kept = 0
    for row in read_csv(mapping_path, {"participant_id", "session_id", "type", "original",
                                       "token", "anonymize"}):
        key = (row["participant_id"].strip(), row["session_id"].strip())
        original = row["original"].strip()
        if key not in transcripts or not original:
            raise ValueError("Every mapping row needs a transcript of this file and an original text")
        identity = _entry_hash(*key, original)
        if identity in decided:
            raise ValueError("Each original text may appear only once per transcript")
        decided.add(identity)
        decision = row["anonymize"].strip()
        if decision not in {"0", "1"}:
            raise ValueError("Set anonymize to 1 (replace) or 0 (keep) in every row")
        if decision == "0":
            kept += 1
        elif not row["token"].strip():
            raise ValueError("A replaced text needs a token")
        else:
            chosen[key][original] = row["token"].strip()
    removed = set(meta["proposed_entries"]) - decided
    if removed:
        raise ValueError(f"{len(removed)} proposed rows are missing; decide every proposal "
                         "with anonymize 1 or 0 instead of deleting it")
    output: list[dict] = []
    replacements = 0
    for row in turns:
        mapping = chosen.get((row["participant_id"], row["session_id"]))
        text = row["text"]
        if mapping:
            text, count = _replace_all(text, mapping)
            replacements += count
        output.append({**row, "text": text})
    write_jsonl(output_path, output)
    receipt = {
        "status": "pending_final_review",
        "source_path": str(transcript_path.resolve()),
        "source_sha256": sha256_file(transcript_path),
        "transcript_approval_path": str(transcript_approval_path),
        "transcript_approval_sha256": sha256_file(transcript_approval_path),
        "reviewed_mapping_sha256": sha256_file(mapping_path),
        "output_sha256": sha256_file(output_path),
        "n_proposed": len(meta["proposed_entries"]),
        "n_added_by_reviewer": len(decided - set(meta["proposed_entries"])),
        "n_replaced_texts": sum(len(mapping) for mapping in chosen.values()),
        "n_kept_texts": kept,
        "n_replacements": replacements,
        "warning": "Automated proposals do not establish de-identification. Review every output turn.",
    }
    write_json(sidecar(output_path, "receipt"), receipt)
    return receipt


def approve(
    output_path: Path, reviewer: str, approval_path: Path, attested: bool,
    require_distinct_reviewer: bool = False,
) -> None:
    if not attested or not reviewer.strip():
        raise ValueError("Explicit final-review attestation and reviewer id are required")
    receipt = json.loads(sidecar(output_path, "receipt").read_text(encoding="utf-8"))
    if receipt["status"] != "pending_final_review":
        raise ValueError("Unexpected receipt status")
    if receipt["output_sha256"] != sha256_file(output_path):
        raise ValueError("Transcript changed after review application")
    transcript_reviewer = None
    if receipt.get("transcript_approval_path"):
        upstream = Path(receipt["transcript_approval_path"])
        if upstream.is_file():
            transcript_reviewer = json.loads(upstream.read_text(encoding="utf-8")).get("reviewer")
    if require_distinct_reviewer and transcript_reviewer == reviewer.strip():
        raise ValueError("Privacy reviewer must differ from the transcript reviewer")
    write_json(
        approval_path,
        {
            "status": "approved",
            "reviewer": reviewer,
            "transcript_reviewer": transcript_reviewer,
            "distinct_reviewers": transcript_reviewer is not None
                                  and transcript_reviewer != reviewer.strip(),
            "approved_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_sha256": receipt["output_sha256"],
            "receipt_sha256": sha256_file(sidecar(output_path, "receipt")),
        },
    )


def verify_approval(output_path: Path, approval_path: Path) -> None:
    approval = json.loads(Path(approval_path).read_text(encoding="utf-8"))
    if approval.get("status") != "approved":
        raise ValueError("Transcript has no final privacy approval")
    if approval.get("output_sha256") != sha256_file(output_path):
        raise ValueError("Approved transcript hash does not match")
    receipt_path = sidecar(output_path, "receipt")
    if approval.get("receipt_sha256") != sha256_file(receipt_path):
        raise ValueError("Privacy receipt changed after approval")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not all(receipt.get(field) for field in
               ("source_path", "source_sha256", "transcript_approval_path",
                "transcript_approval_sha256")):
        raise ValueError("Privacy receipt does not bind the transcript review; "
                         "rerun privacy-apply and privacy-approve")
    transcript_approval = Path(receipt["transcript_approval_path"])
    source = Path(receipt["source_path"])
    if sha256_file(source) != receipt["source_sha256"]:
        raise ValueError("Source transcript changed")
    if sha256_file(transcript_approval) != receipt["transcript_approval_sha256"]:
        raise ValueError("Source transcript approval changed")
    verify_transcript_approval(source, transcript_approval)

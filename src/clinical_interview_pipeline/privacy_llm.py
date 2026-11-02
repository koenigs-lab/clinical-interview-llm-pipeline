"""Local-LLM identifier search over batches of consecutive turns.

Each request carries as many consecutive turns of one transcript as fit the
context budget (at most rows_per_batch), so the model sees neighbouring turns.
The model returns identifier strings per turn as JSON. Replacement follows the
human review, in privacy.apply_reviews.
"""

from __future__ import annotations

import http.client
import json
import re
import time
from collections import Counter
from urllib.parse import urlsplit

# Identifier category -> replacement token prefix. The first six are the default
# categories, and the indirect ones are optional.
CATEGORIES = {
    "persons": "PERSON", "emails": "EMAIL", "phones": "PHONE", "dates": "DATE",
    "addresses": "ADDRESS", "ids": "ID",
    "organizations": "ORGANIZATION", "locations": "LOCATION",
    "occupations": "OCCUPATION", "rare_events": "EVENT",
}
DEFAULT_CATEGORIES = ("persons", "emails", "phones", "dates", "addresses", "ids")
INDIRECT_CATEGORIES = ("organizations", "locations", "occupations", "rare_events")
DEFAULT_ENDPOINT = "http://127.0.0.1:2929/v1"
SYSTEM_PROMPT = ("Return ONLY a valid JSON object that matches the provided JSON Schema. "
                 "No prose. No markdown. No code fences. No extra keys.")
COMPLETION_TOKENS = 1024
SAFETY_TOKENS = 384
ROW_OVERHEAD_TOKENS = 96
TRANSIENT_ATTEMPTS = 5


def loopback_endpoint(endpoint: str) -> tuple[str, int]:
    """Host and port of an OpenAI-compatible /v1 endpoint on this machine only."""
    parsed = urlsplit(endpoint)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "::1"}
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path.rstrip("/") != "/v1"):
        raise ValueError("Identifier detector endpoint must be a loopback HTTP /v1 URL")
    return parsed.hostname, parsed.port or 80


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def _schema(categories: tuple[str, ...]) -> dict:
    row = {"row_id": {"type": "integer"}}
    row.update({name: {"type": "array", "items": {"type": "string"}} for name in categories})
    return {"type": "object", "required": ["rows"], "additionalProperties": False,
            "properties": {"rows": {"type": "array", "items": {
                "type": "object", "properties": row, "required": list(row),
                "additionalProperties": False}}}}


def _instructions(categories: tuple[str, ...]) -> str:
    return ("Extract PII per row.\nOutput JSON only.\nTop key: rows.\n"
            f"Each row object: row_id, {', '.join(categories)}.\n"
            "No other keys.\nUse [] when none.\n"
            "Copy every value exactly as it is written in the row.\nINPUT:")


def _answer_rows(categories: tuple[str, ...]) -> int:
    """Rows whose JSON answer, about one line per category, fits the reserved answer tokens."""
    return max(1, (COMPLETION_TOKENS - 64) // (12 + 8 * len(categories)))


def batches(texts: list[str], categories: tuple[str, ...], context_tokens: int,
            rows_per_batch: int) -> list[list[int]]:
    """Consecutive turn indices per request. A turn larger than the budget goes alone.

    A batch holds at most rows_per_batch turns, and no more than an answer for
    every category of every turn can fit the reserved answer tokens.
    """
    if rows_per_batch < 1:
        raise ValueError("rows_per_batch must be positive")
    rows_per_batch = min(rows_per_batch, _answer_rows(categories))
    overhead = _approx_tokens(SYSTEM_PROMPT + json.dumps(_schema(categories))
                              + _instructions(categories))
    budget = context_tokens - overhead - COMPLETION_TOKENS - SAFETY_TOKENS
    if budget < 64:
        raise ValueError("context_tokens is too small for the prompt and the reserved answer")
    groups: list[list[int]] = []
    current: list[int] = []
    used = 0
    for index, text in enumerate(texts):
        need = _approx_tokens(text) + ROW_OVERHEAD_TOKENS
        if current and (used + need > budget or len(current) >= rows_per_batch):
            groups.append(current)
            current, used = [], 0
        current.append(index)
        used += need
    if current:
        groups.append(current)
    return groups


def _http(address: tuple[str, int], method: str, path: str,
          payload: dict | None = None) -> tuple[int, bytes]:
    connection = http.client.HTTPConnection(*address, timeout=600)
    try:
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        connection.request(method, path, body=body, headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, response.read(4_000_000)
    finally:
        connection.close()


def _retrying(call):
    """Repeat a request while the local server starts, loads, or is busy."""
    wait = 10.0
    for attempt in range(TRANSIENT_ATTEMPTS):
        try:
            return call()
        except OSError:
            if attempt + 1 == TRANSIENT_ATTEMPTS:
                raise
            time.sleep(wait)
            wait = min(wait * 1.5, 60.0)
    raise AssertionError("unreachable")


def _post(address: tuple[str, int], model: str, messages: list[dict]) -> tuple[str, str]:
    """Reply text and finish reason ("length" when the answer was cut off)."""
    status, body = _http(address, "POST", "/v1/chat/completions",
                         {"model": model, "temperature": 0, "max_tokens": COMPLETION_TOKENS,
                          "messages": messages})
    if status in {429, 500, 502, 503, 504}:
        raise ConnectionError(f"Local model busy or loading (HTTP {status})")
    if status != 200:
        raise RuntimeError(f"Local model returned HTTP {status}; no transcript text logged")
    choice = json.loads(body)["choices"][0]
    return choice["message"]["content"], str(choice.get("finish_reason"))


def _chat(address: tuple[str, int], model: str, messages: list[dict]) -> tuple[str, str]:
    return _retrying(lambda: _post(address, model, messages))


def served_models(address: tuple[str, int]) -> list[str] | None:
    """Model names the local server reports at /v1/models, or None if it does not say."""
    status, body = _retrying(lambda: _http(address, "GET", "/v1/models"))
    if status != 200:
        return None
    try:
        data = json.loads(body).get("data") or []
    except (ValueError, AttributeError):
        return None
    return [str(item["id"]) for item in data if isinstance(item, dict) and item.get("id")]


def _json_rows(content: str) -> dict:
    """The JSON reply, tolerating code fences, trailing commas, and a bare list of rows."""
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"\s*```$", "", re.sub(r"^```(?:json)?\s*", "", text))
    starts = [index for index in (text.find("{"), text.find("[")) if index != -1]
    if starts:
        text = text[min(starts):max(text.rfind("}"), text.rfind("]")) + 1]
    try:
        value = json.loads(text)
    except ValueError:
        value = json.loads(re.sub(r",\s*([}\]])", r"\1", text))
    if isinstance(value, list):
        value = {"rows": value}
    if not isinstance(value, dict):
        raise ValueError("Reply is not a JSON object")
    return value


def _search_rows(address: tuple[str, int], model: str, categories: tuple[str, ...],
                 rows: list[tuple[int, str]], counts: Counter) -> dict[int, dict]:
    """Identifiers per turn for consecutive turns.

    An unreadable answer is sent back for repair. An answer cut off at the
    length limit cannot be repaired, so the batch is halved and asked again.
    """
    schema = json.dumps(_schema(categories), ensure_ascii=False, separators=(",", ":"))
    text = _instructions(categories) + json.dumps(
        [[index, value] for index, value in rows], ensure_ascii=False, separators=(",", ":"))
    system = {"role": "system", "content": SYSTEM_PROMPT}
    content, finish = _chat(address, model, [system, {"role": "user",
                                                      "content": "SCHEMA:" + schema + "\nTEXT:" + text}])
    for attempt in range(3):
        try:
            value = _json_rows(content)
            break
        except ValueError:
            if finish == "length" or attempt == 2:
                if len(rows) > 1:
                    counts["batches_split"] += 1
                    half = len(rows) // 2
                    return {**_search_rows(address, model, categories, rows[:half], counts),
                            **_search_rows(address, model, categories, rows[half:], counts)}
                raise ValueError("Local model did not return the identifier JSON object; "
                                 "no transcript text logged") from None
            content, finish = _chat(address, model, [system, {"role": "user", "content": (
                "Fix the JSON to match SCHEMA exactly. Output JSON only.\nSCHEMA:" + schema
                + "\nTEXT:" + text + "\nBAD_JSON:" + content)}])
    return {row["row_id"]: row for row in value.get("rows") or []
            if isinstance(row, dict) and isinstance(row.get("row_id"), int)}


def search_identifiers(transcripts: dict[tuple[str, str], list[dict]], model: str,
                       address: tuple[str, int], categories: tuple[str, ...],
                       context_tokens: int, rows_per_batch: int):
    """Identifier strings per transcript as (transcript key, category, text), plus counts."""
    found: list[tuple[tuple[str, str], str, str]] = []
    counts: Counter[str] = Counter()
    for key, turns in transcripts.items():
        texts = [turn["text"] for turn in turns]
        for batch in batches(texts, categories, context_tokens, rows_per_batch):
            answer = _search_rows(address, model, categories, [(i, texts[i]) for i in batch], counts)
            counts["batches"] += 1
            counts["turns_missing_from_model_output"] += sum(i not in answer for i in batch)
            for index in batch:
                row = answer.get(index, {})
                for category in categories:
                    values = row.get(category)
                    if isinstance(values, str):
                        values = [values]
                    if not isinstance(values, list):
                        continue
                    found.extend((key, category, value.strip()) for value in values
                                 if isinstance(value, str) and value.strip())
    return found, dict(counts)

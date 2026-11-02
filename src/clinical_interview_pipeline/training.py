"""Optional local LoRA training and inference, with models and data read from local paths."""

from __future__ import annotations

import inspect
import math
import os
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .records import read_jsonl, sha256_file, sidecar, write_csv, write_json
from .task import load_task


def _dependencies() -> SimpleNamespace:
    """The optional training stack, imported only when a GPU job runs."""
    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        raise RuntimeError("Install the optional train dependencies (pip install '.[train]')") from error
    return SimpleNamespace(
        torch=torch, LoraConfig=LoraConfig, PeftModel=PeftModel, get_peft_model=get_peft_model,
        prepare_model_for_kbit_training=prepare_model_for_kbit_training,
        AutoModelForCausalLM=AutoModelForCausalLM, AutoTokenizer=AutoTokenizer,
        BitsAndBytesConfig=BitsAndBytesConfig,
    )


def _load_model(deps: SimpleNamespace, model_path: Path, quantization: str, training: bool):
    """Tokenizer and model from a local directory, on the one GPU the job may use."""
    torch = deps.torch
    if quantization not in {"4bit", "none"}:
        raise ValueError("quantization must be 4bit or none")
    if not model_path.is_dir():
        raise FileNotFoundError(f"Local model directory not found: {model_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("Training and prediction require a CUDA GPU")
    tokenizer = deps.AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    options: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": False,
        "device_map": {"": 0},
    }
    if quantization == "4bit":
        options["quantization_config"] = deps.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        options["torch_dtype"] = compute_dtype
    model = deps.AutoModelForCausalLM.from_pretrained(str(model_path), **options)
    if training and quantization == "4bit":
        model = deps.prepare_model_for_kbit_training(
            model, gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return tokenizer, model


def _prompt(tokenizer, row: dict) -> list[int]:
    if [item.get("role") for item in row.get("messages", [])] != ["system", "user", "assistant"]:
        raise ValueError("Every example needs system, user, and assistant messages")
    messages = row["messages"][:-1]
    if not callable(getattr(tokenizer, "apply_chat_template", None)):
        raise ValueError("The local tokenizer needs a chat template")
    try:
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        if hasattr(encoded, "keys"):
            encoded = encoded["input_ids"]
        if not isinstance(encoded, list) or any(not isinstance(token, int) for token in encoded):
            raise ValueError("Chat template returned an unexpected token format")
        return encoded
    except (ValueError, TypeError) as error:
        raise ValueError("The local model needs a usable chat template") from error


OBJECTIVES = ("token_ce", "label_rps")


def _encode(tokenizer, rows: list[dict], max_length: int, objective: str,
            labels: tuple[int, ...]):
    """Training items (row, input ids, target).

    token_ce: next-token cross-entropy on the answer tokens and end-of-sequence,
    with the prompt masked. label_rps: the prompt alone and the index of the
    reference label, scored at the answer position like prediction.
    """
    encoded = []
    skipped: list[dict] = []
    for row in rows:
        prefix = _prompt(tokenizer, row)
        if objective == "label_rps":
            ids, target = prefix, labels.index(int(row["label"]))
        else:
            answer = tokenizer.encode(
                str(row["messages"][-1]["content"]), add_special_tokens=False
            ) + [tokenizer.eos_token_id]
            ids, target = prefix + answer, [-100] * len(prefix) + answer
        if len(ids) > max_length:
            skipped.append(row)
            continue
        encoded.append((row, ids, target))
    return encoded, skipped


def _skip_report(skipped: list[dict]) -> dict[str, Any]:
    """Too-long exclusions by target and label."""
    by_target: dict[str, dict[str, int]] = {}
    for row in skipped:
        counts = by_target.setdefault(str(row["target_id"]), {})
        label = str(row.get("label", ""))
        counts[label] = counts.get(label, 0) + 1
    return {"n": len(skipped), "by_target_and_label": by_target}


def local_tokenizer(model_path: Path):
    """The tokenizer of a local model, loaded on the CPU for the length check."""
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Install the optional train dependencies (pip install '.[train]'), "
                           "or plan with --skip-length-check") from error
    if not model_path.is_dir():
        raise FileNotFoundError(f"Local model directory not found: {model_path}")
    return AutoTokenizer.from_pretrained(str(model_path), local_files_only=True,
                                         trust_remote_code=False)


def too_long_report(job: dict[str, Any], labels: tuple[int, ...], tokenizer) -> dict[str, Any]:
    """Examples longer than the job's max_length, counted before any GPU job runs.

    Training skips such training examples. Prediction marks such validation
    examples too_long, which counts as the worst possible error.
    """
    _, train_skipped = _encode(tokenizer, read_jsonl(job["train_path"]), job["max_length"],
                               job["objective"], labels)
    unscorable = [row for row in read_jsonl(job["validation_path"])
                  if len(_prompt(tokenizer, row)) > job["max_length"]]
    return {"max_length": job["max_length"], "train": _skip_report(train_skipped),
            "validation": _skip_report(unscorable)}


def _batch(torch, tokenizer, items):
    width = max(len(item[1]) for item in items)
    ids = [item[1] + [tokenizer.pad_token_id] * (width - len(item[1])) for item in items]
    attention = [[1] * len(item[1]) + [0] * (width - len(item[1])) for item in items]
    labels = [item[2] + [-100] * (width - len(item[1])) for item in items]
    return {
        "input_ids": torch.tensor(ids, device="cuda"),
        "attention_mask": torch.tensor(attention, device="cuda"),
        "labels": torch.tensor(labels, device="cuda"),
    }


def _label_rps_loss(torch, model, chunk, label_ids: list[int], logit_kwargs: dict):
    """Mean ranked probability score of the renormalised label-token distribution.

    Same quantity as evaluation.ranked_probability_score, so the objective
    respects the ordering of the scale instead of treating labels as nominal.
    """
    boundaries = len(label_ids) - 1
    losses = []
    for _, prompt, target in chunk:
        logits = _last_token_logits(model, torch, prompt, logit_kwargs)
        probabilities = torch.softmax(logits.float()[label_ids], dim=-1)
        cumulative = torch.cumsum(probabilities, dim=-1)[:-1]
        observed = (torch.arange(boundaries, device=cumulative.device) >= target).float()
        losses.append(((cumulative - observed) ** 2).sum() / boundaries)
    return torch.stack(losses).mean()


def train_job(job: dict[str, Any]) -> dict[str, Any]:
    if os.environ.get("CILP_SAFE_GPU_WRAPPER") != "1":
        raise RuntimeError("Start training through cilp train and the GPU safety wrapper")
    deps = _dependencies()
    torch = deps.torch
    run_dir = Path(job["run_dir"])
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    task = load_task(job["task_path"])
    allowed = set(job["target_ids"])
    if not allowed <= set(task.target_map):
        raise ValueError("Job references unknown targets")
    objective = job.get("objective", "token_ce")
    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown training objective: {objective}")
    train = [row for row in read_jsonl(job["train_path"]) if row["target_id"] in allowed]
    validation = [row for row in read_jsonl(job["validation_path"]) if row["target_id"] in allowed]
    if not train or not validation:
        raise ValueError("Job needs train and validation examples for its targets")
    random.seed(job["seed"])
    torch.manual_seed(job["seed"])
    torch.cuda.manual_seed_all(job["seed"])
    tokenizer, model = _load_model(deps, Path(job["model_path"]), job["quantization"],
                                   training=True)
    label_ids = label_token_ids(tokenizer, task.labels) if objective == "label_rps" else []
    train_encoded, train_skipped = _encode(tokenizer, train, job["max_length"], objective,
                                           task.labels)
    val_encoded, val_skipped = _encode(tokenizer, validation, job["max_length"], objective,
                                       task.labels)
    if not train_encoded or not val_encoded:
        raise ValueError("All examples exceed max_length; use sections or a longer-context model")
    model.config.use_cache = False
    model = deps.get_peft_model(model, deps.LoraConfig(
        r=job["lora_r"],
        lora_alpha=job["lora_alpha"],
        lora_dropout=job["lora_dropout"],
        target_modules=job["target_modules"],
        task_type="CAUSAL_LM",
    ))
    logit_kwargs = _logit_kwargs(model)

    def loss_of(chunk):
        if objective == "label_rps":
            return _label_rps_loss(torch, model, chunk, label_ids, logit_kwargs)
        return model(**_batch(torch, tokenizer, chunk)).loss

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=job["learning_rate"],
    )
    batch_size = int(job["batch_size"])
    accumulation = int(job["gradient_accumulation"])
    randomizer = random.Random(job["seed"])
    history = []
    run_dir.mkdir(parents=True)
    for epoch in range(int(job["epochs"])):
        randomizer.shuffle(train_encoded)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        batches = math.ceil(len(train_encoded) / batch_size)
        for number in range(batches):
            chunk = train_encoded[number * batch_size:(number + 1) * batch_size]
            loss = loss_of(chunk)
            group_start = (number // accumulation) * accumulation
            group_size = min(accumulation, batches - group_start)
            (loss / group_size).backward()
            losses.append(float(loss.detach().cpu()))
            if (number + 1) % accumulation == 0 or number + 1 == batches:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        model.eval()
        validation_losses = []
        with torch.no_grad():
            for start in range(0, len(val_encoded), batch_size):
                chunk = val_encoded[start:start + batch_size]
                validation_losses.append(float(loss_of(chunk).detach().cpu()))
        history.append({
            "epoch": epoch + 1,
            "train_loss": sum(losses) / len(losses),
            "validation_loss": sum(validation_losses) / len(validation_losses),
        })
    model.save_pretrained(str(run_dir / "adapter"))
    tokenizer.save_pretrained(str(run_dir / "tokenizer"))
    report = {
        "job": job,
        "objective": objective,
        "train_examples": len(train_encoded),
        "validation_examples": len(val_encoded),
        "train_too_long": _skip_report(train_skipped),
        "validation_too_long": _skip_report(val_skipped),
        "history": history,
        "note": ("Validation loss is a training diagnostic only. Model selection uses the "
                 "prespecified validation task metric computed by cilp evaluate."),
    }
    write_json(run_dir / "result.json", report)
    return report


DECISION_RULE = "ordinal_median"


def label_token_ids(tokenizer, labels: tuple[int, ...]) -> list[int]:
    """Token id of each permitted label, which must match the first trained answer token."""
    ids = []
    for label in labels:
        encoded = tokenizer.encode(str(label), add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"Label {label!r} is not a single token for this tokenizer; "
                "label scoring needs single-token labels"
            )
        ids.append(int(encoded[0]))
    if len(set(ids)) != len(ids):
        raise ValueError("Two labels map to the same token id")
    return ids


def label_distribution(label_logprobs: list[float], labels: tuple[int, ...]) -> dict[str, Any]:
    """Renormalise full-vocabulary log-probabilities of the label tokens.

    ``label_mass`` is the probability the model placed on any permitted label
    before renormalisation, a format-compliance diagnostic reported
    alongside the prediction.
    """
    if len(label_logprobs) != len(labels) or not labels:
        raise ValueError("One log-probability per label is required")
    if not all(math.isfinite(value) for value in label_logprobs):
        raise ValueError("Label log-probabilities must be finite")
    peak = max(label_logprobs)
    weights = [math.exp(value - peak) for value in label_logprobs]
    total = sum(weights)
    probabilities = {label: weight / total for label, weight in zip(labels, weights)}
    return {"probabilities": probabilities, "label_mass": math.exp(peak) * total}


def ordinal_median(probabilities: dict[int, float]) -> int:
    """Smallest label with cumulative probability >= 0.5 (lower median).

    The median of the predictive distribution minimises expected absolute
    error, matching MAE-based selection and the lower-median constant baseline.
    """
    cumulative = 0.0
    ordered = sorted(probabilities)
    for label in ordered:
        cumulative += probabilities[label]
        if cumulative >= 0.5 - 1e-12:
            return label
    return ordered[-1]


def score_record(label_logprobs: list[float], labels: tuple[int, ...]) -> dict[str, Any]:
    distribution = label_distribution(label_logprobs, labels)
    probabilities = distribution["probabilities"]
    mode = max(probabilities, key=lambda label: (probabilities[label], -label))
    record: dict[str, Any] = {
        "prediction": ordinal_median(probabilities),
        "status": "scored",
        "label_mass": round(distribution["label_mass"], 6),
        "mode": mode,
        "expected_value": round(sum(label * p for label, p in probabilities.items()), 6),
    }
    record.update({f"p_{label}": round(p, 6) for label, p in probabilities.items()})
    return record


def prediction_columns(labels: tuple[int, ...]) -> list[str]:
    return (["participant_id", "target_id", "prediction", "status", "label_mass",
             "mode", "expected_value"] + [f"p_{label}" for label in labels])


def _logit_kwargs(model) -> dict[str, int]:
    """Request final-position logits only, under the keyword the model's forward accepts.

    Full-sequence logits of a long prompt can take gigabytes of GPU memory.
    PEFT wrappers pass extra keywords through, so the base model is inspected.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    try:
        parameters = inspect.signature(getattr(base, "forward", base)).parameters
    except (TypeError, ValueError):
        return {}
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in parameters:
            return {name: 1}
    return {}


def _last_token_logits(model, torch, prompt: list[int], logit_kwargs: dict):
    inputs = {
        "input_ids": torch.tensor([prompt], device="cuda"),
        "attention_mask": torch.ones((1, len(prompt)), device="cuda", dtype=torch.long),
    }
    return model(**inputs, **logit_kwargs).logits[0, -1]


def predict_job(job: dict[str, Any], data_path: Path, output_path: Path,
                use_adapter: bool = True, provenance: dict[str, Any] | None = None) -> int:
    """Score every permitted label token, without free-text generation or parsing.

    The prediction is the lower median of the renormalised label distribution.
    Prompts longer than the job's max_length are written with status
    ``too_long`` and an empty prediction, so evaluation counts them as invalid.
    """
    if os.environ.get("CILP_SAFE_GPU_WRAPPER") != "1":
        raise RuntimeError("Start prediction through cilp predict and the GPU safety wrapper")
    deps = _dependencies()
    torch = deps.torch
    adapter = Path(job["run_dir"]) / "adapter"
    if use_adapter and not adapter.is_dir():
        raise FileNotFoundError(f"Trained adapter missing: {adapter}")
    task = load_task(job["task_path"])
    tokenizer, model = _load_model(deps, Path(job["model_path"]), job["quantization"],
                                   training=False)
    if use_adapter:
        model = deps.PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.eval()
    label_ids = label_token_ids(tokenizer, task.labels)
    logit_kwargs = _logit_kwargs(model)
    rows = [
        row for row in read_jsonl(data_path)
        if row["target_id"] in set(job["target_ids"])
    ]
    records = []
    for row in rows:
        prompt = _prompt(tokenizer, row)
        record: dict[str, Any] = {
            "participant_id": row["participant_id"],
            "target_id": row["target_id"],
            "prediction": "",
            "status": "too_long",
        }
        if len(prompt) <= job["max_length"]:
            with torch.no_grad():
                logits = _last_token_logits(model, torch, prompt, logit_kwargs)
                logprobs = torch.log_softmax(logits.float(), dim=-1)[label_ids]
            record.update(score_record([float(v) for v in logprobs.cpu().tolist()],
                                       task.labels))
        records.append(record)
    write_csv(output_path, prediction_columns(task.labels), records)
    meta = {
        **(provenance or {}),
        "jobs": [job["name"]],
        "adapter_used": use_adapter,
        "decision_rule": DECISION_RULE,
        "label_token_ids": dict(zip(map(str, task.labels), label_ids)),
        "model_path": job["model_path"],
        "quantization": job["quantization"],
        "max_length": job["max_length"],
        "data_sha256": sha256_file(Path(data_path)),
        "n_rows": len(records),
        "n_too_long": sum(record["status"] == "too_long" for record in records),
    }
    write_json(sidecar(output_path), meta)
    return len(records)

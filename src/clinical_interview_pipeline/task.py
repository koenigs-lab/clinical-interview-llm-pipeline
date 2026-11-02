"""Task definitions keep construct-specific rules out of the pipeline code."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .records import is_safe_id

ADAPTER_STRATEGIES = ("shared", "by_group", "per_target")
PROMPT_FIELDS = ("construct", "target_id", "target_name", "rubric", "evidence")
PLACEHOLDER = re.compile(r"\bPLACEHOLDER\b")


@dataclass(frozen=True)
class Section:
    """An interview part delimited by markers inserted during transcript correction."""

    id: str
    start_pattern: str
    end_pattern: str | None = None


@dataclass(frozen=True)
class Target:
    id: str
    name: str
    rubric: Path
    group: str | None = None
    parent: str | None = None
    sections: tuple[str, ...] = ()


@dataclass(frozen=True)
class Task:
    id: str
    construct: str
    version: str
    labels: tuple[int, ...]
    participant_speaker: str
    input_mode: str
    evidence_mode: str
    adapter_strategy: str
    system_prompt: str
    user_prompt: str
    targets: tuple[Target, ...]
    sections: tuple[Section, ...]
    path: Path

    @property
    def target_map(self) -> dict[str, Target]:
        return {target.id: target for target in self.targets}

    @property
    def section_map(self) -> dict[str, Section]:
        return {section.id: section for section in self.sections}

    def rubric_text(self, target_id: str) -> str:
        target = self.target_map[target_id]
        value = target.rubric.read_text(encoding="utf-8").strip()
        if not value:
            raise ValueError(f"Empty rubric for target {target_id}")
        return value

    def render_prompts(self, target: Target, rubric: str, evidence: str) -> tuple[str, str]:
        """System and user prompt. Both templates may use every field in PROMPT_FIELDS."""
        fields = {"construct": self.construct, "target_id": target.id,
                  "target_name": target.name, "rubric": rubric, "evidence": evidence}
        return self.system_prompt.format(**fields), self.user_prompt.format(**fields)


def _sections(config: dict) -> tuple[list[Section], dict[str, tuple[str, ...]]]:
    """Declared sections in interview order, and the sections each target draws on.

    A target's ``section_pattern`` is shorthand for a section of its own. A task
    uses either the shorthand or a ``[[sections]]`` table.
    """
    declared = [
        Section(id=str(item["id"]), start_pattern=str(item["start_pattern"]),
                end_pattern=item.get("end_pattern"))
        for item in config.get("sections", [])
    ]
    shorthand = [item for item in config["targets"] if "section_pattern" in item]
    if shorthand and (declared or any("sections" in item for item in config["targets"])):
        raise ValueError("Use either [[sections]] with target 'sections' lists or the "
                         "'section_pattern' shorthand, not both")
    if shorthand:
        declared = [Section(id=str(item["id"]), start_pattern=str(item["section_pattern"]))
                    for item in shorthand]
        return declared, {str(item["id"]): (str(item["id"]),) for item in shorthand}
    return declared, {str(item["id"]): tuple(str(name) for name in item.get("sections", []))
                      for item in config["targets"]}


def load_task(path: str | Path, allow_placeholders: bool = False) -> Task:
    path = Path(path).resolve()
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    details = config["task"]
    prompt = config["prompt"]
    sections, target_sections = _sections(config)
    targets = tuple(
        Target(
            id=str(item["id"]),
            name=str(item["name"]),
            rubric=(path.parent / item["rubric"]).resolve(),
            group=item.get("group"),
            parent=item.get("parent"),
            sections=target_sections.get(str(item["id"]), ()),
        )
        for item in config["targets"]
    )
    task = Task(
        id=str(details["id"]),
        construct=str(details["construct"]),
        version=str(details["version"]),
        labels=tuple(int(value) for value in details["labels"]),
        participant_speaker=str(details.get("participant_speaker", "participant")),
        input_mode=str(details.get("input_mode", "participant_only")),
        evidence_mode=str(details.get("evidence_mode", "full_interview")),
        adapter_strategy=str(details.get("adapter_strategy", "shared")),
        system_prompt=str(prompt["system"]),
        user_prompt=str(prompt["user"]),
        targets=targets,
        sections=tuple(sections),
        path=path,
    )
    validate_task(task, allow_placeholders)
    return task


def _validate_sections(task: Task) -> None:
    ids = [section.id for section in task.sections]
    if len(ids) != len(set(ids)) or not all(map(is_safe_id, ids)):
        raise ValueError("Section ids must be unique and path-safe")
    for section in task.sections:
        re.compile(section.start_pattern)
        if section.end_pattern is not None:
            re.compile(section.end_pattern)
    for target in task.targets:
        if task.evidence_mode == "section" and not target.sections:
            raise ValueError(f"Target {target.id} needs sections or a section_pattern")
        if len(target.sections) != len(set(target.sections)):
            raise ValueError(f"Target {target.id} lists a section twice")
        unknown = set(target.sections) - set(ids)
        if unknown:
            raise ValueError(f"Target {target.id} uses undeclared sections: {sorted(unknown)}")


def validate_task(task: Task, allow_placeholders: bool = False) -> None:
    if not is_safe_id(task.id) or not task.construct.strip():
        raise ValueError("Task id or construct is missing or invalid")
    if len(task.labels) < 2 or sorted(set(task.labels)) != list(task.labels):
        raise ValueError("Labels must be at least two unique integers in ascending order")
    if task.labels != tuple(range(task.labels[0], task.labels[-1] + 1)):
        raise ValueError("Labels must be contiguous integers, for example [0, 1, 2, 3, 4]")
    if task.input_mode not in {"participant_only", "dialogue"}:
        raise ValueError("input_mode must be participant_only or dialogue")
    if task.evidence_mode not in {"full_interview", "section", "tagged"}:
        raise ValueError("evidence_mode must be full_interview, section, or tagged")
    if task.adapter_strategy not in ADAPTER_STRATEGIES:
        raise ValueError(f"adapter_strategy must be one of {', '.join(ADAPTER_STRATEGIES)}")
    ids = [target.id for target in task.targets]
    if not ids or len(ids) != len(set(ids)) or not all(map(is_safe_id, ids)):
        raise ValueError("Target ids must be present, unique, and path-safe")
    _validate_sections(task)
    for target in task.targets:
        if not target.rubric.is_file():
            raise FileNotFoundError(target.rubric)
        if target.parent and target.parent not in ids:
            raise ValueError(f"Unknown parent for {target.id}: {target.parent}")
        rubric = task.rubric_text(target.id)
        if PLACEHOLDER.search(rubric) and not allow_placeholders:
            raise ValueError(f"Rubric for {target.id} is a placeholder; insert the "
                             "rubric text before building data or training")
    if task.adapter_strategy == "by_group" and any(
        not target.group for target in task.targets
    ):
        raise ValueError("Every target needs a group for by_group adapters")
    if "{evidence}" not in task.user_prompt or "{rubric}" not in task.user_prompt:
        raise ValueError("User prompt must contain {evidence} and {rubric}")
    try:
        task.render_prompts(task.targets[0], "rubric", "evidence")
    except (KeyError, IndexError, ValueError) as error:
        raise ValueError(
            f"Prompt template cannot be rendered ({error!r}). Allowed fields: "
            f"{', '.join(PROMPT_FIELDS)}; write literal braces as {{{{ and }}}}"
        ) from error


def adapter_groups(task: Task) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for target in task.targets:
        if task.adapter_strategy == "shared":
            key = "shared"
        elif task.adapter_strategy == "by_group":
            key = str(target.group)
        else:
            key = target.id
        groups.setdefault(key, []).append(target.id)
    return groups

# Adapting the pipeline to another task

Copy `examples/synthetic/` or the [LPFS skeleton](../examples/lpfs-skeleton/README.md) to a private folder and run `cilp task-check --task task.toml` after every change.

## Targets and scale

`[task]` sets the construct, a stable task ID and version, the integer `labels`, the participant's speaker value, the input mode, the evidence mode, and the default adapter strategy. Each `[[targets]]` entry needs a path-safe unique `id`, a display `name`, and a rubric file. `group` is required for `by_group` adapters. `parent` names another target to record a hierarchy. Composite scores, such as domain or total scores, are targets of their own.

Labels are consecutive integers such as `[0, 1, 2, 3, 4]`, each a single token for the model's tokenizer.

Commands refuse rubric files that contain the word `PLACEHOLDER`. `cilp task-check --allow-placeholders` checks a skeleton anyway.

## Evidence selection

`evidence_mode` decides which turns become a target's evidence:

- `full_interview`: every turn of every session.
- `tagged`: turns whose `target_ids` list contains the target.
- `section`: turns between section markers. A marker is a line that transcript reviewers insert during correction, or a standard opening question asked by the interviewer, such as the question that opens each facet of the STiP-5.1.

`input_mode = "participant_only"` keeps only participant text, and `"dialogue"` keeps all turns with speaker labels. Both refuse `speaker=unknown`. When more than one session contributes evidence, each block is headed `[Session 1]`, `[Session 2]`, and so on.

### Sections

Declare sections in the order they occur in the interview, and list the sections each target draws on:

```toml
[[sections]]
id = "identity"
start_pattern = "^SECTION: IDENTITY$"

[[sections]]
id = "self_direction"
start_pattern = "^SECTION: SELF-DIRECTION$"
end_pattern = "^SECTION END$"          # optional explicit end marker

[[sections]]
id = "closing"                          # boundary only: no target uses it
start_pattern = "^SECTION: CLOSING$"

[[targets]]
id = "identity.1"
sections = ["identity"]                 # several targets may share a section
# ...

[[targets]]
id = "self_direction.3"
sections = ["identity", "self_direction"]   # a target may draw on several
# ...
```

For one section per target, `section_pattern = "^SECTION: SLEEP$"` on each target is a shorthand. A task uses either the shorthand or a `[[sections]]` table.

Boundary rules:

- Markers are regular expressions, searched in non-participant turns. A start marker matches at most one turn per session, and each section starts at its own turn.
- A section ends at its own end marker if it declares one, otherwise at the next section's start marker, otherwise at the end of the session.
- **Missing markers.** If a declared section's marker is missing from a session, the section before it could have absorbed its text. That preceding section is then excluded for the session as `ambiguous_section_boundary`, unless it has its own end marker. The same applies when a declared end marker is missing. If markers appear out of declared order while a marker is missing, every section without an end marker in that session is ambiguous.
- A target whose sections are all missing is excluded as `missing_section_anchor`, which points to a data-preparation error. A target that keeps some but not all of its sections is used and counted as `incomplete_section_coverage`.
- A section that runs to the end of a session without an end marker is used and counted as `open_ended_section`, because it may include closing talk. Declare a boundary-only closing section, or end markers, to prevent that.
- When an interview spans several sessions, close the last section of each session with an end marker or a boundary section. Otherwise the sections that follow in other sessions make it ambiguous.

Insert every declared marker, even for a section that was skipped in the interview: the skipped section then has no evidence and its neighbours stay unambiguous. The manifest's `section_report` lists all four counts per target and split.

Training skips evidence longer than the model's context, and prediction marks it `too_long`, which the primary metric counts as the worst possible error. `cilp plan` counts such examples before any GPU job, so `max_length` or the evidence mode can be adjusted first.

## Prompts

Both prompt templates may use `{construct}`, `{target_id}`, `{target_name}`, `{rubric}`, and `{evidence}`, and the user prompt must contain `{rubric}` and `{evidence}`. Write literal braces as `{{` and `}}`. `task-check` renders both templates and reports other placeholders. Ask for one permitted integer so that a bare score is the natural continuation: prediction reads the label-token probabilities at that position.

## Adapters, experiments, and training objective

`shared` trains one adapter for all targets, `by_group` one per `group`, and `per_target` one per target. With all three strategies, a plan has

`(1 + number of groups + number of targets) × learning rates × epoch counts × seeds`

jobs. A shared adapter learns from the examples of all targets, a per-target adapter from those of its own target.

`objective` selects the training loss:

- `token_ce` (default): next-token cross-entropy on the label token and end-of-sequence. It treats the scale as nominal. Ordinality enters through the median decision rule and the ordinal metrics.
- `label_rps`: the ranked probability score of the renormalised label-token distribution at the answer position, the quantity `evaluate` reports for calibration. It penalises probability placed far from the reference more than probability placed nearby. With this objective `label_mass` no longer indicates format compliance, because only the relative label probabilities are trained.

The experiment file also sets the LoRA target modules, rank, alpha, dropout, maximum length, accumulation, and quantization (`4bit` or `none`), which depend on the model architecture and GPU memory. The trainer works with local causal chat models that have a chat template, loaded from disk.

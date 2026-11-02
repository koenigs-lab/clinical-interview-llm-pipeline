# LPFS task skeleton

A structure-only starting point for rating personality functioning along the Level of Personality Functioning Scale: four elements (identity, self-direction, empathy, intimacy) with three facets each, rated 0–4, in the self- and interpersonal-functioning domains. Each facet has its own interview section, as in an interview that asks about each facet in turn. Facet names are numbered, and every rubric is a placeholder.

To use it, copy the folder into your project and:

1. Rename the facets to your instrument's labels and paste the rubric text into each file under `rubrics/`. Commands refuse rubrics that still contain the word `PLACEHOLDER`.
2. Replace the placeholder section patterns in `task.toml` with your markers, such as each facet's opening question as asked by the interviewer or markers your transcript reviewers insert, and keep them in interview order (see [adapting a task](../../docs/adapting-a-task.md#evidence-selection)).
3. Run `cilp task-check --task task.toml`.

Inspect the unfilled skeleton with `cilp task-check --task examples/lpfs-skeleton/task.toml --allow-placeholders`.

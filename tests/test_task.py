"""Task configuration: targets, sections, prompts, and placeholder rubrics."""

from __future__ import annotations

import unittest
from dataclasses import replace

from clinical_interview_pipeline.task import adapter_groups, load_task

from support import EXAMPLE, LPFS, section, target, write_task, WorkspaceTest


class TaskTest(WorkspaceTest):
    def test_task_is_independent_of_number_of_targets(self):
        task = load_task(EXAMPLE / "task.toml")
        self.assertEqual(len(task.targets), 3)
        self.assertEqual(len(adapter_groups(task)), 1)
        self.assertEqual(set(task.target_map),
                         {"routine.sleep", "routine.planning", "social.support"})
        # The section_pattern shorthand declares one section per target, in order.
        self.assertEqual([s.id for s in task.sections], [t.id for t in task.targets])

    def test_lpfs_skeleton_is_structure_only(self):
        with self.assertRaises(ValueError):
            load_task(LPFS / "task.toml")
        task = load_task(LPFS / "task.toml", allow_placeholders=True)
        self.assertEqual(len(task.targets), 12)
        self.assertEqual(task.labels, (0, 1, 2, 3, 4))
        self.assertEqual(len(adapter_groups(replace(task, adapter_strategy="by_group"))), 4)
        self.assertIn("closing", task.section_map)
        self.assertFalse(any("closing" in t.sections for t in task.targets))
        # One section per facet, as in an interview that asks about each facet in turn.
        self.assertEqual(len({t.sections for t in task.targets}), 12)

    def test_prompt_templates_are_rendered_at_task_check(self):
        body = section("a", "^A$") + target("t1", ["a"])
        task = load_task(write_task(self.root, body, system="Rate {target_name} of {construct}."))
        system, user = task.render_prompts(task.targets[0], "R", "E")
        self.assertEqual(system, "Rate t1 of Fictional construct.")
        self.assertEqual(user, "R\nE")
        for broken in ("Use {unknown}.", 'Answer as {"score": 0}.'):
            other = self.root / "other"
            other.mkdir(exist_ok=True)
            with self.assertRaises(ValueError):
                load_task(write_task(other, body, system=broken))

    def test_sections_and_shorthand_cannot_be_mixed(self):
        body = (section("a", "^A$") + target("t1", ["a"])
                + target("t2", extra='section_pattern = "^B$"\n'))
        with self.assertRaises(ValueError):
            load_task(write_task(self.root, body))

    def test_targets_need_declared_sections_in_section_mode(self):
        with self.assertRaises(ValueError):
            load_task(write_task(self.root, section("a", "^A$") + target("t1", ["b"])))
        other = self.root / "other"
        other.mkdir()
        with self.assertRaises(ValueError):
            load_task(write_task(other, section("a", "^A$") + target("t1")))
        tagged = self.root / "tagged"
        tagged.mkdir()
        self.assertEqual(len(load_task(write_task(tagged, target("t1"), "tagged")).targets), 1)


if __name__ == "__main__":
    unittest.main()

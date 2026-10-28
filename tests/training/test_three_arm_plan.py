"""Full-cohort coverage, unchanged human labels and held-out data boundaries."""

import json
import unittest

from reporting.berrutti.monitor_status import event_summary, experiment_focus
from reporting.berrutti.report_to_chat import report_prompt
from finetune.recipes.berrutti_qwen38.three_arm_plan import (
    all_regions,
    evaluation_schedule,
    make_plan,
    select_test_epoch,
)


class ThreeArmPlan(unittest.TestCase):
    def setUp(self):
        self.summary = {
            "train_pages": 125,
            "train_lines": 5212,
            "validation_pages": 25,
            "test_pages": 25,
            "page_line_annotation_disagreements": 34,
        }

    def test_all_three_epochs_keep_every_image_and_do_not_inherit_adapters(self):
        plan = make_plan(self.summary)
        self.assertEqual(set(plan["arms"]), {"pages", "lines", "combined"})
        for name, arm in plan["arms"].items():
            self.assertFalse(arm["inherits_trained_checkpoint"])
            self.assertEqual([row["epoch"] for row in arm["epochs"]], [1, 2, 3])
            self.assertEqual(
                len({row["image_presentations"] for row in arm["epochs"]}), 1
            )
            self.assertEqual(arm["new_optimizer_steps"], 48)
            for epoch in arm["epochs"]:
                self.assertEqual(
                    epoch["berrutti_line_images"], 0 if name == "pages" else 5212
                )
                self.assertEqual(epoch["luisa_line_images"], 0)
                self.assertEqual(
                    epoch["full_page_images"], 0 if name == "lines" else 125
                )
            self.assertAlmostEqual(sum(arm["weight_per_training_unit"].values()), 1.0)

    def test_a_smaller_cohort_or_changed_heldout_is_rejected(self):
        with self.assertRaises(ValueError):
            make_plan({**self.summary, "train_pages": 91})
        with self.assertRaises(ValueError):
            make_plan({**self.summary, "test_pages": 1})

    def test_shared_baseline_and_selected_test_reduce_evaluation_not_training(self):
        schedule = evaluation_schedule()
        self.assertEqual(len(schedule), 14)
        self.assertEqual(sum(row["pages"] for row in schedule), 350)
        self.assertEqual(len([r for r in schedule if r["phase"] == "baseline"]), 2)
        for arm in ("pages", "lines", "combined"):
            tasks = [row for row in schedule if row["arm"] == arm]
            self.assertEqual(
                [row["epoch"] for row in tasks if row["split"] == "validation"],
                [1, 2, 3],
            )
            self.assertEqual(len([row for row in tasks if row["split"] == "test"]), 1)
        plan = make_plan(self.summary)
        self.assertEqual(plan["evaluation"]["scheduled_page_evaluations"], 350)
        self.assertFalse(plan["evaluation"]["test_after_each_epoch"])
        self.assertEqual(plan["training"]["steps_per_epoch"], 16)

    @staticmethod
    def validation(epoch, value, **changes):
        return {
            "split": "validation",
            "epoch": epoch,
            "pages": 25,
            "all_pages_completed": True,
            "failed_pages": 0,
            "retried_pages": 0,
            "cer": value,
            **changes,
        }

    def test_selected_test_waits_for_third_validation_and_ignores_test_and_baseline(
        self,
    ):
        rows = [self.validation(1, 0.03), self.validation(2, 0.02)]
        self.assertIsNone(select_test_epoch(rows))
        rows.extend(
            [
                self.validation(3, 0.04),
                self.validation(0, 0.001, stage="baseline"),
                {"split": "test", "epoch": 3, "cer": 0.001},
            ]
        )
        self.assertEqual(select_test_epoch(rows)["epoch"], 2)

    def test_incomplete_or_partial_validation_cannot_win_selected_test(self):
        for changes in (
            {"all_pages_completed": False},
            {"failed_pages": 1},
            {"pages": 1},
            {"cer": float("inf")},
            {"cer": float("nan")},
        ):
            with self.subTest(changes=changes):
                rows = [
                    self.validation(1, 0.001, **changes),
                    self.validation(2, 0.02, retried_pages=2),
                    self.validation(3, 0.02),
                ]
                self.assertEqual(select_test_epoch(rows)["epoch"], 2)
        rows = [self.validation(epoch, 0.01, pages=1) for epoch in (1, 2, 3)]
        self.assertIsNone(select_test_epoch(rows))

    def test_duplicate_validation_is_rejected_before_selected_test(self):
        row = self.validation(1, 0.02)
        with self.assertRaises(ValueError):
            select_test_epoch([row, row])

    def test_short_and_whitespace_labels_are_kept_exactly(self):
        regions = [
            {
                "shape_attributes": {
                    "name": "rect",
                    "x": 0,
                    "y": i * 10,
                    "width": 20,
                    "height": 10,
                },
                "region_attributes": {"text": target},
            }
            for i, target in enumerate(("  é  ", "7"))
        ]
        result = all_regions(regions, (40, 40))
        self.assertEqual([row["target"] for row in result], ["  é  ", "7"])
        regions[0]["shape_attributes"]["width"] = 41
        with self.assertRaises(ValueError):
            all_regions(regions, (40, 40))

    def test_missing_lines_do_not_silently_remove_a_training_page(self):
        with self.assertRaises(ValueError):
            all_regions([], (40, 40))

    def test_luisa_proposal_never_adds_examples_to_the_three_main_arms(self):
        plan = make_plan(self.summary)
        self.assertEqual(
            plan["optional_luisa_experiment"]["state"], "PROPOSED_NOT_LAUNCHED"
        )
        self.assertEqual(len(plan["optional_luisa_experiment"]["variants"]), 2)
        self.assertTrue(
            plan["optional_luisa_experiment"][
                "requires_exclusion_audit_before_training"
            ]
        )
        for name in ("lines", "combined"):
            arm = plan["arms"][name]
            self.assertEqual(arm["weight_per_training_unit"]["luisa_group"], 0)
            self.assertEqual(arm["epochs"][0]["luisa_line_images"], 0)
            self.assertAlmostEqual(sum(arm["weight_per_training_unit"].values()), 1)

    def test_old_training_does_not_become_a_result_of_a_new_arm(self):
        plan = make_plan(self.summary)
        coordination = {"experiment_plan": plan, "runs": []}
        old = {"old_mixture": {"state": "RUNNING", "evaluations": [{"cer": 0.02}]}}
        focused = experiment_focus(coordination, old)
        self.assertEqual(len(focused["primary_experiments"]), 3)
        for arm in focused["primary_experiments"].values():
            self.assertEqual(arm["state"], "PLANNED")
            self.assertEqual(arm["evaluations"], [])
        self.assertNotIn("historical_running", focused)
        prompt = report_prompt({"checked_at_utc": None, **focused})
        self.assertIn("todavía no lanzado", prompt)
        self.assertNotIn("threeway_pages, threeway_lines", prompt)

    def test_pretty_runtime_receipts_do_not_break_status_refresh(self):
        probe = {
            "event": "gpu_probe",
            "runtime": True,
            "success": True,
            "cuInit": 0,
            "nvmlInit": 0,
            "torch": {"device_count": 2},
            "generation_guards": {"success": True, "tests_run": 8},
        }
        self.assertEqual(
            event_summary(json.dumps(probe, indent=2)), event_summary(json.dumps(probe))
        )
        self.assertEqual(
            event_summary(json.dumps(probe, indent=2))["diagnosis"][
                "cuda_device_count"
            ],
            2,
        )


if __name__ == "__main__":
    unittest.main()

import copy
import json
import math
from pathlib import Path
import random
import tempfile
import unittest

from reporting.berrutti.audit_statistics import paired_difference
from evaluation.scoring import score_pairs
from reporting.berrutti.final_audit import (
    check_attempts,
    selected_epoch,
    training_epoch_receipt,
    validate_completion_header,
    validate_updates,
)
from finetune.recipes.berrutti_qwen38.generation_guard import GenerationPolicy


def update_rows():
    return [
        {
            "optimizer_steps": step,
            "epoch": (step - 1) // 16 + 1,
            "page": min((((step - 1) % 16) + 1) * 8, 125),
            "total": 125,
            "stage": "pages",
            "loss": 0.2,
            "gradient_norm": 0.1,
            "lr": 2e-6 + (5e-5 - 2e-6) * (1 + math.cos(math.pi * (step - 1) / 47)) / 2,
        }
        for step in range(1, 49)
    ]


class UpdateAuditTests(unittest.TestCase):
    def verify(self, rows):
        validate_updates(rows, epochs=3, stage="pages", maximum=5e-5, minimum=2e-6)

    def test_all_confirmed_updates_pass(self):
        self.verify(update_rows())

    def test_missing_repeated_and_reordered_updates_fail(self):
        rows = update_rows()
        for changed in (rows[:-1], rows + rows[-1:], rows[1:] + rows[:1]):
            with self.assertRaisesRegex(ValueError, "Missing, repeated"):
                self.verify(changed)

    def test_nonfinite_gradient_and_changed_schedule_fail(self):
        for key, value in (("gradient_norm", float("nan")), ("lr", 0.001)):
            rows = update_rows()
            rows[19][key] = value
            with self.assertRaises(ValueError):
                self.verify(rows)

    def test_wrong_epoch_or_incomplete_page_cursor_fails(self):
        for key, value in (("epoch", 2), ("page", 124), ("stage", "combined")):
            rows = update_rows()
            rows[-1][key] = value
            with self.assertRaisesRegex(ValueError, "cursor"):
                self.verify(rows)


class FrozenReceiptSchemaTests(unittest.TestCase):
    def test_original_six_field_completion_and_training_event(self):
        completion = dict(
            best_epoch=2,
            best_validation_cer=0.02,
            test_cer_at_selected_epoch=0.03,
            selected_test_failed_pages=0,
            optimizer_steps=48,
            selected_checkpoint="checkpoint",
        )
        validate_completion_header(completion, exploratory=False)
        event = {
            "event": "training_epoch",
            "epoch": 1,
            "elapsed_segment": 12,
            "samples": 125,
            "optimizer_steps": 16,
        }
        row = training_epoch_receipt(Path("unused"), [event], 1, exploratory=False)
        self.assertEqual(row, {"epoch": 1, "samples": 125, "optimizer_steps": 16})
        with self.assertRaises(ValueError):
            training_epoch_receipt(Path("unused"), [event, event], 1, exploratory=False)
        with self.assertRaises(ValueError):
            validate_completion_header(completion, exploratory=True)
        with self.assertRaises(ValueError):
            validate_completion_header(
                {**completion, "success": False}, exploratory=False
            )


class SelectionAuditTests(unittest.TestCase):
    def validations(self):
        return [
            {
                "epoch": e,
                "split": "validation",
                "pages": 25,
                "all_pages_completed": True,
                "failed_pages": 0,
                "cer": 0.03 - e * 0.001,
            }
            for e in range(1, 5)
        ]

    def test_intermediate_evaluation_cannot_select(self):
        rows = self.validations()
        rows[3]["epoch"] = 412
        with self.assertRaisesRegex(ValueError, "end-of-epoch"):
            selected_epoch(rows, 4)

    def test_incomplete_lowest_cer_is_ineligible(self):
        rows = self.validations()
        rows[3].update(all_pages_completed=False, failed_pages=1, cer=0.0)
        self.assertEqual(selected_epoch(rows, 4)["epoch"], 3)

    def test_complete_validation_tie_uses_earlier_epoch(self):
        rows = self.validations()
        rows[2]["cer"] = rows[3]["cer"]
        self.assertEqual(selected_epoch(rows, 4)["epoch"], 3)

    def test_no_complete_validation_fails(self):
        rows = self.validations()
        for row in rows:
            row["all_pages_completed"] = False
        with self.assertRaisesRegex(ValueError, "No complete"):
            selected_epoch(rows, 4)


class AttemptAuditTests(unittest.TestCase):
    def fixture(self, root, *, failed=False):
        policy = GenerationPolicy()
        count = 2 if failed else 1
        for number in range(1, count + 1):
            parameters = policy.attempt_parameters(
                "repetition" if number == 2 else None
            )
            started = {"attempt": number, "parameters": parameters}
            (root / f"attempt-{number}.started.json").write_text(json.dumps(started))
            (root / f"attempt-{number}.json").write_text(
                json.dumps(
                    {
                        **started,
                        "prediction": "axc",
                        "seconds": 3.0,
                        "generated_tokens": 3,
                        "stop_reason": "repetition" if failed else "eos",
                    }
                )
            )
        row = {
            "attempts": count,
            "status": "incomplete" if failed else "complete",
            "generation_stop_counts": {"repetition" if failed else "eos": count},
            "seconds": 3.0 * count,
            **score_pairs([("abc", "axc")]),
        }
        return row, policy

    def test_raw_selected_attempt_rescored_without_generation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            row, policy = self.fixture(root)
            self.assertEqual(len(check_attempts(root, row, "abc", policy)), 1)
            self.assertEqual(row["char_errors"], 1)

    def test_exhausted_failure_remains_incomplete(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            row, policy = self.fixture(root, failed=True)
            before = copy.deepcopy(row)
            self.assertEqual(len(check_attempts(root, row, "abc", policy)), 2)
            self.assertEqual(row, before)
            row["status"] = "complete"
            with self.assertRaisesRegex(ValueError, "status"):
                check_attempts(root, row, "abc", policy)

    def test_third_attempt_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            row, policy = self.fixture(root, failed=True)
            (root / "attempt-3.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "coverage"):
                check_attempts(root, row, "abc", policy)

    def test_wrong_metric_or_modified_parameters_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            row, policy = self.fixture(root)
            row["char_errors"] = 0
            with self.assertRaisesRegex(ValueError, "rescore"):
                check_attempts(root, row, "abc", policy)
            row["char_errors"] = 1
            attempt = json.loads((root / "attempt-1.json").read_text())
            attempt["parameters"]["max_new_tokens"] = 1
            (root / "attempt-1.json").write_text(json.dumps(attempt))
            with self.assertRaisesRegex(ValueError, "parameters"):
                check_attempts(root, row, "abc", policy)


class PairedStatisticsTests(unittest.TestCase):
    def rows(self):
        return [
            {"page_id": "high_r0001_0001", "char_errors": 1, "char_count": 10},
            {"page_id": "high_r0001_0002", "char_errors": 2, "char_count": 1000},
            {"page_id": "high_r0002_0003", "char_errors": 3, "char_count": 2000},
        ]

    def test_equal_pairs_have_zero_difference_and_interval(self):
        result = paired_difference(self.rows(), self.rows(), resamples=100)
        self.assertEqual(result["percentile_95_interval"], [0, 0])

    def test_micro_difference_preserves_denominators_and_rng(self):
        left = self.rows()
        right = copy.deepcopy(left)
        right[0]["char_errors"] = 0
        before = random.getstate()
        result = paired_difference(left, right, resamples=100)
        self.assertAlmostEqual(result["difference_left_minus_right"], 1 / 3010)
        self.assertEqual(random.getstate(), before)
        self.assertEqual(result, paired_difference(left, right, resamples=100))

    def test_roll_resampling_keeps_shared_scan_rolls_together(self):
        result = paired_difference(
            self.rows(), self.rows(), resamples=100, by_roll=True
        )
        self.assertEqual(result["resampling_units"], 2)
        self.assertEqual(result["pages"], 3)

    def test_mismatched_missing_duplicate_or_negative_counts_rejected(self):
        rows = self.rows()
        altered = copy.deepcopy(rows)
        altered[0]["char_count"] = 11
        negative = copy.deepcopy(rows)
        negative[0]["char_errors"] = -1
        for right in (rows[:-1], rows + rows[:1], altered, negative):
            with self.assertRaises(ValueError):
                paired_difference(rows, right, resamples=100)


if __name__ == "__main__":
    unittest.main()

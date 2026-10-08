"""Small checks of the private experiment's split and score contracts."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from experiments.berrutti_qwen38.augmentation import (
    conservative_augment,
    padded_crop,
    presentation_seed,
)
from experiments.berrutti_qwen38.common import score_pairs
from experiments.berrutti_qwen38.generation_guard import (
    DecodeGuard,
    GenerationPolicy,
    eos_ids,
    generate_with_retries,
    repeated_suffix,
    select_complete_validation_epoch,
)
from experiments.berrutti_qwen38.monitor_status import event_summary
from experiments.berrutti_qwen38.report_to_chat import Deferred, deliver, report_prompt


class ExperimentContractTests(unittest.TestCase):
    def test_micro_cer_uses_total_characters_and_ascii_text_policy(self) -> None:
        scored = score_pairs([("éééé", "eeex"), ("a \t b", "a b")])
        self.assertEqual(scored["char_count"], 7)
        self.assertEqual(scored["char_errors"], 1)
        self.assertAlmostEqual(scored["cer"], 1 / 7)

    def test_split_covers_manifest_with_disjoint_rolls(self) -> None:
        path = Path("experiments/berrutti_qwen38/split-v1.json")
        split = json.loads(path.read_text())
        self.assertEqual(
            [len(split["splits"][name]) for name in ("train", "validation", "test")],
            [125, 25, 25],
        )
        ids = [
            page
            for name in ("train", "validation", "test")
            for page in split["splits"][name]
        ]
        self.assertEqual(len(set(ids)), 175)
        owners = {}
        for name, pages in split["splits"].items():
            for page in pages:
                roll = page.split("_")[1].split("-")[0]
                self.assertIn(owners.setdefault(roll, name), (name,))

    def test_file_hash_detects_changed_reference(self) -> None:
        from experiments.berrutti_qwen38.common import file_sha256

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "reference.txt"
            path.write_bytes(b"abc")
            self.assertEqual(file_sha256(path), hashlib.sha256(b"abc").hexdigest())
            path.write_bytes(b"abd")
            self.assertNotEqual(file_sha256(path), hashlib.sha256(b"abc").hexdigest())

    def test_page_augmentation_is_deterministic_and_preserves_source(self) -> None:
        source = Image.new("RGB", (64, 80), (180, 180, 180))
        seed = presentation_seed(
            20261007,
            stage="page",
            epoch=1,
            position=3,
            page_id="synthetic",
            region=None,
        )
        first, first_operations = conservative_augment(source, seed, page=True)
        second, second_operations = conservative_augment(source, seed, page=True)
        self.assertEqual(first_operations, second_operations)
        self.assertEqual(first.tobytes(), second.tobytes())
        self.assertEqual(source.getpixel((0, 0)), (180, 180, 180))

    def test_line_crop_includes_alignment_margin(self) -> None:
        source = Image.new("RGB", (100, 80), "white")
        crop = padded_crop(source, [20, 30, 80, 50])
        self.assertGreater(crop.width, 60)
        self.assertGreater(crop.height, 20)
        self.assertLessEqual(crop.width, source.width)

    def test_chat_report_uses_snapshot_without_remote_secrets(self) -> None:
        prompt = report_prompt(
            {
                "checked_at_utc": "2026-10-07T17:00:00+00:00",
                "runs": {
                    "full": {
                        "state": "RUNNING",
                        "evaluations": [
                            {"epoch": 0, "split": "validation", "cer": 0.0275}
                        ],
                    }
                },
            }
        )
        self.assertIn("ESTE MISMO chat", prompt)
        self.assertIn("línea base", prompt)
        self.assertIn('"cer":0.0275', prompt)
        self.assertNotIn("/gpfs/", prompt)


class ChatDeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_busy_chat_does_not_start_or_resume_another_writer(self) -> None:
        class Server:
            calls = []

            async def call(self, method, params):
                self.calls.append(method)
                return {"thread": {"status": {"type": "active"}}}

        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(
                thread_id="00000000-0000-4000-8000-000000000001",
                receipt=Path(temporary) / "receipt.json",
            )
            server = Server()
            with self.assertRaises(Deferred):
                await deliver(server, args)
            self.assertEqual(server.calls, ["thread/read"])
            self.assertFalse(args.receipt.exists())

    async def test_loaded_idle_chat_receives_turn_without_changing_its_settings(self):
        class Server:
            calls = []

            async def call(self, method, params):
                self.calls.append((method, params))
                if method == "turn/start":
                    return {"turn": {"id": "report-turn"}}
                return {"thread": {"status": {"type": "idle"}}}

            async def wait_turn(self, turn_id):
                return {"id": turn_id, "status": "completed"}

        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(
                thread_id="00000000-0000-4000-8000-000000000001",
                receipt=Path(temporary) / "receipt.json",
                snapshot=Path(temporary) / "snapshot.json",
                refresh_unit=None,
            )
            server = Server()
            result = await deliver(server, args)
            self.assertEqual(result["phase"], "completed")
            start = next(p for m, p in server.calls if m == "turn/start")
            self.assertEqual(start["threadId"], args.thread_id)
            self.assertNotIn("model", start)
            self.assertNotIn("sandboxPolicy", start)
            self.assertNotIn("approvalPolicy", start)

    async def test_recovery_confirms_previous_delivery_without_duplicate_turn(self):
        class Server:
            calls = []

            async def call(self, method, params):
                self.calls.append(method)
                return {"data": [{"id": "previous", "status": "completed"}]}

        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(
                thread_id="00000000-0000-4000-8000-000000000001",
                receipt=Path(temporary) / "receipt.json",
            )
            args.receipt.write_text(
                json.dumps({"phase": "accepted", "turn_id": "previous"})
            )
            server = Server()
            result = await deliver(server, args)
            self.assertEqual(result["phase"], "completed")
            self.assertEqual(server.calls, ["thread/turns/list"])


class GenerationGuardTests(unittest.TestCase):
    def test_sustained_loops_detected_but_normal_repetition_retained(self):
        policy = GenerationPolicy()
        self.assertEqual(repeated_suffix([5] * 128, policy), 1)
        self.assertEqual(repeated_suffix(list(range(32)) * 4, policy), 32)
        self.assertIsNone(repeated_suffix(list(range(32)) * 3, policy))
        self.assertIsNone(repeated_suffix(list(range(1024)), policy))
        self.assertIsNone(repeated_suffix([1, 2, 3, 1, 2, 3], policy))

    def test_secondary_eos_at_exact_token_budget_is_success(self):
        policy = GenerationPolicy(max_new_tokens=4)
        self.assertEqual(eos_ids([7, 9], 7), (7, 9))
        guard = DecodeGuard(policy, policy.attempt_parameters(None), (7, 9))
        self.assertEqual(guard.observe([1, 2, 3, 9], generated=4, elapsed=0), "eos")

    def test_length_time_and_unexpected_stop_are_distinct(self):
        policy = GenerationPolicy(max_new_tokens=4, max_seconds=2)
        for tokens, elapsed, final, expected in (
            ([1, 2, 3, 4], 1, False, "token_limit"),
            ([1], 2, False, "time_limit"),
            ([1], 1, True, "unexpected_stop"),
        ):
            guard = DecodeGuard(policy, policy.attempt_parameters(None), (9,))
            self.assertEqual(
                guard.observe(
                    tokens, generated=len(tokens), elapsed=elapsed, final=final
                ),
                expected,
            )

    def test_repetition_with_eos_still_flagged_and_state_sticky(self):
        policy = GenerationPolicy()
        guard = DecodeGuard(policy, policy.attempt_parameters(None), (9,))
        self.assertEqual(
            guard.observe([5] * 128 + [9], generated=129, elapsed=0), "repetition"
        )
        self.assertEqual(guard.observe([9], generated=130, elapsed=0), "repetition")

    def test_retry_changes_only_reason_specific_budget_without_reference(self):
        policy = GenerationPolicy()
        for reason in ("token_limit", "repetition", "time_limit"):
            calls, records = [], []

            def generate(parameters):
                calls.append(parameters)
                return {
                    "stop_reason": reason if len(calls) == 1 else "eos",
                    "prediction": "raw",
                }

            result = generate_with_retries(generate, policy, records.append)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(len(records), 2)
            self.assertEqual(calls[1]["max_new_tokens"], 8192)
            self.assertEqual(
                calls[1]["repetition_penalty"], 1.05 if reason == "repetition" else 1.0
            )
            self.assertEqual(calls[0]["no_repeat_ngram_size"], 0)
            self.assertIsNone(calls[0]["forced_eos_token_id"])

    def test_exhausted_retry_retains_last_raw_output(self):
        records = []
        result = generate_with_retries(
            lambda p: {"stop_reason": "repetition", "prediction": "abc" * 100},
            GenerationPolicy(),
            records.append,
        )
        self.assertEqual(len(records), 2)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["prediction"], "abc" * 100)
        score = score_pairs([("abc", result["prediction"]), ("def", "def")])
        self.assertEqual(score["pages"], 2)
        self.assertGreater(score["cer"], 1.0)

    def test_resume_reuses_recorded_attempt_without_third_retry(self):
        policy = GenerationPolicy()
        records = []
        result = generate_with_retries(
            lambda p: {"stop_reason": "token_limit", "prediction": "partial"},
            policy,
            records.append,
        )

        def no_generation(parameters):
            self.fail("Completed attempts must not be regenerated")

        resumed = generate_with_retries(
            no_generation, policy, lambda a: None, previous_attempts=records
        )
        self.assertEqual(resumed, result)
        with self.assertRaises(ValueError):
            generate_with_retries(
                no_generation,
                GenerationPolicy(max_new_tokens=8),
                lambda a: None,
                previous_attempts=records,
            )

    def test_hardware_failure_is_not_silenced_as_bad_ocr(self):
        def generate(parameters):
            raise RuntimeError("CUDA unavailable")

        with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
            generate_with_retries(generate, GenerationPolicy(), lambda a: None)

    def test_checkpoint_selection_excludes_incomplete_validation_not_test(self):
        summaries = [
            {"epoch": 1, "validation": {"cer": 0.01, "failed_pages": 1}},
            {
                "epoch": 2,
                "validation": {"cer": 0.03, "failed_pages": 0},
                "test": {"cer": 1.0, "failed_pages": 1},
            },
        ]
        self.assertEqual(select_complete_validation_epoch(summaries)["epoch"], 2)
        self.assertIsNone(select_complete_validation_epoch(summaries[:1]))

    def test_monitor_keeps_guard_failures_and_curriculum_stage(self):
        event = {
            "event": "evaluation",
            "epoch": 0,
            "split": "validation",
            "cer": 0.1,
            "wer": 0.2,
            "pages": 25,
            "seconds": 30,
            "stage": "region",
            "failed_pages": 1,
            "retried_pages": 2,
        }
        result = event_summary(json.dumps(event))["evaluations"][0]
        self.assertEqual(result["failed_pages"], 1)
        self.assertEqual(result["stage"], "region")

    def test_diagnostic_monitor_exposes_cause_without_native_text(self):
        event = {
            "event": "legacy_probe_result",
            "generated_tokens": 4096,
            "recognized_model_eos": False,
            "legacy_would_raise": True,
            "repeat_period_tokens": 32,
            "reference_tokens": 1200,
            "seconds": 300,
            "prediction": "private OCR",
            "page_id": "private page",
        }
        result = event_summary(json.dumps(event))["diagnosis"]
        self.assertTrue(result["legacy_would_raise"])
        self.assertEqual(result["repeat_period_tokens"], 32)
        self.assertNotIn("prediction", result)
        self.assertNotIn("page_id", result)


if __name__ == "__main__":
    unittest.main()

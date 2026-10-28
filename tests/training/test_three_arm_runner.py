"""Coverage, loss weighting and interrupted decoding of the actual runner."""

import json
import sqlite3
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from finetune.recipes.berrutti_qwen38.generation_guard import (
    GenerationPolicy,
    generate_with_retries,
)
from finetune.recipes.berrutti_qwen38.runner import (
    RUNTIME_PROFILE,
    Runtime,
    check_gpu_names,
    closed_json,
    example_weights,
    learning_rate,
    main,
    resumed_attempts,
    training_groups,
)


class ThreeArmRunner(unittest.TestCase):
    def test_cpu_preflight_checks_actual_launch_config_without_loading_gpu_runtime(
        self,
    ):
        from data.integrity import file_sha256

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "runner.py"
            source.write_text("frozen source")
            pages = {
                split: [
                    {
                        "page_id": f"{split}-{i}",
                        "image_sha256": "i",
                        "reference_sha256": "r",
                    }
                    for i in range(count)
                ]
                for split, count in (("train", 125), ("validation", 25), ("test", 25))
            }
            cohort = root / "cohort.json"
            cohort.write_text(
                json.dumps({"pages": pages["train"], "summary": {"train_lines": 5212}})
            )
            split = root / "split.json"
            split.write_text("sealed split")
            cache = root / "cache.sqlite"
            with sqlite3.connect(cache) as db:
                db.execute("CREATE TABLE metadata (key TEXT, value TEXT)")
                db.execute(
                    "INSERT INTO metadata VALUES (?,?)",
                    (
                        "closed_receipt",
                        json.dumps(
                            {
                                "complete": True,
                                "cohort_sha256": file_sha256(cohort),
                                "line_images": 5212,
                            }
                        ),
                    ),
                )
            config = {
                "protocol": "three-arms-125-v1",
                "runtime_profile": RUNTIME_PROFILE,
                "code_root": str(root),
                "source_hashes": {source.name: file_sha256(source)},
                "cohort_path": str(cohort),
                "cohort_sha256": file_sha256(cohort),
                "split_path": str(split),
                "split_sha256": file_sha256(split),
                "dataset_root": str(root),
                "line_cache": str(cache),
                "maximum_segments": 8,
            }
            path = root / "config.json"
            module = "finetune.recipes.berrutti_qwen38.runner"
            with (
                patch(module + ".load_pages", return_value=pages),
                patch(module + ".Runtime") as gpu,
            ):
                for index, profile in enumerate(
                    (
                        RUNTIME_PROFILE,
                        "bf16-sdpa-fp32-projection-no-visual-cache-v1",
                        "unknown",
                    )
                ):
                    config["runtime_profile"] = profile
                    path.write_text(json.dumps(config))
                    run = root / str(index)
                    with patch(
                        "sys.argv",
                        [
                            "runner",
                            "--config",
                            str(path),
                            "--root",
                            str(run),
                            "--mode",
                            "preflight",
                        ],
                    ):
                        if index == 0:
                            self.assertEqual(main(), 0)
                            receipt = json.loads((run / "completed.json").read_text())
                            self.assertEqual(
                                (receipt["source_pages"], receipt["line_images"]),
                                (125, 5212),
                            )
                            self.assertFalse(receipt["gpu_check"])
                        else:
                            with self.assertRaisesRegex(
                                ValueError, "unapproved runtime"
                            ):
                                main()
                            self.assertFalse((run / "completed.json").exists())
                gpu.assert_not_called()

    def test_backward_releases_unused_outputs_before_cache_and_gradient_work(self):
        calls = []
        references = []
        test = self

        class Loss:
            def __mul__(self, weight):
                calls.append(("weight", weight))
                return self

            def backward(self):
                test.assertIsNone(references[0]())
                test.assertEqual(calls[0], "cache_released")
                calls.append("backward")

            def detach(self):
                return 2.0

        class Output:
            pass

        class Model:
            config = SimpleNamespace()

            def train(self):
                pass

            def gradient_checkpointing_enable(self, **kwargs):
                pass

            def __call__(self, **kwargs):
                output = Output()
                output.loss = Loss()
                references.append(weakref.ref(output))
                return output

        def release_cache():
            test.assertIsNone(references[0]())
            calls.append("cache_released")

        runtime = object.__new__(Runtime)
        runtime.model = Model()
        runtime.visual = SimpleNamespace(eval=lambda: None)
        runtime.torch = SimpleNamespace(
            isfinite=lambda loss: True,
            cuda=SimpleNamespace(empty_cache=release_cache),
        )
        runtime.backward_diagnostic_emitted = True
        tensor = SimpleNamespace(to=lambda device: None)
        value = runtime.backward(
            {"inputs": {"input_ids": tensor}, "labels": tensor, "keep": 7}, 0.25
        )
        self.assertEqual(value, 2.0)
        self.assertEqual(calls, ["cache_released", ("weight", 0.25), "backward"])

    def test_hardware_requires_an_explicit_two_gpu_profile(self):
        check_gpu_names(["NVIDIA A100-SXM4-40GB"] * 2)
        check_gpu_names(["NVIDIA A40"] * 2, "A40")
        for names, expected in (
            (["NVIDIA A40"] * 2, "A100"),
            (["NVIDIA A100-SXM4-40GB"], "A100"),
            (["NVIDIA A40"] * 3, "A40"),
            (["NVIDIA RTX A4000"] * 2, "A40"),
            (["NVIDIA V100"] * 2, "V100"),
        ):
            with self.subTest(names=names, expected=expected):
                with self.assertRaises(RuntimeError):
                    check_gpu_names(names, expected)

    def test_each_epoch_and_resumed_suffix_cover_all_125_sources_once(self):
        pages = [{"page_id": f"source-{i:03d}"} for i in range(125)]
        orders = []
        for epoch in (1, 2, 3):
            groups = training_groups(pages, 20261007, epoch)
            self.assertEqual([len(g) for g in groups], [8] * 15 + [5])
            ids = [p["page_id"] for g in groups for p in g]
            self.assertEqual(len(set(ids)), 125)
            self.assertEqual(
                groups, training_groups(list(reversed(pages)), 20261007, epoch)
            )
            # The persisted group cursor resumes whole, uncommitted groups.
            committed = [p for g in groups[:7] for p in g]
            remaining = [
                p for g in training_groups(pages, 20261007, epoch)[7:] for p in g
            ]
            self.assertEqual(committed + remaining, [p for g in groups for p in g])
            orders.append(ids)
        self.assertNotEqual(orders[0], orders[1])
        with self.assertRaises(ValueError):
            training_groups(pages + pages[:1], 1, 1)

    def test_longer_lines_keep_token_weight_without_inflating_a_source_page(self):
        expected = {
            "pages": (1.0, [0.0, 0.0]),
            "lines": (0.0, [0.1, 0.9]),
            "combined": (0.5, [0.05, 0.45]),
        }
        for arm, weights in expected.items():
            page, lines = example_weights(arm, [1, 9])
            self.assertAlmostEqual(page, weights[0])
            for observed, desired in zip(lines, weights[1], strict=True):
                self.assertAlmostEqual(observed, desired)
            self.assertAlmostEqual(page + sum(lines), 1)
            # Eight full units and the final five both average to one update.
            for count in (8, 5):
                self.assertAlmostEqual(
                    sum((page + sum(lines)) / count for _ in range(count)), 1
                )

    def test_scheduler_resumes_at_the_next_update_with_the_same_48_step_schedule(self):
        schedule = [learning_rate(i, 48, 5e-5, 2e-6) for i in range(48)]
        self.assertAlmostEqual(schedule[0], 5e-5)
        self.assertAlmostEqual(schedule[-1], 2e-6)
        self.assertEqual(
            schedule[19:], [learning_rate(i, 48, 5e-5, 2e-6) for i in range(19, 48)]
        )
        self.assertTrue(all(a >= b for a, b in zip(schedule, schedule[1:])))

    def test_an_interrupted_page_consumes_attempt_one_and_retry_is_saved_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root, policy = Path(directory), GenerationPolicy()
            closed_json(
                root / "attempt-1.started.json",
                {"attempt": 1, "parameters": policy.attempt_parameters(None)},
            )
            previous = resumed_attempts(root, policy)
            self.assertEqual(previous[0]["stop_reason"], "interrupted")
            calls = []

            def generate(parameters):
                calls.append(parameters)
                return {
                    "prediction": "diagnostic",
                    "stop_reason": "time_limit",
                    "seconds": 1.0,
                }

            def record(row):
                path = root / f"attempt-{row['attempt']}.json"
                if not path.exists():
                    closed_json(path, row)

            result = generate_with_retries(
                generate, policy, record, previous_attempts=previous
            )
            self.assertEqual(len(calls), 1)
            self.assertEqual(result["status"], "incomplete")
            previous = resumed_attempts(root, policy)
            generate_with_retries(generate, policy, record, previous_attempts=previous)
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(previous), 2)

    def test_closed_evidence_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            closed_json(path, {"complete": True})
            with self.assertRaises(FileExistsError):
                closed_json(path, {"complete": False})
            self.assertEqual(json.loads(path.read_text()), {"complete": True})


if __name__ == "__main__":
    unittest.main()

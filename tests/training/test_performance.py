"""Parallel evaluation must preserve exact coverage, micro scores and failures."""

import hashlib
import io
import json
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from reporting.berrutti.monitor_status import (
    configure_cluster,
    job_states,
    pending_queue,
    performance_summary,
    read_performance_checks,
)

from finetune.recipes.berrutti_qwen38.performance import (
    cache_key,
    merge_page_metrics,
    page_shard,
    prepare_line_cache,
)


class PerformanceContracts(unittest.TestCase):
    def test_diagnostic_reads_whole_pretty_receipt_without_accepting_failed_check(self):
        from reporting.berrutti import monitor_status as monitor

        receipt = {
            "phase": "performance_identity_diagnostic_completed",
            "success": False,
            "generation_checked": False,
            "optimization_accepted": False,
            "kernel_comparisons": [{"private_metric": n} for n in range(128)],
        }
        payload = json.dumps(receipt, indent=2)
        self.assertGreater(len(payload.splitlines()), 64)

        def read(command):
            return (
                payload
                if command.startswith("head -c 131072 ")
                else "\n".join(payload.splitlines()[-64:])
            )

        for source in (
            {
                "events_path": "/private/diagnostic/receipt",
                "receipt_kind": "performance_diagnostic",
            },
            {"events_path": "/private/diagnostic/diagnosis.json"},
        ):
            spec = {"name": "diagnostic", "job_id": 120, **source}
            with (
                self.subTest(source=source),
                patch.object(monitor, "job_states", return_value={120: "FAILED"}),
                patch.object(monitor, "pending_queue", return_value={}),
                patch.object(monitor, "remote", side_effect=read),
            ):
                result = read_performance_checks([spec], "Ruche")["diagnostic"]
            self.assertEqual(result["state"], "FAILED")
            self.assertFalse(result["completed"])
            self.assertFalse(result["latest_progress"]["success"])
            self.assertFalse(result["latest_progress"]["generation_checked"])
            self.assertFalse(result["latest_progress"]["optimization_accepted"])
            self.assertNotIn("private_metric", json.dumps(result))

    def test_bad_receipt_preserves_other_checks_and_does_not_approve_completion(self):
        from reporting.berrutti import monitor_status as monitor

        specs = [
            {
                "name": name,
                "job_id": job,
                "events_path": f"/private/{name}/completed.json",
                "receipt_kind": "performance_diagnostic",
            }
            for name, job in (("bad", 120), ("good", 121))
        ]
        for payload in ('{\n  "success": true,', "[]"):
            with (
                self.subTest(payload=payload),
                patch.object(
                    monitor,
                    "job_states",
                    return_value={120: "COMPLETED", 121: "COMPLETED"},
                ),
                patch.object(monitor, "pending_queue", return_value={}),
                patch.object(
                    monitor,
                    "remote",
                    side_effect=[
                        payload,
                        json.dumps({"phase": "preflight_completed", "success": True}),
                    ],
                ),
            ):
                result = read_performance_checks(specs, "Ruche")
            self.assertEqual(result["bad"]["state"], "COMPLETED")
            self.assertFalse(result["bad"]["completed"])
            self.assertIsNone(result["bad"]["latest_progress"])
            self.assertIn("monitor_error", result["bad"])
            self.assertTrue(result["good"]["completed"])
            self.assertNotIn("/private/", json.dumps(result))

    def test_arm_audit_reads_closed_receipt_and_preserves_incomplete_test(self):
        from reporting.berrutti import monitor_status as monitor

        receipt = {
            key: True
            for key in (
                "success",
                "execution_verified",
                "diagnostic_only",
                "committed_updates_contiguous_without_repetition",
                "all_epoch_checkpoint_sha256_and_crc_valid",
                "source_hashes_match",
                "baseline_identical",
                "selection_uses_complete_validation",
                "test_only_selected_checkpoint",
                "generation_attempt_policy_preserved",
                "test_cer_provisional",
            )
        }
        receipt.update(
            result_complete=False,
            epochs_verified=3,
            optimizer_updates_performed=0,
            test={"cer": 0.2, "failed_pages": 1},
            private_path="/private/checkpoint",
        )
        specs = [
            {
                "name": "combined_final_audit",
                "job_id": 120,
                "events_path": "/private/audit/completed.json",
                "receipt_kind": "arm_verification",
            }
        ]
        with (
            patch.object(monitor, "job_states", return_value={120: "COMPLETED"}),
            patch.object(monitor, "pending_queue", return_value={}),
            patch.object(monitor, "remote", return_value=json.dumps(receipt)) as remote,
        ):
            result = read_performance_checks(specs, "Ruche")["combined_final_audit"]
        self.assertIn("/private/audit/completed.json", remote.call_args.args[0])
        self.assertTrue(remote.call_args.args[0].startswith("head -c 131072 "))
        self.assertTrue(result["completed"])
        self.assertFalse(result["latest_progress"]["result_complete"])
        self.assertTrue(result["latest_progress"]["test_cer_provisional"])
        self.assertNotIn('cer"', json.dumps(result))
        self.assertNotIn("/private/", json.dumps(result))
        for state in ("RUNNING", "FAILED", "TIMEOUT"):
            self.assertFalse(
                monitor.arm_verification_summary(json.dumps(receipt), state)[
                    "completed"
                ]
            )
        for changes in ({"success": False}, {"optimizer_updates_performed": 1}):
            self.assertFalse(
                monitor.arm_verification_summary(
                    json.dumps({**receipt, **changes}), "COMPLETED"
                )["completed"]
            )

    def test_checks_keep_cluster_namespaces_separate_and_restore_after_failure(self):
        from reporting.berrutti import monitor_status as monitor

        self.addCleanup(configure_cluster, "Ruche")
        specs = [
            {"name": "ruche", "job_id": 120, "events_path": "/private/ruche"},
            {
                "name": "uy",
                "cluster": "ClusterUY",
                "job_id": 120,
                "events_path": "/private/uy",
            },
        ]

        def reply(command):
            alias = monitor.SSH[-1]
            if command.startswith("squeue"):
                return "120|RUNNING\n" if alias == "ruche" else "120|COMPLETED\n"
            if command.startswith("sacct"):
                return ""
            self.assertIn(
                "/private/" + ("uy" if alias == "clusteruy" else alias), command
            )
            return json.dumps({"phase": "runtime_and_data_verified", "success": True})

        with patch.object(monitor, "remote", side_effect=reply):
            result = read_performance_checks(specs, "Ruche")
        self.assertFalse(result["ruche"]["completed"])
        self.assertTrue(result["uy"]["completed"])
        self.assertEqual(result["uy"]["cluster"], "ClusterUY")
        self.assertEqual(monitor.SSH[-1], "ruche")

        def disconnected(command):
            if monitor.SSH[-1] == "clusteruy":
                raise subprocess.TimeoutExpired("private hostname", 40)
            return reply(command)

        with patch.object(monitor, "remote", side_effect=disconnected):
            result = read_performance_checks(specs, "Ruche")
        self.assertEqual(result["ruche"]["state"], "RUNNING")
        self.assertEqual(result["uy"]["state"], "UNKNOWN")
        self.assertNotIn("private hostname", json.dumps(result))
        self.assertEqual(monitor.SSH[-1], "ruche")

    def test_pretty_preparation_receipt_requires_scheduler_completion(self):
        loading = performance_summary(
            '{"event":"phase","phase":"model_loading"}', "RUNNING"
        )
        self.assertEqual(loading["latest_progress"]["phase"], "model_loading")
        self.assertFalse(loading["completed"])
        receipt = json.dumps(
            {"phase": "runtime_and_data_verified", "success": True, "gpu_check": False},
            indent=2,
        )
        self.assertFalse(performance_summary(receipt, "RUNNING")["completed"])
        result = performance_summary(receipt, "COMPLETED")
        self.assertTrue(result["completed"])
        self.assertFalse(result["latest_progress"]["gpu_check"])

    def test_assay_failure_preserves_phase_after_successful_gpu_runtime(self):
        payload = "\n".join(
            json.dumps(row)
            for row in (
                {"event": "gpu_probe", "success": True, "cuda_device_count": 2},
                {
                    "event": "failed",
                    "diagnosis": {
                        "phase": "training_pages_epoch_1",
                        "error_type": "RuntimeError",
                        "private_traceback": "must not escape",
                    },
                },
            )
        )
        result = performance_summary(payload, "FAILED")
        self.assertFalse(result["completed"])
        self.assertEqual(
            result["latest_progress"],
            {
                "phase": "training_pages_epoch_1",
                "error_type": "RuntimeError",
                "success": False,
            },
        )
        self.assertNotIn("must not escape", json.dumps(result))

    def test_memory_recovery_requires_one_committed_update_reload_and_generation(self):
        receipt = {
            "event": "memory_recovery_completed",
            "success": True,
            "training_committed": True,
            "checkpoint_reload": True,
            "generation_checked": True,
            "optimizer_steps_before": 9,
            "optimizer_steps_after": 10,
            "selected_checkpoint": "/private/checkpoint",
            "cer": 0.01,
        }
        for state in ("PENDING", "RUNNING", "FAILED", "TIMEOUT"):
            self.assertFalse(
                performance_summary(json.dumps(receipt), state)["completed"]
            )
        accepted = performance_summary(json.dumps(receipt), "COMPLETED")
        self.assertTrue(accepted["completed"])
        self.assertNotIn("cer", accepted["latest_progress"])
        self.assertNotIn("/private/", json.dumps(accepted))
        for changes in (
            {"training_committed": False},
            {"checkpoint_reload": False},
            {"generation_checked": False},
            {"optimizer_steps_after": 9},
            {"optimizer_steps_after": 11},
            {"optimizer_steps_before": None},
        ):
            with self.subTest(changes=changes):
                self.assertFalse(
                    performance_summary(
                        json.dumps({**receipt, **changes}), "COMPLETED"
                    )["completed"]
                )

    def test_scheduler_queries_use_only_the_configured_cluster(self):
        self.addCleanup(configure_cluster, "Ruche")
        with patch("reporting.berrutti.monitor_status.subprocess.run") as run:
            run.return_value.stdout = "120|PENDING\n"
            for cluster, alias in (("ClusterUY", "clusteruy"), ("Ruche", "ruche")):
                configure_cluster(cluster)
                run.reset_mock()
                self.assertEqual(job_states([120]), {120: "PENDING"})
                for call in run.call_args_list:
                    self.assertEqual(call.args[0][-2], alias)
                    self.assertIn("120", call.args[0][-1])
            run.reset_mock()
            with self.assertRaises(ValueError):
                configure_cluster("clusteruy -o ProxyCommand=unexpected")
            run.assert_not_called()

    def test_independent_workers_cover_each_page_exactly_once(self):
        pages = [f"page-{index}" for index in range(25)]
        shards = [page_shard(pages, rank, 2) for rank in range(2)]
        self.assertEqual([len(shard) for shard in shards], [13, 12])
        self.assertEqual(set(shards[0]) | set(shards[1]), set(pages))
        self.assertFalse(set(shards[0]) & set(shards[1]))
        with self.assertRaises(ValueError):
            page_shard(pages + pages[:1], 0, 2)
        with self.assertRaises(ValueError):
            page_shard(pages, 2, 2)

    @staticmethod
    def row(page, characters, errors, **changes):
        return {
            "page_id": page,
            "char_count": characters,
            "char_errors": errors,
            "word_count": characters,
            "word_errors": errors,
            "status": "complete",
            "attempts": 1,
            "evaluation_key": "fixed-checkpoint-and-policy",
            **changes,
        }

    def test_parallel_micro_cer_matches_serial_with_unequal_reference_lengths(self):
        rows = [self.row("a", 100, 0), self.row("b", 1, 1)]
        serial = merge_page_metrics([rows], ["a", "b"], "fixed-checkpoint-and-policy")
        parallel = merge_page_metrics(
            [[rows[1]], [rows[0]]], ["a", "b"], "fixed-checkpoint-and-policy"
        )
        self.assertEqual(serial, parallel)
        self.assertAlmostEqual(parallel["cer"], 1 / 101)

    def test_missing_and_duplicate_pages_are_rejected_and_failures_are_preserved(self):
        first = self.row("a", 100, 3)
        failed = self.row("b", 10, 12, status="incomplete", attempts=2)
        for shards in ([[first]], [[first], [first, failed]]):
            with self.assertRaises(ValueError):
                merge_page_metrics(shards, ["a", "b"], "fixed-checkpoint-and-policy")
        merged = merge_page_metrics(
            [[first], [failed]], ["a", "b"], "fixed-checkpoint-and-policy"
        )
        self.assertEqual((merged["failed_pages"], merged["retried_pages"]), (1, 1))
        self.assertFalse(merged["all_pages_completed"])
        self.assertAlmostEqual(merged["cer"], 15 / 110)
        with self.assertRaises(ValueError):
            merge_page_metrics(
                [[{**first, "attempts": 3}], [failed]],
                ["a", "b"],
                "fixed-checkpoint-and-policy",
            )
        with self.assertRaises(ValueError):
            merge_page_metrics(
                [[first], [{**failed, "evaluation_key": "other-checkpoint"}]],
                ["a", "b"],
                "fixed-checkpoint-and-policy",
            )

    def test_cache_keys_bind_model_inputs_and_policies(self):
        identity = {"weights": "fixed", "pixels": "original", "dtype": "bf16"}
        self.assertEqual(
            cache_key(identity), cache_key(dict(reversed(list(identity.items()))))
        )
        for field in identity:
            self.assertNotEqual(
                cache_key(identity), cache_key({**identity, field: "changed"})
            )

    def test_array_accounting_does_not_confuse_two_workers_with_one_job(self):
        with patch(
            "reporting.berrutti.monitor_status.remote",
            side_effect=[
                "120_0|RUNNING\n120_1|PENDING\n",
                "120_0|RUNNING|\n120_1|PENDING|\n80|TIMEOUT|\n",
            ],
        ) as remote:
            self.assertEqual(
                job_states(["120_0", "120_1", 80]),
                {"120_0": "RUNNING", "120_1": "PENDING", 80: "TIMEOUT"},
            )
            self.assertIn("squeue -r", remote.call_args_list[0].args[0])
        with self.assertRaises(ValueError):
            job_states(["120; unexpected command"])

    def test_queue_distinguishes_dependencies_and_resources_without_promising_start(
        self,
    ):
        with patch(
            "reporting.berrutti.monitor_status.remote",
            return_value=(
                "120|PENDING|Resources|2026-10-09T16:18:32|7894\n"
                "121|PENDING|Dependency|N/A|7880\n"
                "122|RUNNING|None|2026-10-09T10:00:00|8000\n"
            ),
        ) as remote:
            queued = pending_queue([120, 121, 122])
        self.assertEqual(queued[120]["reason"], "Resources")
        self.assertEqual(
            queued[120]["estimated_start_utc"], "2026-10-09T16:18:32+00:00"
        )
        self.assertTrue(queued[120]["start_is_estimate"])
        self.assertIsNone(queued[121]["estimated_start_utc"])
        self.assertEqual(queued[121]["reason"], "Dependency")
        self.assertNotIn(122, queued)
        self.assertIn("TZ=UTC", remote.call_args.args[0])

    def test_queue_omits_private_free_form_reasons_and_rejects_invalid_ids(self):
        with patch(
            "reporting.berrutti.monitor_status.remote",
            return_value=(
                "120|PENDING|ReqNodeNotAvail, UnavailableNodes:private-node|Unknown|5\n"
                "121|PENDING|private node details|Unknown|5\n"
            ),
        ) as remote:
            queued = pending_queue([120, 121])
            self.assertEqual(queued[120]["reason"], "ReqNodeNotAvail")
            self.assertEqual(queued[121]["reason"], "Other")
            self.assertNotIn("private", json.dumps(queued))
            with self.assertRaises(ValueError):
                pending_queue(["120; unexpected command"])
            self.assertEqual(remote.call_count, 1)

    def test_optimization_receipt_needs_slurm_completion_and_omits_private_text(self):
        receipt = json.dumps(
            {
                "event": "performance_assay_completed",
                "success": True,
                "optimizer_steps": 0,
                "prediction": "private OCR",
                "path": "/private/path",
                "full_generated_pages": 1,
            }
        )
        self.assertFalse(performance_summary(receipt, "TIMEOUT")["completed"])
        complete = performance_summary(receipt, "COMPLETED")
        self.assertTrue(complete["completed"])
        self.assertNotIn("private", json.dumps(complete))
        diagnostic = performance_summary(
            json.dumps(
                {
                    "event": "performance_identity_diagnostic_completed",
                    "success": True,
                    "optimization_accepted": False,
                    "training_identity_passed": False,
                    "generation_checked": False,
                }
            ),
            "COMPLETED",
        )
        self.assertTrue(diagnostic["completed"])
        self.assertFalse(diagnostic["latest_progress"]["optimization_accepted"])
        self.assertFalse(diagnostic["latest_progress"]["generation_checked"])

    def test_parallel_crop_preparation_preserves_pixels_labels_and_full_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.new("RGB", (9, 7), (37, 83, 141))
            image.putpixel((4, 2), (1, 2, 3))
            image.save(root / "page.png")
            digest = hashlib.sha256((root / "page.png").read_bytes()).hexdigest()
            (root / "manifest.json").write_text(
                json.dumps({"pages": [{"page_id": "source", "image": "page.png"}]})
            )
            target = "  é  \n original label  "
            line = {
                "region_index": 0,
                "crop_xyxy": [2, 1, 8, 5],
                "target": target,
                "target_sha256": hashlib.sha256(target.encode()).hexdigest(),
            }
            source = {
                "pages": [
                    {"page_id": "source", "image_sha256": digest, "regions": [line]}
                ],
                "summary": {"train_pages": 1, "train_lines": 1},
            }
            cohort = root / "cohort.json"
            cohort.write_text(json.dumps(source))
            output = root / "cache.sqlite"
            summary = prepare_line_cache(root, cohort, output, 2)
            self.assertEqual((summary["source_pages"], summary["line_images"]), (1, 1))
            self.assertTrue(summary["complete"])
            with sqlite3.connect(output) as db:
                saved_target, payload = db.execute(
                    "SELECT target, png FROM crops"
                ).fetchone()
                self.assertIsNotNone(
                    db.execute(
                        "SELECT value FROM metadata WHERE key='closed_receipt'"
                    ).fetchone()
                )
            self.assertEqual(saved_target, target)
            with Image.open(io.BytesIO(payload)) as cached:
                self.assertEqual(
                    cached.tobytes(), image.crop(line["crop_xyxy"]).tobytes()
                )
            with self.assertRaises(FileExistsError):
                prepare_line_cache(root, cohort, output, 1)
            source["summary"]["train_lines"] = 2
            cohort.write_text(json.dumps(source))
            with self.assertRaises(ValueError):
                prepare_line_cache(root, cohort, root / "partial.sqlite", 1)
            with sqlite3.connect(root / "partial.sqlite") as db:
                self.assertIsNone(
                    db.execute(
                        "SELECT value FROM metadata WHERE key='closed_receipt'"
                    ).fetchone()
                )


if __name__ == "__main__":
    unittest.main()

"""Viewer export preserves closed evidence, exact crops and publication boundaries."""

import hashlib
import io
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from reporting.berrutti.results_viewer import (
    Bundle,
    baseline_status,
    build,
    export_evaluation,
    jailed_file,
    method_status,
    public_summary,
    training_history,
    validate_cohort,
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False))


class ResultsViewer(unittest.TestCase):
    @unittest.skipUnless(
        shutil.which("node"), "Node.js is required for viewer alignment"
    )
    def test_browser_alignment_and_baseline_contract(self):
        result = subprocess.run(
            ["node", "--test", str(Path(__file__).with_name("viewer_alignment.cjs"))],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def dataset(self):
        dataset = self.root / "dataset"
        dataset.mkdir()
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), (41, 67, 98)).save(buffer, "PNG")
        image = buffer.getvalue()
        (dataset / "image.png").write_bytes(image)
        (dataset / "ref.txt").write_text("téxto humano")
        image_hash = hashlib.sha256(image).hexdigest()
        reference_hash = hashlib.sha256("téxto humano".encode()).hexdigest()
        splits = {"train": [], "validation": [], "test": []}
        rows, cohort = [], []
        for roll, (split, count) in enumerate(
            (("train", 125), ("validation", 25), ("test", 25)), 1
        ):
            for index in range(count):
                page_id = f"high_r{roll:04d}_{index:04d}"
                row = {
                    "page_id": page_id,
                    "image": "image.png",
                    "image_sha256": image_hash,
                    "reference": "ref.txt",
                    "reference_sha256": reference_hash,
                }
                rows.append(row)
                splits[split].append(page_id)
                if split == "train":
                    regions = [
                        {
                            "region_index": region,
                            "crop_xyxy": [1, 2, 6, 5],
                            "target": "línea humana",
                            "target_sha256": hashlib.sha256(
                                "línea humana".encode()
                            ).hexdigest(),
                        }
                        for region in range(42 if index < 87 else 41)
                    ]
                    cohort.append({**row, "source_size": [8, 8], "regions": regions})
        write(dataset / "manifest.json", {"content_sha256": "sealed", "pages": rows})
        split_path, cohort_path = self.root / "split.json", self.root / "cohort.json"
        write(split_path, {"dataset_content_sha256": "sealed", "splits": splits})
        write(cohort_path, {"summary": {"train_lines": 5212}, "pages": cohort})
        return dataset, split_path, cohort_path

    def evaluation(self):
        root = self.root / "evaluation"
        pages = [{"page_id": f"page-{i:02d}", "reference_text": "a"} for i in range(25)]
        totals = {
            "char_errors": 0,
            "char_count": 25,
            "word_errors": 0,
            "word_count": 25,
        }
        for index, page in enumerate(pages):
            failed = index == 0
            source = root / "pages" / page["page_id"]
            attempts = 2 if failed else 1
            write(
                source / "closed.json",
                {
                    "page_id": page["page_id"],
                    "evaluation_key": "private-checkpoint-identity",
                    "status": "incomplete" if failed else "complete",
                    "attempts": attempts,
                    "char_errors": int(failed),
                    "char_count": 1,
                    "word_errors": int(failed),
                    "word_count": 1,
                    "runtime_path": "/private/not-for-publication",
                },
            )
            for number in range(1, attempts + 1):
                write(
                    source / f"attempt-{number}.json",
                    {
                        "attempt": number,
                        "prediction": "first attempt"
                        if failed and number == 1
                        else "a"
                        if not failed
                        else "b",
                        "stop_reason": "repetition" if failed else "eos",
                        "seconds": 2.0,
                        "job_id": 99999,
                        "parameters": {"max_new_tokens": 4096},
                    },
                )
        totals.update(char_errors=1, word_errors=1)
        write(
            root / "closed.json",
            {
                **totals,
                "pages": 25,
                "failed_pages": 1,
                "retried_pages": 1,
                "all_pages_completed": False,
                "cer": 0.04,
                "wer": 0.04,
                "epoch": 2,
                "split": "test",
                "stage": "lines",
                "private_path": "/private",
            },
        )
        return root, pages

    def test_full_cohort_exact_training_crops_closed_bundle_and_create_only(self):
        dataset, split, cohort = self.dataset()
        evidence = self.root / "evidence"
        evidence.mkdir()
        assets = self.root / "assets"
        assets.mkdir()
        for name in ("index.html", "app.js", "style.css"):
            (assets / name).write_text("fixture")
        snapshot = self.root / "snapshot.json"
        write(
            snapshot,
            {
                "primary_experiments": {
                    "pages": {
                        "state": "PLANNED",
                        "job_id": 3333,
                        "checkpoint": "/private/checkpoint",
                    }
                }
            },
        )
        output = self.root / "bundle"
        result = build(
            dataset, split, cohort, evidence, output, assets=assets, snapshot=snapshot
        )
        self.assertEqual((result["pages"], result["berrutti_lines"]), (175, 5212))
        catalog = json.loads((output / "catalog.json").read_text())
        self.assertEqual(
            CounterSplits(catalog["pages"]),
            {"train": 125, "validation": 25, "test": 25},
        )
        page = next(row for row in catalog["pages"] if row["split"] == "train")
        detail = json.loads((output / page["detail"]).read_text())
        self.assertEqual(detail["reference"], "téxto humano")
        self.assertEqual(detail["reference_normalized"], "texto humano")
        self.assertNotIn("prediction", detail)
        line = detail["lines"][0]
        self.assertEqual(line["reference"], "línea humana")
        self.assertEqual(line["box"], [1, 2, 5, 3])
        with Image.open(output / line["image"]) as crop:
            self.assertEqual((crop.size, crop.getpixel((0, 0))), ((5, 3), (41, 67, 98)))
        manifest = json.loads((output / "manifest.json").read_text())
        actual = {
            p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()
        }
        self.assertEqual(actual, set(manifest["files"]) | {"manifest.json"})
        self.assertEqual(manifest["artifact_type"], "berrutti-benchmark-viewer")
        for path in output.rglob("*.json"):
            payload = path.read_text()
            self.assertNotIn("/private", payload)
            self.assertNotIn('"job_id"', payload)
        with self.assertRaises(FileExistsError):
            build(dataset, split, cohort, evidence, output, assets=assets)

    def test_incomplete_last_attempt_is_preserved_and_private_fields_dropped(self):
        root, pages = self.evaluation()
        bundle = Bundle(self.root / "bundle")
        exported = export_evaluation(bundle, root, "lines", 2, "test", pages)
        self.assertEqual(exported["summary"]["pages_completed"], 24)
        self.assertTrue(exported["summary"]["provisional"])
        first = json.loads(
            (bundle.root / exported["documents"][0]["detail"]).read_text()
        )
        self.assertEqual(first["prediction"], "b")
        self.assertFalse(first["complete"])
        self.assertTrue(first["failed"])
        self.assertEqual(len(first["attempts"]), 2)
        for path in bundle.root.rglob("*.json"):
            self.assertNotIn("private", path.read_text())
            self.assertNotIn("job_id", path.read_text())
        write(root / "pages" / "page-00" / "attempt-3.json", {"attempt": 3})
        with self.assertRaisesRegex(ValueError, "extra generation"):
            export_evaluation(bundle, root, "lines", 2, "test", pages)

    def test_changed_reference_counts_or_split_coverage_are_rejected(self):
        root, pages = self.evaluation()
        bundle = Bundle(self.root / "bundle")
        pages[0]["reference_text"] = "different reference"
        with self.assertRaisesRegex(ValueError, "reference counts"):
            export_evaluation(bundle, root, "lines", 2, "test", pages)
        pages[0]["reference_text"] = "a"
        (root / "pages" / "foreign").mkdir()
        with self.assertRaisesRegex(ValueError, "cover the split"):
            export_evaluation(bundle, root, "lines", 2, "test", pages)

    def test_selection_ignores_intermediate_and_incomplete_metrics(self):
        rows = []
        for epoch, cer in ((1, 0.03), (2, 0.02), (3, 0.04), (4, 0.05), (104, 0)):
            summary = public_summary(
                {
                    "pages": 25,
                    "failed_pages": 0,
                    "retried_pages": 0,
                    "all_pages_completed": True,
                    "cer": cer,
                }
            )
            rows.append(
                {
                    "method": "experiment4",
                    "epoch": epoch,
                    "split": "validation",
                    "summary": summary,
                }
            )
        status = method_status("experiment4", {}, rows, self.root)
        self.assertEqual(status["selection"], {"epoch": 2, "validation_cer": 0.02})
        self.assertEqual(status["epochs_completed"], 4)
        rows[1]["summary"].update(all_pages_completed=False, provisional=True)
        status = method_status("experiment4", {}, rows, self.root)
        self.assertEqual(status["selection"]["epoch"], 1)
        self.assertEqual(status["epochs_completed"], 3)
        rows.append(
            {
                "method": "experiment4",
                "epoch": 104,
                "split": "test",
                "summary": rows[0]["summary"],
            }
        )
        with self.assertRaisesRegex(ValueError, "minimum complete"):
            method_status("experiment4", {}, rows, self.root)

    def test_baseline_closed_receipts_do_not_hide_incomplete_generation(self):
        rows = [
            {
                "id": f"initial-{split}",
                "method": "experiment4_baseline",
                "split": split,
                "summary": {"all_pages_completed": split == "validation"},
            }
            for split in ("validation", "test")
        ]
        self.assertEqual(
            baseline_status(rows, "experiment4_baseline")["state"], "INCOMPLETE"
        )
        rows[1]["summary"]["all_pages_completed"] = True
        self.assertEqual(
            baseline_status(rows, "experiment4_baseline")["state"], "COMPLETED"
        )
        self.assertEqual(
            baseline_status([], "experiment4_baseline")["state"], "PLANNED"
        )
        self.assertEqual(
            baseline_status(rows[:1], "experiment4_baseline")["state"], "RUNNING"
        )

    def test_finite_metrics_and_scheduler_completion_do_not_hide_failure(self):
        base = {
            "pages": 25,
            "failed_pages": 1,
            "retried_pages": 2,
            "all_pages_completed": False,
            "cer": 0.2,
        }
        for value in (float("inf"), float("nan"), -1, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                public_summary({**base, "cer": value})
        summary = public_summary(base)
        row = {"method": "lines", "epoch": 2, "split": "test", "summary": summary}
        validations = [
            {
                "method": "lines",
                "epoch": epoch,
                "split": "validation",
                "summary": {
                    **summary,
                    "cer": 0.02 if epoch == 2 else 0.04,
                    "all_pages_completed": True,
                },
            }
            for epoch in (1, 2, 3)
        ]
        status = method_status(
            "lines",
            {"runs": {"lines": {"state": "COMPLETED"}}},
            validations + [row],
            self.root,
        )
        self.assertEqual(status["state"], "INCOMPLETE_TEST")
        self.assertTrue(status["test"]["provisional"])

    def test_escaped_symlinks_missing_lines_and_changed_labels_fail(self):
        outside = self.root / "outside"
        outside.write_text("sensitive")
        source = self.root / "source"
        source.mkdir()
        (source / "link").symlink_to(outside)
        with self.assertRaises(ValueError):
            jailed_file(source, "link")
        with self.assertRaises(ValueError):
            jailed_file(source, "../outside")
        dataset, split, cohort = self.dataset()
        from data.berrutti import load_pages

        pages = load_pages(dataset, split)
        value = json.loads(cohort.read_text())
        value["pages"][0]["regions"][0]["target"] = "changed"
        with self.assertRaisesRegex(ValueError, "reference changed"):
            validate_cohort(value, pages)
        value["pages"][0]["regions"].pop()
        with self.assertRaisesRegex(ValueError, "5212"):
            validate_cohort(value, pages)

    def test_training_curve_labels_partial_recovery_and_omits_operational_fields(self):
        event = {
            "event": "optimizer_step",
            "epoch": 1,
            "optimizer_steps": 10,
            "page": 80,
            "loss": 0.1,
            "job_id": 123,
            "checkpoint": "/private",
        }
        path = self.root / "combined" / "events.jsonl"
        path.parent.mkdir()
        path.write_text(json.dumps(event) + "\n")
        history = training_history(self.root, "combined")
        self.assertFalse(history["updates_complete"])
        self.assertEqual(
            history["updates"],
            [{"epoch": 1, "step": 10, "source_pages": 80, "loss": 0.1}],
        )
        self.assertNotIn("private", json.dumps(history))
        self.assertNotIn("job_id", json.dumps(history))
        event["loss"] = float("nan")
        path.write_text(json.dumps(event) + "\n")
        with self.assertRaises(ValueError):
            training_history(self.root, "combined")

    def test_running_extension_exposes_confirmed_updates_before_final_selection(self):
        path = self.root / "experiment4" / "events.jsonl"
        path.parent.mkdir()
        path.write_text(
            json.dumps(
                {
                    "event": "optimizer_step",
                    "epoch": 2,
                    "optimizer_steps": 25,
                    "page": 72,
                    "loss": 0.36,
                }
            )
            + "\n"
        )
        source = {"additional_experiments": {"experiment4": {"state": "RUNNING"}}}
        status = method_status("experiment4", source, [], self.root)
        self.assertEqual(status["state"], "RUNNING")
        self.assertEqual(status["optimizer_steps"], 25)
        self.assertIsNone(status["selection"])
        self.assertIsNone(status["test"])
        self.assertEqual(status["epochs_completed"], 0)


def CounterSplits(pages):
    return {
        split: sum(row["split"] == split for row in pages)
        for split in ("train", "validation", "test")
    }


if __name__ == "__main__":
    unittest.main()

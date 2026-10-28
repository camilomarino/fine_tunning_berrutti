"""Stage counts and ETA must describe current work, including existing workers."""

from __future__ import annotations

import json
import unittest
import zipfile
from datetime import datetime, timezone
from unittest.mock import patch

from tests.data.test_archive_library import LibraryFixture
from archive_berrutti import library, manage as core, monitor, tui


class Estimates(unittest.TestCase):
    def test_recent_speed_excludes_reused_units_and_rejects_stale_or_reset_samples(
        self,
    ):
        samples = [(100, 1000), (102, 1001), (104, 1002)]
        self.assertEqual(monitor.estimate(samples, 1002, 1010, 104), (0.5, 16))
        self.assertEqual(monitor.estimate(samples, 1002, 1010, 200), (None, None))
        self.assertEqual(monitor.estimate(samples, 1010, 1010, 104), (None, None))
        self.assertEqual(
            monitor.estimate(samples + [(106, 0), (108, 1)], 1, 1010, 108), (None, None)
        )
        self.assertEqual(monitor.estimate(samples[:1], 1000, 1010, 100), (None, None))


class StageTests(LibraryFixture):
    def test_existing_worker_uses_its_current_zip_total_and_recent_page_timestamps(
        self,
    ):
        output = self.base / "render"
        output.mkdir()
        records = [
            {
                "event": "page_complete",
                "page_id": str(i),
                "at_utc": datetime.fromtimestamp(at, timezone.utc).isoformat(),
            }
            for i, at in enumerate([20, 22, 100, 102, 104])
        ]
        journal = output / "events.jsonl"
        journal.write_text(
            "\n".join(json.dumps(record) for record in records) + '\n{"unfinished":'
        )
        roll = {"roll": "099", "files": list(enumerate(self.data["files"], 1))}
        local = [
            library.status(self.root, row, include_renders=False)
            for row in self.data["files"]
        ]
        final = [
            {"images": 900, "expected": 900, "output": str(self.base / "other")},
            {"images": 54, "expected": 100, "output": str(output)},
        ]
        job = {
            "phase": "running",
            "created": 100,
            "message": f"Convirtiendo Rollo 099 → {output}",
            "job": str(self.base / "job"),
        }
        with patch.object(monitor.time, "time", return_value=104):
            progress = monitor.snapshot(self.root, roll, local, final, job)
        self.assertEqual(
            (progress["step"], progress["zip_index"], progress["zip_total"]), (3, 2, 2)
        )
        self.assertEqual(
            (progress["done"], progress["total"], progress["percent"]), (54, 100, 54)
        )
        self.assertEqual(progress["eta_seconds"], 92)
        self.assertIn(
            "3/3 Imágenes 54/100",
            tui.state_label({"active": True, "progress": progress}),
        )
        self.assertIn("ZIP 2/2", tui.progress_line({"progress": progress}))
        self.assertTrue(journal.read_text().endswith('{"unfinished":'))

    def test_extraction_exposes_pdf_total_partial_bytes_and_observed_eta(self):
        row = self.data["files"][0]
        self.operate([row], "download")
        directory = library.entry_root(self.root, row)
        with zipfile.ZipFile(
            core.zip_path(directory, library.local_entry(row))
        ) as archive:
            total = sum(info.file_size for info in archive.infolist())
        pdfs = directory / "pdfs" / row["key"]
        pdfs.mkdir(parents=True)
        (pdfs / "0001-example.pdf.part").write_bytes(b"x" * (total // 2))
        job = {
            "phase": "running",
            "stage": {"name": "extract", "key": row["key"], "started": 100},
            "job": "synthetic",
        }
        progress = monitor.snapshot(
            self.root,
            {"roll": "099", "files": [(1, row)]},
            [library.status(self.root, row, include_renders=False)],
            [{"output": None}],
            job,
        )
        self.assertEqual(
            (progress["step"], progress["done"], progress["total"]), (2, 0, 1)
        )
        self.assertEqual(
            (progress["bytes_done"], progress["bytes_total"]), (total // 2, total)
        )
        self.assertIn(
            "Extrayendo PDF 0/1",
            tui.state_label({"active": True, "progress": progress}),
        )
        windows = {}
        for instant, done in [(100, 0), (102, 250), (104, 500)]:
            current = {**progress, "bytes_done": done, "bytes_total": 1000}
            with patch.object(monitor.time, "time", return_value=instant):
                monitor.observe(current, "job-one", windows)
        self.assertEqual(current["eta_seconds"], 4)
        with patch.object(monitor.time, "time", return_value=106):
            current["eta_seconds"] = None
            monitor.observe(current, "new-job", windows)
        self.assertIsNone(current["eta_seconds"])

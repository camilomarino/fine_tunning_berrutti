"""Exercise the machine contract, real queued workers and bounded downloads."""

from __future__ import annotations

import contextlib
import fcntl
import io
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from unittest import skipUnless
from unittest.mock import patch

from tests.data.test_archive_data import server
from tests.data.test_archive_library import LibraryFixture

from archive_berrutti import cli, library, manage as core, rolls, workflow

POPPLER = bool(shutil.which("pdfinfo") and shutil.which("pdftoppm"))


class AgentTests(LibraryFixture):
    def invoke(self, command, *arguments, code=0):
        output, errors = io.StringIO(), io.StringIO()
        actual_code = 0
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            try:
                cli.main(["agent", command, "--root", str(self.root), *arguments])
            except SystemExit as error:
                actual_code = error.code
        self.assertEqual(actual_code, code, output.getvalue() + errors.getvalue())
        # json.loads rejects both extra JSON documents and leaked progress text.
        result = json.loads(output.getvalue())
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["ok"], code == 0)
        return result

    def test_queries_are_offline_read_only_paginated_and_have_absolute_paths(self):
        before = sorted(str(path) for path in self.root.rglob("*"))
        with patch.object(core.urllib.request, "urlopen") as network:
            first = self.invoke("list", "--limit", "1")["data"]
            second = self.invoke("list", "--limit", "1", "--offset", "1")["data"]
            self.assertEqual(first["total"], 2)
            self.assertEqual(first["next_offset"], 1)
            self.assertIsNone(second["next_offset"])
            self.assertEqual(first["items"][0]["state"], "not_downloaded")
            self.assertEqual(
                self.invoke("list", "--state", "local")["data"]["total"], 0
            )
            self.assertEqual(
                self.invoke("list", "--search", "absent")["data"]["total"], 0
            )
            info = self.invoke("info", "--rolls", "99")["data"]["items"][0]
            self.assertEqual(info["roll"], "099")
            self.assertEqual(info["next_step"], "download")
            archive = info["archives"][0]["zip"]
            self.assertTrue(Path(archive["path"]).is_absolute())
            self.assertFalse(archive["present"])
            self.assertEqual(self.invoke("files")["data"]["total"], 0)
            missing = self.invoke("files", "--kind", "zip", "--availability", "missing")
            self.assertEqual(missing["data"]["total"], 2)
            self.invoke("plan", "--rolls", "099,100")
            self.invoke("complete", "--all", "--dry-run")
        network.assert_not_called()
        self.assertEqual(before, sorted(str(path) for path in self.root.rglob("*")))

    def test_bad_selection_and_budget_fail_before_any_work_with_json_errors(self):
        cases = [
            ("download", []),
            ("complete", ["--all", "--rolls", "099"]),
            ("download", ["--rolls", "099", "--max-download-gib", "0"]),
            ("download", ["--rolls", "999"]),
            ("complete", ["--rolls", ""]),
            ("complete", ["--rolls", "099,"]),
            ("download", ["--all", "--reserve-gib", "nan"]),
            ("list", ["--offset", "-1"]),
            ("files", ["--limit", "0"]),
            ("job", ["../outside"]),
        ]
        with (
            patch.object(workflow, "start") as start,
            patch.object(core, "fetch") as fetch,
        ):
            for command, args in cases:
                with self.subTest(command=command, args=args):
                    result = self.invoke(command, *args, code=2)
                    self.assertEqual(result["error"]["code"], "invalid_arguments")
        start.assert_not_called()
        fetch.assert_not_called()
        self.assertFalse((self.root / "jobs").exists())

    def test_download_over_http_resumes_and_never_puts_progress_in_stdout(self):
        row = self.data["files"][0]
        payload = self.payloads[row["name"]]
        archive = core.zip_path(
            library.entry_root(self.root, row), library.local_entry(row)
        )
        archive.parent.mkdir(parents=True)
        archive.with_suffix(".zip.part").write_bytes(payload[:7])
        estimate = self.invoke("plan", "--rolls", "99")["data"]
        self.assertEqual(estimate["remaining_download_bytes"], len(payload) - 7)
        original = core.urllib.request.urlopen
        with server(payload) as (url, requests):

            def redirected(request, **kwargs):
                request.full_url = url
                return original(request, **kwargs)

            with patch.object(core.urllib.request, "urlopen", side_effect=redirected):
                result = self.invoke("download", "--rolls", "99", "--reserve-gib", "0")
        self.assertEqual(requests, ["bytes=7-"])
        job = result["data"]["results"][0]["job"]
        self.assertEqual(job["status"]["phase"], "done")
        self.assertEqual(core.hashes(archive)["sha1"], row["sha1"])
        self.assertFalse(archive.with_suffix(".zip.part").exists())
        state = self.invoke("list", "--state", "local")["data"]["items"][0]
        self.assertEqual(state["zips"], {"downloaded": 1, "total": 1})
        self.assertEqual(state["next_step"], "extract")
        self.assertEqual(state["images"]["available"], 0)
        queried = self.invoke("job", job["id"])["data"]
        self.assertEqual(
            queried["paths"]["progress"],
            str(Path(job["paths"]["directory"]) / "progress.jsonl"),
        )
        self.assertEqual(self.invoke("files", "--kind", "zip")["data"]["total"], 1)

    @skipUnless(POPPLER, "Poppler missing")
    def test_complete_lists_all_zips_pdfs_final_images_and_separate_previews(self):
        raw = core.load(self.metadata)
        extra = {**raw["files"][0], "name": "Archivo - Rollo 099 - Anexo.zip"}
        self.payloads[extra["name"]] = self.payloads[raw["files"][0]["name"]]
        raw["files"].append(extra)
        self.metadata.write_text(json.dumps(raw))
        self.data = library.refresh(self.root, self.metadata)
        # A JPEG preview must never count toward final PNG completion.
        self.operate([self.data["files"][0]], "prepare", first_pages=1, dpi=72)
        with patch.object(core, "fetch", side_effect=self.fake_fetch):
            result = self.invoke("complete", "--rolls", "99", "--reserve-gib", "0")
        first_job = result["data"]["results"][0]["job"]
        self.assertEqual(
            core.load(Path(first_job["paths"]["request"]))["options"]["reserve_gib"], 0
        )
        info = self.invoke("info", "--rolls", "99")["data"]["items"][0]
        self.assertTrue(info["complete"])
        self.assertEqual(info["images"], {"available": 6, "expected": 6})
        self.assertEqual(info["zips"], {"downloaded": 2, "total": 2})
        self.assertEqual(info["pdfs"], 2)
        self.assertEqual(len(info["archives"]), 2)
        for entry in info["archives"]:
            self.assertTrue(Path(entry["zip"]["path"]).is_file())
            self.assertTrue(Path(entry["pdfs"][0]["path"]).is_file())
            self.assertTrue(Path(entry["final"]["output"], "images").is_dir())
        images = self.invoke(
            "files", "--rolls", "99", "--kind", "image", "--limit", "2"
        )["data"]
        self.assertEqual((images["total"], images["next_offset"]), (6, 2))
        image = images["items"][0]
        self.assertEqual(
            (image["format"], image["dpi"], image["scope"]), ("png", 200, "final")
        )
        self.assertTrue(image["size_matches"])
        self.assertTrue(image["pdf_member"].endswith(".pdf"))
        self.assertEqual(image["sha256"], core.hashes(Path(image["path"]))["sha256"])
        previews = self.invoke(
            "files", "--rolls", "99", "--kind", "image", "--scope", "other"
        )["data"]
        self.assertEqual(previews["total"], 1)
        self.assertEqual(previews["items"][0]["format"], "jpeg")
        before = Path(image["path"]).stat().st_mtime_ns
        with patch.object(core, "fetch", side_effect=AssertionError("No new download")):
            self.invoke(
                "complete",
                "--rolls",
                "99",
                "--reserve-gib",
                "0",
                "--max-download-gib",
                "0",
            )
        self.assertEqual(Path(image["path"]).stat().st_mtime_ns, before)
        # Exact job IDs remain queryable after a later invocation replaces the pointer.
        self.assertEqual(
            self.invoke("job", first_job["id"])["data"]["id"], first_job["id"]
        )
        Path(image["path"]).unlink()
        missing = self.invoke(
            "files", "--rolls", "99", "--kind", "image", "--availability", "missing"
        )["data"]
        self.assertEqual(missing["total"], 1)
        self.assertFalse(missing["items"][0]["present"])
        broken = self.invoke("list", "--rolls", "99")["data"]["items"][0]
        self.assertEqual(broken["images"]["available"], 5)
        self.assertEqual(broken["state"], "error")

    def test_failure_preserves_job_and_other_batch_results_then_can_retry(self):
        def fetch(url, destination, entry, attempts, reserve, **kwargs):
            if entry["roll"] == "099":
                raise OSError("Synthetic network failure")
            return self.fake_fetch(url, destination, entry, attempts, reserve, **kwargs)

        with patch.object(core, "fetch", side_effect=fetch):
            result = self.invoke(
                "download", "--rolls", "099,100", "--reserve-gib", "0", code=1
            )
        failed, done = result["data"]["results"]
        self.assertEqual(
            (failed["disposition"], done["disposition"]), ("failed", "done")
        )
        job = failed["job"]
        state = self.invoke("job", job["id"])["data"]["status"]
        self.assertEqual(state["phase"], "failed")
        self.assertIn("Synthetic network failure", state["error"])
        self.assertEqual(self.invoke("list", "--state", "error")["data"]["total"], 1)
        with patch.object(core, "fetch", side_effect=self.fake_fetch):
            self.invoke("download", "--rolls", "99", "--reserve-gib", "0")
        self.assertEqual(self.invoke("list", "--state", "error")["data"]["total"], 0)

    @skipUnless(POPPLER, "Poppler missing")
    def test_background_job_outlives_command_and_busy_roll_is_not_queued_twice(self):
        self.operate([self.data["files"][0]], "download")
        (self.root / "operations").mkdir(exist_ok=True)
        job_id = None
        try:
            with (self.root / "operations" / ".worker").open("a") as scheduler:
                fcntl.flock(scheduler, fcntl.LOCK_EX)
                process = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "archive_berrutti",
                        "agent",
                        "complete",
                        "--root",
                        str(self.root),
                        "--rolls",
                        "99",
                        "--background",
                        "--reserve-gib",
                        "0",
                    ],
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
                job_id = json.loads(process.stdout)["data"]["results"][0]["job"]["id"]
                self.assertEqual(
                    self.invoke("job", job_id)["data"]["status"]["phase"], "queued"
                )
                busy = self.invoke("complete", "--rolls", "99", "--background", code=1)
                self.assertIn(job_id, busy["error"]["message"])
                self.assertEqual(len(list((self.root / "jobs").iterdir())), 1)
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                job = self.invoke("job", job_id)["data"]
                if not job["active"]:
                    break
                time.sleep(0.05)
            self.assertEqual(job["status"]["phase"], "done", job)
            self.assertTrue(Path(job["paths"]["log"]).is_file())
            self.assertEqual(
                self.invoke("list", "--state", "complete")["data"]["total"], 1
            )
        finally:
            if job_id:
                workflow.stop_job(self.root / "jobs" / job_id)

    def test_refresh_can_bootstrap_from_local_metadata_without_network(self):
        self.root = self.base / "new-library"
        with patch.object(core.urllib.request, "urlopen") as network:
            result = self.invoke("refresh", "--metadata", str(self.metadata))
        network.assert_not_called()
        self.assertEqual(result["data"]["zips"], 2)
        self.assertFalse((self.root / "rolls").exists())

    def test_job_detects_interruption_and_reports_error_without_relaunching(self):
        roll = rolls.group(self.data)[0]
        with core.locked(self.root / "operations"):
            job = workflow.create_job(self.root, self.data, roll)
        library.replace_json(job / "state.json", {"phase": "running", "created": 0})
        state = self.invoke("job", job.name)["data"]
        self.assertFalse(state["active"])
        self.assertEqual(state["status"]["phase"], "interrupted")
        self.assertTrue(state["status"]["error"])

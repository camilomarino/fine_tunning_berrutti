"""The basic contract: all roll pages, useful retry, bounded deletion, direct UI."""

from __future__ import annotations

import contextlib
import curses
import fcntl
import io
import json
import shutil
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.data.test_archive_library import LibraryFixture
from tests.data.test_archive_data import server

from archive_berrutti import (
    cli,
    library,
    locations,
    manage as core,
    rolls,
    tui,
    workflow,
)


class WorkflowTests(LibraryFixture):
    def roll(self, number="099"):
        return next(roll for roll in rolls.group(self.data) if roll["roll"] == number)

    def complete(self, roll=None):
        with patch.object(core, "fetch", side_effect=self.fake_fetch):
            with contextlib.redirect_stdout(io.StringIO()):
                return workflow.complete_roll(
                    self.root, self.data, roll or self.roll(), progress=lambda _: None
                )

    def job(self, roll=None):
        roll = roll or self.roll()
        job = self.root / "jobs" / library.identifier()
        core.save_new(
            job / "request.json",
            {
                "root": str(self.root),
                "data": self.data,
                "roll": roll,
                "profile": workflow.PROFILE,
            },
        )
        library.replace_json(
            job / "state.json", {"phase": "queued", "created": time.time()}
        )
        library.replace_json(
            workflow.control(self.root, roll) / "current.json",
            {"job": str(job.relative_to(self.root))},
        )
        return job

    def test_starts_with_every_roll_and_cancel_never_downloads(self):
        # Enter -> Complete -> Escape cancels; no basket, tab or selection needed.
        with patch.object(workflow, "start") as start:
            app, screen = self.interface(
                [curses.KEY_RIGHT, curses.KEY_RIGHT, "\x1b", curses.KEY_LEFT, "\x1b"]
            )
        start.assert_not_called()
        self.assertIn("0/2 completos", screen.frames[0])
        self.assertIn("099", screen.frames[0])
        self.assertIn("100", screen.frames[0])
        self.assertIn("Sin descargar", screen.frames[0])
        self.assertNotIn("Pendiente", screen.frames[0])
        header = next(
            line for line in screen.frames[0].splitlines() if line.startswith("Rollo ")
        )
        self.assertEqual(header.split(), ["Rollo", "Imágenes", "Estado"])
        self.assertFalse(app.status(self.roll())["complete"])
        self.assertFalse((self.root / "rolls").exists())

    def test_stopping_finished_job_preserves_its_result(self):
        job = self.job()
        saved = {"phase": "done", "message": "Rollo completo"}
        library.replace_json(job / "state.json", saved)
        with patch.object(workflow.os, "killpg") as kill:
            workflow.stop_job(job)
        kill.assert_not_called()
        self.assertEqual(core.load(job / "state.json"), saved)

    def test_stopping_job_that_just_exited_never_signals_another_process(self):
        job = self.job()
        core.save_new(
            job / "process.json",
            {
                "pid": 99999999,
                "start_ticks": "123",
                "host": workflow.socket.gethostname(),
            },
        )
        with (
            patch.object(workflow, "process_ticks", side_effect=FileNotFoundError),
            patch.object(workflow.os, "killpg") as kill,
        ):
            workflow.stop_job(job)
        kill.assert_not_called()
        self.assertEqual(core.load(job / "state.json")["phase"], "stopped")

    def test_direct_action_ignores_legacy_basket_and_targets_focused_roll(self):
        library.replace_json(
            self.root / "session.json",
            {"selected_keys": [self.data["files"][1]["key"]]},
        )
        with patch.object(workflow, "start") as start:
            self.interface(["\n", "\n", "\n", "\x1b", "\x1b"])
        start.assert_called_once()
        self.assertEqual(start.call_args.args[2]["roll"], "099")

    def test_search_and_completion_filter_do_not_change_action_scope(self):
        with patch.object(workflow, "start") as start:
            _, screen = self.interface(
                [
                    "/",
                    *"100",
                    "\n",
                    "f",
                    curses.KEY_DOWN,
                    "\n",
                    "\n",
                    "\n",
                    "\n",
                    "\x1b",
                    "\x1b",
                ]
            )
        self.assertEqual(start.call_args.args[2]["roll"], "100")
        self.assertIn("Incompletos", "\n".join(screen.frames))

    def test_completion_reorders_list_without_changing_focused_action(self):
        states = {
            number: {
                **workflow.roll_status(self.root, self.roll(number)),
                "complete": number == "100",
                "active": number == "099",
                "label": "Completo" if number == "100" else "Procesando",
            }
            for number in ("099", "100")
        }

        def keys():
            yield curses.KEY_DOWN  # Completed 100 is first; focus running 099.
            states["099"].update(complete=True, active=False, label="Completo")
            yield "r"  # 099 now moves above 100, in alphabetical order.
            yield "c"
            yield "\n"
            yield "\x1b"

        with (
            patch.object(
                workflow,
                "roll_status",
                side_effect=lambda _, r: states[r["roll"]].copy(),
            ),
            patch.object(workflow, "start") as start,
        ):
            _, screen = self.interface(keys())
        initial, refreshed = screen.frames[0], screen.frames[2]
        self.assertLess(initial.index("100"), initial.index("099"))
        self.assertLess(refreshed.index("099"), refreshed.index("100"))
        start.assert_called_once()
        self.assertEqual(start.call_args.args[2]["roll"], "099")

    def test_download_command_persists_transfer_history_and_only_downloads_selected_roll(
        self,
    ):
        row = self.data["files"][0]
        payload = self.payloads[row["name"]]
        destination = core.zip_path(
            library.entry_root(self.root, row), library.local_entry(row)
        )
        destination.parent.mkdir(parents=True)
        destination.with_suffix(".zip.part").write_bytes(payload[:7])
        fetch = core.fetch
        with server(payload) as (url, requests):
            with (
                patch.object(
                    core,
                    "fetch",
                    side_effect=lambda _, *args, **kwargs: fetch(url, *args, **kwargs),
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                cli.main(
                    [
                        "download",
                        "--root",
                        str(self.root),
                        "--rolls",
                        "99",
                        "--reserve-gib",
                        "0",
                    ]
                )
        self.assertEqual(requests, ["bytes=7-"])
        state = workflow.roll_status(self.root, self.roll())
        self.assertEqual(
            (state["job"]["phase"], state["download"]["percent"]), ("done", 100)
        )
        self.assertEqual(state["zip_done"], 1)
        self.assertFalse(state["complete"])
        self.assertEqual((state["pdfs"], state["images"]), (0, 0))
        self.assertEqual(state["next_step"], "extract")
        self.assertEqual(tui.state_label(state), "Falta extraer PDF")
        records = [
            json.loads(line)
            for line in (Path(state["job"]["job"]) / "progress.jsonl")
            .read_text()
            .splitlines()
        ]
        transfers = [record["transfer"] for record in records if record.get("transfer")]
        self.assertEqual(transfers[0]["bytes"], 7)
        self.assertIn("verifying", [item["phase"] for item in transfers])
        self.assertEqual(transfers[-1]["phase"], "done")
        self.assertEqual(
            workflow.roll_status(self.root, self.roll("100"))["label"], "Sin descargar"
        )

    @unittest.skipUnless(
        shutil.which("pdftoppm") and shutil.which("pdfinfo"), "Poppler missing"
    )
    def test_right_opens_roll_files_and_left_returns_without_starting_work(self):
        self.complete()
        resources = locations.resources(self.root, self.roll())
        images = next(
            item["path"]
            for item in resources
            if item["label"] == "Carpeta de imágenes finales"
        )
        pdf = next(item["path"] for item in resources if item["path"].suffix == ".pdf")
        archive = next(
            item["path"] for item in resources if item["path"].suffix == ".zip"
        )
        self.assertTrue(images.is_dir() and pdf.is_file() and archive.is_file())
        right, left = curses.KEY_RIGHT, curses.KEY_LEFT
        with (
            patch.object(workflow, "start") as start,
            patch.dict(
                tui.os.environ,
                {
                    "DISPLAY": "",
                    "WAYLAND_DISPLAY": "",
                    "ARCHIVE_BERRUTTI_VIEWER": "blocks",
                },
            ),
            patch.object(tui.subprocess, "Popen") as viewer,
        ):
            _, screen = self.interface(
                [
                    right,
                    right,
                    right,
                    left,
                    left,
                    curses.KEY_DOWN,
                    "p",
                    left,
                    left,
                    "\x1b",
                ]
            )
        start.assert_not_called()
        viewer.assert_not_called()
        self.assertIn("Carpeta de imágenes finales", screen.frames[1])
        self.assertIn("── ARCHIVOS", screen.frames[1])
        self.assertIn("── ACCIONES", screen.frames[1])
        self.assertLess(
            screen.frames[1].index("── ARCHIVOS"), screen.frames[1].index("ZIP ·")
        )
        self.assertLess(
            screen.frames[1].index("ZIP ·"), screen.frames[1].index("── ACCIONES")
        )
        self.assertIn("PDF ·", screen.frames[1])
        self.assertIn("ZIP ·", screen.frames[1])
        self.assertIn("Ruta completa:", screen.frames[1])
        self.assertIn("/images", screen.frames[1])
        self.assertIn("Carpeta · images", screen.frames[2])
        self.assertIn(".png", screen.frames[2])
        self.assertIn("Imagen 1/3", "\n".join(screen.frames))
        self.assertIn("Vista aproximada por caracteres", "\n".join(screen.frames))
        self.assertIn(pdf.name, "\n".join(screen.frames))

    def test_download_only_queues_and_returns_to_roll_sheet(self):
        down = curses.KEY_DOWN
        with patch.object(workflow, "start") as start:
            self.interface(
                [
                    curses.KEY_RIGHT,
                    "a",
                    down,
                    down,
                    down,
                    down,
                    curses.KEY_RIGHT,
                    curses.KEY_RIGHT,
                    curses.KEY_LEFT,
                    "\x1b",
                ]
            )
        start.assert_called_once()
        self.assertEqual(start.call_args.args[2]["roll"], "099")
        self.assertEqual(start.call_args.kwargs, {"action": "download"})

    def test_open_status_refreshes_partial_bytes_without_user_input(self):
        row = self.data["files"][0]
        archive = core.zip_path(
            library.entry_root(self.root, row), library.local_entry(row)
        )
        archive.parent.mkdir(parents=True)
        partial = archive.with_suffix(".zip.part")
        partial.write_bytes(self.payloads[row["name"]][: row["bytes"] // 4])
        job = self.job()
        library.replace_json(
            job / "state.json",
            {
                "phase": "running",
                "created": time.time(),
                "transfer": {"phase": "downloading"},
            },
        )

        def keys():
            yield "e"
            partial.write_bytes(self.payloads[row["name"]][: row["bytes"] // 2])
            yield None  # Timed refresh, without a keypress.
            yield "\x1b"
            yield "\x1b"

        with (job / ".lease").open("a") as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            _, screen = self.interface(keys())
            _, reopened = self.interface(["e", curses.KEY_NPAGE, "\x1b", "\x1b"])
        quarter = f"{100 * (row['bytes'] // 4) / row['bytes']:.1f}%"
        half = f"{100 * (row['bytes'] // 2) / row['bytes']:.1f}%"
        self.assertIn(f"Descarga {quarter}", screen.frames[0])
        self.assertIn(quarter, screen.frames[1])
        self.assertIn(half, screen.frames[2])
        self.assertIn(half, reopened.frames[1])
        self.assertIn("Descarga parcial", reopened.frames[1])
        self.assertNotIn("› Descarga parcial", reopened.frames[1])

    @unittest.skipUnless(
        shutil.which("pdftoppm") and shutil.which("pdfinfo"), "Poppler missing"
    )
    def test_preview_is_not_complete_and_all_zips_and_pages_are_required(self):
        first = self.data["files"][0]
        self.operate([first], "prepare", first_pages=1, dpi=200, format="png")
        raw = core.load(self.metadata)
        extra = {**raw["files"][0], "name": "Archivo - Rollo 099 - Otra parte.zip"}
        self.payloads[extra["name"]] = self.payloads[first["name"]]
        raw["files"].append(extra)
        self.metadata.write_text(json.dumps(raw))
        self.data = library.refresh(self.root, self.metadata)
        roll = self.roll()
        before = workflow.roll_status(self.root, roll)
        self.assertFalse(before["complete"])
        self.assertEqual(
            (before["zip_done"], before["zip_total"], before["images"]), (1, 2, 0)
        )
        # Even a fully converted first ZIP cannot complete a two-ZIP roll.
        self.complete({"roll": roll["roll"], "files": [roll["files"][0]]})
        partial = workflow.roll_status(self.root, roll)
        self.assertFalse(partial["complete"])
        self.assertEqual(partial["images"], 3)
        self.assertEqual(tui.state_label(partial), "Falta descargar")
        # A PDF from one ZIP must not hide the extraction needed by another.
        second = roll["files"][1][1]
        self.operate([second], "download")
        partial = workflow.roll_status(self.root, roll)
        self.assertEqual(partial["pdfs"], 1)
        self.assertEqual(tui.state_label(partial), "Falta extraer PDF")
        self.operate([second], "extract")
        self.assertEqual(
            tui.state_label(workflow.roll_status(self.root, roll)),
            "Falta convertir a imágenes",
        )
        self.complete(roll)
        state = workflow.roll_status(self.root, roll)
        self.assertTrue(state["complete"])
        self.assertEqual(
            (state["zip_done"], state["pdfs"], state["images"], state["expected"]),
            (2, 2, 6, 6),
        )
        self.assertEqual(len(state["outputs"]), 2)
        resources = locations.resources(self.root, roll)
        self.assertEqual(sum(item["path"].suffix == ".zip" for item in resources), 2)
        self.assertEqual(sum(item["path"].suffix == ".pdf" for item in resources), 2)
        self.assertEqual(
            sum(
                item["label"].startswith("Carpeta de imágenes finales")
                for item in resources
            ),
            2,
        )
        self.assertTrue(
            any(item["label"].startswith("Carpeta de muestras") for item in resources)
        )
        _, screen = self.interface(["\x1b"])
        self.assertIn("1/2 completos", screen.frames[0])
        self.assertIn("Completo", screen.frames[0])
        self.assertIsNone(state["next_step"])

    @unittest.skipUnless(
        shutil.which("pdftoppm") and shutil.which("pdfinfo"), "Poppler missing"
    )
    def test_failed_conversion_keeps_error_and_retry_reuses_completed_pages(self):
        job = self.job()
        original_run = core.run

        def fail_second_page(command, timeout):
            if command[0] == "pdftoppm" and command[command.index("-f") + 1] == "2":
                raise ValueError("smoke: página 2 falló")
            return original_run(command, timeout)

        with (
            patch.object(core, "fetch", side_effect=self.fake_fetch),
            patch.object(core, "run", side_effect=fail_second_page),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            with self.assertRaisesRegex(ValueError, "página 2"):
                workflow.work(job)
        failed = workflow.roll_status(self.root, self.roll())
        self.assertEqual(failed["job"]["stage"]["name"], "render")
        stages = [
            json.loads(line).get("stage", {}).get("name")
            for line in (job / "progress.jsonl").read_text().splitlines()
        ]
        self.assertEqual(
            list(dict.fromkeys(stage for stage in stages if stage)),
            ["verify", "download", "extract", "count", "render"],
        )
        self.assertFalse(failed["complete"])
        self.assertEqual(failed["label"], "Error")
        self.assertEqual((failed["images"], failed["expected"]), (1, 3))
        self.assertIn("página 2", " ".join(failed["errors"]))

        first = next((Path(failed["outputs"][0]) / "images").glob("*.png"))
        timestamp = first.stat().st_mtime_ns
        with (
            patch.object(core, "fetch", side_effect=self.fake_fetch),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            workflow.work(self.job())
        self.assertEqual(first.stat().st_mtime_ns, timestamp)
        self.assertTrue(workflow.roll_status(self.root, self.roll())["complete"])

    @unittest.skipUnless(
        shutil.which("pdftoppm") and shutil.which("pdfinfo"), "Poppler missing"
    )
    def test_missing_final_image_is_incomplete_and_retry_preserves_old_evidence(self):
        self.complete()
        state = workflow.roll_status(self.root, self.roll())

        output = Path(state["outputs"][0])
        first, second, _ = sorted((output / "images").glob("*.png"))
        second_time = second.stat().st_mtime_ns
        first.unlink()
        broken = workflow.roll_status(self.root, self.roll())
        self.assertFalse(broken["complete"])
        self.assertEqual(broken["images"], 2)
        self.complete()
        repaired = workflow.roll_status(self.root, self.roll())
        self.assertTrue(repaired["complete"])
        self.assertEqual(repaired["images"], 3)
        self.assertNotEqual(repaired["outputs"], state["outputs"])
        self.assertTrue((output / "complete.json").exists())
        self.assertEqual(second.stat().st_mtime_ns, second_time)

    @unittest.skipUnless(
        shutil.which("pdftoppm") and shutil.which("pdfinfo"), "Poppler missing"
    )
    def test_two_real_background_jobs_queue_survive_ui_exit_and_reject_active_delete(
        self,
    ):
        # Seed test ZIPs only. Children exercise the real pipeline without network.
        self.operate(self.data["files"], "download")
        (self.root / "operations").mkdir(exist_ok=True)
        with (self.root / "operations" / ".worker").open("a") as scheduler:
            fcntl.flock(scheduler, fcntl.LOCK_EX)
            jobs = [
                workflow.start(self.root, self.data, self.roll(number))
                for number in ("099", "100")
            ]
            with self.assertRaisesRegex(ValueError, "ya está"):
                workflow.start(self.root, self.data, self.roll())
            with self.assertRaisesRegex(ValueError, "procesándose"):
                workflow.remove(self.root, self.roll())
            self.interface(["\x1b"])
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if all(
                core.load(job / "state.json")["phase"] in {"done", "failed"}
                for job in jobs
            ):
                break
            time.sleep(0.05)
        for job in jobs:
            self.assertEqual(
                core.load(job / "state.json")["phase"],
                "done",
                (job / "worker.log").read_text(),
            )
        self.assertTrue(workflow.roll_status(self.root, self.roll())["complete"])
        self.assertTrue(workflow.roll_status(self.root, self.roll("100"))["complete"])

    def test_delete_requires_confirmation_and_only_removes_focused_roll(self):
        self.operate(self.data["files"], "download")
        first, second = self.data["files"]
        self.interface(["d", "\n", "\x1b"])
        self.assertTrue(library.entry_root(self.root, first).exists())
        self.interface(
            [
                curses.KEY_RIGHT,
                curses.KEY_END,
                curses.KEY_UP,
                curses.KEY_RIGHT,
                curses.KEY_DOWN,
                curses.KEY_RIGHT,
                curses.KEY_LEFT,
                "\x1b",
            ]
        )
        self.assertFalse(library.entry_root(self.root, first).exists())
        self.assertTrue(library.entry_root(self.root, second).exists())
        self.assertEqual(library.load_catalog(self.root)["files"], self.data["files"])
        self.assertEqual(
            workflow.roll_status(self.root, self.roll())["label"], "Sin descargar"
        )
        self.assertTrue(list((self.root / "jobs").glob("*/deletion.json")))

    def test_delete_refuses_symlink_outside_library(self):
        target = self.base / "unrelated"
        target.mkdir()
        protected = target / "keep.txt"
        protected.write_text("keep")
        entry = library.entry_root(self.root, self.data["files"][0])
        entry.parent.parent.mkdir(parents=True)
        entry.parent.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "fuera"):
            workflow.remove(self.root, self.roll())
        self.assertEqual(protected.read_text(), "keep")

    def test_confirmed_delete_can_cancel_its_own_queued_worker(self):
        self.operate([self.data["files"][0]], "download")
        (self.root / "operations").mkdir(exist_ok=True)
        with (self.root / "operations" / ".worker").open("a") as scheduler:
            fcntl.flock(scheduler, fcntl.LOCK_EX)
            job = workflow.start(self.root, self.data, self.roll())
            workflow.remove(self.root, self.roll(), stop_active=True)
        self.assertEqual(core.load(job / "state.json")["phase"], "stopped")
        self.assertFalse(library.entry_root(self.root, self.data["files"][0]).exists())

    @unittest.skipUnless(
        shutil.which("pdftoppm") and shutil.which("pdfinfo"), "Poppler missing"
    )
    def test_batch_complete_and_status_share_the_same_final_contract(self):
        with (
            patch.object(core, "fetch", side_effect=self.fake_fetch),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            cli.main(["complete", "--root", str(self.root), "--rolls", "99"])
        with contextlib.redirect_stdout(io.StringIO()) as output:
            cli.main(["roll-status", "--root", str(self.root), "--rolls", "99"])
        state = json.loads(output.getvalue())[0]
        self.assertTrue(state["complete"])
        self.assertEqual((state["images"], state["expected"]), (3, 3))

    def test_interrupted_worker_offers_retry_and_error_details(self):
        job = self.job()
        library.replace_json(job / "state.json", {"phase": "running", "created": 0})
        state = workflow.roll_status(self.root, self.roll())
        self.assertFalse(state["active"])
        self.assertIn("interrumpió", " ".join(state["errors"]))
        _, screen = self.interface([curses.KEY_RIGHT, curses.KEY_LEFT, "\x1b"])
        text = "\n".join(screen.frames)
        self.assertIn("Reintentar procesamiento", text)
        self.assertIn("El proceso se interrumpió", text)

    @unittest.skipUnless(
        shutil.which("pdftoppm") and shutil.which("pdfinfo"), "Poppler missing"
    )
    def test_corrupt_pdf_is_repaired_and_delete_also_removes_recovery_data(self):
        self.complete()
        entry = library.entry_root(self.root, self.data["files"][0])
        pdf = next((entry / "pdfs").glob("*/*.pdf"))
        pdf.write_bytes(b"broken")
        self.assertEqual(workflow.roll_status(self.root, self.roll())["label"], "Error")
        self.complete()
        self.assertTrue(workflow.roll_status(self.root, self.roll())["complete"])
        self.assertTrue(list((self.root / "recovery").rglob("*.pdf")))
        workflow.remove(self.root, self.roll())
        self.assertFalse(list((self.root / "recovery").rglob("*.pdf")))

    @unittest.skipUnless(
        shutil.which("pdftoppm") and shutil.which("pdfinfo"), "Poppler missing"
    )
    def test_bad_preview_cannot_prevent_basic_completion(self):
        self.operate([self.data["files"][0]], "prepare", first_pages=1, dpi=72)
        entry = library.entry_root(self.root, self.data["files"][0])
        next((entry / "renders").glob("*/config.json")).write_text("broken preview")
        self.complete()
        state = workflow.roll_status(self.root, self.roll())
        self.assertTrue(state["complete"])
        self.assertEqual(state["images"], 3)

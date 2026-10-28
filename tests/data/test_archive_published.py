"""Read-only remote browsing, incremental publishing and nonselectable missing files."""

from __future__ import annotations

import contextlib
import curses
import io
import json
import os
import shlex
import subprocess
from pathlib import Path
from unittest.mock import patch

from tests.data.test_archive_library import LibraryFixture
from archive_berrutti import (
    agent,
    locations,
    manage as core,
    published,
    remote,
    remote_library,
    rolls,
    tui,
    workflow,
)


class PublishedTests(LibraryFixture):
    def setUp(self):
        super().setUp()
        self.groups = rolls.group(self.data)
        self.profile = {
            "name": "test",
            "host": "test-host",
            "workdir": str(self.base),
            "root": str(self.root),
        }
        self.reads = []

    def transport(self, arguments, **kwargs):
        self.assertEqual(arguments[0], "ssh")
        command = shlex.split(arguments[-1])
        self.assertEqual(
            command[0], "head", "Queries must never launch Python, srun or sbatch"
        )
        self.assertEqual(command[1], "-c")
        self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
        path = Path(command[-1])
        self.reads.append(path)
        return subprocess.CompletedProcess(
            arguments, 0, path.read_bytes()[: int(command[2])], b""
        )

    def test_machine_queries_and_pagination_need_no_compute(self):
        published.publish(self.root, self.data)
        with patch.object(remote_library.subprocess, "run", side_effect=self.transport):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                remote.launch(self.profile, ["agent", "list", "--limit", "1"])
            result = json.loads(output.getvalue())
            self.assertEqual(result["data"]["total"], 2)
            self.assertEqual(result["location"]["access"], "published_index")
            client = remote_library.Client(self.profile)
            response = client.query(
                agent.parser().parse_args(
                    ["files", "--availability", "missing", "--limit", "1"]
                )
            )
            self.assertEqual(response["total"], 2)
            self.assertEqual(len(response["items"]), 1)
            self.assertEqual(response["items"][0]["kind"], "zip")
            self.assertEqual(response["next_offset"], 1)
            self.assertEqual(sum(path.parent.name == "rolls" for path in self.reads), 1)

    def test_unchanged_revision_only_reads_the_small_pointer(self):
        published.publish(self.root, self.data)
        with patch.object(remote_library.subprocess, "run", side_effect=self.transport):
            client = remote_library.Client(self.profile)
            before = len(self.reads)
            client.poll(force=True)
            self.assertEqual(self.reads[before:], [self.root / "browse/revision.json"])

    def test_incremental_publish_updates_states_without_changing_other_rolls(self):
        published.publish(self.root, self.data)
        before = core.load(self.root / "browse/index.json")
        self.operate([self.data["files"][0]], "download")
        published.publish(self.root, self.data, [self.groups[0]])
        after = core.load(self.root / "browse/index.json")
        self.assertEqual(after["items"]["100"], before["items"]["100"])
        with patch.object(remote_library.subprocess, "run", side_effect=self.transport):
            client = remote_library.Client(self.profile)
            info = client.query(agent.parser().parse_args(["info", "--rolls", "99"]))[
                "items"
            ][0]
            self.assertEqual(info["zips"]["downloaded"], 1)
            self.assertEqual(info["next_step"], "extract")
            self.assertEqual(
                client.plan([self.groups[0]])["remaining_download_bytes"], 0
            )

    def test_missing_resources_are_dimmed_and_skipped_by_navigation(self):
        items = locations.resources(self.root, self.groups[0])
        self.assertTrue(all(item["disabled"] for item in items))
        with (
            patch.object(tui.Application, "complete") as complete,
            patch.object(tui.Application, "open_location") as opened,
        ):
            _, screen = self.interface(
                [
                    curses.KEY_RIGHT,
                    curses.KEY_HOME,
                    curses.KEY_UP,
                    curses.KEY_RIGHT,
                    curses.KEY_LEFT,
                    "q",
                ]
            )
        complete.assert_called_once()
        opened.assert_not_called()
        text = "\n".join(screen.frames)
        self.assertIn("› Descargar y procesar", text)
        self.assertNotIn("Descargar solo los ZIP", text)
        self.assertNotIn("› PDF · aún sin extraer", text)

    def test_present_files_remain_selectable_and_complete_action_is_unique(self):
        self.operate([self.data["files"][0]], "download")
        items = locations.resources(self.root, self.groups[0])
        self.assertEqual(sum(not item["disabled"] for item in items), 1)
        self.assertEqual(items[0]["path"].suffix, ".zip")
        _, screen = self.interface([curses.KEY_RIGHT, curses.KEY_LEFT, "q"])
        self.assertIn("Continuar procesamiento", "\n".join(screen.frames))

    def test_reader_keeps_previous_state_on_connection_failure(self):
        published.publish(self.root, self.data)
        with patch.object(remote_library.subprocess, "run", side_effect=self.transport):
            client = remote_library.Client(self.profile)
        old = client.index
        with patch.object(
            client, "fetch_index", side_effect=ValueError("SSH disconnected")
        ):
            client._last_poll = 0
            client.poll()
            client._thread.join(timeout=1)
            client.poll()
        self.assertIs(client.index, old)
        self.assertIn("Sin conexión", client.warning)
        self.assertEqual(client.state(self.groups[0])["label"], "Sin descargar")

    def test_cached_file_is_bounded_validated_and_does_not_fetch_other_data(self):
        self.operate([self.data["files"][0]], "download")
        published.publish(self.root, self.data)
        with patch.object(remote_library.subprocess, "run", side_effect=self.transport):
            client = remote_library.Client(self.profile)
            try:
                file = next(client.files([self.groups[0]]))
                before = len(self.reads)
                cached = client.cached_file(file)
                self.assertEqual(cached.read_bytes(), Path(file["path"]).read_bytes())
                self.assertEqual(self.reads[before:], [Path(file["path"])])
                client.cached_file(file)
                self.assertEqual(len(self.reads), before + 1)
                with self.assertRaises(ValueError):
                    client.cached_file({**file, "bytes": remote_library.MAX_FILE + 1})
                with self.assertRaises(ValueError):
                    client.read(self.base / "outside")
            finally:
                client.close()
            self.assertFalse(cached.exists())

    def test_scheduler_timeout_updates_snapshot_status_without_a_worker(self):
        published.publish(self.root, self.data)
        with patch.object(remote_library.subprocess, "run", side_effect=self.transport):
            client = remote_library.Client(self.profile)
        job = str(self.root / "jobs" / "owned")
        state = client.index["items"]["099"]["state"]
        state.update(
            active=True, job={"phase": "running", "job": job, "slurm_job_id": "123"}
        )
        name = "berrutti-" + core.digest(job)[:24]
        with patch.object(
            client,
            "shell",
            side_effect=["", f"123|TIMEOUT|{name}|{os.getuid()}|0:15\n"],
        ):
            client.reconcile(client.index)
        self.assertFalse(state["active"])
        self.assertEqual(state["label"], "Error")
        self.assertIn("TIMEOUT", state["errors"][0])

    def test_finished_worker_publishes_the_actual_final_image_count(self):
        with core.locked(self.root / "operations"):
            job = workflow.create_job(self.root, self.data, self.groups[0])
        published.publish(self.root, self.data)
        with (
            patch.object(core, "fetch", side_effect=self.fake_fetch),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            workflow.work(job)
        index = core.load(self.root / "browse/index.json")
        self.assertTrue(index["items"]["099"]["state"]["complete"])
        self.assertEqual(index["items"]["099"]["state"]["images"], 3)
        self.assertEqual(index["items"]["099"]["info"]["images"]["expected"], 3)

    def test_completed_scheduler_job_cannot_claim_unpublished_images_are_complete(self):
        published.publish(self.root, self.data)
        with patch.object(remote_library.subprocess, "run", side_effect=self.transport):
            client = remote_library.Client(self.profile)
        job = str(self.root / "jobs" / "unpublished")
        state = client.index["items"]["099"]["state"]
        state.update(
            active=True, job={"phase": "running", "job": job, "slurm_job_id": "124"}
        )
        name = "berrutti-" + core.digest(job)[:24]
        with patch.object(
            client,
            "shell",
            side_effect=["", f"124|COMPLETED|{name}|{os.getuid()}|0:0\n"],
        ):
            client.reconcile(client.index)
        self.assertFalse(state["active"])
        self.assertFalse(state["complete"])
        self.assertIn("agent index", state["errors"][0])

    def test_historical_job_queries_reconcile_the_requested_job_after_pointer_changes(
        self,
    ):
        published.publish(self.root, self.data)
        job = self.root / "jobs" / "old"
        core.save_new(
            job / "request.json", {"roll": self.groups[0], "action": "complete"}
        )
        core.save_new(job / "state.json", {"phase": "running"})
        token = "berrutti-" + core.digest(str(job))[:24]
        core.save_new(
            job / "slurm.json", {"uid": os.getuid(), "token": token, "job_id": "125"}
        )
        with patch.object(remote_library.subprocess, "run", side_effect=self.transport):
            client = remote_library.Client(self.profile)
            with patch.object(
                client,
                "shell",
                side_effect=["", f"125|TIMEOUT|{token}|{os.getuid()}|0:15\n"],
            ):
                result = client.query(agent.parser().parse_args(["job", "old"]))
        self.assertFalse(result["active"])
        self.assertEqual(result["status"]["slurm_state"], "TIMEOUT")

    def test_accepted_action_still_returns_its_id_when_followup_refresh_fails(self):
        published.publish(self.root, self.data)
        with patch.object(remote_library.subprocess, "run", side_effect=self.transport):
            client = remote_library.Client(self.profile)
        response = {"ok": True, "data": {"id": "accepted"}}
        with (
            patch.object(remote, "execution_script", return_value="script"),
            patch.object(
                remote,
                "ssh_script",
                return_value=subprocess.CompletedProcess(
                    [], 0, json.dumps(response), ""
                ),
            ),
            patch.object(client, "poll", side_effect=ValueError("offline")),
        ):
            self.assertEqual(
                client.action(["agent", "complete", "--rolls", "099", "--background"]),
                response,
            )
        self.assertIn("Acción enviada", client.warning)

    def test_remote_browser_uses_snapshots_not_local_filesystem_probes(self):
        published.publish(self.root, self.data)
        with patch.object(remote_library.subprocess, "run", side_effect=self.transport):
            client = remote_library.Client(self.profile)
            app, screen = self.interface(
                [curses.KEY_RIGHT, curses.KEY_LEFT, "q"], run=False
            )
            app.backend = client
            app.load()
            with patch.object(
                workflow,
                "roll_status",
                side_effect=AssertionError("No local filesystem state"),
            ):
                app.run()
        self.assertIn("Descargar y procesar", "\n".join(screen.frames))

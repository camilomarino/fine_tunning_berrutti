"""Destination routing, isolated Slurm requests, failure/retry and owned cancellation."""

from __future__ import annotations

import contextlib
import io
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from unittest.mock import patch

from tests.data.test_archive_library import LibraryFixture

from archive_berrutti import (
    cli,
    library,
    manage as core,
    remote,
    rolls,
    slurm,
    workflow,
)


class RemoteTests(LibraryFixture):
    def setUp(self):
        super().setUp()
        self.profile = {
            "name": "ruche",
            "host": "ruche",
            "account": "project",
            "workdir": str(self.base),
            "root": str(self.root),
            "python": "/some/env/bin/python",
            "code_root": str(self.base / "code/release"),
            "runtime_file": str(self.base / "code/release/profile.json"),
            "session": {
                "partition": "cpu_med",
                "cpus": 1,
                "mem": "2G",
                "time": "04:00:00",
            },
            "worker": {
                "partition": "cpu_med",
                "cpus": 2,
                "mem": "8G",
                "time": "04:00:00",
            },
        }
        self.config = self.base / "profiles.json"
        library.replace_json(
            self.config,
            {
                "schema_version": 1,
                "default": "ruche",
                "profiles": {"ruche": self.profile},
            },
        )
        self.environment = patch.dict(os.environ, {remote.CONFIG_ENV: str(self.config)})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        slurm._CACHE.clear()
        self.addCleanup(slurm._CACHE.clear)
        self.groups = rolls.group(self.data)

    def make_job(self, index=0, identifier="700"):
        with core.locked(self.root / "operations"):
            job = workflow.create_job(self.root, self.data, self.groups[index])
        core.save_new(
            job / "slurm.json",
            {
                "schema_version": 1,
                "token": slurm.token(job),
                "uid": os.getuid(),
                "job_id": identifier,
                "created_date": "2026-10-08",
            },
        )
        library.replace_json(
            job / "state.json", {"phase": "queued", "created": time.time() - 3600}
        )
        return job

    def row(self, job, state="PENDING", identifier="700"):
        return f"{identifier}|{state}|{slurm.token(job)}|{os.getuid()}|Resources\n"

    def test_default_is_remote_but_explicit_local_and_roots_never_connect(self):
        args, profile = remote.extract_destination([])
        self.assertEqual(profile["root"], str(self.root))
        for args in (
            ["--local"],
            ["agent", "list", "--root", str(self.root)],
            ["--help"],
        ):
            self.assertIsNone(remote.extract_destination(args)[1])
        args, profile = remote.extract_destination(
            ["agent", "list", "--profile", "ruche", "--root", str(self.root)]
        )
        self.assertEqual(profile["name"], "ruche")
        self.assertNotIn("--profile", args)
        with self.assertRaises(ValueError):
            remote.extract_destination(["--local", "--profile", "ruche"])
        with patch.object(
            remote, "launch", side_effect=AssertionError("Network must not run")
        ):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                cli.main(["--local", "agent", "list", "--root", str(self.root)])
            self.assertTrue(json.loads(output.getvalue())["ok"])

    def test_unknown_default_or_unreachable_remote_does_not_fall_back_to_local(self):
        with patch.object(remote, "launch", side_effect=OSError("ssh unavailable")):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                with self.assertRaises(SystemExit) as stopped:
                    cli.main(["agent", "list"])
        self.assertEqual(stopped.exception.code, 1)
        self.assertFalse(json.loads(output.getvalue())["ok"])
        self.assertFalse((self.root / "rolls").exists())
        with self.assertRaises(ValueError):
            remote.extract_destination(["--profile", "unknown"])

    def test_remote_script_quotes_arguments_and_runs_python_only_after_srun(self):
        malicious = "$(touch injected);`uname` ' quoted"
        script = remote.execution_script(
            self.profile, ["agent", "list", "--search", malicious]
        )
        command = shlex.split(script.split("exec ", 1)[1])
        self.assertEqual(command[0], "srun")
        self.assertIn(malicious, command)
        self.assertIn("--time=00:10:00", command)
        self.assertIn("WORKDIR=" + str(self.base), command)
        self.assertEqual(command[-2:], ["--root", str(self.root)])
        self.assertNotIn("--pty", command)
        interactive = remote.execution_script(self.profile, [], interactive=True)
        self.assertIn("--pty", interactive)
        self.assertIn("04:00:00", interactive)
        with self.assertRaises(ValueError):
            remote.ssh_base("-oProxyCommand=bad")

    def test_remote_root_must_be_in_workdir_and_separate_from_code(self):
        for path in (self.base, self.base.parent / "other", self.base / "code"):
            with self.assertRaises(ValueError):
                remote.check_root(path, self.profile)
        self.assertEqual(remote.check_root(self.root, self.profile), self.root)
        link = self.base / "escape"
        link.symlink_to(self.base.parent, target_is_directory=True)
        with self.assertRaises(ValueError):
            remote.check_root(link / "elsewhere", self.profile)

    def test_pending_job_stays_queued_without_local_pid_and_scheduler_calls_are_cached(
        self,
    ):
        job = self.make_job()
        with patch.object(slurm, "run", return_value=self.row(job)) as calls:
            for _ in range(3):
                status = workflow.read_job_status(job)
                self.assertEqual(status["phase"], "queued")
                self.assertEqual(status["slurm_job_id"], "700")
            calls.assert_called_once()
        self.assertFalse((job / "process.json").exists())

    def test_timeout_retains_data_and_allows_new_request(self):
        job = self.make_job()
        evidence = self.root / "partial.data"
        evidence.write_bytes(b"keep")

        def invoke(arguments):
            return (
                ""
                if arguments[0] == "squeue"
                else f"700|TIMEOUT|{slurm.token(job)}|{os.getuid()}|0:15\n"
            )

        with patch.object(slurm, "run", side_effect=invoke):
            status = workflow.read_job_status(job)
            self.assertEqual(status["phase"], "interrupted")
            self.assertIn("TIMEOUT", status["error"])
            replacement = workflow.create_job(self.root, self.data, self.groups[0])
        self.assertNotEqual(job, replacement)
        self.assertEqual(evidence.read_bytes(), b"keep")

    def test_detached_jobs_use_dependency_and_reject_duplicate_roll(self):
        submitted = []

        def invoke(arguments):
            if arguments[0] == "sbatch":
                submitted.append(arguments)
                return str(700 + len(submitted) - 1) + "\n"
            if arguments[0] == "squeue":
                job = Path(submitted[0][-1]).parent
                return self.row(job)
            return ""

        with (
            patch.object(remote, "runtime", return_value=self.profile),
            patch.object(slurm, "run", side_effect=invoke),
        ):
            first = workflow.start(self.root, self.data, self.groups[0])
            second = workflow.start(self.root, self.data, self.groups[1])
            with self.assertRaisesRegex(ValueError, "ya está"):
                workflow.start(self.root, self.data, self.groups[0])
        self.assertEqual(len(submitted), 2)
        self.assertIn("--dependency=afterany:700", submitted[1])
        self.assertIn("--export=NONE", submitted[0])
        self.assertNotEqual(first, second)
        self.assertIn(
            "archive_berrutti.workflow", (first / "worker.sbatch").read_text()
        )
        self.assertFalse((first / "process.json").exists())

    def test_lost_submission_ack_is_recovered_without_second_submission(self):
        submissions = []

        def invoke(arguments):
            if arguments[0] == "sbatch":
                submissions.append(arguments)
                raise subprocess.TimeoutExpired(arguments, 20)
            if arguments[0] == "squeue":
                return self.row(Path(submissions[0][-1]).parent)
            return ""

        with (
            patch.object(remote, "runtime", return_value=self.profile),
            patch.object(slurm, "run", side_effect=invoke),
        ):
            job = workflow.start(self.root, self.data, self.groups[0])
            self.assertEqual(slurm.receipt(job)["job_id"], "700")
            with self.assertRaises(ValueError):
                workflow.start(self.root, self.data, self.groups[0])
        self.assertEqual(len(submissions), 1)

    def test_cancelled_tail_does_not_drop_an_earlier_running_predecessor(self):
        submitted = []

        def invoke(arguments):
            if arguments[0] == "sbatch":
                submitted.append(arguments)
                return str(700 + len(submitted) - 1)
            if arguments[0] == "squeue":
                return self.row(Path(submitted[0][-1]).parent, "RUNNING")
            return ""

        with (
            patch.object(remote, "runtime", return_value=self.profile),
            patch.object(slurm, "run", side_effect=invoke),
        ):
            workflow.start(self.root, self.data, self.groups[0])
            cancelled = workflow.start(self.root, self.data, self.groups[1])
            library.replace_json(cancelled / "state.json", {"phase": "stopped"})
            workflow.start(self.root, self.data, self.groups[1])
        self.assertIn("--dependency=afterany:700", submitted[2])

    def test_invalid_remote_arguments_return_json_without_an_allocation(self):
        with (
            patch.object(remote.subprocess, "run") as launch,
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            with self.assertRaises(SystemExit) as result:
                cli.main(["agent", "complete"])
        self.assertEqual(result.exception.code, 2)
        self.assertEqual(
            json.loads(output.getvalue())["error"]["code"], "invalid_arguments"
        )
        launch.assert_not_called()

    def test_cancel_refuses_foreign_scheduler_identity(self):
        job = self.make_job()
        with patch.object(
            slurm, "run", return_value=f"700|RUNNING|other-job|{os.getuid()}|None\n"
        ) as commands:
            with self.assertRaisesRegex(ValueError, "ajeno"):
                workflow.stop_job(job)
        self.assertTrue(
            all(call.args[0][0] != "scancel" for call in commands.call_args_list)
        )
        self.assertEqual(core.load(job / "state.json")["phase"], "queued")

    def test_cancel_waits_for_terminal_ack_before_marking_stopped(self):
        job = self.make_job()
        cancelled = []

        def invoke(arguments):
            if arguments[0] == "squeue":
                return "" if cancelled else self.row(job)
            if arguments[0] == "scancel":
                cancelled.append(arguments[1:])
                return ""
            return f"700|CANCELLED|{slurm.token(job)}|{os.getuid()}|0:15\n"

        with (
            patch.object(slurm, "run", side_effect=invoke),
            patch.object(slurm.time, "sleep"),
        ):
            workflow.stop_job(job)
        self.assertEqual(cancelled, [["700"]])
        self.assertEqual(workflow.read_job_status(job)["phase"], "stopped")

    def test_agent_remove_requires_explicit_confirmation(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(SystemExit):
                cli.main(
                    [
                        "--local",
                        "agent",
                        "remove",
                        "--root",
                        str(self.root),
                        "--rolls",
                        "99",
                    ]
                )
        self.assertEqual(
            json.loads(output.getvalue())["error"]["code"], "invalid_arguments"
        )
        self.assertFalse((self.root / "jobs").exists())

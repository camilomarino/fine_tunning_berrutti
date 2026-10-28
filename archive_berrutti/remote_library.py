"""Client-side browsing of published metadata; no Python or scans on login."""

from __future__ import annotations

import copy
import contextlib
import json
import os
import shlex
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from archive_berrutti import agent, manage as core, remote, workflow

MAX_FILE = 64 * 1024**2
CACHE_LIMIT = 128 * 1024**2


class Client:
    def __init__(self, profile, root=None):
        self.profile = dict(profile)
        self.root = Path(root or profile["root"])
        if (
            not self.root.is_absolute()
            or ".." in self.root.parts
            or not self.root.is_relative_to(Path(profile["workdir"]))
        ):
            raise ValueError("Biblioteca remota fuera de WORKDIR")
        self.profile["root"] = str(self.root)
        self.details = {}
        self.warning = ""
        self._pending = None
        self._thread = None
        self._last_poll = time.monotonic()
        self._cache = None
        self._schedule_at = 0
        self._schedule_rows = {}
        self.index = self.fetch_index()

    def path(self, value):
        path = Path(value)
        if (
            not path.is_absolute()
            or ".." in path.parts
            or not path.is_relative_to(self.root)
        ):
            raise ValueError("Ruta publicada fuera de la biblioteca remota")
        return path

    def read(self, path, limit=MAX_FILE):
        path = self.path(path)
        command = [
            *remote.ssh_base(self.profile["host"]),
            shlex.join(["head", "-c", str(limit + 1), "--", str(path)]),
        ]
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=35,
        )
        if result.returncode:
            raise ValueError(
                "No se pudo leer "
                + str(path)
                + ": "
                + result.stderr.decode(errors="replace").strip()
            )
        if len(result.stdout) > limit:
            raise ValueError("El archivo supera el límite de lectura del cliente")
        return result.stdout

    def read_json(self, path):
        return json.loads(self.read(path))

    def shell(self, arguments):
        result = subprocess.run(
            [*remote.ssh_base(self.profile["host"]), shlex.join(arguments)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=25,
        )
        if result.returncode:
            raise ValueError(result.stderr.strip() or "No se pudo consultar Slurm")
        return result.stdout

    def reconcile(self, index):
        active = [
            item["state"]
            for item in index["items"].values()
            if item["state"]["active"]
            and (item["state"].get("job") or {}).get("slurm_job_id")
        ]
        if not active:
            return
        uid = str(index["uid"])
        if not uid.isdigit():
            raise ValueError("UID del índice inválido")
        found = self._schedule_rows if time.monotonic() - self._schedule_at < 15 else {}
        if time.monotonic() - self._schedule_at < 15:
            lines = []
        else:
            lines = self.shell(
                ["squeue", "--noheader", "--user", uid, "--format=%i|%T|%j|%U|%r"]
            ).splitlines()
        for line in lines:
            fields = line.strip().split("|", 4)
            if len(fields) == 5:
                found[fields[0]] = fields[1:]
        missing = [
            str(state["job"]["slurm_job_id"])
            for state in active
            if str(state["job"]["slurm_job_id"]) not in found
        ]
        if missing and time.monotonic() - self._schedule_at >= 15:
            if not all(value.isdigit() for value in missing):
                raise ValueError("ID Slurm inválido")
            for line in self.shell(
                [
                    "sacct",
                    "--noheader",
                    "--parsable2",
                    "--allocations",
                    "--jobs",
                    ",".join(missing),
                    "--format=JobIDRaw,State,JobName%80,UID,ExitCode",
                ]
            ).splitlines():
                fields = line.strip().split("|")
                if len(fields) >= 5:
                    found[fields[0]] = fields[1:5]
        if time.monotonic() - self._schedule_at >= 15:
            self._schedule_rows = found
            self._schedule_at = time.monotonic()
        for state in active:
            job = state["job"]
            row = found.get(str(job["slurm_job_id"]))
            if not row:
                continue
            phase, name, owner, reason = row
            if (
                owner != uid
                or name != "berrutti-" + core.digest(str(self.path(job["job"])))[:24]
            ):
                raise ValueError("La identidad Slurm no coincide con el índice")
            phase = phase.split()[0].rstrip("+")
            job.update(slurm_state=phase)
            if phase == "PENDING":
                state.update(label="En cola", progress=None)
                job.update(
                    phase="queued",
                    message="En cola Slurm"
                    + (" · " + reason if reason != "None" else ""),
                )
            elif phase == "RUNNING" and job["phase"] == "queued":
                state.update(label="Procesando", progress=None)
                job.update(phase="running", message="Iniciando procesamiento")
            elif phase in {
                "COMPLETED",
                "CANCELLED",
                "TIMEOUT",
                "FAILED",
                "NODE_FAIL",
                "OUT_OF_MEMORY",
                "PREEMPTED",
                "BOOT_FAIL",
                "DEADLINE",
            }:
                error = (
                    "Slurm finalizó pero falta el estado final publicado; actualizar con agent index"
                    if phase == "COMPLETED"
                    else "Slurm " + phase + "; reintentar conserva los datos válidos"
                )
                job.update(phase="interrupted", error=error)
                state.update(
                    active=False,
                    label="Error",
                    complete=False,
                    progress=None,
                    errors=[*state["errors"], error],
                )

    def fetch_index(self):
        try:
            previous = getattr(self, "index", None)
            revision = (
                self.read_json(self.root / "browse/revision.json") if previous else None
            )
            index = (
                copy.deepcopy(previous)
                if revision and revision.get("revision") == previous["revision"]
                else self.read_json(self.root / "browse/index.json")
            )
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            raise ValueError(
                f"No se pudo leer el índice remoto. Para crearlo/actualizarlo: archive-berrutti --profile {self.profile['name']} agent index --root {self.root}. {error}"
            ) from error
        if (
            index.get("schema_version") != 1
            or index.get("root") != str(self.root)
            or index["catalog"]["identifier"] != core.ITEM
        ):
            raise ValueError("El índice corresponde a otra biblioteca o esquema")
        try:
            self.reconcile(index)
            self.warning = ""
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            self.warning = "Slurm sin confirmar: " + str(error)
        return index

    def poll(self, *, force=False):
        if force:
            self.index = self.fetch_index()
            self._last_poll = time.monotonic()
            return True
        changed = False
        if self._pending is not None:
            index, error = self._pending
            self._pending = None
            if index is not None:
                self.index = index
                changed = True
            if error:
                self.warning = "Sin conexión; mostrando el último estado: " + error
        if time.monotonic() - self._last_poll > 5 and (
            self._thread is None or not self._thread.is_alive()
        ):
            self._last_poll = time.monotonic()

            def fetch():
                try:
                    self._pending = (self.fetch_index(), None)
                except Exception as error:
                    self._pending = (None, str(error))

            self._thread = threading.Thread(target=fetch, daemon=True)
            self._thread.start()
        return changed

    def state(self, roll):
        return copy.deepcopy(self.index["items"][roll["roll"]]["state"])

    def detail(self, roll):
        item = self.index["items"][roll["roll"]]
        cached = self.details.get(roll["roll"])
        if cached is None or cached[0] != item["revision"]:
            body = self.read_json(self.root / item["path"])
            if body.get("root") != str(self.root) or body.get("roll") != roll["roll"]:
                raise ValueError("Detalle remoto de otro rollo")
            self.details[roll["roll"]] = (item["revision"], body)
        return self.details[roll["roll"]][1]

    def resources(self, roll):
        return [
            {**item, "path": self.path(item["path"])}
            for item in self.detail(roll)["resources"]
        ]

    def files(self, groups):
        for roll in groups:
            for file in self.detail(roll)["files"]:
                self.path(file["path"])
                yield {"roll": roll["roll"], **file}

    @staticmethod
    def matches_file(file, args):
        return (
            (
                args.kind == "all"
                or file["kind"] == args.kind
                or args.kind == "zip"
                and file["kind"] == "zip_partial"
            )
            and (
                file["kind"] != "image"
                or args.scope == "all"
                or file["scope"] == args.scope
            )
            and agent.available(file, args.availability)
        )

    def file_page(self, groups, args):
        total, selected = 0, []
        for roll in groups:
            counts = self.index["items"][roll["roll"]]["file_counts"]
            count = 0
            for key, value in counts.items():
                kind, scope, present = key.split("|")
                if self.matches_file(
                    {"kind": kind, "scope": scope, "present": present == "1"}, args
                ):
                    count += value
            if (
                count
                and total < args.offset + args.limit
                and total + count > args.offset
            ):
                candidates = [
                    item for item in self.files([roll]) if self.matches_file(item, args)
                ]
                start = max(0, args.offset - total)
                stop = min(count, args.offset + args.limit - total)
                selected.extend(candidates[start:stop])
            total += count
        return {
            "items": selected,
            "total": total,
            "offset": args.offset,
            "limit": args.limit,
            "next_offset": args.offset + args.limit
            if args.offset + args.limit < total
            else None,
        }

    def plan(self, groups, maximum=None):
        states = [self.state(roll) for roll in groups]
        total = sum(state["download"]["total_bytes"] for state in states)
        remaining = sum(
            state["download"]["total_bytes"] - state["download"]["bytes"]
            for state in states
        )
        if maximum is not None and remaining > maximum * core.GIB:
            raise agent.UsageError("Los ZIP pendientes superan --max-download-gib")
        return {
            "rolls": [roll["roll"] for roll in groups],
            "zip_total": sum(s["zip_total"] for s in states),
            "zip_downloaded": sum(s["zip_done"] for s in states),
            "zip_total_bytes": total,
            "remaining_download_bytes": remaining,
            "additional_pdf_image_bytes": None,
            "final_profile": workflow.PROFILE,
        }

    def summary(self, roll):
        state = self.state(roll)
        info = copy.deepcopy(self.index["items"][roll["roll"]]["info"])
        info.update(
            active=state["active"], errors=state["errors"], progress=state["progress"]
        )
        if state["label"] == "Error":
            info.update(state="error", complete=False)
        elif state["active"]:
            info["state"] = (
                "queued" if state["job"]["phase"] == "queued" else "processing"
            )
        if info["job"] and state.get("job"):
            info["job"]["phase"] = state["job"]["phase"]
        return info

    def query(self, args):
        groups = agent.select(self.index["catalog"], args)
        if args.command == "list":
            items = [
                self.summary(roll)
                for roll in groups
                if args.search.casefold()
                in (
                    roll["roll"]
                    + " "
                    + " ".join(row["name"] for _, row in roll["files"])
                ).casefold()
            ]
            return agent.page(
                (
                    item
                    for item in items
                    if args.state == "all"
                    or args.state == item["state"]
                    or args.state == "local"
                    and item["state"] != "not_downloaded"
                ),
                args.offset,
                args.limit,
            )
        if args.command == "info":
            return {
                "items": [
                    {
                        **self.summary(roll),
                        "archives": self.detail(roll)["info"]["archives"],
                    }
                    for roll in groups
                ]
            }
        if args.command == "files":
            return self.file_page(groups, args)
        if args.command == "plan" or getattr(args, "dry_run", False):
            plan = self.plan(groups, args.max_download_gib)
            return (
                {"dry_run": True, "plan": plan}
                if getattr(args, "dry_run", False)
                else plan
            )
        if args.command == "job":
            if Path(args.id).name != args.id or args.id in {".", ".."}:
                raise agent.UsageError("ID de trabajo inválido")
            directory = self.root / "jobs" / args.id
            request = self.read_json(directory / "request.json")
            state = self.read_json(directory / "state.json")
            state["job"] = str(directory)
            roll_state = (
                self.index["items"].get(request["roll"]["roll"], {}).get("state")
            )
            current = (roll_state or {}).get("job") or {}
            if current.get("job") == str(directory):
                # The worker file can already be final while its index is older.
                state = (
                    {**current, **state}
                    if state["phase"] in workflow.FINISHED
                    else {**state, **current}
                )
            elif state["phase"] not in workflow.FINISHED:
                receipt = self.read_json(directory / "slurm.json")
                if (
                    receipt.get("uid") != self.index["uid"]
                    or receipt.get("token")
                    != "berrutti-" + core.digest(str(directory))[:24]
                ):
                    raise ValueError(
                        "El recibo Slurm no corresponde al trabajo solicitado"
                    )
                state.update(scheduler="slurm", slurm_job_id=receipt.get("job_id"))
                historical = {"active": True, "job": state, "errors": []}
                self.reconcile(
                    {
                        "uid": self.index["uid"],
                        "items": {args.id: {"state": historical}},
                    }
                )
            active = state["phase"] not in workflow.FINISHED
            result = (
                self.read_json(directory / "result.json")
                if state["phase"] == "done"
                else None
            )
            return {
                "id": args.id,
                "roll": request["roll"]["roll"],
                "action": request.get("action", "complete"),
                "active": active,
                "status": state,
                "progress": roll_state.get("progress")
                if active and current.get("job") == str(directory)
                else None,
                "paths": agent.job_paths(directory),
                "result": result,
            }
        raise ValueError("Consulta remota no soportada")

    def action(self, arguments):
        if not any(
            value == "--root" or value.startswith("--root=") for value in arguments
        ):
            arguments = [*arguments, "--root", str(self.root)]
        script = remote.execution_script(self.profile, arguments)
        try:
            result = remote.ssh_script(
                self.profile, script, stdin=subprocess.DEVNULL, capture_output=True
            )
        except subprocess.CalledProcessError as error:
            try:
                message = json.loads(error.stdout)["error"]["message"]
            except (ValueError, TypeError, KeyError):
                message = error.stderr or str(error)
            raise ValueError(message) from error
        if arguments[0] == "agent":
            output = json.loads(result.stdout)
            if not output["ok"]:
                raise ValueError(output["error"]["message"])
        else:
            output = result.stdout
        try:
            self.poll(force=True)
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            # An accepted action is still accepted if the following read fails.
            self.warning = "Acción enviada; no se pudo actualizar el índice: " + str(
                error
            )
        return output

    def cached_file(self, file):
        size = file.get("bytes")
        if not file.get("present") or not isinstance(size, int) or size > MAX_FILE:
            raise ValueError(
                "El archivo no está disponible o supera 64 MiB de caché local"
            )
        if self._cache is None:
            base = core.REPO / ".private/archive_berrutti"
            base.mkdir(parents=True, exist_ok=True)
            self._cache = tempfile.TemporaryDirectory(
                prefix="remote-preview-", dir=base
            )
        folder = Path(self._cache.name)
        path = folder / (core.digest(file)[:24] + Path(file["path"]).suffix)
        if not path.exists():
            data = self.read(file["path"], size)
            if len(data) != size:
                raise ValueError("El tamaño remoto cambió; actualizar el índice")
            path.write_bytes(data)
            if file.get("sha256") and core.hashes(path)["sha256"] != file["sha256"]:
                path.unlink()
                raise ValueError("La copia de vista previa no coincide con su hash")
        os.utime(path, None)
        files = sorted(folder.iterdir(), key=lambda item: item.stat().st_mtime)
        total = sum(item.stat().st_size for item in files)
        for item in files:
            if total <= CACHE_LIMIT:
                break
            if item != path:
                total -= item.stat().st_size
                item.unlink()
        return path

    def close(self):
        if self._cache:
            self._cache.cleanup()


def query(profile, arguments):
    args = agent.parser().parse_args(arguments[1:])
    with contextlib.closing(Client(profile, args.root)) as client:
        result = client.query(args)
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "ok": True,
                    "command": args.command,
                    "root": str(client.root),
                    "data": result,
                    "observed_at": core.now(),
                    "snapshot_at": client.index["updated_at"],
                    "warning": client.warning or None,
                    "location": {
                        "profile": profile["name"],
                        "host": profile["host"],
                        "root": str(client.root),
                        "scheduler": "slurm",
                        "access": "published_index",
                    },
                },
                ensure_ascii=False,
            )
        )
    return 0

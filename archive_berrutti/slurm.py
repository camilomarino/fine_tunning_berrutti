"""Detached, owned Slurm workers and their durable, queryable job receipts."""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path

from archive_berrutti import library, manage as core, remote

ACTIVE = {
    "PENDING",
    "RUNNING",
    "CONFIGURING",
    "COMPLETING",
    "SUSPENDED",
    "RESIZING",
    "REQUEUED",
    "REQUEUE_FED",
    "REQUEUE_HOLD",
    "SIGNALING",
    "STAGE_OUT",
}
_CACHE = {}


def run(arguments):
    return subprocess.run(
        arguments, check=True, text=True, capture_output=True, timeout=20
    ).stdout


def token(job):
    return "berrutti-" + core.digest(str(job.resolve()))[:24]


def receipt(job):
    value = core.load(job / "slurm.json")
    if value.get("token") != token(job) or value.get("uid") != os.getuid():
        raise ValueError("El recibo Slurm no corresponde a esta biblioteca/usuario")
    if value.get("job_id") is not None and not str(value["job_id"]).isdigit():
        raise ValueError("ID Slurm inválido")
    return value


def queue(*, fresh=False):
    now = time.monotonic()
    cached = _CACHE.get("queue")
    if cached and not fresh and now - cached[0] < 15:
        return cached[1]
    rows = {}
    for line in run(
        ["squeue", "--noheader", "--user", str(os.getuid()), "--format=%i|%T|%j|%U|%r"]
    ).splitlines():
        fields = line.strip().split("|", 4)
        if len(fields) != 5:
            raise ValueError("Respuesta squeue inválida")
        identifier, state, name, uid, reason = fields
        if identifier.isdigit():
            rows[identifier] = {
                "job_id": identifier,
                "state": state,
                "name": name,
                "uid": uid,
                "reason": reason,
            }
    _CACHE["queue"] = (now, rows)
    return rows


def owns(row, saved):
    return row["name"] == saved["token"] and str(row["uid"]) == str(saved["uid"])


def accounting(saved, *, fresh=False):
    key = saved["token"]
    now = time.monotonic()
    cached = _CACHE.get(key)
    if cached and not fresh and now - cached[0] < 15:
        return cached[1]
    args = [
        "sacct",
        "--noheader",
        "--parsable2",
        "--allocations",
        "--user",
        str(saved["uid"]),
        "--format=JobIDRaw,State,JobName%80,UID,ExitCode",
    ]
    if saved.get("job_id"):
        args += ["--jobs", saved["job_id"]]
    else:
        args += ["--name", saved["token"], "--starttime", saved["created_date"]]
    matches = []
    for line in run(args).splitlines():
        fields = line.strip().split("|")
        if len(fields) < 5:
            continue
        identifier, state, name, uid, exit_code = fields[:5]
        row = {
            "job_id": identifier,
            "state": state.split()[0].rstrip("+"),
            "name": name,
            "uid": uid,
            "exit_code": exit_code,
        }
        if (
            identifier.isdigit()
            and owns(row, saved)
            and (not saved.get("job_id") or identifier == saved["job_id"])
        ):
            matches.append(row)
    if len(matches) > 1:
        raise ValueError(
            "Más de un trabajo Slurm para la misma solicitud; revisar antes de actuar"
        )
    found = matches[0] if matches else None
    _CACHE[key] = (now, found)
    return found


def observed(job, *, fresh=False):
    saved = receipt(job)
    rows = queue(fresh=fresh)
    if saved.get("job_id"):
        row = rows.get(saved["job_id"])
        if row and not owns(row, saved):
            raise ValueError("Slurm devolvió un trabajo ajeno; no se modifica")
    else:
        matches = [row for row in rows.values() if owns(row, saved)]
        if len(matches) > 1:
            raise ValueError(
                "Solicitud duplicada en Slurm; revisar sin cancelar trabajos"
            )
        row = matches[0] if matches else None
    row = row or accounting(saved, fresh=fresh)
    if row and not saved.get("job_id"):
        saved["job_id"] = row["job_id"]
        library.replace_json(job / "slurm.json", saved)
    return row


def status(job, state):
    """A queued allocation is alive even before its worker creates a lease/PID."""
    saved = receipt(job)
    result = {**state, "scheduler": "slurm", "slurm_job_id": saved.get("job_id")}
    try:
        row = observed(job)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return {
            **result,
            "scheduler_warning": str(error),
            "message": "Estado Slurm sin confirmar; reintentar consulta",
        }
    if not row:
        return {
            **result,
            "message": "Esperando confirmación de Slurm; no reenviar la solicitud",
            "scheduler_warning": "Sin registro todavía en squeue/sacct",
        }
    result.update(
        slurm_job_id=row["job_id"],
        slurm_state=row["state"],
        slurm_reason=row.get("reason"),
    )
    if row["state"] in ACTIVE:
        if row["state"] in {"PENDING", "CONFIGURING", "REQUEUED", "REQUEUE_HOLD"}:
            reason = row.get("reason", "")
            result.update(
                phase="queued",
                message="En cola Slurm"
                + (" · " + reason if reason and reason != "None" else ""),
            )
        elif result["phase"] == "queued":
            result.update(phase="running", message="Slurm: iniciando trabajador")
        return result
    # A terminal scheduler record without the successful worker receipt is a failure.
    result.update(
        phase="interrupted"
        if row["state"] in {"TIMEOUT", "PREEMPTED", "NODE_FAIL", "CANCELLED"}
        else "failed",
        error=f"Slurm {row['state']} ({row.get('exit_code', '?')}); se conserva lo terminado para reintentar",
    )
    return result


def script(job, profile):
    return "\n".join(
        [
            "#!/bin/bash",
            "set -euo pipefail",
            "umask 077",
            ': "${SLURM_JOB_ID:?Ejecutar mediante sbatch}"',
            "cd " + shlex.quote(profile["code_root"]),
            "export PYTHONDONTWRITEBYTECODE=1",
            "export WORKDIR=" + shlex.quote(profile["workdir"]),
            "export LANG=C.UTF-8",
            "export OMP_NUM_THREADS=" + str(profile["worker"]["cpus"]),
            "export " + remote.RUNTIME_ENV + "=" + shlex.quote(profile["runtime_file"]),
            "exec "
            + shlex.join(
                [
                    profile["python"],
                    "-u",
                    "-m",
                    "archive_berrutti.workflow",
                    "--job",
                    str(job),
                ]
            ),
            "",
        ]
    )


def submit(job, profile):
    """Caller holds the library operations lock across creation and submission."""
    root = Path(core.load(job / "request.json")["root"])
    tail = root / "operations" / "slurm-tail.json"
    dependencies, pending = [], []
    if tail.exists():
        from archive_berrutti import workflow

        saved_queue = core.load(tail)
        # Keep all unfinished predecessors: cancelling the last queued job must
        # not allow a new worker to overtake an earlier one still running.
        queue(fresh=True)
        for relative in saved_queue.get("pending", [saved_queue["job"]]):
            previous = workflow.confined(root, relative)
            if workflow.read_job_status(previous)["phase"] in workflow.FINISHED:
                continue
            observed(previous)
            previous_receipt = receipt(previous)
            if not previous_receipt.get("job_id"):
                library.replace_json(
                    job / "state.json",
                    {
                        "phase": "failed",
                        "error": "La solicitud previa no tiene confirmación Slurm; consultarla antes de enviar más",
                    },
                )
                raise ValueError("Slurm aún no confirmó la solicitud anterior")
            dependencies.append(previous_receipt["job_id"])
            pending.append(relative)
    saved = {
        "schema_version": 1,
        "token": token(job),
        "uid": os.getuid(),
        "job_id": None,
        "created_date": time.strftime("%Y-%m-%d"),
        "profile": profile["name"],
        "release": profile.get("release"),
        "partition": profile["worker"]["partition"],
    }
    core.save_new(job / "slurm.json", saved)
    relative = str(job.relative_to(root))
    library.replace_json(tail, {"job": relative, "pending": [*pending, relative]})
    contents = script(job, profile)
    (job / "worker.sbatch").write_text(contents)
    args = [
        "sbatch",
        "--parsable",
        "--export=NONE",
        "--propagate=NONE",
        "--job-name",
        saved["token"],
        "--comment",
        saved["token"],
        *remote.resource_args(profile, "worker"),
        *(["--dependency=afterany:" + ":".join(dependencies)] if dependencies else []),
        "--chdir",
        profile["code_root"],
        "--output",
        str(job / "worker.log"),
        "--error",
        str(job / "worker.log"),
        str(job / "worker.sbatch"),
    ]
    try:
        result = run(args).strip().split(";", 1)[0]
        if not result.isdigit():
            raise ValueError("sbatch no devolvió un ID válido")
    except subprocess.CalledProcessError as error:
        library.replace_json(
            job / "state.json",
            {"phase": "failed", "error": error.stderr.strip() or str(error)},
        )
        raise ValueError(
            "Slurm rechazó el trabajo: " + (error.stderr.strip() or str(error))
        ) from error
    except FileNotFoundError as error:
        library.replace_json(
            job / "state.json", {"phase": "failed", "error": str(error)}
        )
        raise
    except (OSError, ValueError, subprocess.TimeoutExpired):
        # An acknowledgement can be lost after acceptance. Keep this request active;
        # subsequent queries recover its ID by the unique name, never resubmit it.
        _CACHE.clear()
        try:
            observed(job, fresh=True)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass  # Receipt and active request remain recoverable on the next query.
        return job
    saved["job_id"] = result
    library.replace_json(job / "slurm.json", saved)
    _CACHE.clear()
    return job


def stop(job):
    row = observed(job, fresh=True)
    if row is None:
        raise ValueError(
            "Slurm no confirma el trabajo; no es seguro cancelar ni borrar"
        )
    if row["state"] in ACTIVE:
        # observed() checks the exact token and UID; never signal a copied PID.
        run(["scancel", row["job_id"]])
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            time.sleep(1)
            row = observed(job, fresh=True)
            if row and row["state"] not in ACTIVE:
                break
        else:
            raise ValueError(
                "Slurm aún no confirmó la detención; los datos se conservaron"
            )
    library.replace_json(
        job / "state.json",
        {
            "phase": "stopped",
            "message": "Detenido por el usuario en Slurm",
            "updated": core.now(),
        },
    )
    _CACHE.clear()

"""One-roll lifecycle: complete every page, inspect progress, retry and remove."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import re
import signal
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from archive_berrutti import library, manage as core, monitor

PROFILE = {"format": "png", "dpi": 200, "all_pages": True}
PROFILE_LABEL = "Todas las páginas · PNG · 200 DPI"
FINISHED = {"done", "failed", "removed", "interrupted", "stopped"}


def settings(**overrides):
    return SimpleNamespace(
        **{
            "pages": None,
            "first_pages": None,
            "all_pages": True,
            "format": "png",
            "dpi": 200,
            "quality": 95,
            "timeout": 180,
            "max_total_pages": sys.maxsize,
            "reserve_gib": 5,
            "max_pdf_gib": 8,
            "attempts": 3,
            "quota_free_gib": None,
            "render_generation": None,
            **overrides,
        }
    )


def control(root, roll):
    return root / "operations" / core.digest(roll["roll"])[:20]


def confined(base, relative):
    path = base / relative
    if path.resolve() != path.absolute() or not path.is_relative_to(base):
        raise ValueError(f"Ruta de datos fuera de su carpeta: {path}")
    return path


def final_status(root, row, local=None, *, verify=False):
    """Count one canonical output, checking each expected file, not preview totals."""
    directory = library.entry_root(root, row)
    result = {
        "complete": False,
        "images": 0,
        "expected": None,
        "problem": None,
        "output": None,
    }
    target_path = directory / "final_target.json"
    if not target_path.exists():
        return result
    try:
        target = core.load(target_path)
        if target["profile"] != PROFILE or target["source_sha1"] != row["sha1"]:
            raise ValueError("El destino final pertenece a otro original o formato")
        manifest = core.load(directory / "pdfs" / row["key"] / "manifest.json")
        if core.digest(manifest) != target["pdf_manifest_sha256"]:
            raise ValueError("El inventario de PDF cambió")
        output = confined(directory, target["output"])
        if output.parent != directory / "renders":
            raise ValueError("Destino final fuera de renders")
        result["output"] = str(output)
        expected = {}
        if len(target["pdfs"]) != len(manifest["pdfs"]) or not target["pdfs"]:
            raise ValueError("Inventario final de PDF incompleto")
        for index, (pdf, original) in enumerate(
            zip(target["pdfs"], manifest["pdfs"]), 1
        ):
            if pdf["sha256"] != original["sha256"] or pdf["pages"] < 1:
                raise ValueError("Inventario final de páginas inválido")
            for page in range(1, pdf["pages"] + 1):
                expected[f"{row['key']}-pdf{index:04d}-p{page:06d}"] = (
                    pdf["sha256"],
                    page,
                )
        result["expected"] = len(expected)
        seen = set()
        journal = output / "pages.jsonl"
        if journal.exists():
            with journal.open(encoding="utf-8") as stream:
                for line in stream:
                    if (
                        not line.endswith("\n")
                        and not (output / "complete.json").exists()
                    ):
                        continue  # A worker may be appending this record right now.
                    try:
                        record = json.loads(line)
                        page_id = record["page_id"]
                        if page_id in seen or page_id not in expected:
                            raise ValueError(
                                "Página duplicada o ajena en la salida final"
                            )
                        source_hash, page = expected[page_id]
                        relative = f"images/{page_id}.png"
                        if (
                            record["image"] != relative
                            or record["format"] != "png"
                            or record["dpi"] != 200
                            or record["pdf_sha256"] != source_hash
                            or record["page_number"] != page
                        ):
                            raise ValueError(
                                "Una imagen no corresponde a su página o formato final"
                            )
                        path = confined(output, relative)
                        if path.stat().st_size != record["bytes"]:
                            raise ValueError(f"Imagen final incompleta: {path.name}")
                        if (
                            verify
                            and core.hashes(path)["sha256"] != record["image_sha256"]
                        ):
                            raise ValueError(f"Imagen final modificada: {path.name}")
                        seen.add(page_id)
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        result["problem"] = str(error)
        result["images"] = len(seen)
        if not (output / "complete.json").exists():
            return result
        config = core.load(output / "config.json")
        plan = core.load(output / "render_plan.json")
        complete = core.load(output / "complete.json")
        if (
            any(config.get(key) != value for key, value in PROFILE.items())
            or config.get("pages")
            or config.get("first_pages")
        ):
            raise ValueError(
                "La conversión no incluye todas las páginas en el formato final"
            )
        if plan["config_sha256"] != core.digest(config) or complete[
            "config_sha256"
        ] != core.digest(config):
            raise ValueError("Los recibos finales no corresponden a la configuración")
        if plan["expected_pages"] != len(expected) or complete["expected_pages"] != len(
            expected
        ):
            raise ValueError("La conversión no cubre todas las páginas de los PDF")
        local = local or library.status(root, row, include_renders=False)
        result["complete"] = (
            not result["problem"]
            and len(seen) == len(expected)
            and local["download"] == "downloaded"
            and local["pdf"] == "extracted"
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["problem"] = str(error)
    return result


def job_status(root, roll):
    pointer = control(root, roll) / "current.json"
    if not pointer.exists():
        return None
    job = confined(root, core.load(pointer)["job"])
    return read_job_status(job)


def read_job_status(job):
    """Inspect a specific job, including history no longer pointed to by a roll."""
    state = core.load(job / "state.json")
    state["job"] = str(job)
    if (job / "slurm.json").exists():
        from archive_berrutti import slurm

        if state["phase"] in FINISHED:
            receipt = slurm.receipt(job)
            return {
                **state,
                "scheduler": "slurm",
                "slurm_job_id": receipt.get("job_id"),
            }
        return slurm.status(job, state)
    if state["phase"] in FINISHED:
        return state
    # A worker holds its lease both while queued and while doing work.
    alive = False
    lease = job / ".lease"
    if lease.exists():
        with lease.open("a") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                alive = True
    process = job / "process.json"
    if not alive and process.exists():
        process_info = core.load(process)
        if process_info["host"] == socket.gethostname():
            try:
                alive = process_info["start_ticks"] == process_ticks(
                    process_info["pid"]
                )
            except (OSError, ValueError):
                pass
    if not alive and time.time() - state.get("created", 0) > 5:
        state = {
            **state,
            "phase": "interrupted",
            "error": "El proceso se interrumpió. Reintentar conserva el trabajo válido.",
        }
    return state


def roll_status(root, roll):
    local = [
        library.status(root, row, include_renders=False) for _, row in roll["files"]
    ]
    final = [
        final_status(root, row, item) for (_, row), item in zip(roll["files"], local)
    ]
    job = job_status(root, roll)
    errors = [item["problem"] for item in [*local, *final] if item["problem"]]
    for (_, row), item in zip(roll["files"], local):
        if (
            item["download"] in {"failed", "inconsistent"}
            or item["pdf"] == "inconsistent"
        ):
            last = library.last_event(library.entry_root(root, row) / "events.jsonl")
            errors.append(
                last.get("error")
                or f"ZIP/PDF local faltante o inconsistente: {row['name']}"
            )
    if job and job.get("error"):
        errors.append(job["error"])
    downloaded = sum(item["download"] == "downloaded" for item in local)
    download_total = sum(row["bytes"] for _, row in roll["files"])
    download_bytes = sum(
        row["bytes"]
        if item["download"] == "downloaded"
        else min(row["bytes"], item["partial_bytes"])
        for (_, row), item in zip(roll["files"], local)
    )
    complete = all(item["complete"] for item in final)
    active = bool(job and job["phase"] not in FINISHED)
    if active:
        label = "En cola" if job["phase"] == "queued" else "Procesando"
        complete = False
    elif complete:
        label = "Completo"
    elif errors:
        label = "Error"
    elif any(item["download"] != "pending" or item["pdf_count"] for item in local):
        label = "Incompleto"
    else:
        label = "Sin descargar"
    if complete:
        next_step = None
    elif downloaded < len(local):
        next_step = "download"
    elif any(item["pdf"] != "extracted" for item in local):
        next_step = "extract"
    else:
        next_step = "render"
    return {
        "complete": complete,
        "label": label,
        "next_step": next_step,
        "active": active,
        "zip_done": downloaded,
        "zip_total": len(local),
        "download": {
            "bytes": download_bytes,
            "total_bytes": download_total,
            "percent": round(100 * download_bytes / download_total, 1)
            if download_total
            else 0,
        },
        "pdfs": sum(item["pdf_available"] for item in local),
        "images": sum(item["images"] for item in final),
        "expected": sum(item["expected"] for item in final)
        if all(item["expected"] is not None for item in final)
        else None,
        "errors": errors,
        "job": job,
        "progress": monitor.snapshot(root, roll, local, final, job) if active else None,
        "outputs": [item["output"] for item in final if item["output"]],
    }


def quarantine(root, path, job_id):
    if path.exists():
        relative = path.relative_to(root)
        confined(root, str(relative))
        destination = root / "recovery" / job_id / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        path.rename(destination)


def repair_originals(root, row, job_id):
    """Keep damaged evidence in a new recovery path instead of overwriting it."""
    directory = library.entry_root(root, row)
    archive = core.zip_path(directory, library.local_entry(row))
    if archive.exists():
        try:
            core.verify_zip(archive, row)
        except (OSError, ValueError):
            quarantine(root, directory, job_id)
            return
    receipt = directory / "receipts" / f"{row['key']}.json"
    if receipt.exists():
        try:
            saved = core.load(receipt)
            if saved["sha1"] != row["sha1"] or saved["md5"] != row["md5"]:
                raise ValueError("Recibo distinto")
        except (OSError, ValueError, KeyError):
            quarantine(root, receipt, job_id)
    pdfs = directory / "pdfs" / row["key"]
    if (pdfs / "manifest.json").exists():
        try:
            for pdf in core.load(pdfs / "manifest.json")["pdfs"]:
                if core.hashes(confined(pdfs, pdf["file"]))["sha256"] != pdf["sha256"]:
                    raise ValueError("PDF modificado")
        except (OSError, ValueError, KeyError):
            quarantine(root, pdfs, job_id)


def complete_roll(
    root,
    data,
    roll,
    progress=print,
    *,
    job_id=None,
    transfer=None,
    stage=None,
    options=None,
):
    """Use all member ZIPs and all PDF pages with the fixed common profile."""
    job_id = job_id or library.identifier()
    outputs = []
    for position, (_, row) in enumerate(roll["files"], 1):
        label = f"ZIP {position}/{len(roll['files'])} · Rollo {roll['roll']}"
        if stage:
            stage("verify", row)
        progress(f"{label} · comprobando originales")
        with core.locked(root), core.locked(library.entry_root(root, row)):
            repair_originals(root, row, job_id)
        current = final_status(root, row, verify=True)
        if current["complete"]:
            outputs.append(current["output"])
            continue
        progress(f"{label} · descargando ZIP")
        args = settings(**(options or {}))
        library.operate(
            root,
            data,
            [row],
            "download",
            args,
            progress=progress,
            transfer=transfer,
            stage=stage,
        )
        progress(f"{label} · extrayendo PDF")
        library.operate(
            root, data, [row], "extract", args, progress=progress, stage=stage
        )
        if stage:
            stage("count", row)
        directory = library.entry_root(root, row)
        manifest = core.load(directory / "pdfs" / row["key"] / "manifest.json")
        inventory = []
        for pdf in manifest["pdfs"]:
            path = confined(directory / "pdfs" / row["key"], pdf["file"])
            info = core.run(["pdfinfo", str(path)], args.timeout)
            match = re.search(r"^Pages:\s+(\d+)$", info, re.MULTILINE)
            if not match or int(match[1]) < 1:
                raise ValueError(f"No se pudo contar páginas de {path}")
            inventory.append({"sha256": pdf["sha256"], "pages": int(match[1])})
        if not inventory:
            raise ValueError("El rollo no contiene PDF para convertir")
        previous = directory / "final_target.json"
        if previous.exists():
            try:
                args.render_generation = core.load(previous).get("generation")
            except ValueError:
                pass
        if current["problem"]:
            args.render_generation = job_id
        _, render_id = library.render_spec(args)
        target = {
            "profile": PROFILE,
            "source_sha1": row["sha1"],
            "pdf_manifest_sha256": core.digest(manifest),
            "pdfs": inventory,
            "output": f"renders/{render_id}",
            "generation": args.render_generation,
        }
        with core.locked(root):
            core.save_new(directory / "final_attempts" / f"{job_id}.json", target)
            library.replace_json(previous, target)
        progress(f"{label} · convirtiendo todas las páginas a PNG 200 DPI")
        library.operate(
            root, data, [row], "render", args, progress=progress, stage=stage
        )
        final = final_status(root, row, verify=True)
        if not final["complete"]:
            raise ValueError(final["problem"] or "Faltan páginas finales")
        outputs.append(final["output"])
    return {"outputs": outputs, "profile": PROFILE}


def process_ticks(pid):
    # Linux /proc start time disambiguates recycled process IDs.
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def create_job(root, data, roll, *, action="complete", options=None):
    """Caller holds the operations lock until a request has been registered."""
    if action not in {"complete", "download"}:
        raise ValueError("Acción de trabajo inválida")
    previous = job_status(root, roll)
    if previous and previous["phase"] not in FINISHED:
        raise ValueError("Este rollo ya está en cola o procesándose")
    job = root / "jobs" / library.identifier()
    core.save_new(
        job / "request.json",
        {
            "root": str(root),
            "data": data,
            "roll": roll,
            "profile": PROFILE,
            "action": action,
            "options": {
                key: getattr(options, key)
                for key in ("reserve_gib", "attempts", "quota_free_gib")
            }
            if options is not None
            else {},
        },
    )
    library.replace_json(
        job / "state.json",
        {"phase": "queued", "created": time.time(), "message": "Esperando turno"},
    )
    library.replace_json(
        control(root, roll) / "current.json", {"job": str(job.relative_to(root))}
    )
    return job


def foreground(root, data, roll, *, action="complete", options=None):
    with core.locked(root / "operations"):
        job = create_job(root, data, roll, action=action, options=options)
    work(job)
    return core.load(job / "result.json")


def start(root, data, roll, *, action="complete", options=None):
    """Queue a detached worker; one library processes one roll at a time."""
    with core.locked(root / "operations"):
        job = create_job(root, data, roll, action=action, options=options)
        from archive_berrutti import remote, slurm

        if profile := remote.runtime():
            try:
                return slurm.submit(job, profile)
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                if not (job / "slurm.json").exists():
                    library.replace_json(
                        job / "state.json", {"phase": "failed", "error": str(error)}
                    )
                raise
        try:
            with (job / "worker.log").open("w") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "archive_berrutti.workflow",
                        "--job",
                        str(job),
                    ],
                    cwd=core.REPO,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            ticks = process_ticks(process.pid)
            core.save_new(
                job / "process.json",
                {
                    "pid": process.pid,
                    "host": socket.gethostname(),
                    "start_ticks": ticks,
                },
            )
            threading.Thread(target=process.wait, daemon=True).start()
        except (OSError, ValueError) as error:
            library.replace_json(
                job / "state.json", {"phase": "failed", "error": str(error)}
            )
            raise
    return job


def work(job):
    from archive_berrutti import published

    request = core.load(job / "request.json")
    with published.reporting(Path(request["root"]), request["data"], request["roll"]):
        return _work(job)


def _work(job):
    request = core.load(job / "request.json")
    root = library.open_library(Path(request["root"]))
    if socket.gethostname().split(".")[0] in {"ruche01", "ruche02"}:
        raise ValueError("En Ruche ejecutar dentro de Slurm")
    with (job / ".lease").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        with (root / "operations" / ".worker").open("a") as scheduler:
            fcntl.flock(scheduler, fcntl.LOCK_EX)
            state = core.load(job / "state.json")

            def save(**values):
                state.update(values, updated=core.now())
                library.replace_json(job / "state.json", state)
                with (job / "progress.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(state, ensure_ascii=False) + "\n")

            def progress(message):
                save(phase="running", message=message, transfer=None)
                print(message, flush=True)

            def stage(name, row):
                save(
                    phase="running",
                    stage={"name": name, "key": row["key"], "started": time.time()},
                    transfer=None,
                )

            def transfer(value):
                labels = {
                    "connecting": "Conectando",
                    "downloading": "Descargando",
                    "verifying": "Verificando ZIP",
                    "done": "ZIP verificado",
                    "retrying": "Reintentando descarga",
                    "failed": "Descarga fallida",
                }
                message = f"{labels[value['phase']]} · {value['bytes']}/{value['total_bytes']} bytes · {value['name']}"
                save(phase="running", message=message, transfer=value)
                print(message, flush=True)

            try:
                action = request.get("action", "complete")
                progress(
                    "Descargando ZIP del rollo"
                    if action == "download"
                    else "Completando rollo"
                )
                if action == "download":
                    result = library.operate(
                        root,
                        request["data"],
                        [row for _, row in request["roll"]["files"]],
                        "download",
                        settings(**request.get("options", {})),
                        progress=progress,
                        transfer=transfer,
                        stage=stage,
                    )
                else:
                    result = complete_roll(
                        root,
                        request["data"],
                        request["roll"],
                        progress,
                        job_id=job.name,
                        transfer=transfer,
                        stage=stage,
                        options=request.get("options", {}),
                    )
                core.save_new(job / "result.json", result)
                save(
                    phase="done",
                    message="ZIP descargados y verificados"
                    if action == "download"
                    else "Rollo completo",
                )
            except BaseException as error:
                save(
                    phase="failed",
                    error=f"{type(error).__name__}: {error}",
                )
                raise


def stop_job(job):
    """Stop only our recorded process group, checking host and process identity."""
    if core.load(job / "state.json")["phase"] in FINISHED:
        return
    if (job / "slurm.json").exists():
        from archive_berrutti import slurm

        return slurm.stop(job)
    process = core.load(job / "process.json")
    pid = process["pid"]
    if process["host"] != socket.gethostname():
        raise ValueError(
            f"Este proceso está en {process['host']}; eliminar desde esa máquina."
        )
    try:
        if process_ticks(pid) != process["start_ticks"] or os.getpgid(pid) != pid:
            raise ValueError(
                "La identidad del proceso cambió; no se envió ninguna señal"
            )
        os.killpg(pid, signal.SIGTERM)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                if process_ticks(pid) != process["start_ticks"]:
                    break
            except FileNotFoundError:
                break
            time.sleep(0.05)
        else:
            raise ValueError("El proceso no se detuvo; se conservaron los datos")
    except (ProcessLookupError, FileNotFoundError):
        pass
    library.replace_json(
        job / "state.json",
        {
            "phase": "stopped",
            "message": "Detenido por el usuario",
            "updated": core.now(),
        },
    )


def remove(root, roll, *, stop_active=False):
    """Delete only this roll, including its recovery copies; retain audit history."""
    with core.locked(root / "operations"):
        previous = job_status(root, roll)
        if previous and previous["phase"] not in FINISHED:
            if not stop_active:
                raise ValueError(
                    "El rollo está en cola o procesándose; esperar antes de eliminar"
                )
            stop_job(Path(previous["job"]))
        with core.locked(root):
            directories = {library.entry_root(root, row) for _, row in roll["files"]}
            recovery = root / "recovery"
            if recovery.exists():
                directories.update(
                    recovered / path.relative_to(root)
                    for recovered in recovery.iterdir()
                    if recovered.is_dir()
                    for path in list(directories)
                    if path.is_relative_to(root / "rolls")
                    and (recovered / path.relative_to(root)).exists()
                )
            for path in directories:
                confined(root, str(path.relative_to(root)))
            with contextlib.ExitStack() as stack:
                for path in sorted(directories):
                    if path.exists():
                        stack.enter_context(core.locked(path))
                audit = root / "jobs" / library.identifier()
                core.save_new(
                    audit / "deletion.json",
                    {
                        "roll": roll,
                        "paths": [str(path) for path in sorted(directories)],
                        "at_utc": core.now(),
                    },
                )
                for path in directories:
                    if path.exists():
                        shutil.rmtree(path)
                library.replace_json(
                    audit / "state.json",
                    {"phase": "removed", "message": "Datos locales eliminados"},
                )
                library.replace_json(
                    control(root, roll) / "current.json",
                    {"job": str(audit.relative_to(root))},
                )
    return len(directories)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    work(args.job)


if __name__ == "__main__":
    main()

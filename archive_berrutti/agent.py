"""Versioned, non-interactive JSON commands over the same library as the TUI."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import socket
import sys
from pathlib import Path

from archive_berrutti import library, manage as core, monitor, rolls, workflow

SCHEMA_VERSION = 1
STATES = ("complete", "queued", "processing", "error", "incomplete", "not_downloaded")


class UsageError(ValueError):
    pass


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise UsageError(message)


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Debe ser un entero no negativo")
    return number


def budget(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("Debe ser un número finito no negativo")
    return number


def parser():
    cli = Parser(
        prog="archive-berrutti agent",
        description="Sin prompts ni TTY. stdout: un documento JSON; progreso: stderr.",
    )
    commands = cli.add_subparsers(dest="command", required=True)
    for name, help_text in {
        "list": "Estado de rollos, con filtros y paginación",
        "info": "Estado y rutas de ZIP, PDF y conversiones de rollos elegidos",
        "files": "Archivos con rutas absolutas y procedencia, paginados",
        "plan": "Estimar una selección sin descargar ni escribir",
        "download": "Descargar los ZIP de los rollos elegidos",
        "complete": "Descargar, extraer y completar todas las imágenes PNG 200 DPI",
        "job": "Consultar un trabajo por ID, aunque ya haya otro posterior",
        "cancel": "Detener solo el trabajo propio indicado, conservando los datos",
        "remove": "Eliminar los datos de los rollos seleccionados con confirmación explícita",
        "refresh": "Crear/actualizar el catálogo; no descarga ZIP",
        "index": "Publicar el índice de exploración, sin descargar datos",
    }.items():
        command = commands.add_parser(name, help=help_text)
        command.add_argument(
            "--root", type=Path, help="Biblioteca; por defecto, la de la CLI visual"
        )
        if name in {"job", "cancel"}:
            command.add_argument("id")
            continue
        if name == "refresh":
            command.add_argument(
                "--metadata", type=Path, help="Snapshot local; evita la red"
            )
            continue
        selection = command.add_mutually_exclusive_group(
            required=name in {"info", "plan", "download", "complete", "remove"}
        )
        selection.add_argument(
            "--rolls", help="Rollos exactos separados por coma, p. ej. 099,284"
        )
        selection.add_argument(
            "--all", action="store_true", help="Todo el catálogo explícitamente"
        )
        if name == "remove":
            command.add_argument(
                "--yes",
                action="store_true",
                help="Confirmar borrado de los rollos indicados",
            )
            command.add_argument(
                "--stop-active",
                action="store_true",
                help="Detener primero el trabajo de esos rollos",
            )
        if name in {"list", "files"}:
            command.add_argument("--offset", type=nonnegative_int, default=0)
            command.add_argument("--limit", type=core.positive, default=50)
        if name == "list":
            command.add_argument("--search", default="", help="Texto en rollo o título")
            command.add_argument(
                "--state", choices=("all", "local", *STATES), default="all"
            )
        if name == "files":
            command.add_argument(
                "--kind", choices=("all", "zip", "pdf", "image"), default="all"
            )
            command.add_argument(
                "--scope",
                choices=("final", "other", "all"),
                default="final",
                help="Conversiones de imágenes que incluir",
            )
            command.add_argument(
                "--availability",
                choices=("present", "missing", "all"),
                default="present",
            )
        if name in {"plan", "download", "complete"}:
            command.add_argument(
                "--max-download-gib",
                type=budget,
                help="Límite de bytes ZIP pendientes, descontando parciales",
            )
        if name in {"download", "complete"}:
            command.add_argument(
                "--background",
                action="store_true",
                help="Devuelve IDs inmediatamente; seguir con agent job",
            )
            command.add_argument(
                "--dry-run",
                action="store_true",
                help="Solo estimación; no crea trabajos",
            )
            command.add_argument("--reserve-gib", type=budget, default=5)
            command.add_argument("--quota-free-gib", type=budget)
            command.add_argument("--attempts", type=core.positive, default=3)
    return cli


def select(data, args):
    """Select whole rolls: a roll with two ZIPs must never silently lose one."""
    groups = rolls.group(data)
    spec = getattr(args, "rolls", None)
    if spec is not None:
        if not spec.strip() or any(not token.strip() for token in spec.split(",")):
            raise UsageError("--rolls requiere identificadores separados por coma")
        try:
            requested = {core.roll_key(token) for token in spec.split(",")}
        except ValueError as error:
            raise UsageError(str(error)) from error
        available = {core.roll_key(row["roll"]) for row in data["files"] if row["roll"]}
        if missing := requested - available:
            raise UsageError(f"Rollos ausentes del inventario: {sorted(missing)}")
        groups = [
            group
            for group in groups
            if group["files"][0][1]["roll"]
            and core.roll_key(group["roll"]) in requested
        ]
    return sorted(groups, key=lambda group: group["roll"].casefold())


def job_paths(job):
    return {
        name: str(job / filename)
        for name, filename in {
            "directory": ".",
            "request": "request.json",
            "state": "state.json",
            "progress": "progress.jsonl",
            "log": "worker.log",
            "result": "result.json",
        }.items()
    }


def job_info(root, job):
    job = workflow.confined(root, str(job.relative_to(root)))
    state = workflow.read_job_status(job)
    request = core.load(job / "request.json")
    active = state["phase"] not in workflow.FINISHED
    progress = None
    if active:
        roll = request["roll"]
        local = [
            library.status(root, row, include_renders=False) for _, row in roll["files"]
        ]
        final = [
            workflow.final_status(root, row, item)
            for (_, row), item in zip(roll["files"], local)
        ]
        progress = monitor.snapshot(root, roll, local, final, state)
    return {
        "id": job.name,
        "roll": request["roll"]["roll"],
        "action": request.get("action", "complete"),
        "active": active,
        "progress": progress,
        "status": state,
        "paths": job_paths(job),
        "result": core.load(job / "result.json")
        if (job / "result.json").exists()
        else None,
    }


def roll_info(root, roll, *, status=None):
    status = status if status is not None else workflow.roll_status(root, roll)
    if status["active"]:
        state = "queued" if status["job"]["phase"] == "queued" else "processing"
    elif status["complete"]:
        state = "complete"
    elif status["errors"]:
        state = "error"
    elif status["label"] == "Sin descargar":
        state = "not_downloaded"
    else:
        state = "incomplete"
    return {
        "roll": roll["roll"],
        "state": state,
        "complete": status["complete"],
        "next_step": status["next_step"],
        "active": status["active"],
        "zips": {"downloaded": status["zip_done"], "total": status["zip_total"]},
        "pdfs": status["pdfs"],
        "images": {"available": status["images"], "expected": status["expected"]},
        "download": status["download"],
        "progress": status["progress"],
        "errors": status["errors"],
        "job": {
            "id": Path(status["job"]["job"]).name,
            "phase": status["job"]["phase"],
            "scheduler": status["job"].get("scheduler"),
            "slurm_job_id": status["job"].get("slurm_job_id"),
        }
        if status["job"]
        else None,
    }


def file_info(path, **values):
    """Current presence/size; hashes in records are provenance, not rehashed here."""
    try:
        size = path.stat().st_size if path.is_file() else None
    except FileNotFoundError:
        size = None  # A worker may publish or remove a file during a query.
    return {
        **values,
        "path": str(path),
        "present": size is not None,
        "bytes": size,
        "size_matches": size == values["expected_bytes"]
        if values.get("expected_bytes") is not None
        else None,
    }


def artifacts(root, row, *, include_images=True):
    directory = library.entry_root(root, row)
    local = library.status(root, row, include_renders=include_images)
    final = workflow.final_status(root, row, local) if include_images else None
    archive = workflow.confined(
        directory,
        str(core.zip_path(directory, library.local_entry(row)).relative_to(directory)),
    )
    pdf_dir = directory / "pdfs" / row["key"]
    manifest = pdf_dir / "manifest.json"
    pdf_records = core.load(manifest)["pdfs"] if manifest.exists() else []
    pdfs = []
    for pdf in pdf_records:
        pdfs.append(
            file_info(
                workflow.confined(pdf_dir, pdf["file"]),
                kind="pdf",
                member=pdf["member"],
                sha256=pdf["sha256"],
                expected_bytes=pdf["bytes"],
                recorded=True,
            )
        )
    known = {pdf["path"] for pdf in pdfs}
    for path in sorted(pdf_dir.glob("*.pdf")):
        if str(path) not in known:
            pdfs.append(
                file_info(
                    workflow.confined(pdf_dir, path.name), kind="pdf", recorded=False
                )
            )
    conversions = []
    for run in local["renders"]:
        output = workflow.confined(
            directory, str(Path(run["path"]).relative_to(directory))
        )
        conversions.append(
            {
                "id": output.name,
                "scope": "final" if str(output) == final["output"] else "other",
                "directory": str(output),
                "images_directory": str(output / "images"),
                "manifest": str(output / "pages.jsonl"),
                "config": run["config"],
                "recorded_images": run["images"],
                "expected_images": run["expected"],
            }
        )
    return {
        "key": row["key"],
        "name": row["name"],
        "directory": str(directory),
        "zip": file_info(
            archive,
            kind="zip",
            state=local["download"],
            expected_bytes=row["bytes"],
            md5=row["md5"],
            sha1=row["sha1"],
        ),
        "partial_zip": file_info(archive.with_suffix(".zip.part"), kind="zip_partial"),
        "pdf_directory": str(pdf_dir),
        "pdf_manifest": str(manifest),
        "pdfs": pdfs,
        "final": final,
        "conversions": conversions,
    }


def image_files(conversion):
    output = Path(conversion["directory"])
    journal = Path(conversion["manifest"])
    if not journal.exists():
        return
    with journal.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.endswith("\n") and not (output / "complete.json").exists():
                continue  # Last line may still be in flight.
            record = json.loads(line)
            path = workflow.confined(output, record["image"])
            yield file_info(
                path,
                kind="image",
                scope=conversion["scope"],
                conversion=conversion["id"],
                page_id=record["page_id"],
                page_number=record["page_number"],
                pdf_member=record["pdf_member"],
                pdf_sha256=record["pdf_sha256"],
                sha256=record["image_sha256"],
                expected_bytes=record["bytes"],
                width=record["width"],
                height=record["height"],
                format=record["format"],
                dpi=record["dpi"],
            )


def files(root, groups, args):
    for roll in groups:
        for _, row in roll["files"]:
            entry = artifacts(root, row, include_images=args.kind in {"all", "image"})
            candidates = []
            if args.kind in {"all", "zip"}:
                candidates.append(entry["zip"])
                if entry["partial_zip"]["present"]:
                    candidates.append(entry["partial_zip"])
            if args.kind in {"all", "pdf"}:
                candidates.extend(entry["pdfs"])
            if args.kind in {"all", "image"}:
                for conversion in entry["conversions"]:
                    if args.scope == "all" or args.scope == conversion["scope"]:
                        # Stream image records instead of collecting an entire roll.
                        for item in image_files(conversion):
                            if available(item, args.availability):
                                yield {"roll": roll["roll"], "key": row["key"], **item}
            for item in candidates:
                if available(item, args.availability):
                    yield {"roll": roll["roll"], "key": row["key"], **item}


def available(item, selection):
    return selection == "all" or item["present"] == (selection == "present")


def page(items, offset, limit):
    """Bound response memory; count totals without keeping every image record."""
    selected, total = [], 0
    for index, item in enumerate(items):
        if offset <= index < offset + limit:
            selected.append(item)
        total = index + 1
    return {
        "items": selected,
        "total": total,
        "offset": offset,
        "limit": limit,
        "next_offset": offset + limit if offset + limit < total else None,
    }


def estimate(root, groups, args):
    entries = [row for roll in groups for _, row in roll["files"]]
    remaining, downloaded = 0, 0
    for row in entries:
        state = library.status(root, row, include_renders=False)
        downloaded += state["download"] == "downloaded"
        if state["download"] != "downloaded":
            remaining += max(0, row["bytes"] - state["partial_bytes"])
    if (
        args.max_download_gib is not None
        and remaining > args.max_download_gib * core.GIB
    ):
        raise UsageError("Los ZIP pendientes superan --max-download-gib")
    return {
        "rolls": [roll["roll"] for roll in groups],
        "zip_total": len(entries),
        "zip_downloaded": downloaded,
        "zip_total_bytes": sum(row["bytes"] for row in entries),
        "remaining_download_bytes": remaining,
        "additional_pdf_image_bytes": None,
        "final_profile": workflow.PROFILE,
    }


def perform(root, data, groups, args):
    plan = estimate(root, groups, args)
    if args.dry_run:
        return {"dry_run": True, "plan": plan}, 0
    for roll in groups:
        job = workflow.job_status(root, roll)
        if job and job["phase"] not in workflow.FINISHED:
            raise ValueError(
                f"Rollo {roll['roll']} ya activo; consultar agent job {Path(job['job']).name}"
            )
    entries = [row for roll in groups for _, row in roll["files"]]
    library.check_download_space(root, entries, args)
    results, failed = [], False
    for roll in groups:
        job = None
        try:
            if args.background:
                job = workflow.start(
                    root, data, roll, action=args.command, options=args
                )
                disposition = "started"
            else:
                with core.locked(root / "operations"):
                    job = workflow.create_job(
                        root, data, roll, action=args.command, options=args
                    )
                workflow.work(job)
                disposition = "done"
            results.append(
                {
                    "roll": roll["roll"],
                    "disposition": disposition,
                    "job": job_info(root, job),
                }
            )
        except Exception as error:
            failed = True
            results.append(
                {
                    "roll": roll["roll"],
                    "disposition": "failed",
                    "error": str(error),
                    "job": job_info(root, job) if job else None,
                }
            )
    return {
        "plan": plan,
        "background": args.background,
        "results": results,
    }, 1 if failed else 0


def execute(root, args):
    if args.command == "refresh":
        data = library.refresh(root, args.metadata)
        return {
            "rolls": len(rolls.group(data)),
            "zips": len(data["files"]),
            "zip_bytes": data["zip_bytes"],
        }, 0
    if args.command in {"job", "cancel"}:
        if Path(args.id).name != args.id or args.id in {".", ".."}:
            raise UsageError("ID de trabajo inválido; usar solo el nombre, sin ruta")
        if args.command == "cancel":
            job = workflow.confined(root, str(Path("jobs") / args.id))
            workflow.stop_job(job)
        return job_info(root, root / "jobs" / args.id), 0
    data = library.load_catalog(root)
    groups = select(data, args)
    if args.command == "index":
        from archive_berrutti import published

        return published.publish(root, data, groups), 0
    if args.command == "list":
        groups = [
            roll
            for roll in groups
            if args.search.casefold()
            in (
                roll["roll"] + " " + " ".join(row["name"] for _, row in roll["files"])
            ).casefold()
        ]
        if args.state == "all":
            # Pagination of the catalog must not stat images in unrequested rolls.
            result = page(groups, args.offset, args.limit)
            result["items"] = [roll_info(root, roll) for roll in result["items"]]
            return result, 0

        def items():
            for roll in groups:
                item = roll_info(root, roll)
                if (
                    args.state == "all"
                    or args.state == item["state"]
                    or (args.state == "local" and item["state"] != "not_downloaded")
                ):
                    yield item

        return page(items(), args.offset, args.limit), 0
    if args.command == "info":
        return {
            "items": [
                {
                    **roll_info(root, roll),
                    "archives": [artifacts(root, row) for _, row in roll["files"]],
                }
                for roll in groups
            ]
        }, 0
    if args.command == "files":
        return page(files(root, groups, args), args.offset, args.limit), 0
    if args.command == "remove":
        if not args.yes:
            raise UsageError("remove requiere --yes; consultar info antes de confirmar")
        return {
            "removed": [
                {
                    "roll": roll["roll"],
                    "directories": workflow.remove(
                        root, roll, stop_active=args.stop_active
                    ),
                }
                for roll in groups
            ]
        }, 0
    if args.command == "plan":
        return estimate(root, groups, args), 0
    return perform(root, data, groups, args)


def main(argv=None):
    args, root = None, None
    response = {"schema_version": SCHEMA_VERSION, "ok": False}
    exit_code = 0
    try:
        args = parser().parse_args(argv)
        response["command"] = args.command
        if socket.gethostname().split(".")[0] in {"ruche01", "ruche02"}:
            raise ValueError("En Ruche ejecutar dentro de Slurm")
        root = library.open_library(
            args.root or library.default_root(), create=args.command == "refresh"
        )
        response["root"] = str(root)
        # Existing stages print progress and render summaries; none leak to stdout.
        with contextlib.redirect_stdout(sys.stderr):
            result, exit_code = execute(root, args)
            if args.command in {"refresh", "complete", "download", "remove", "cancel"}:
                from archive_berrutti import published

                data = library.load_catalog(root)
                selected = (
                    select(data, args)
                    if args.command != "cancel"
                    else [core.load(root / "jobs" / args.id / "request.json")["roll"]]
                )
                published.maybe_publish(root, data, selected)
        response.update(ok=exit_code == 0, data=result)
        if exit_code:
            response["error"] = {
                "code": "operation_failed",
                "message": "Uno o más rollos fallaron; ver data.results",
            }
    except KeyboardInterrupt:
        exit_code = 130
        response["error"] = {
            "code": "interrupted",
            "message": "Interrumpido; los parciales se conservan. Consultar info/job antes de reintentar.",
        }
    except Exception as error:
        exit_code = 2 if isinstance(error, UsageError) else 1
        response["error"] = {
            "code": "invalid_arguments" if exit_code == 2 else "operation_failed",
            "type": type(error).__name__,
            "message": str(error),
        }
    response["observed_at"] = core.now()
    from archive_berrutti import remote

    response["location"] = remote.location(root)
    print(json.dumps(response, ensure_ascii=False, separators=(",", ":")))
    if exit_code:
        raise SystemExit(exit_code)

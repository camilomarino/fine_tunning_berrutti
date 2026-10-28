"""Select Archive ZIPs, download verified originals, and render chosen PDF pages."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ITEM = "archivos-del-terror-de-uruguay_sid-berrutti_indice"
REPO = Path(__file__).resolve().parents[1]
GIB = 1024**3
CHUNK = 1024**2


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def private_path(path: Path) -> Path:
    path = path.resolve()
    if (REPO / ".git").exists() and path.is_relative_to(REPO):
        if not path.is_relative_to(REPO / ".private"):
            raise ValueError(
                "Los datos y manifiestos del repositorio deben ir en .private/"
            )
    return path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_new(path: Path, value: dict) -> None:
    """Publish complete JSON atomically, without replacing earlier evidence."""
    private_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(
                (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
            )
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def freeze(path: Path, value: dict) -> None:
    if path.exists():
        if load(path) != value:
            raise ValueError(
                f"Configuración distinta: utilizar una ruta nueva ({path})"
            )
    else:
        save_new(path, value)


@contextlib.contextmanager
def locked(root: Path):
    private_path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"Otra operación está usando {root}") from error
        yield


def event(root: Path, kind: str, **fields) -> None:
    record = {"at_utc": now(), "event": kind, **fields}
    with (root / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def hashes(path: Path) -> dict:
    algorithms = {name: hashlib.new(name) for name in ("md5", "sha1", "sha256")}
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK):
            for algorithm in algorithms.values():
                algorithm.update(chunk)
    return {name: algorithm.hexdigest() for name, algorithm in algorithms.items()}


def verify_zip(path: Path, entry: dict) -> dict:
    if path.stat().st_size != entry["bytes"]:
        raise ValueError(f"Tamaño incorrecto: {path}")
    actual = hashes(path)
    if any(actual[name] != entry[name] for name in ("md5", "sha1")):
        raise ValueError(f"Checksum incorrecto: {path}")
    return actual


def zip_path(root: Path, entry: dict) -> Path:
    name = entry.get("local_zip_name", f"{entry['key']}.zip")
    if Path(name).name != name or len(name.encode()) > 255:
        raise ValueError("Nombre local de ZIP inválido")
    return root / "zips" / name


def space(
    path: Path, needed: int, reserve_gib: float, *, quota: bool = False, files: int = 0
) -> None:
    available = shutil.disk_usage(path).free
    workdir = os.environ.get("WORKDIR", "")
    if quota and workdir.startswith("/gpfs/workdir/"):
        if path.resolve().is_relative_to(Path(workdir).resolve()):
            # Numeric UID works on compute nodes without GPFS name resolution.
            report = run(["mmlsquota", "-u", str(os.getuid()), "gpfs"], 30)
            rows = [
                line.replace("|", " ").split()
                for line in report.splitlines()
                if line.startswith("gpfs") and "workdir" in line and "USR" in line
            ]
            if len(rows) != 1:
                raise ValueError("No se pudo determinar la cuota de WORKDIR")
            row = rows[0]
            # GPFS reports KiB; account conservatively for uncommitted allocation.
            available = min(available, (int(row[4]) - int(row[3]) - int(row[6])) * 1024)
            if files > int(row[9]) - int(row[8]) - int(row[11]):
                raise ValueError(
                    "La operación supera la cuota de cantidad de archivos de WORKDIR"
                )
    if available < needed + reserve_gib * GIB:
        raise ValueError(
            f"Espacio insuficiente: {available / GIB:.2f} GiB libres, "
            f"{needed / GIB:.2f} GiB necesarios y {reserve_gib:g} GiB de reserva. "
            "La comprobación física no sustituye la cuota de Ruche."
        )


def roll_key(label: str) -> str:
    match = re.fullmatch(r"(\d+)([a-z]*)", label.strip().lower())
    if not match:
        raise ValueError(f"Identificador de rollo inválido: {label}")
    return str(int(match[1])) + match[2]


def catalog(metadata: dict) -> dict:
    identifier = metadata["metadata"]["identifier"]
    if not re.fullmatch(r"[\w.-]+", identifier, flags=re.ASCII):
        raise ValueError("Identificador de Archive inválido")
    entries = []
    for item in metadata["files"]:
        if item.get("format") != "ZIP" or item.get("private"):
            continue
        name = item["name"]
        match = re.search(r"\bRollo\s+(\d+[a-z]*)\b", name, re.IGNORECASE)
        entry = {
            "key": hashlib.sha256(name.encode()).hexdigest()[:20],
            "name": name,
            "roll": match[1] if match else None,
            "bytes": int(item["size"]),
            "md5": item["md5"],
            "sha1": item["sha1"],
        }
        if entry["bytes"] <= 0 or any(
            not re.fullmatch(rf"[0-9a-f]{{{length}}}", entry[field])
            for field, length in (("md5", 32), ("sha1", 40))
        ):
            raise ValueError(f"Metadatos incompletos: {name}")
        entries.append(entry)
    entries.sort(key=lambda row: row["name"])
    if not entries or len({row["key"] for row in entries}) != len(entries):
        raise ValueError("Inventario vacío o nombres de ZIP duplicados")
    return {
        "schema_version": 1,
        "identifier": identifier,
        "observed_utc": now(),
        "metadata_sha256": digest(metadata),
        "files": entries,
        "zip_bytes": sum(row["bytes"] for row in entries),
    }


def select(data: dict, args: argparse.Namespace) -> list[dict]:
    rolls = {roll_key(label) for label in (args.rolls or "").split(",") if label}
    selectors = bool(rolls or args.roll_range or args.match)
    if args.all == selectors:
        raise ValueError("Elegir --all o filtros (--rolls, --roll-range, --match)")
    available = {roll_key(row["roll"]) for row in data["files"] if row["roll"]}
    if missing := rolls - available:
        raise ValueError(f"Rollos ausentes del inventario: {sorted(missing)}")
    bounds = None
    if args.roll_range:
        match = re.fullmatch(r"(\d+):(\d+)", args.roll_range)
        if not match or int(match[1]) > int(match[2]):
            raise ValueError("--roll-range requiere INICIO:FIN, inclusive")
        bounds = int(match[1]), int(match[2])
    selected = []
    for row in data["files"]:
        label = row["roll"]
        if rolls and (not label or roll_key(label) not in rolls):
            continue
        if bounds and (
            not label or not bounds[0] <= int(re.match(r"\d+", label)[0]) <= bounds[1]
        ):
            continue
        if args.match and args.match.casefold() not in row["name"].casefold():
            continue
        selected.append(row)
    if args.limit:
        selected = selected[: args.limit]
    if not selected:
        raise ValueError("Los filtros no seleccionaron ningún ZIP")
    total = sum(row["bytes"] for row in selected)
    if args.max_download_gib is not None and total > args.max_download_gib * GIB:
        raise ValueError(f"La selección ({total / GIB:.2f} GiB) supera el presupuesto")
    return selected


def fetch(
    url: str,
    destination: Path,
    entry: dict,
    attempts: int,
    reserve: float,
    *,
    progress=None,
):
    """Resume only validated HTTP ranges; never publish unverified bytes."""
    last_report, last_phase = 0.0, None

    def report(phase, received, attempt=0, error=None):
        nonlocal last_report, last_phase
        instant = time.monotonic()
        if progress and (phase != last_phase or instant - last_report >= 1):
            progress(
                {
                    "key": entry["key"],
                    "name": entry["name"],
                    "phase": phase,
                    "bytes": received,
                    "total_bytes": entry["bytes"],
                    "attempt": attempt + 1,
                    "error": error,
                }
            )
            last_report, last_phase = instant, phase

    if destination.exists():
        report("verifying", destination.stat().st_size)
        actual = verify_zip(destination, entry)
        report("done", entry["bytes"])
        return actual
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".zip.part")
    for attempt in range(attempts):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > entry["bytes"]:
                raise ValueError("El archivo parcial supera el tamaño declarado")
            report("connecting", offset, attempt)
            if offset < entry["bytes"]:
                space(destination.parent, entry["bytes"] - offset, reserve)
                headers = {"User-Agent": "Berrutti-data-preparation/1.0"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status == 206:
                        match = re.fullmatch(
                            r"bytes (\d+)-(\d+)/(\d+)",
                            response.headers.get("Content-Range", ""),
                        )
                        if not match or tuple(map(int, match.groups())) != (
                            offset,
                            entry["bytes"] - 1,
                            entry["bytes"],
                        ):
                            raise ValueError("Content-Range inconsistente")
                    elif response.status == 200:
                        offset = (
                            0  # Server ignored Range: restart this mutable partial.
                        )
                        space(destination.parent, entry["bytes"], reserve)
                    else:
                        raise ValueError(
                            f"Respuesta HTTP inesperada: {response.status}"
                        )
                    with partial.open("ab" if offset else "wb") as stream:
                        report("downloading", offset, attempt)
                        while chunk := response.read(CHUNK):
                            if stream.tell() + len(chunk) > entry["bytes"]:
                                raise ValueError(
                                    "La respuesta supera el tamaño declarado"
                                )
                            space(destination.parent, len(chunk), reserve)
                            stream.write(chunk)
                            report("downloading", stream.tell(), attempt)
                        stream.flush()
                        os.fsync(stream.fileno())
            report("verifying", partial.stat().st_size, attempt)
            actual = verify_zip(partial, entry)
            os.replace(partial, destination)
            report("done", entry["bytes"], attempt)
            return actual
        except (OSError, ValueError) as error:
            report(
                "failed" if attempt + 1 == attempts else "retrying",
                partial.stat().st_size if partial.exists() else 0,
                attempt,
                str(error),
            )
            if attempt + 1 == attempts:
                raise
            if partial.exists() and partial.stat().st_size >= entry["bytes"]:
                # Preserve the rejected full payload and retry into a new partial.
                rejected = partial.with_name(
                    f"{partial.name}.rejected-{time.time_ns()}"
                )
                partial.rename(rejected)
            time.sleep(min(2**attempt, 8))
    raise RuntimeError("Descarga incompleta")


def download(args) -> None:
    plan = args.plan if isinstance(args.plan, dict) else load(args.plan)
    root = private_path(args.root)
    with locked(root):
        freeze(root / "selection.json", plan)
        pending = sum(
            max(
                0,
                row["bytes"]
                - (
                    zip_path(root, row).with_suffix(".zip.part").stat().st_size
                    if zip_path(root, row).with_suffix(".zip.part").exists()
                    else 0
                ),
            )
            for row in plan["files"]
            if not zip_path(root, row).exists()
        )
        space(root, pending, args.reserve_gib, quota=True)
        if args.quota_free_gib is not None:
            if pending + args.reserve_gib * GIB > args.quota_free_gib * GIB:
                raise ValueError(
                    "La selección supera la cuota libre indicada y su reserva"
                )
        for row in plan["files"]:
            destination = zip_path(root, row)
            url = (
                f"https://archive.org/download/{plan['identifier']}/"
                + urllib.parse.quote(row["name"], safe="")
            )
            try:
                actual = fetch(
                    url,
                    destination,
                    row,
                    args.attempts,
                    args.reserve_gib,
                    progress=getattr(args, "transfer", None),
                )
                freeze(root / "receipts" / f"{row['key']}.json", {**row, **actual})
                event(root, "download_complete", key=row["key"], bytes=row["bytes"])
            except Exception as error:
                event(root, "download_failed", key=row["key"], error=str(error))
                raise


def publish(partial: Path, destination: Path) -> str:
    actual = hashes(partial)["sha256"]
    if destination.exists():
        if hashes(destination)["sha256"] != actual:
            raise ValueError(
                f"El archivo existente difiere; no se reemplaza: {destination}"
            )
        partial.unlink()
    else:
        os.replace(partial, destination)
    return actual


def extract(args) -> None:
    root = private_path(args.root)
    with locked(root):
        for row in load(root / "selection.json")["files"]:
            archive = zip_path(root, row)
            actual = verify_zip(archive, row)
            output = root / "pdfs" / row["key"]
            output.mkdir(parents=True, exist_ok=True)
            manifest = output / "manifest.json"
            if manifest.exists():
                previous = load(manifest)
                if previous["zip_sha256"] != actual["sha256"]:
                    raise ValueError("El ZIP difiere de la extracción anterior")
                for pdf in previous["pdfs"]:
                    if hashes(output / pdf["file"])["sha256"] != pdf["sha256"]:
                        raise ValueError("Un PDF extraído cambió")
                continue
            try:
                with zipfile.ZipFile(archive) as zipped:
                    members = [
                        info
                        for info in zipped.infolist()
                        if not info.is_dir() and info.filename.lower().endswith(".pdf")
                    ]
                    if not members:
                        raise ValueError("El ZIP no contiene PDF")
                    space(
                        output,
                        sum(info.file_size for info in members),
                        args.reserve_gib,
                        quota=True,
                    )
                    records = []
                    for index, info in enumerate(members, 1):
                        if (
                            info.flag_bits & 1
                            or info.file_size > args.max_pdf_gib * GIB
                        ):
                            raise ValueError("PDF cifrado o superior a --max-pdf-gib")
                        # Flatten member paths; never let ZIP names choose destinations.
                        member_name = Path(info.filename.replace("\\", "/")).name
                        if len(member_name.encode()) > 220 or member_name in {
                            "",
                            ".",
                            "..",
                        }:
                            member_name = f"pdf-{index:04d}.pdf"
                        destination = output / f"{index:04d}-{member_name}"
                        partial = destination.with_suffix(".pdf.part")
                        with zipped.open(info) as source, partial.open("wb") as target:
                            while chunk := source.read(CHUNK):
                                space(output, len(chunk), args.reserve_gib)
                                target.write(chunk)
                        if partial.stat().st_size != info.file_size:
                            raise ValueError("PDF incompleto")
                        records.append(
                            {
                                "file": destination.name,
                                "member": info.filename,
                                "bytes": info.file_size,
                                "sha256": publish(partial, destination),
                            }
                        )
                    save_new(
                        manifest, {"zip_sha256": actual["sha256"], "pdfs": records}
                    )
                    event(root, "extract_complete", key=row["key"], pdfs=len(records))
            except Exception as error:
                event(root, "extract_failed", key=row["key"], error=str(error))
                raise


def page_selection(spec: str | None, first: int | None, all_pages: bool, count: int):
    if all_pages or first is not None:
        return range(1, min(count, first) + 1 if first is not None else count + 1)
    pages = set()
    for token in (spec or "").split(","):
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", token.strip())
        if not match:
            raise ValueError("--pages requiere números o rangos: 1-5,10")
        start, end = int(match[1]), int(match[2] or match[1])
        if not 1 <= start <= end <= count:
            raise ValueError(f"Páginas fuera del PDF ({count} páginas): {token}")
        pages.update(range(start, end + 1))
    return sorted(pages)


def run(command: list[str], timeout: int) -> str:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "LC_ALL": "C"},
    )
    if result.returncode:
        raise ValueError(
            f"Falló {command[0]} ({result.returncode}): {result.stderr[-4000:]}"
        )
    return result.stdout


def renderer_version() -> str:
    result = subprocess.run(
        ["pdftoppm", "-v"], capture_output=True, text=True, check=True, timeout=30
    )
    return (result.stderr or result.stdout).splitlines()[0]


def read_journal(journal: Path) -> dict:
    """Keep evidence of a torn final append, then recover only complete records."""
    if not journal.exists():
        return {}
    raw = journal.read_bytes()
    if raw and not raw.endswith(b"\n"):
        boundary = raw.rfind(b"\n") + 1
        tail = raw[boundary:]
        evidence = journal.with_name(f"{journal.name}.incomplete-{time.time_ns()}")
        evidence.write_bytes(tail)
        with journal.open("r+b") as stream:
            stream.truncate(boundary)
            stream.flush()
            os.fsync(stream.fileno())
        raw = raw[:boundary]
    records = {}
    for line in raw.splitlines():
        record = json.loads(line)
        key = record["page_id"]
        if key in records:
            raise ValueError(f"Página duplicada en el manifiesto: {key}")
        records[key] = record
    return records


def render(args) -> None:
    root, output = private_path(args.root), private_path(args.output)
    if root == output:
        raise ValueError("La salida de imágenes requiere su propia ruta")
    for tool in ("pdfinfo", "pdftoppm"):
        if not shutil.which(tool):
            raise ValueError(f"Falta {tool}: instalar/cargar Poppler")
    version = renderer_version()
    plan = load(root / "selection.json")
    config = {
        "schema_version": 1,
        "source_root": str(root),
        "selection_sha256": digest(plan),
        "dpi": args.dpi,
        "format": args.format,
        "jpeg_quality": args.quality,
        "pages": args.pages,
        "first_pages": args.first_pages,
        "all_pages": args.all_pages,
        "renderer": version,
        "script_sha256": hashes(Path(__file__))["sha256"],
        "timeout_seconds": args.timeout,
    }
    with locked(root), locked(output):
        freeze(output / "config.json", config)
        jobs = []
        for row in plan["files"]:
            directory = root / "pdfs" / row["key"]
            for index, pdf in enumerate(load(directory / "manifest.json")["pdfs"], 1):
                path = directory / pdf["file"]
                if hashes(path)["sha256"] != pdf["sha256"]:
                    raise ValueError(f"PDF modificado: {path}")
                info = run(["pdfinfo", str(path)], args.timeout)
                match = re.search(r"^Pages:\s+(\d+)$", info, re.MULTILINE)
                if not match:
                    raise ValueError(f"No se pudo contar páginas: {path}")
                pages = page_selection(
                    args.pages, args.first_pages, args.all_pages, int(match[1])
                )
                jobs.append((row, index, pdf, path, pages))
        total = sum(len(job[-1]) for job in jobs)
        if total > args.max_total_pages:
            raise ValueError(
                f"{total} páginas superan --max-total-pages={args.max_total_pages}"
            )
        journal = output / "pages.jsonl"
        freeze(
            output / "render_plan.json",
            {
                "config_sha256": digest(config),
                "expected_pages": total,
                "pdfs": [
                    {
                        "key": row["key"],
                        "pdf_sha256": pdf["sha256"],
                        "selected_pages": len(pages),
                    }
                    for row, _, pdf, _, pages in jobs
                ],
            },
        )
        completed = read_journal(journal)
        space(
            output,
            0,
            args.reserve_gib,
            quota=True,
            files=max(0, total - len(completed)) + 4,
        )
        extension = "jpg" if args.format == "jpeg" else "png"
        for row, index, pdf, path, pages in jobs:
            for number in pages:
                page_id = f"{row['key']}-pdf{index:04d}-p{number:06d}"
                destination = output / "images" / f"{page_id}.{extension}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                if page_id in completed:
                    if (
                        hashes(destination)["sha256"]
                        != completed[page_id]["image_sha256"]
                    ):
                        raise ValueError(f"Imagen completada modificada: {destination}")
                    continue
                prefix = destination.with_suffix(".part")
                temporary = prefix.with_suffix(f".part.{extension}")
                command = [
                    "pdftoppm",
                    "-f",
                    str(number),
                    "-l",
                    str(number),
                    "-singlefile",
                    "-r",
                    str(args.dpi),
                    f"-{args.format}",
                ]
                if args.format == "jpeg":
                    command += ["-jpegopt", f"quality={args.quality}"]
                command += [str(path), str(prefix)]
                try:
                    space(
                        output,
                        0,
                        args.reserve_gib,
                        quota=number == pages[0] or number % 100 == 0,
                    )
                    run(command, args.timeout)
                    with Image.open(temporary) as image:
                        image.load()
                        width, height = image.size
                        mode = image.mode
                    image_hash = publish(temporary, destination)
                    record = {
                        "page_id": page_id,
                        "identifier": plan["identifier"],
                        "roll": row["roll"],
                        "archive_file": row["name"],
                        "archive_sha1": row["sha1"],
                        "pdf_member": pdf["member"],
                        "pdf_sha256": pdf["sha256"],
                        "page_number": number,
                        "image": str(destination.relative_to(output)),
                        "image_sha256": image_hash,
                        "bytes": destination.stat().st_size,
                        "width": width,
                        "height": height,
                        "mode": mode,
                        "dpi": args.dpi,
                        "format": args.format,
                        "label_status": "unlabeled",
                    }
                    with journal.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    event(output, "page_complete", page_id=page_id)
                except Exception as error:
                    event(output, "page_failed", page_id=page_id, error=str(error))
                    raise
        freeze(
            output / "complete.json",
            {
                "config_sha256": digest(config),
                "expected_pages": total,
            },
        )
        print(json.dumps({"pages": total, "manifest": str(journal)}))


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Se requiere un entero positivo")
    return number


def nonnegative(value: str) -> float:
    number = float(value)
    if not 0 <= number < float("inf"):
        raise argparse.ArgumentTypeError("Se requiere un número finito no negativo")
    return number


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("catalog", help="Guardar inventario, sin bajar ZIP")
    inventory.add_argument("--metadata", type=Path, help="Snapshot local de la API")
    inventory.add_argument("--output", type=Path, required=True)
    selection = commands.add_parser(
        "plan", help="Seleccionar y estimar antes de descargar"
    )
    selection.add_argument("--catalog", type=Path, required=True)
    selection.add_argument("--output", type=Path, required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument(
        "--rolls", help="Identificadores separados por coma; sufijos exactos"
    )
    selection.add_argument(
        "--roll-range", help="INICIO:FIN; incluye variantes con sufijo"
    )
    selection.add_argument(
        "--match", help="Texto literal en el nombre; filtros se intersectan"
    )
    selection.add_argument("--limit", type=positive)
    selection.add_argument("--max-download-gib", type=nonnegative)
    transfer = commands.add_parser(
        "download", help="Descargar/resumir ZIP con checksum"
    )
    transfer.add_argument("--plan", type=Path, required=True)
    transfer.add_argument("--attempts", type=positive, default=3)
    transfer.add_argument("--quota-free-gib", type=nonnegative)
    unpack = commands.add_parser("extract", help="Extraer PDF sin convertir imágenes")
    unpack.add_argument("--max-pdf-gib", type=nonnegative, default=8)
    images = commands.add_parser(
        "render", help="Convertir páginas elegidas con Poppler"
    )
    images.add_argument("--output", type=Path, required=True)
    page_options = images.add_mutually_exclusive_group(required=True)
    page_options.add_argument("--pages", help="Páginas numeradas desde 1: 1-5,10")
    page_options.add_argument("--first-pages", type=positive)
    page_options.add_argument("--all-pages", action="store_true")
    images.add_argument("--dpi", type=positive, default=200)
    images.add_argument("--format", choices=("jpeg", "png"), default="jpeg")
    images.add_argument("--quality", type=int, choices=range(1, 101), default=95)
    images.add_argument("--timeout", type=positive, default=180)
    images.add_argument("--max-total-pages", type=positive, default=1000)
    for command in (transfer, unpack, images):
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--reserve-gib", type=nonnegative, default=5)
    return cli


def main() -> None:
    cli = parser()
    args = cli.parse_args()
    os.umask(0o077)
    try:
        if socket.gethostname().split(".")[0] in {"ruche01", "ruche02"}:
            raise ValueError("En Ruche, ejecutar este script dentro de Slurm")
        if args.command == "catalog":
            if args.metadata:
                metadata = load(args.metadata)
            else:
                with urllib.request.urlopen(
                    f"https://archive.org/metadata/{ITEM}", timeout=60
                ) as response:
                    metadata = json.load(response)
            data = catalog(metadata)
            save_new(args.output, data)
            print(
                json.dumps({"zips": len(data["files"]), "gib": data["zip_bytes"] / GIB})
            )
        elif args.command == "plan":
            data = load(args.catalog)
            rows = select(data, args)
            plan = {
                "schema_version": 1,
                "identifier": data["identifier"],
                "catalog_sha256": digest(data),
                "created_utc": now(),
                "files": rows,
                "zip_bytes": sum(row["bytes"] for row in rows),
                "selection": {
                    key: getattr(args, key)
                    for key in (
                        "all",
                        "rolls",
                        "roll_range",
                        "match",
                        "limit",
                        "max_download_gib",
                    )
                },
            }
            save_new(args.output, plan)
            print(
                json.dumps(
                    {
                        "zips": len(rows),
                        "download_gib": plan["zip_bytes"] / GIB,
                        "pdf_image_bytes": "unknown until extraction/rendering",
                    }
                )
            )
        else:
            {"download": download, "extract": extract, "render": render}[args.command](
                args
            )
    except (
        OSError,
        ValueError,
        KeyError,
        zipfile.BadZipFile,
        subprocess.SubprocessError,
    ) as error:
        cli.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()

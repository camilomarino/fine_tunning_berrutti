"""Incremental library with immutable catalogs, plans, originals and image runs."""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
import unicodedata
import urllib.request
import uuid
from pathlib import Path
from types import SimpleNamespace

from archive_berrutti import manage as core

DEFAULT_ROOT = core.REPO / ".private" / "archive_berrutti" / "library"
PREFERENCES = core.REPO / ".private" / "archive_berrutti" / "preferences.json"


def identifier() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]


def replace_json(path: Path, value: dict) -> None:
    """Replace only mutable UI/cache pointers; durable artifacts use save_new."""
    core.private_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(
                (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
            )
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def default_root() -> Path:
    return (
        Path(core.load(PREFERENCES)["root"]) if PREFERENCES.exists() else DEFAULT_ROOT
    )


def remember(root: Path) -> None:
    from archive_berrutti import remote

    if remote.runtime():
        return  # The client profile owns its default; keep deployed code immutable.
    replace_json(PREFERENCES, {"root": str(root.resolve())})


def open_library(root: Path, *, create: bool = False) -> Path:
    from archive_berrutti import remote

    if profile := remote.runtime():
        root = remote.check_root(root, profile)
    root = core.private_path(root.expanduser())
    manifest = root / "library.json"
    if not manifest.exists():
        if not create:
            raise ValueError(
                "Biblioteca sin inventario. Ejecutar refresh o abrir browse."
            )
        core.save_new(manifest, {"schema_version": 1, "identifier": core.ITEM})
    if core.load(manifest) != {"schema_version": 1, "identifier": core.ITEM}:
        raise ValueError("El destino pertenece a otro ítem o esquema de biblioteca")
    return root


def refresh(root: Path, metadata: Path | None = None) -> dict:
    root = open_library(root, create=True)
    with core.locked(root):
        if metadata:
            raw = core.load(metadata)
        else:
            with urllib.request.urlopen(
                f"https://archive.org/metadata/{core.ITEM}", timeout=60
            ) as response:
                raw = json.load(response)
        data = core.catalog(raw)
        if data["identifier"] != core.ITEM:
            raise ValueError("El inventario no corresponde al Archivo SID configurado")
        # Numeric roll order makes browsing predictable; IDs refer to this snapshot.
        data["files"].sort(
            key=lambda row: (
                int(re.match(r"\d+", row["roll"])[0]) if row["roll"] else 10**9,
                core.roll_key(row["roll"]) if row["roll"] else "",
                row["name"],
            )
        )
        snapshot = Path("catalogs") / f"{identifier()}.json"
        core.save_new(root / snapshot, data)
        replace_json(
            root / "active_catalog.json",
            {"path": str(snapshot), "sha256": core.digest(data)},
        )
        return data


def load_catalog(root: Path) -> dict:
    pointer = core.load(root / "active_catalog.json")
    path = (root / pointer["path"]).resolve()
    if not path.is_relative_to(root.resolve() / "catalogs"):
        raise ValueError("Ruta de catálogo inválida")
    data = core.load(path)
    if core.digest(data) != pointer["sha256"]:
        raise ValueError("El snapshot del inventario cambió")
    return data


def local_entry(row: dict) -> dict:
    return {**row, "local_zip_name": row["name"]}


def entry_root(root: Path, row: dict) -> Path:
    label = row["roll"] or "sin-rollo"
    title = re.sub(r"^.*? - Rollo [^ ]+ - ", "", row["name"]).removesuffix(".zip")
    title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", title).strip("-").lower()[:90] or "archivo"
    return (
        root / "rolls" / f"rollo-{label}" / f"{slug}-{row['key'][:8]}-{row['sha1'][:8]}"
    )


def last_event(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 16384))
        lines = stream.read().splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)
        except (ValueError, UnicodeError):
            continue
    return {}


def status(root: Path, row: dict, *, include_renders=True) -> dict:
    """Bounded metadata inspection; full content validation is an explicit action."""
    directory = entry_root(root, row)
    archive = core.zip_path(directory, local_entry(row))
    partial = archive.with_suffix(".zip.part")
    receipt = directory / "receipts" / f"{row['key']}.json"
    downloaded = "pending"
    problem = None
    if archive.exists():
        downloaded = "inconsistent"
        if receipt.exists() and archive.stat().st_size == row["bytes"]:
            try:
                saved = core.load(receipt)
                if saved["sha1"] == row["sha1"] and saved["md5"] == row["md5"]:
                    downloaded = "downloaded"
            except (ValueError, KeyError, OSError) as error:
                problem = str(error)
    elif partial.exists():
        downloaded = "partial"
    elif last_event(directory / "events.jsonl").get("event", "").endswith("failed"):
        downloaded = "failed"
    pdf_state, pdfs, pdf_available = "none", [], 0
    pdf_directory = directory / "pdfs" / row["key"]
    manifest = pdf_directory / "manifest.json"
    try:
        if manifest.exists():
            pdfs = core.load(manifest)["pdfs"]
            pdf_available = sum(
                (pdf_directory / pdf["file"]).is_file()
                and (pdf_directory / pdf["file"]).stat().st_size == pdf["bytes"]
                for pdf in pdfs
            )
            pdf_state = (
                "extracted"
                if all(
                    (pdf_directory / pdf["file"]).exists()
                    and (pdf_directory / pdf["file"]).stat().st_size == pdf["bytes"]
                    for pdf in pdfs
                )
                else "inconsistent"
            )
        elif pdf_directory.exists():
            pdf_state = "partial"
        renders = []
        image_pages = set()
        if include_renders and (directory / "renders").exists():
            for output in sorted((directory / "renders").iterdir()):
                if not output.is_dir() or not (output / "config.json").exists():
                    continue
                journal = output / "pages.jsonl"
                count = 0
                pages = set()
                journal_valid = True
                if journal.exists():
                    with journal.open(encoding="utf-8") as stream:
                        for line in stream:
                            try:
                                record = json.loads(line)
                                if record["page_id"] in pages:
                                    raise ValueError(
                                        "Página duplicada en el manifiesto"
                                    )
                                pages.add(record["page_id"])
                                count += 1
                            except (ValueError, KeyError, TypeError):
                                journal_valid = False
                expected_path = output / "render_plan.json"
                expected = (
                    core.load(expected_path)["expected_pages"]
                    if expected_path.exists()
                    else None
                )
                complete = (
                    (output / "complete.json").exists()
                    and expected == count
                    and journal_valid
                )
                renders.append(
                    {
                        "path": str(output),
                        "images": count,
                        "image_pages": len(pages),
                        "expected": expected,
                        "state": "complete" if complete else "partial",
                        "config": core.load(output / "config.json"),
                    }
                )
                image_pages.update(pages)
    except (OSError, ValueError, KeyError) as error:
        problem = str(error)
        pdf_state = "inconsistent"
        renders = []
        image_pages = set()
    try:
        partial_bytes = partial.stat().st_size
    except FileNotFoundError:
        partial_bytes = 0  # The downloader may just have published the ZIP.
    return {
        "download": downloaded,
        "partial_bytes": partial_bytes,
        "pdf": pdf_state,
        "pdf_count": len(pdfs),
        "pdf_available": pdf_available,
        "renders": renders,
        "images": sum(run["images"] for run in renders),
        "image_pages": len(image_pages),
        "path": str(directory),
        "problem": problem,
    }


def parse_ids(spec: str, count: int) -> set[int]:
    result = set()
    for token in spec.split(","):
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", token.strip())
        if not match:
            raise ValueError("IDs inválidos; utilizar 1,3,5-8")
        first, last = int(match[1]), int(match[2] or match[1])
        if not 1 <= first <= last <= count:
            raise ValueError(f"ID fuera del inventario (1..{count}): {token}")
        result.update(range(first, last + 1))
    return result


def choose(data: dict, args) -> list[dict]:
    if getattr(args, "ids", None):
        if any(
            getattr(args, key, None) for key in ("all", "rolls", "roll_range", "match")
        ):
            raise ValueError("--ids no se combina con otros filtros")
        selected = [
            data["files"][index - 1]
            for index in sorted(parse_ids(args.ids, len(data["files"])))
        ]
        if args.max_download_gib is not None:
            if sum(row["bytes"] for row in selected) > args.max_download_gib * core.GIB:
                raise ValueError("La selección supera --max-download-gib")
        return selected
    return core.select(data, args)


def plan(data: dict, rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("La selección está vacía")
    return {
        "schema_version": 1,
        "identifier": data["identifier"],
        "catalog_sha256": core.digest(data),
        "created_utc": core.now(),
        "files": rows,
        "zip_bytes": sum(row["bytes"] for row in rows),
    }


def preview(root: Path, rows: list[dict]) -> dict:
    remaining, done = 0, 0
    for row in rows:
        state = status(root, row)
        if state["download"] == "downloaded":
            done += 1
        else:
            remaining += max(0, row["bytes"] - state["partial_bytes"])
    return {
        "selected": len(rows),
        "already_downloaded": done,
        "zip_gib": sum(row["bytes"] for row in rows) / core.GIB,
        "remaining_download_gib": remaining / core.GIB,
        "destination": str(root),
        "pdf_image_bytes": "No calculados todavía",
    }


def render_spec(options):
    parameters = {
        key: getattr(options, key)
        for key in (
            "pages",
            "first_pages",
            "all_pages",
            "dpi",
            "format",
            "quality",
            "timeout",
        )
    }
    parameters["script_sha256"] = core.hashes(Path(core.__file__))["sha256"]
    parameters["poppler_version"] = core.renderer_version()
    render_id = f"{options.dpi}dpi-{options.format}-{core.digest(parameters)[:12]}"
    generation = getattr(options, "render_generation", None)
    if generation:
        if not re.fullmatch(r"[a-zA-Z0-9-]+", generation):
            raise ValueError("Identificador de conversión inválido")
        render_id += f"-{generation}"
    return parameters, render_id


def check_download_space(root, rows, options):
    estimate = preview(root, rows)
    needed = int(estimate["remaining_download_gib"] * core.GIB)
    core.space(root, needed, options.reserve_gib, quota=True)
    if (
        options.quota_free_gib is not None
        and needed + options.reserve_gib * core.GIB > options.quota_free_gib * core.GIB
    ):
        raise ValueError("La selección supera la cuota libre indicada")


def operate(
    root: Path,
    data: dict,
    rows: list[dict],
    action: str,
    options,
    progress=print,
    *,
    transfer=None,
    stage=None,
) -> dict:
    """One library lock, immutable action plan, and resumable per-original storage."""
    with core.locked(root):
        selected = plan(data, rows)
        run_id = identifier()
        core.save_new(root / "plans" / f"{run_id}-{action}.json", selected)
        if action in {"download", "prepare"}:
            check_download_space(root, rows, options)
        total_images = 0
        outputs = []
        for position, row in enumerate(rows, 1):
            directory = entry_root(root, row)
            progress(f"[{position}/{len(rows)}] Rollo {row['roll']}: {row['name']}")
            try:
                if action in {"download", "prepare"}:
                    if stage:
                        stage("download", row)
                    core.download(
                        SimpleNamespace(
                            plan={
                                "identifier": data["identifier"],
                                "files": [local_entry(row)],
                            },
                            root=directory,
                            attempts=options.attempts,
                            reserve_gib=options.reserve_gib,
                            quota_free_gib=None,
                            transfer=transfer,
                        )
                    )
                if action in {"extract", "prepare"}:
                    if stage:
                        stage("extract", row)
                    core.extract(
                        SimpleNamespace(
                            root=directory,
                            reserve_gib=options.reserve_gib,
                            max_pdf_gib=options.max_pdf_gib,
                        )
                    )
            except Exception as error:
                core.event(
                    root,
                    "operation_failed",
                    run_id=run_id,
                    action=action,
                    key=row["key"],
                    error=str(error),
                )
                raise
        if action in {"render", "prepare"}:
            # Preflight the global page budget before creating any images.
            for row in rows:
                directory = entry_root(root, row) / "pdfs" / row["key"]
                for pdf in core.load(directory / "manifest.json")["pdfs"]:
                    info = core.run(
                        ["pdfinfo", str(directory / pdf["file"])], options.timeout
                    )
                    match = re.search(r"^Pages:\s+(\d+)$", info, re.MULTILINE)
                    if not match:
                        raise ValueError("No se pudo contar páginas del PDF")
                    total_images += len(
                        core.page_selection(
                            options.pages,
                            options.first_pages,
                            options.all_pages,
                            int(match[1]),
                        )
                    )
            if total_images > options.max_total_pages:
                raise ValueError(
                    f"{total_images} páginas superan el presupuesto global de {options.max_total_pages}"
                )
            _, render_id = render_spec(options)
            for row in rows:
                directory = entry_root(root, row)
                output = directory / "renders" / render_id
                if stage:
                    stage("render", row)
                progress(f"Convirtiendo Rollo {row['roll']} → {output}")
                try:
                    core.render(
                        SimpleNamespace(
                            root=directory,
                            output=output,
                            **{
                                key: getattr(options, key)
                                for key in (
                                    "pages",
                                    "first_pages",
                                    "all_pages",
                                    "dpi",
                                    "format",
                                    "quality",
                                    "timeout",
                                    "max_total_pages",
                                    "reserve_gib",
                                )
                            },
                        )
                    )
                except Exception as error:
                    core.event(
                        root,
                        "operation_failed",
                        run_id=run_id,
                        action=action,
                        key=row["key"],
                        error=str(error),
                    )
                    raise
                outputs.append(str(output))
        core.event(
            root, "operation_complete", run_id=run_id, action=action, files=len(rows)
        )
        return {
            "action": action,
            "files": len(rows),
            "images": total_images,
            "outputs": outputs,
        }


def verify(root: Path, rows: list[dict]) -> dict:
    checked = 0
    with core.locked(root):
        for row in rows:
            directory = entry_root(root, row)
            actual = core.verify_zip(core.zip_path(directory, local_entry(row)), row)
            pdf_directory = directory / "pdfs" / row["key"]
            manifest = pdf_directory / "manifest.json"
            if manifest.exists():
                saved = core.load(manifest)
                if saved["zip_sha256"] != actual["sha256"]:
                    raise ValueError("La extracción pertenece a otro ZIP")
                for pdf in saved["pdfs"]:
                    if (
                        core.hashes(pdf_directory / pdf["file"])["sha256"]
                        != pdf["sha256"]
                    ):
                        raise ValueError("Un PDF cambió")
            state = status(root, row)
            if state["problem"]:
                raise ValueError(state["problem"])
            for rendered in state["renders"]:
                output = Path(rendered["path"])
                for record in core.read_journal(output / "pages.jsonl").values():
                    path = (output / record["image"]).resolve()
                    if (
                        not path.is_relative_to(output)
                        or core.hashes(path)["sha256"] != record["image_sha256"]
                    ):
                        raise ValueError("Una imagen cambió")
            checked += 1
        core.event(root, "verification_complete", files=checked)
    return {"verified_files": checked}

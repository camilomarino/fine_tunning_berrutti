"""Validate, copy and seal immutable page/line datasets, on local or compute nodes."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import shutil
import socket
import sys
import tarfile
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from PIL import Image

GIB = 1024**3
LUISA_REVISION = "9b9ccccddf0b1e68199bd6cc66c6e90a2434494c"
LUISA_TREE = "519237ab13cc3322f263eba0cfa6344e11d6cb56"
BERRUTTI_HASH = "4af41cdd70771c0c8b0f3caa3739f534507d83a2cd3a49184f75f4af28161ab6"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_tree(root):
    """Verify all upstream bytes against the pinned subtree, without a Git clone."""
    entries = []
    for path in root.iterdir():
        if path.is_symlink():
            raise ValueError("Enlace inesperado en el árbol de datos")
        if path.is_dir():
            mode, digest, order = b"40000", git_tree(path), path.name.encode() + b"/"
        else:
            mode, order = b"100644", path.name.encode()
            blob = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    blob.update(chunk)
            digest = blob.hexdigest()
        entries.append(
            (order, mode + b" " + path.name.encode() + b"\0" + bytes.fromhex(digest))
        )
    content = b"".join(value for _, value in sorted(entries))
    return hashlib.sha1(f"tree {len(content)}\0".encode() + content).hexdigest()


def tree_from_blobs(blobs):
    """The same Git identity, using hashes collected while writing the files."""
    tree = {}
    for relative, digest in blobs.items():
        node = tree
        parts = PurePosixPath(relative).parts
        for name in parts[:-1]:
            node = node.setdefault(name, {})
        node[parts[-1]] = digest

    def encode(node):
        entries = []
        for name, value in node.items():
            directory = isinstance(value, dict)
            digest = encode(value) if directory else value
            entries.append(
                (
                    name.encode() + (b"/" if directory else b""),
                    (b"40000" if directory else b"100644")
                    + b" "
                    + name.encode()
                    + b"\0"
                    + bytes.fromhex(digest),
                )
            )
        content = b"".join(value for _, value in sorted(entries))
        return hashlib.sha1(f"tree {len(content)}\0".encode() + content).hexdigest()

    return encode(tree)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def confined(root, relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Ruta de datos inválida: {relative}")
    target = root / str(path)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Ruta fuera del conjunto: {relative}")
    return target


def check_compute():
    if socket.gethostname().split(".")[0] in {"ruche01", "ruche02"}:
        raise ValueError(
            "Importar/verificar en Ruche requiere un nodo de cómputo Slurm"
        )


def staging(store, dataset, version):
    check_compute()
    store = Path(store).expanduser().resolve()
    base = store / dataset
    target = base / "releases" / version
    if target.exists():
        raise ValueError(f"La versión ya existe; verificar sin sobrescribir: {target}")
    root = Path(__file__).resolve().parents[1]
    if base.is_relative_to(root) and store != root / "data":
        raise ValueError("Dentro del repositorio, guardar los datasets solo en data/")
    base.mkdir(parents=True, exist_ok=True)
    temporary = base / ".staging" / uuid.uuid4().hex
    temporary.mkdir(parents=True, mode=0o700)
    return temporary, target


def activate(target):
    link = target.parent.parent / "current"
    if link.exists() and not link.is_symlink():
        raise ValueError(f"No reemplazar una carpeta existente: {link}")
    temporary = link.with_name(".current-" + uuid.uuid4().hex)
    temporary.symlink_to(target.relative_to(link.parent), target_is_directory=True)
    temporary.replace(link)


def seal(temporary, target, metadata, *, written_inventory=None):
    inventory = list(written_inventory or [])
    # LUISA supplies verified bytes from extraction; don't reopen 71k GPFS files.
    paths = (
        [temporary / "records.jsonl"]
        if written_inventory is not None
        else sorted(temporary.rglob("*"))
    )
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"El snapshot no admite enlaces: {path}")
        if path.is_file():
            inventory.append(
                {
                    "path": path.relative_to(temporary).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    index = temporary / "files.jsonl"
    with index.open("x", encoding="utf-8") as stream:
        for item in inventory:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    metadata = {
        "schema_version": 1,
        "artifact_type": "berrutti-data-release",
        "created_at": datetime.now(timezone.utc).isoformat(),
        **metadata,
        "raw": "raw",
        "records": "records.jsonl",
        "files": "files.jsonl",
        "files_sha256": sha256(index),
        "file_count": len(inventory),
        "bytes": sum(item["bytes"] for item in inventory),
    }
    write_json(temporary / "release.json", metadata)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Different invocations never merge partial trees into an existing release.
    temporary.rename(target)
    activate(target)
    return {"root": str(target), **metadata}


def verify(root):
    check_compute()
    root = Path(root).resolve()
    metadata = json.loads((root / "release.json").read_text())
    inventory = confined(root, metadata["files"])
    if sha256(inventory) != metadata["files_sha256"]:
        raise ValueError("Cambió el inventario de la versión")
    count, size, seen = 0, 0, set()
    for line in inventory.read_text().splitlines():
        entry = json.loads(line)
        path = confined(root, entry["path"])
        if (
            path.is_symlink()
            or entry["path"] in seen
            or not path.is_file()
            or path.stat().st_size != entry["bytes"]
            or sha256(path) != entry["sha256"]
        ):
            raise ValueError(f"Archivo ausente o modificado: {entry['path']}")
        seen.add(entry["path"])
        count += 1
        size += entry["bytes"]
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual != seen | {"release.json", "files.jsonl"}:
        raise ValueError("Hay archivos adicionales en la versión")
    if count != metadata["file_count"] or size != metadata["bytes"]:
        raise ValueError("Los totales no corresponden al inventario")
    return {
        "ok": True,
        "root": str(root),
        "dataset": metadata["dataset"],
        "records": metadata["record_count"],
        "bytes": size,
        "files": count,
    }


def import_berrutti(source, store, split):
    check_compute()
    source = Path(source).resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    if (
        manifest["content_sha256"] != BERRUTTI_HASH
        or manifest["page_count"] != 175
        or len(manifest["pages"]) != 175
    ):
        raise ValueError("Se requiere la versión Berrutti v3 de 175 páginas")
    split_data = json.loads(Path(split).read_text())
    if split_data["dataset_content_sha256"] != BERRUTTI_HASH:
        raise ValueError("Split de otra versión de Berrutti")
    version = "v3.0-" + BERRUTTI_HASH[:12]
    temporary, target = staging(store, "berrutti_175", version)
    raw = temporary / "raw"
    raw.mkdir()
    paths = [entry["path"] for entry in manifest["files"]]
    if len(paths) != len(set(paths)):
        raise ValueError("Rutas duplicadas en el manifiesto")
    for entry in manifest["files"]:
        path = confined(source, entry["path"])
        if (
            not path.is_file()
            or path.stat().st_size != entry["bytes"]
            or sha256(path) != entry["sha256"]
        ):
            raise ValueError(f"Fuente modificada: {entry['path']}")
        destination = confined(raw, entry["path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        if sha256(destination) != entry["sha256"]:
            raise ValueError(f"Copia incorrecta: {entry['path']}")
    shutil.copy2(source / "manifest.json", raw / "manifest.json")
    shutil.copy2(split, temporary / "split-v1.json")
    from data.berrutti import load_pages

    # Retain the same loader, roll grouping, hashes and 125/25/25 partition.
    load_pages(raw, temporary / "split-v1.json")
    partition = {
        identifier: name
        for name, ids in split_data["splits"].items()
        for identifier in ids
    }
    with (temporary / "records.jsonl").open("x", encoding="utf-8") as stream:
        for page in manifest["pages"]:
            stream.write(
                json.dumps(
                    {
                        "id": page["page_id"],
                        "dataset": "berrutti_175",
                        "unit": "page",
                        "image": "raw/" + page["image"],
                        "reference": "raw/" + page["reference"],
                        "image_sha256": page["image_sha256"],
                        "reference_sha256": page["reference_sha256"],
                        "project_split": partition[page["page_id"]],
                        "source_document": None,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return seal(
        temporary,
        target,
        {
            "dataset": "berrutti_175",
            "version": version,
            "record_count": 175,
            "source_content_sha256": BERRUTTI_HASH,
            "split_counts": {k: len(v) for k, v in split_data["splits"].items()},
            "source_manifest_sha256": sha256(source / "manifest.json"),
        },
    )


def extract_luisa(archive, raw, *, maximum=4 * GIB, inspect=None):
    """Extract only the pinned subtree; reject links, duplicates and traversal."""
    raw.mkdir(parents=True)
    total, seen = 0, set()
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            parts = PurePosixPath(member.name).parts
            if member.name.startswith("/") or ".." in parts:
                raise ValueError("Ruta insegura en el archivo de GitLab")
            if member.isdir():
                continue
            if (
                not member.isfile()
                or len(parts) < 3
                or parts[1] != "luisa_tesseract_lines"
            ):
                raise ValueError(
                    f"Archivo fuera del subconjunto solicitado: {member.name}"
                )
            relative = PurePosixPath(*parts[2:]).as_posix()
            total += member.size
            if (
                relative in seen
                or total > maximum
                or len(seen) >= 80000
                or member.size > 64 * 1024**2
            ):
                raise ValueError("Archivo duplicado o extracción supera su presupuesto")
            seen.add(relative)
            path = confined(raw, relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, path.open("xb") as destination:
                if inspect is None:
                    shutil.copyfileobj(source, destination)
                else:
                    payload = source.read()
                    if len(payload) != member.size:
                        raise ValueError("Extracción truncada")
                    inspect(relative, payload)
                    destination.write(payload)
            if path.stat().st_size != member.size:
                raise ValueError("Extracción truncada")
            if len(seen) % 5000 == 0:
                print(
                    f"LUISA: {len(seen)} archivos extraídos y comprobados",
                    file=sys.stderr,
                    flush=True,
                )
    return total


def csv_records(path, *, text=None):
    with (
        io.StringIO(text, newline="")
        if text is not None
        else path.open(newline="", encoding="utf-8-sig")
    ) as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["id", "content"]:
            raise ValueError(f"Esquema CSV inesperado: {path.name}")
        result = {}
        for row in reader:
            if not re.fullmatch(r"\d{1,8}", row["id"]) or not isinstance(
                row["content"], str
            ):
                raise ValueError("ID o texto de línea inválido")
            identifier = f"{int(row['id']):08d}"
            if identifier in result:
                raise ValueError("ID duplicado en el CSV")
            result[identifier] = row["content"]
    return result


def index_luisa(raw, records, *, expected=None, inspected=None, consistency=None):
    expected = expected or {"train": 21611, "test": 2163}
    consistency = Counter() if consistency is None else consistency
    ids, counts = set(), {}
    with records.open("x", encoding="utf-8") as stream:
        for split, suffix in (("train", "21611"), ("test", "2163")):
            gt = csv_records(
                raw / f"gts{suffix}.csv",
                text=inspected[f"gts{suffix}.csv"]["text"]
                if inspected is not None
                else None,
            )
            pred = csv_records(
                raw / f"preds{suffix}.csv",
                text=inspected[f"preds{suffix}.csv"]["text"]
                if inspected is not None
                else None,
            )
            if (
                len(gt) != expected[split]
                or gt.keys() != pred.keys()
                or ids & gt.keys()
            ):
                raise ValueError(
                    "No coinciden IDs/cantidades o hay cruce entre particiones LUISA"
                )
            ids.update(gt)
            counts[split] = len(gt)
            for identifier in sorted(gt):
                image = raw / "images" / (identifier + ".gif")
                truth = image.with_suffix(".gt.txt")
                prediction = image.with_suffix(".pred.txt")
                truth_text = (
                    inspected["images/" + truth.name]["text"]
                    if inspected is not None
                    else truth.read_text(encoding="utf-8")
                )
                prediction_text = (
                    inspected["images/" + prediction.name]["text"]
                    if inspected is not None
                    else prediction.read_text(encoding="utf-8")
                )
                if not gt[identifier].strip():
                    raise ValueError(f"Referencia vacía: {identifier}")
                agreement = {}
                for field, table_text, sidecar_text in (
                    ("reference", gt[identifier], truth_text),
                    ("ocr", pred[identifier], prediction_text),
                ):
                    if table_text.strip() == sidecar_text.strip():
                        agreement[field] = "exact"
                    elif (
                        re.sub(r"\s+", " ", table_text).strip()
                        == re.sub(r"\s+", " ", sidecar_text).strip()
                    ):
                        agreement[field] = "whitespace_only"
                    else:
                        raise ValueError(f"No coincide CSV/GT/predicción: {identifier}")
                    consistency[f"{split}.{field}.{agreement[field]}"] += 1
                if inspected is not None:
                    info = inspected["images/" + image.name]
                    width, height = info["size"]
                    image_hash = info["sha256"]
                    image_format = info["format"]
                else:
                    with Image.open(image) as picture:
                        if getattr(picture, "n_frames", 1) != 1:
                            raise ValueError(
                                f"Imagen de línea inesperada: {identifier}"
                            )
                        width, height = picture.size
                        image_format = picture.format
                        picture.load()
                    image_hash = sha256(image)
                row = {
                    "id": identifier,
                    "dataset": "luisa_tesseract_lines",
                    "unit": "line",
                    "image": "raw/images/" + image.name,
                    "reference": "raw/images/" + truth.name,
                    "ocr_prediction": "raw/images/" + prediction.name,
                    "reference_text": gt[identifier],
                    "ocr_text": pred[identifier],
                    "csv_sidecar_agreement": agreement,
                    "image_sha256": image_hash,
                    "image_format": image_format,
                    "width": width,
                    "height": height,
                    "source_split": split,
                    "project_split": None,
                    "source_document": None,
                    "source_page": None,
                    "requires_overlap_audit": True,
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"LUISA {split}: {len(gt)} líneas verificadas",
                file=sys.stderr,
                flush=True,
            )
    actual = (
        {
            name.removeprefix("images/")
            for name in inspected
            if name.startswith("images/")
        }
        if inspected is not None
        else {p.name for p in (raw / "images").iterdir()}
    )
    wanted = {
        identifier + suffix
        for identifier in ids
        for suffix in (".gif", ".gt.txt", ".pred.txt")
    }
    if actual != wanted:
        raise ValueError("Faltan archivos o hay líneas fuera de los CSV")
    return counts


def import_luisa(archive, store, revision=LUISA_REVISION):
    check_compute()
    if revision != LUISA_REVISION:
        raise ValueError(
            "Revisión no registrada; revisar el catálogo antes de importarla"
        )
    temporary, target = staging(store, "luisa_tesseract_lines", revision)
    inspected, blobs, inventory = {}, {}, []

    def inspect(relative, payload):
        digest = hashlib.sha256(payload).hexdigest()
        blobs[relative] = hashlib.sha1(
            f"blob {len(payload)}\0".encode() + payload
        ).hexdigest()
        inventory.append(
            {"path": "raw/" + relative, "bytes": len(payload), "sha256": digest}
        )
        info = {"sha256": digest}
        if relative.endswith(".gif"):
            with Image.open(io.BytesIO(payload)) as picture:
                if getattr(picture, "n_frames", 1) != 1:
                    raise ValueError(f"Imagen de línea inesperada: {relative}")
                info["size"] = picture.size
                info["format"] = picture.format
                picture.load()
        else:
            info["text"] = payload.decode("utf-8-sig")
        inspected[relative] = info

    extract_luisa(Path(archive), temporary / "raw", inspect=inspect)
    if tree_from_blobs(blobs) != LUISA_TREE:
        raise ValueError(
            "El contenido no corresponde al árbol Git de la revisión fijada"
        )
    consistency = Counter()
    counts = index_luisa(
        temporary / "raw",
        temporary / "records.jsonl",
        inspected=inspected,
        consistency=consistency,
    )
    return seal(
        temporary,
        target,
        {
            "dataset": "luisa_tesseract_lines",
            "version": revision,
            "record_count": sum(counts.values()),
            "published_split_counts": counts,
            "csv_sidecar_consistency": dict(consistency),
            "source": "https://gitlab.fing.edu.uy/memoria/datasets-publicos",
            "source_subdirectory": "luisa_tesseract_lines",
            "source_tree_sha1": LUISA_TREE,
            "image_formats": dict(
                Counter(
                    info["format"] for info in inspected.values() if "format" in info
                )
            ),
            "source_archive_sha256": sha256(Path(archive)),
            "training_eligibility": "requires_source_overlap_audit",
        },
        written_inventory=inventory,
    )

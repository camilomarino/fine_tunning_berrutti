"""Indexed access to sealed datasets; bounded SSH reads, computation on client."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import sqlite3
import subprocess
import tempfile
import uuid
from pathlib import Path, PurePosixPath

from archive_berrutti import remote
from data import manage, releases

MAX_METADATA = 64 * 1024**2
MAX_PREVIEW = 64 * 1024**2
CACHE_VERSION = 1


def digest(value):
    return hashlib.sha256(value).hexdigest()


def dataset(identifier):
    return manage.describe(identifier)["datasets"][0]


def location(identifier, where=None):
    where = where or manage.locations()["default"]
    item = dataset(identifier)
    if where not in item["locations"]:
        raise ValueError(f"{identifier}: sin ubicación en {where}; usar data list")
    result = dict(item["locations"][where])
    result["where"] = where
    root = Path(result["root"])
    if not root.is_absolute() or ".." in root.parts:
        raise ValueError("La ubicación debe tener una raíz absoluta")
    return result


def cache_root():
    base = Path(
        os.environ.get(
            "BERRUTTI_DATA_CACHE", manage.BASE.parent / ".private/data-browser"
        )
    ).expanduser()
    base.mkdir(parents=True, exist_ok=True, mode=0o700)
    return base


class Source:
    """One immutable release. Queries never walk payload directories."""

    def __init__(self, identifier, where=None):
        releases.check_compute()
        if identifier == "archive_berrutti":
            raise ValueError("Archive usa su índice de rollos y el adaptador archive")
        self.identifier = identifier
        self.description = dataset(identifier)
        self.location = location(identifier, where)
        self.root = Path(self.location["root"])
        self.local = self.location.get("host", "local") == "local"
        key = json.dumps(
            [CACHE_VERSION, identifier, self.location], sort_keys=True
        ).encode()
        self.cache = cache_root() / digest(key)
        self.cache.mkdir(exist_ok=True, mode=0o700)
        self._db = None
        self._annotations = None
        self._preview = None
        self.metadata = self.json_file("release.json")
        if (
            self.metadata.get("schema_version") != 1
            or self.metadata.get("dataset") != identifier
            or self.metadata.get("version") != self.location.get("version")
        ):
            raise ValueError("La versión registrada no coincide con release.json")

    def path(self, relative):
        rel = PurePosixPath(relative)
        if rel.is_absolute() or ".." in rel.parts or not rel.parts:
            raise ValueError("Ruta fuera del conjunto")
        target = self.root / str(rel)
        if self.local and not target.resolve().is_relative_to(self.root.resolve()):
            raise ValueError("Enlace fuera del conjunto")
        return target

    def read(self, relative, limit=MAX_METADATA):
        path = self.path(relative)
        if self.local:
            with path.open("rb") as stream:
                content = stream.read(limit + 1)
        else:
            # Only transfer bytes on login. Parsing, filtering and rendering run here.
            result = subprocess.run(
                [
                    *remote.ssh_base(self.location["host"]),
                    shlex.join(["head", "-c", str(limit + 1), "--", str(path)]),
                ],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=60,
            )
            if result.returncode:
                raise ValueError(
                    f"No se pudo leer {path}: "
                    + result.stderr.decode(errors="replace").strip()
                )
            content = result.stdout
        if len(content) > limit:
            raise ValueError(f"Archivo mayor que el límite de lectura: {path}")
        return content

    def cached_bytes(self, relative, sha256=None):
        path = self.cache / (digest(relative.encode()) + ".bin")
        if path.exists():
            content = path.read_bytes()
        else:
            content = self.read(relative)
            if sha256 and digest(content) != sha256:
                raise ValueError(f"Hash incorrecto: {relative}")
            temporary = path.with_suffix("." + uuid.uuid4().hex)
            try:
                temporary.write_bytes(content)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        if sha256 and digest(content) != sha256:
            raise ValueError(
                f"Caché modificada: {relative}; usar browse --refresh-cache"
            )
        return content

    def json_file(self, relative, sha256=None):
        return json.loads(self.cached_bytes(relative, sha256))

    def db(self):
        if self._db is not None:
            return self._db
        path = self.cache / "records.sqlite3"
        if not path.exists():
            payload = self.cached_bytes(self.metadata["records"])
            temporary = path.with_suffix("." + uuid.uuid4().hex)
            connection = sqlite3.connect(temporary)
            try:
                connection.execute(
                    "CREATE TABLE records (id TEXT PRIMARY KEY, project_split TEXT, "
                    "source_split TEXT, search TEXT, payload TEXT)"
                )
                count = 0
                for line in payload.splitlines():
                    row = json.loads(line)
                    if row.get("dataset") != self.identifier:
                        raise ValueError("Registro de otro conjunto")
                    for field in ("image", "reference", "ocr_prediction"):
                        if row.get(field):
                            self.path(row[field])
                    connection.execute(
                        "INSERT INTO records VALUES (?, ?, ?, ?, ?)",
                        (
                            row["id"],
                            row.get("project_split"),
                            row.get("source_split"),
                            (
                                row["id"] + " " + row.get("reference_text", "")
                            ).casefold(),
                            json.dumps(row, ensure_ascii=False),
                        ),
                    )
                    count += 1
                if count != self.metadata["record_count"]:
                    raise ValueError("Cantidad de registros distinta de la versión")
                connection.execute(
                    "CREATE INDEX project_split ON records(project_split)"
                )
                connection.execute("CREATE INDEX source_split ON records(source_split)")
                connection.commit()
                connection.close()
                temporary.replace(path)
            finally:
                connection.close()
                temporary.unlink(missing_ok=True)
        self._db = sqlite3.connect(path)
        return self._db

    def records(self, *, offset=0, limit=50, identifier=None, search="", split=None):
        if offset < 0 or not 1 <= limit <= 500:
            raise ValueError("offset >= 0; limit entre 1 y 500")
        clauses, values = [], []
        if identifier is not None:
            clauses.append("id = ?")
            values.append(identifier)
        if search:
            clauses.append("instr(search, ?) > 0")
            values.append(search.casefold())
        if split:
            field = (
                "project_split" if self.identifier == "berrutti_175" else "source_split"
            )
            clauses.append(field + " = ?")
            values.append(split)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        db = self.db()
        total = db.execute("SELECT count(*) FROM records" + where, values).fetchone()[0]
        rows = db.execute(
            "SELECT payload FROM records" + where + " ORDER BY id LIMIT ? OFFSET ?",
            [*values, limit, offset],
        )
        items = [self.expand(json.loads(row[0])) for row in rows]
        return {
            "items": items,
            "total": total,
            "offset": offset,
            "limit": limit,
            "next_offset": offset + len(items) if offset + len(items) < total else None,
        }

    def expand(self, row):
        result = dict(row)
        result["paths"] = {
            field: str(self.path(row[field]))
            for field in ("image", "reference", "ocr_prediction")
            if row.get(field)
        }
        result["host"] = self.location["host"]
        result["version"] = self.metadata["version"]
        return result

    def record(self, identifier):
        rows = self.records(identifier=identifier)["items"]
        if not rows:
            raise ValueError(f"Ejemplo desconocido: {identifier}")
        return rows[0]

    def annotation_file(self):
        manifest = self.json_file(
            "raw/manifest.json", self.metadata.get("source_manifest_sha256")
        )
        annotation = manifest["annotations"]
        return {**annotation, "path": "raw/" + annotation["path"]}

    def annotations(self, identifier):
        row = self.record(identifier)
        if self.identifier != "berrutti_175":
            raise ValueError("LUISA no publica cajas ni correspondencia con páginas")
        annotation = self.annotation_file()
        if self._annotations is None:
            via = self.json_file(annotation["path"], annotation["sha256"])
            self._annotations = {}
            for page in via["_via_img_metadata"].values():
                name = Path(page["filename"]).name
                if name in self._annotations:
                    raise ValueError("Anotación ambigua para una imagen")
                self._annotations[name] = page
        page = self._annotations[Path(row["image"]).name]
        lines = []
        for index, region in enumerate(page["regions"]):
            shape = region["shape_attributes"]
            if shape.get("name") != "rect":
                raise ValueError("Anotación no rectangular")
            x, y, width, height = (shape[k] for k in ("x", "y", "width", "height"))
            lines.append(
                {
                    "region_index": index,
                    "bbox_xywh": [x, y, width, height],
                    "crop_xyxy": [x, y, x + width, y + height],
                    "text": region["region_attributes"]["text"],
                }
            )
        return {
            "id": identifier,
            "path": str(self.path(annotation["path"])),
            "format": "via2",
            "coordinates": "pixels_in_original_image",
            "order": "annotation_order_not_verified_reading_order",
            "total": len(lines),
            "items": lines,
        }

    def text(self, identifier, kind="reference"):
        row = self.record(identifier)
        field = "reference" if kind == "reference" else "ocr_prediction"
        if not row.get(field):
            raise ValueError(f"Este ejemplo no contiene {kind}")
        content = self.cached_bytes(row[field], row.get(field + "_sha256")).decode(
            "utf-8"
        )
        result = {
            "id": identifier,
            "kind": kind,
            "path": str(self.path(row[field])),
            "text": content,
            "source": "human_transcription" if kind == "reference" else "tesseract",
        }
        csv_field = "reference_text" if kind == "reference" else "ocr_text"
        if csv_field in row:
            result["csv_text"] = row[csv_field]
            result["csv_sidecar_agreement"] = row.get("csv_sidecar_agreement")
        return result

    def files(self, row, kind="all"):
        result = []
        fields = {"image": "image", "reference": "reference", "ocr_prediction": "ocr"}
        for field, role in fields.items():
            if row.get(field) and kind in {"all", role}:
                result.append(
                    {
                        "id": row["id"],
                        "kind": role,
                        "path": str(self.path(row[field])),
                        "sha256": row.get(field + "_sha256"),
                        "availability": "indexed",
                        "host": self.location["host"],
                    }
                )
        if self.identifier == "berrutti_175" and kind in {"all", "annotations"}:
            entry = self.annotation_file()
            result.append(
                {
                    **entry,
                    "path": str(self.path(entry["path"])),
                    "kind": "annotations",
                    "id": row["id"],
                    "host": self.location["host"],
                    "availability": "indexed",
                    "shared_file": True,
                }
            )
        return result

    def image_path(self, row):
        if self.local:
            return self.path(row["image"])
        if self._preview is None:
            self._preview = tempfile.TemporaryDirectory(
                prefix="preview-", dir=self.cache
            )
        folder = Path(self._preview.name)
        path = folder / (digest(row["image"].encode()) + Path(row["image"]).suffix)
        if not path.exists():
            content = self.read(row["image"], MAX_PREVIEW)
            if digest(content) != row["image_sha256"]:
                raise ValueError("Imagen remota distinta del hash registrado")
            # Only one selected image is kept; no growing local corpus.
            for previous in folder.iterdir():
                previous.unlink()
            path.write_bytes(content)
        return path

    def close(self):
        if self._db is not None:
            self._db.close()
        if self._preview:
            self._preview.cleanup()


def clear_cache(identifier, where=None):
    """Invalidate only our derived metadata, never the source data or previews."""
    import shutil

    loc = location(identifier, where)
    key = json.dumps([CACHE_VERSION, identifier, loc], sort_keys=True).encode()
    folder = cache_root() / digest(key)
    if folder.exists():
        for path in folder.iterdir():
            if path.is_file():
                path.unlink()
            elif not path.name.startswith("preview-"):
                shutil.rmtree(path)

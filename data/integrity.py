"""Streaming hashes and immutable receipts shared by data, training and reports."""

import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def closed_json(path: Path, value: dict) -> None:
    """Publish an immutable, complete receipt, never a partially written JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}-{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()

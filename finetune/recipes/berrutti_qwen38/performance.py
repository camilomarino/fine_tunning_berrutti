"""Shared, precision-preserving cache and independent evaluation shards."""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from pathlib import Path


def cache_key(identity: dict) -> str:
    """Bind caches to every declared input/runtime dependency, never a file path."""
    return hashlib.sha256(
        json.dumps(
            identity, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def prepare_line_cache(dataset: Path, cohort: Path, output: Path, workers: int) -> dict:
    """Build one lossless SQLite crop cache with CPU workers; preserve every label.

    A closed cache is reusable by all three arms. A partial build is never
    advertised as complete. No model runtime or GPU is imported here.
    """
    from PIL import Image

    from data.integrity import file_sha256

    if not 1 <= workers <= 16:
        raise ValueError("Crop preparation requires 1 to 16 allocated CPU workers")
    manifest = json.loads((dataset / "manifest.json").read_text())
    source = json.loads(cohort.read_text())
    by_page = {row["page_id"]: row for row in manifest["pages"]}
    pages = source["pages"]
    if len({row["page_id"] for row in pages}) != len(pages):
        raise ValueError("Duplicate source page in sealed cohort")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb"):
        pass
    started = time.monotonic()

    def crop_page(page):
        row = by_page[page["page_id"]]
        image_path = (dataset / row["image"]).resolve()
        if not image_path.is_relative_to(dataset.resolve()):
            raise ValueError("Image is outside the immutable dataset")
        if file_sha256(image_path) != page["image_sha256"]:
            raise ValueError("Source image changed after sealing")
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
        crops = []
        for line in page["regions"]:
            target = line["target"]
            if hashlib.sha256(target.encode()).hexdigest() != line["target_sha256"]:
                raise ValueError("Human line label changed after sealing")
            crop = image.crop(line["crop_xyxy"])
            raw = crop.tobytes()
            blob = io.BytesIO()
            crop.save(blob, format="PNG", compress_level=1)
            payload = blob.getvalue()
            with Image.open(io.BytesIO(payload)) as restored:
                if restored.mode != "RGB" or restored.tobytes() != raw:
                    raise RuntimeError("Crop compression did not preserve exact pixels")
            crops.append(
                (
                    page["page_id"],
                    line["region_index"],
                    target,
                    payload,
                    hashlib.sha256(raw).hexdigest(),
                )
            )
        return crops

    database = sqlite3.connect(output)
    try:
        database.execute(
            "CREATE TABLE crops(page_id TEXT, region_index INTEGER, target TEXT, png BLOB, rgb_sha256 TEXT, PRIMARY KEY(page_id, region_index))"
        )
        database.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT)")
        count = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for crops in pool.map(crop_page, pages):
                database.executemany("INSERT INTO crops VALUES(?, ?, ?, ?, ?)", crops)
                count += len(crops)
                database.commit()
        if (
            count != source["summary"]["train_lines"]
            or len(pages) != source["summary"]["train_pages"]
        ):
            raise ValueError(
                "Crop cache does not cover the complete sealed training cohort"
            )
        summary = {
            "complete": True,
            "source_pages": len(pages),
            "line_images": count,
            "workers": workers,
            "seconds": time.monotonic() - started,
            "cohort_sha256": file_sha256(cohort),
            "pixels_preserved_exactly": True,
            "labels_modified": False,
            "training": False,
        }
        database.execute(
            "INSERT INTO metadata VALUES('closed_receipt', ?)", (json.dumps(summary),)
        )
        database.commit()
    finally:
        database.close()
    return {**summary, "bytes": output.stat().st_size}


def page_shard(page_ids: list[str], rank: int, workers: int) -> list[str]:
    if len(set(page_ids)) != len(page_ids):
        raise ValueError("Duplicate page in evaluation input")
    if not 0 <= rank < workers <= len(page_ids):
        raise ValueError("Invalid evaluation shard")
    return page_ids[rank::workers]


def merge_page_metrics(
    shards: list[list[dict]], expected_ids: list[str], evaluation_key: str
) -> dict:
    """Micro-average exact coverage; partial or retried failures remain visible."""
    if not expected_ids or len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Expected a nonempty, unique evaluation split")
    rows = [row for shard in shards for row in shard]
    ids = [row["page_id"] for row in rows]
    if len(set(ids)) != len(ids) or set(ids) != set(expected_ids):
        raise ValueError("Evaluation shards overlap or do not cover the complete split")
    totals = dict.fromkeys(
        ("char_errors", "char_count", "word_errors", "word_count"), 0
    )
    failed = retried = 0
    for row in rows:
        if row.get("evaluation_key") != evaluation_key or not evaluation_key:
            raise ValueError("Cannot combine checkpoints or evaluation policies")
        if row.get("status") not in ("complete", "incomplete"):
            raise ValueError("Each page must preserve its completion status")
        if type(row.get("attempts")) is not int or not 1 <= row["attempts"] <= 2:
            raise ValueError("Each page is allowed at most two generation attempts")
        failed += row["status"] != "complete"
        retried += row["attempts"] == 2
        for key in totals:
            if type(row.get(key)) is not int or row[key] < 0:
                raise ValueError("Invalid per-page score counts")
            totals[key] += row[key]
    if not totals["char_count"] or not totals["word_count"]:
        raise ValueError("Empty normalized references")
    return {
        **totals,
        "cer": totals["char_errors"] / totals["char_count"],
        "wer": totals["word_errors"] / totals["word_count"],
        "pages": len(rows),
        "failed_pages": failed,
        "retried_pages": retried,
        "all_pages_completed": failed == 0,
    }


def map_tensors(value, function):
    """Preserve the installed vision output type, including ModelOutput dataclasses."""
    import torch

    if isinstance(value, torch.Tensor):
        return function(value)
    if is_dataclass(value):
        return type(value)(
            **{
                field.name: map_tensors(getattr(value, field.name), function)
                for field in fields(value)
            }
        )
    if isinstance(value, dict):
        return {key: map_tensors(item, function) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(map_tensors(item, function) for item in value)
    if value is None:
        return None
    raise TypeError(f"Unsupported visual output type: {type(value).__name__}")


class FrozenVisionCache:
    """CPU cache of a frozen, deterministic visual encoder; never decoder states.

    The caller hashes pixels, grid, frozen weights and runtime/processor policy
    into the active key. A context cannot cross an optimizer update that changes
    visual weights. The wrapper retains Accelerate routing on every cache miss.
    """

    def __init__(self, visual):
        if any(parameter.requires_grad for parameter in visual.parameters()):
            raise ValueError("Visual feature caching requires a fully frozen encoder")
        self.visual = visual
        self.original = visual.forward
        self.entries = {}
        self.active_key = None
        self.hits = self.misses = 0
        visual.forward = self.forward

    @contextmanager
    def activate(self, key: str):
        if self.active_key is not None:
            raise RuntimeError("Nested visual cache contexts are unsupported")
        self.active_key = key
        try:
            yield
        finally:
            self.active_key = None

    def forward(self, *args, **kwargs):
        import torch

        if self.active_key is None:
            return self.original(*args, **kwargs)
        if self.visual.training or any(
            p.requires_grad for p in self.visual.parameters()
        ):
            raise RuntimeError(
                "Visual cache cannot bypass a trainable or training encoder"
            )
        key = self.active_key
        if key not in self.entries:
            with torch.no_grad():
                result = self.original(*args, **kwargs)
            devices = []

            def save(tensor):
                devices.append(tensor.device)
                return tensor.detach().to("cpu", copy=True)

            self.entries[key] = (map_tensors(result, save), devices)
            self.misses += 1
            return result
        result, devices = self.entries[key]
        iterator = iter(devices)
        self.hits += 1
        return map_tensors(result, lambda tensor: tensor.to(next(iterator)))

    def close(self):
        self.visual.forward = self.original
        self.entries.clear()

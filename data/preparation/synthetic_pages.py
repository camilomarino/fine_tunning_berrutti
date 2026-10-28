"""Seeded one/two-column compositions of complete, human-labelled line crops."""

from __future__ import annotations

import hashlib
import io
import random
import sqlite3
import threading
from collections import OrderedDict
from pathlib import Path

from PIL import Image, ImageDraw


class CropStore:
    """Verified lossless crop bytes, with a bounded RAM cache and one backing file."""

    def __init__(self, path, cache_bytes=512 * 1024 * 1024):
        Path(path).open("xb").close()
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.execute(
            "CREATE TABLE crops (line_id TEXT PRIMARY KEY, reference_sha TEXT, "
            "png BLOB, rgb_sha TEXT)"
        )
        self.lock = threading.Lock()
        self.cache, self.bytes, self.limit = OrderedDict(), 0, cache_bytes

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self.connection.close()
        self.cache.clear()

    def _remember(self, key, row):
        if len(row[1]) > self.limit:
            return
        self.cache[key] = row
        self.bytes += len(row[1])
        while self.bytes > self.limit:
            _, removed = self.cache.popitem(last=False)
            self.bytes -= len(removed[1])

    def add(self, line, encoded, *, expected_rgb=None):
        with Image.open(io.BytesIO(encoded)) as source:
            image = source.convert("RGB")
        try:
            rgb = hashlib.sha256(image.tobytes()).hexdigest()
            if image.size != tuple(line["size"]) or (
                expected_rgb is not None and rgb != expected_rgb
            ):
                raise ValueError("Cached source crop pixels or dimensions changed")
        finally:
            image.close()
        reference = hashlib.sha256(line["reference_text"].encode()).hexdigest()
        row = (reference, encoded, rgb)
        self.connection.execute(
            "INSERT INTO crops VALUES (?,?,?,?)", (line["line_id"], *row)
        )
        self._remember(line["line_id"], row)

    def seal(self):
        self.connection.commit()
        self.connection.execute("PRAGMA query_only=ON")

    def load(self, line):
        key = line["line_id"]
        with self.lock:
            row = self.cache.get(key)
            if row is None:
                row = self.connection.execute(
                    "SELECT reference_sha,png,rgb_sha FROM crops WHERE line_id=?",
                    (key,),
                ).fetchone()
                if row is not None:
                    self._remember(key, row)
            else:
                self.cache.move_to_end(key)
        if (
            row is None
            or row[0] != hashlib.sha256(line["reference_text"].encode()).hexdigest()
        ):
            raise ValueError("Missing source crop or changed human reference")
        with Image.open(io.BytesIO(row[1])) as source:
            image = source.convert("RGB")
        if (
            image.size != tuple(line["size"])
            or hashlib.sha256(image.tobytes()).hexdigest() != row[2]
        ):
            image.close()
            raise ValueError("Cached source crop pixels or dimensions changed")
        return image


def layout(lines: list[dict], size: tuple[int, int], seed: int, columns: int):
    """Pack a prefix without clipping; remaining lines start the next page."""
    if not lines or columns not in (1, 2) or min(size) < 64:
        raise ValueError("Nonempty lines, a page-sized canvas and 1/2 columns required")
    rng = random.Random(seed)
    width, height = size
    margin = round(width * rng.uniform(0.055, 0.09))
    top, bottom = round(height * rng.uniform(0.045, 0.085)), round(height * 0.93)
    gutter = round(width * rng.uniform(0.045, 0.075)) if columns == 2 else 0
    available = width - 2 * margin - gutter
    first = round(available * rng.uniform(0.44, 0.56)) if columns == 2 else available
    widths = [first, available - first] if columns == 2 else [first]
    nominal_height = height / rng.uniform(55, 72)
    placements, column, y = [], 0, top
    for line in lines[:48]:
        if columns == 2 and column == 0 and len(placements) == 24:
            column, y = 1, top
        original_width, original_height = line["size"]
        if min(original_width, original_height) < 1:
            raise ValueError("Invalid line dimensions")
        while True:
            scale = min(
                widths[column] / original_width,
                nominal_height * rng.uniform(0.9, 1.1) / original_height,
            )
            w, h = (
                max(1, int(original_width * scale)),
                max(1, int(original_height * scale)),
            )
            if y + h <= bottom:
                break
            column += 1
            if column == columns:
                return placements
            y = top
        x = margin + (widths[0] + gutter if column else 0)
        x += rng.randint(0, max(0, widths[column] - w))
        placements.append(
            {"line_id": line["line_id"], "column": column, "box": [x, y, x + w, y + h]}
        )
        y += h + round(nominal_height * rng.uniform(0.2, 0.55))
        if rng.random() < 0.09:
            y += round(nominal_height * rng.uniform(0.5, 1.1))
    return placements


def render(
    lines: list[dict],
    size: tuple[int, int],
    seed: int,
    columns: int,
    output: Path,
    *,
    crop_loader=None,
):
    """Use original pixels as crops; only non-text background marks are invented."""
    placements = layout(lines, size, seed, columns)
    if not placements:
        raise ValueError("No line fits the requested page")
    rng = random.Random(seed + 7919)
    shade = rng.randint(244, 255)
    canvas = Image.new("RGB", size, (shade, shade, max(0, shade - rng.randint(0, 5))))
    draw = ImageDraw.Draw(canvas)
    if rng.random() < 0.35:
        y = int(size[1] * 0.025)
        draw.line(
            (int(size[0] * 0.08), y, int(size[0] * 0.92), y),
            fill=(190, 190, 185),
            width=1,
        )
    if columns == 2 and rng.random() < 0.3:
        right_edge = max(p["box"][2] for p in placements if p["column"] == 0)
        second = [p["box"][0] for p in placements if p["column"] == 1]
        if second:
            x = (right_edge + min(second)) // 2
            draw.line(
                (x, int(size[1] * 0.04), x, int(size[1] * 0.94)),
                fill=(210, 210, 205),
                width=1,
            )
    sources = []
    for line, placement in zip(lines, placements, strict=False):
        if crop_loader is None:
            with Image.open(line["image"]) as image:
                crop = image.convert("RGB")
            if line.get("crop_xyxy"):
                crop = crop.crop(line["crop_xyxy"])
        else:
            crop = crop_loader(line)
        try:
            if crop.size != tuple(line["size"]):
                raise ValueError("Source crop dimensions changed")
            x, y, right, bottom = placement["box"]
            crop = crop.resize((right - x, bottom - y), Image.Resampling.LANCZOS)
            canvas.paste(crop, (x, y))
        finally:
            crop.close()
        sources.append(
            {
                **placement,
                "source": line["source"],
                "image_sha256": line["image_sha256"],
                "reference_sha256": hashlib.sha256(
                    line["reference_text"].encode()
                ).hexdigest(),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        canvas.save(stream, format="PNG")
    canvas.close()
    return {
        "image": str(output),
        "size": list(size),
        "columns": columns,
        "seed": seed,
        "reading_order": "left column top-to-bottom, then right column top-to-bottom",
        "sources": sources,
        "reference_text": "\n".join(
            line["reference_text"] for line in lines[: len(placements)]
        ),
        "no_source_line_clipped": True,
        "artificial_text": False,
    }

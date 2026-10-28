"""Original line labels, geometric bounds and audited reading-order alignment."""

import hashlib
import unicodedata
from collections import defaultdict


def canonical_text(value: str) -> str:
    """Match content faithfully, allowing only NFC and whitespace differences."""
    return " ".join(unicodedata.normalize("NFC", value).split())


def reference_order(reference: str, rows: list[dict], *, max_states: int = 10000):
    """Verify the complete target multiset, without guessing visual reading order."""
    pools = defaultdict(list)
    for row in rows:
        pools[canonical_text(row["target"])].append(row)
    texts = sorted(pools, key=lambda text: (-len(text.split()), text))
    pieces = [tuple(text.split()) for text in texts]
    words = tuple(canonical_text(reference).split())
    initial = tuple(len(pools[text]) for text in texts)
    seen = set()
    states = 0

    def search(position, remaining):
        nonlocal states
        key = (position, remaining)
        if key in seen:
            return None
        seen.add(key)
        states += 1
        if states > max_states:
            return None
        if position == len(words):
            return [] if not any(remaining) else None
        for index, piece in enumerate(pieces):
            if remaining[index] and words[position : position + len(piece)] == piece:
                counts = list(remaining)
                counts[index] -= 1
                tail = search(position + len(piece), tuple(counts))
                if tail is not None:
                    return [index, *tail]
        return None

    indices = search(0, initial)
    if indices is None:
        return None
    used = defaultdict(int)
    ordered = []
    for index in indices:
        text = texts[index]
        ordered.append(pools[text][used[text]])
        used[text] += 1
    return ordered


def aligned_regions(reference: str, regions: list[dict], size: tuple[int, int]):
    """Accept a page only if all line targets reconstruct its complete content."""
    width, height = size
    rows = []
    for index, region in enumerate(regions):
        shape = region["shape_attributes"]
        if shape.get("name") != "rect":
            raise ValueError("Line annotation is not rectangular")
        box = [shape[key] for key in ("x", "y", "width", "height")]
        if not all(
            isinstance(value, int) and not isinstance(value, bool) for value in box
        ):
            raise ValueError("Line coordinates must be integers")
        x, y, w, h = box
        if not (0 <= x < x + w <= width and 0 <= y < y + h <= height):
            raise ValueError("Line annotation lies outside its source image")
        target = region["region_attributes"]["text"].strip()
        if not canonical_text(target):
            raise ValueError("Empty annotated line target")
        rows.append(
            {
                "region_index": index,
                "crop_xyxy": [x, y, x + w, y + h],
                "target": target,
                "target_sha256": hashlib.sha256(target.encode()).hexdigest(),
            }
        )
    candidates = (
        ("annotation", rows),
        (
            "yx",
            sorted(
                rows,
                key=lambda row: (
                    row["crop_xyxy"][1],
                    row["crop_xyxy"][0],
                    row["region_index"],
                ),
            ),
        ),
    )
    expected = canonical_text(reference)
    if not expected:
        raise ValueError("Empty page reference")
    for order, ordered in candidates:
        if canonical_text(" ".join(row["target"] for row in ordered)) == expected:
            return {"order": order, "regions": ordered}
    verified = reference_order(reference, rows)
    if verified is not None:
        return {"order": "reference_verified_multiset", "regions": verified}
    return None


def all_regions(regions: list[dict], size: tuple[int, int]) -> list[dict]:
    """Keep every original human line label, independently of the page transcript."""
    width, height = size
    result = []
    for index, region in enumerate(regions):
        shape = region["shape_attributes"]
        if shape.get("name") != "rect":
            raise ValueError("Line annotation must be rectangular")
        values = [shape[key] for key in ("x", "y", "width", "height")]
        if any(
            not isinstance(value, int) or isinstance(value, bool) for value in values
        ):
            raise ValueError("Line coordinates must be integers")
        x, y, w, h = values
        if not (0 <= x < x + w <= width and 0 <= y < y + h <= height):
            raise ValueError("Line annotation lies outside its source image")
        target = region["region_attributes"]["text"]
        if not isinstance(target, str) or not canonical_text(target):
            raise ValueError("Every line must have a nonempty human transcript")
        result.append(
            {
                "region_index": index,
                "crop_xyxy": [x, y, x + w, y + h],
                "target": target,
                "target_sha256": hashlib.sha256(target.encode()).hexdigest(),
            }
        )
    if not result:
        raise ValueError(
            "Every one of the 125 training pages must have annotated lines"
        )
    return result

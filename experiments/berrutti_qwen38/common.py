"""Data and metric contracts for the private Qwen3.8 OCR training experiment."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from experiments.berrutti_qwen38.scoring import (
    NORMALIZERS,
    levenshtein_distance,
    normalize_text,
)

SPLITS = ("train", "validation", "test")
ROLL = re.compile(r"^[a-z]+_(r[0-9]+)[_-]")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pages(dataset_root: Path, split_path: Path) -> dict[str, list[dict]]:
    """Verify every image/reference and a disjoint, roll-grouped 175-page split."""
    dataset_root = dataset_root.resolve()
    manifest = json.loads((dataset_root / "manifest.json").read_text())
    split = json.loads(split_path.read_text())
    pages = {row["page_id"]: row for row in manifest["pages"]}
    if len(pages) != 175 or len(manifest["pages"]) != 175:
        raise ValueError("Expected the immutable 175-page Berrutti v3 snapshot")
    if set(split["splits"]) != set(SPLITS):
        raise ValueError("Split must contain train, validation, test")
    if manifest["content_sha256"] != split["dataset_content_sha256"]:
        raise ValueError("Split is bound to a different dataset")
    seen, roll_owner = set(), {}
    result = {}
    for name in SPLITS:
        selected = []
        for page_id in split["splits"][name]:
            if page_id in seen or page_id not in pages:
                raise ValueError(f"Unknown or duplicate page: {page_id}")
            seen.add(page_id)
            match = ROLL.match(page_id)
            if not match:
                raise ValueError(f"Cannot determine roll: {page_id}")
            roll = match.group(1)
            if roll in roll_owner and roll_owner[roll] != name:
                raise ValueError(f"Roll {roll} crosses train/validation/test")
            roll_owner[roll] = name
            row = pages[page_id].copy()
            for key in ("image", "reference"):
                path = (dataset_root / row[key]).resolve()
                if not path.is_relative_to(dataset_root) or not path.is_file():
                    raise ValueError(f"Unsafe or missing {key}: {page_id}")
                if file_sha256(path) != row[f"{key}_sha256"]:
                    raise ValueError(f"Changed {key}: {page_id}")
                row[key] = path
            row["reference_text"] = row["reference"].read_text(encoding="utf-8")
            selected.append(row)
        result[name] = selected
    if seen != set(pages):
        raise ValueError("The split does not cover exactly all 175 pages")
    return result


def score_pairs(pairs: list[tuple[str, str]]) -> dict:
    """Micro CER/WER under the repository's ascii-text-v1 normalizers."""
    char_errors = char_count = word_errors = word_count = 0
    for reference, prediction in pairs:
        ref = normalize_text(reference, NORMALIZERS)
        hyp = normalize_text(prediction, NORMALIZERS)
        if not ref:
            raise ValueError("Empty normalized reference")
        char_errors += levenshtein_distance(ref, hyp)
        char_count += len(ref)
        ref_words, hyp_words = ref.split(), hyp.split()
        word_errors += levenshtein_distance(ref_words, hyp_words)
        word_count += len(ref_words)
    return {
        "cer": char_errors / char_count,
        "wer": word_errors / word_count,
        "char_errors": char_errors,
        "char_count": char_count,
        "word_errors": word_errors,
        "word_count": word_count,
        "pages": len(pairs),
    }

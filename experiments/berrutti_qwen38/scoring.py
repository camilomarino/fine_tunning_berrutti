"""Pinned OCR scoring shared by this standalone training project.

This implements the benchmark's ascii-text-v1 policy without importing the
benchmark package. Keep the normalizer order and micro aggregation unchanged
when comparing existing runs with future ones.
"""

from __future__ import annotations

from collections.abc import Sequence

from anyascii import anyascii
from rapidfuzz.distance import Levenshtein


NORMALIZERS = ("anyascii@1", "all-whitespace@1")


def normalize_text(text: str, normalizers: Sequence[str] = NORMALIZERS) -> str:
    """Apply the exact ascii-text-v1 normalizers in their declared order."""
    if not isinstance(text, str):
        raise TypeError("Text to normalize must be a string")
    normalized = text
    for normalizer in normalizers:
        if normalizer == "anyascii@1":
            normalized = anyascii(normalized)
        elif normalizer == "all-whitespace@1":
            normalized = " ".join(normalized.split())
        else:
            raise ValueError(f"Unsupported OCR normalizer: {normalizer}")
    return normalized


def levenshtein_distance(left: str | list[str], right: str | list[str]) -> int:
    """Count substitutions, insertions and deletions with RapidFuzz."""
    return int(Levenshtein.distance(left, right))

"""Pinned OCR scoring shared by this standalone training project.

This implements the benchmark's ascii-text-v1 policy without importing the
benchmark package. Keep the normalizer order and micro aggregation unchanged
when comparing existing runs with future ones.
"""

from __future__ import annotations

import math
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


def select_complete_validation_epoch(summaries: list[dict]) -> dict | None:
    eligible = [
        item
        for item in summaries
        if item["validation"].get("failed_pages", 0) == 0
        and math.isfinite(item["validation"]["cer"])
    ]
    return min(
        eligible,
        key=lambda item: (item["validation"]["cer"], item["epoch"]),
        default=None,
    )

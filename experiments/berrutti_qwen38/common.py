"""Compatibility exports for data access and scoring; no implementation here."""

from data.berrutti import ROLL, SPLITS, load_pages  # noqa: F401
from data.integrity import file_sha256  # noqa: F401
from evaluation.scoring import (  # noqa: F401
    NORMALIZERS,
    levenshtein_distance,
    normalize_text,
    score_pairs,
)

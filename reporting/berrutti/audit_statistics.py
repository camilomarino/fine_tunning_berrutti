"""Paired, descriptive uncertainty for sealed OCR evaluations, without inference."""

from __future__ import annotations

import math
import random
import re


def quantile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def paired_difference(left, right, *, resamples=50000, seed=20261022, by_roll=False):
    """Bootstrap the difference in micro CER, preserving pairs and denominators.

    These intervals condition on the already selected checkpoints and one training
    seed. Roll resampling is a sensitivity analysis, not certified document IDs.
    Missing/failed pages must be handled explicitly by the caller.
    """
    if resamples < 100:
        raise ValueError("At least 100 resamples are required")
    a = {row["page_id"]: row for row in left}
    b = {row["page_id"]: row for row in right}
    if len(a) != len(left) or len(b) != len(right) or not a or a.keys() != b.keys():
        raise ValueError("Paired evaluations require identical unique page coverage")
    groups = {}
    for page_id in sorted(a):
        x, y = a[page_id], b[page_id]
        if x["char_count"] != y["char_count"] or x["char_count"] <= 0:
            raise ValueError("Paired references have different or empty denominators")
        if any(
            type(row[key]) is not int or row[key] < 0
            for row in (x, y)
            for key in ("char_count", "char_errors")
        ):
            raise ValueError("Edit counts must be nonnegative integers")
        if by_roll:
            match = re.match(r"^[a-z]+_(r[0-9]+)[_-]", page_id)
            if match is None:
                raise ValueError("Cannot identify scan roll")
            group = match.group(1)
        else:
            group = page_id
        delta, count = groups.get(group, (0, 0))
        groups[group] = (
            delta + x["char_errors"] - y["char_errors"],
            count + x["char_count"],
        )
    units = list(groups.values())
    total_count = sum(value[1] for value in units)
    difference = sum(value[0] for value in units) / total_count
    rng = random.Random(seed)
    samples = []
    for _ in range(resamples):
        delta = denominator = 0
        for _ in units:
            errors, characters = units[rng.randrange(len(units))]
            delta += errors
            denominator += characters
        samples.append(delta / denominator)
    return {
        "difference_left_minus_right": difference,
        "percentile_95_interval": [quantile(samples, 0.025), quantile(samples, 0.975)],
        "resamples": resamples,
        "seed": seed,
        "pages": len(a),
        "resampling_units": len(units),
        "unit": "scan_roll" if by_roll else "page",
        "left_lower_error_pages": sum(
            a[p]["char_errors"] < b[p]["char_errors"] for p in a
        ),
        "equal_error_pages": sum(a[p]["char_errors"] == b[p]["char_errors"] for p in a),
        "left_higher_error_pages": sum(
            a[p]["char_errors"] > b[p]["char_errors"] for p in a
        ),
        "conditional_on_selected_checkpoints_and_single_training_seed": True,
    }

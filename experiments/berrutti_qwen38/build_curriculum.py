"""Create a private, alignment-checked train-only line-crop curriculum."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image

from experiments.berrutti_qwen38.common import NORMALIZERS, file_sha256, load_pages
from experiments.berrutti_qwen38.scoring import normalize_text


def build(dataset: Path, split: Path, candidates: Path) -> tuple[list[dict], dict]:
    pages = load_pages(dataset, split)
    train = {page["page_id"]: page for page in pages["train"]}
    normalized = {
        page_id: normalize_text(page["reference_text"], NORMALIZERS)
        for page_id, page in train.items()
    }
    dimensions = {}
    grouped = defaultdict(list)
    seen = set()
    rejected = defaultdict(int)
    for line in candidates.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        page_id = item["source_page_id"]
        if page_id not in train or item.get("split") != "train":
            raise ValueError(f"Candidate outside training split: {page_id}")
        page = train[page_id]
        if item["image_sha256"] != page["image_sha256"]:
            raise ValueError(f"Changed candidate image: {page_id}")
        key = (page_id, item["source_region_index"])
        if key in seen:
            raise ValueError(f"Duplicate region: {key}")
        seen.add(key)
        box = item["crop_xyxy"]
        if not (
            isinstance(box, list)
            and len(box) == 4
            and all(isinstance(v, int) for v in box)
        ):
            raise ValueError(f"Invalid region box: {key}")
        if page_id not in dimensions:
            with Image.open(page["image"]) as image:
                dimensions[page_id] = image.size
        width, height = dimensions[page_id]
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(f"Out-of-image region box: {key}")
        target = item["target"].strip()
        if (
            len(target) < 10
            or sum(char.isalpha() for char in target) / len(target) < 0.5
        ):
            rejected["low_information"] += 1
            continue
        if normalize_text(target, NORMALIZERS) not in normalized[page_id]:
            rejected["not_in_page_reference"] += 1
            continue
        grouped[page_id].append(
            {
                "page_id": page_id,
                "region_index": item["source_region_index"],
                "image": str(page["image"].relative_to(dataset.resolve())),
                "image_sha256": page["image_sha256"],
                "crop_xyxy": box,
                "target": target,
                "target_sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
            }
        )
    rng = random.Random(20261007)
    selected = []
    for page_id in sorted(grouped):
        rows = sorted(grouped[page_id], key=lambda row: row["region_index"])
        rng.shuffle(rows)
        selected.extend(rows[:8])
    summary = {
        "schema_version": 1,
        "source_candidates_sha256": file_sha256(candidates),
        "dataset_manifest_sha256": file_sha256(dataset / "manifest.json"),
        "split_sha256": file_sha256(split),
        "candidate_count": len(seen),
        "selected_count": len(selected),
        "source_pages": len(grouped),
        "max_per_page": 8,
        "rejected": dict(rejected),
        "selection_seed": 20261007,
        "restriction": "train-only exact source boxes; no test-time GT boxes",
    }
    return selected, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    rows, summary = build(dataset, args.split, args.candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with args.output.with_suffix(".summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

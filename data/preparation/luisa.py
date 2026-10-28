"""Audit all LUISA lines and seal experiment four's complete mixed curriculum."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import re
import sqlite3
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from data.integrity import file_sha256
from data.integrity import closed_json
from data.preparation.contracts import verify_inputs
from data.preparation.synthetic_pages import CropStore, layout, render
from data.annotations import all_regions

EXTENSION = "berrutti-luisa-synthetic-v1"


def canonical(text):
    return re.sub(r"\W+", "", text.casefold())


def visual_key(image):
    thumbnail = image.convert("L").resize((128, 32), Image.Resampling.LANCZOS)
    return hashlib.sha256(thumbnail.tobytes()).hexdigest()


def overlap_reasons(line, text_lines, text_pages, pixels):
    text = canonical(line["reference_text"])
    reasons = []
    if len(text) >= 12 and text in text_lines:
        reasons.append("heldout_line_text")
    if len(text) >= 24 and any(text in page for page in text_pages):
        reasons.append("heldout_page_text")
    if line["visual_key"] in pixels:
        reasons.append("heldout_resized_pixels")
    return reasons


def choose_lines(lines, selection):
    if selection not in {"include-all", "exclude-known-overlap"}:
        raise ValueError("An explicit LUISA inclusion policy is required")
    chosen, excluded = [], []
    for line in lines:
        reasons = list(line["overlap_reasons"])
        if line["source_split"] == "test":
            reasons.append("published_luisa_test")
        if selection == "exclude-known-overlap" and reasons:
            excluded.append({"line_id": line["line_id"], "reasons": reasons})
        else:
            chosen.append(line)
    return chosen, excluded


def distribute(rows, count, seed):
    ordered = sorted(
        rows, key=lambda row: row["line_id"] if "line_id" in row else row["page_id"]
    )
    random.Random(seed).shuffle(ordered)
    buckets = [[] for _ in range(count)]
    for index, row in enumerate(ordered):
        buckets[index % count].append(row)
    assert sum(map(len, buckets)) == len(rows)
    return buckets


def compose_epoch(
    pool, sizes, seed, epoch, output, store, workers, progress, *, check_identity=False
):
    """Plan serially with the original RNG; render independent pages concurrently."""
    rng = random.Random(seed + 10000 * epoch)
    ordered = pool.copy()
    rng.shuffle(ordered)
    plans, cursor = [], 0
    while cursor < len(ordered):
        index = len(plans)
        size = sizes[rng.randrange(len(sizes))]
        page_seed, columns = seed + epoch * 100000 + index, 1 + index % 2
        count = len(layout(ordered[cursor : cursor + 48], size, page_seed, columns))
        if not count:
            raise ValueError("A synthetic page could not retain its source lines")
        plans.append(
            (index, size, page_seed, columns, ordered[cursor : cursor + count])
        )
        cursor += count
    progress("synthetic_epoch_planned", epoch=epoch, total=len(plans))

    def materialize(plan):
        index, size, page_seed, columns, lines = plan
        page_id = f"synthetic-e{epoch}-{index:05d}"
        item = render(
            lines,
            size,
            page_seed,
            columns,
            output / "images" / f"{page_id}.png",
            crop_loader=store.load,
        )
        item.update(
            page_id=page_id,
            source="synthetic",
            image_sha256=file_sha256(Path(item["image"])),
        )
        if check_identity and index in (0, 1):
            reference = render(
                lines,
                size,
                page_seed,
                columns,
                output / "identity" / f"original-{page_id}.png",
            )
            if file_sha256(Path(reference["image"])) != item["image_sha256"] or any(
                reference[key] != item[key]
                for key in ("sources", "reference_text", "size", "seed", "columns")
            ):
                raise ValueError("Cached rendering differs from the original sources")
            closed_json(
                output / "identity" / f"{page_id}.json",
                {
                    "success": True,
                    "columns": columns,
                    "png_bytes_identical": True,
                    "references_and_provenance_identical": True,
                    "source_families": dict(Counter(line["source"] for line in lines)),
                    "image_sha256": item["image_sha256"],
                },
            )
        # Every completed page has a closed receipt, even if preparation is cut.
        closed_json(output / "pages" / f"{page_id}.json", item)
        return item

    synthetic = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for item in executor.map(materialize, plans):
            synthetic.append(item)
            if len(synthetic) % 10 == 0 or len(synthetic) == len(plans):
                progress(
                    "synthetic_rendering",
                    epoch=epoch,
                    completed=len(synthetic),
                    total=len(plans),
                )
    return synthetic


def build(config, output, selection):
    output.mkdir(parents=True, exist_ok=False)
    tick = time.monotonic()

    def progress(phase, **counts):
        event = {
            "phase": phase,
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.monotonic() - tick,
            **counts,
        }
        with (output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        temporary = output / ".progress.json.tmp"
        temporary.write_text(json.dumps(event) + "\n")
        temporary.replace(output / "progress.json")
        print(json.dumps(event), flush=True)

    progress("verifying_original_inputs")
    cohort, pages = verify_inputs(config)
    workers = config.get("preparation_workers", 1)
    if workers not in (1, 2, 3, 4):
        raise ValueError("Preparation requires one to four allocated CPU workers")
    with CropStore(output / "source-crops.sqlite") as store:
        return build_corpus(
            config, output, selection, cohort, pages, store, workers, progress
        )


def build_corpus(config, output, selection, cohort, pages, store, workers, progress):
    luisa_root = Path(config["luisa_root"]).resolve()
    records = luisa_root / "records.jsonl"
    annotations = json.loads(Path(config["annotations_path"]).read_text())[
        "_via_img_metadata"
    ]
    by_page = {
        Path(row["filename"]).stem: row["regions"] for row in annotations.values()
    }
    held_lines, held_pages, held_pixels = set(), [], set()
    for page in pages["validation"] + pages["test"]:
        held_pages.append(canonical(page["reference_text"]))
        with Image.open(page["image"]) as image:
            for region in all_regions(by_page[page["page_id"]], image.size):
                held_lines.add(canonical(region["target"]))
                held_pixels.add(visual_key(image.crop(region["crop_xyxy"])))
    progress("auditing_luisa", completed=0, total=23774)
    lines = []
    for encoded in records.read_text().splitlines():
        row = json.loads(encoded)
        image_path = (luisa_root / row["image"]).resolve()
        if not image_path.is_relative_to(luisa_root):
            raise ValueError("LUISA source image changed or escaped its release")
        encoded_image = image_path.read_bytes()
        if hashlib.sha256(encoded_image).hexdigest() != row["image_sha256"]:
            raise ValueError("LUISA source image changed or escaped its release")
        if not row["reference_text"].strip():
            raise ValueError("LUISA human transcript is empty")
        with Image.open(io.BytesIO(encoded_image)) as image:
            if image.size != (row["width"], row["height"]):
                raise ValueError("LUISA indexed dimensions changed")
            visual = visual_key(image)
        line = {
            "line_id": "luisa-" + row["id"],
            "page_id": "luisa-" + row["id"],
            "source": "luisa",
            "source_split": row["source_split"],
            "image": str(image_path),
            "image_sha256": row["image_sha256"],
            "size": [row["width"], row["height"]],
            "reference_text": row["reference_text"],
            "max_pixels": config["line_max_pixels"],
            "visual_key": visual,
        }
        line["overlap_reasons"] = overlap_reasons(
            line, held_lines, held_pages, held_pixels
        )
        store.add(line, encoded_image)
        lines.append(line)
        if len(lines) % 1000 == 0 or len(lines) == 23774:
            progress("auditing_luisa", completed=len(lines), total=23774)
    if len(lines) != 23774 or len({line["line_id"] for line in lines}) != 23774:
        raise ValueError("All 23,774 unique original LUISA lines must be audited")
    if Counter(line["source_split"] for line in lines) != {
        "train": 21611,
        "test": 2163,
    }:
        raise ValueError("LUISA's published partitions changed")
    chosen, excluded = choose_lines(lines, selection)
    if not chosen:
        raise ValueError("No LUISA lines remain")
    audit = {
        "selection": selection,
        "luisa_total_audited": len(lines),
        "luisa_selected": len(chosen),
        "luisa_excluded": len(excluded),
        "known_overlap_lines": sum(bool(line["overlap_reasons"]) for line in lines),
        "known_overlap_in_training": sum(
            bool(line["overlap_reasons"]) for line in chosen
        ),
        "published_test_in_training": sum(
            line["source_split"] == "test" for line in chosen
        ),
        "source_document_independence_certified": False,
        "limitations": "No source-page/document mapping in LUISA; line-level exact-text/substring/resized-pixel screening cannot certify document independence.",
        "validation_interpretation": "exploratory; known test contamination"
        if any(line["overlap_reasons"] for line in chosen)
        else "exploratory; document independence unproven",
        "excluded": excluded,
        "overlaps": [
            {"line_id": line["line_id"], "reasons": line["overlap_reasons"]}
            for line in lines
            if line["overlap_reasons"]
        ],
    }
    closed_json(output / "audit.json", audit)
    progress("luisa_audit_completed", audited=len(lines), selected=len(chosen))
    real = {page["page_id"]: page for page in pages["train"]}
    berrutti_lines = []
    for page in cohort["pages"]:
        for region in page["regions"]:
            box = region["crop_xyxy"]
            berrutti_lines.append(
                {
                    "line_id": f"berrutti-{page['page_id']}-{region['region_index']}",
                    "page_id": f"berrutti-line-{page['page_id']}-{region['region_index']}",
                    "parent_page_id": page["page_id"],
                    "region_index": region["region_index"],
                    "source": "berrutti_line",
                    "image": str(real[page["page_id"]]["image"]),
                    "image_sha256": page["image_sha256"],
                    "crop_xyxy": box,
                    "size": [box[2] - box[0], box[3] - box[1]],
                    "reference_text": region["target"],
                    "max_pixels": config["line_max_pixels"],
                }
            )
    assert len(berrutti_lines) == 5212
    with sqlite3.connect(f"file:{config['line_cache']}?mode=ro", uri=True) as cache:
        for line in berrutti_lines:
            row = cache.execute(
                "SELECT target,png,rgb_sha256 FROM crops WHERE page_id=? AND region_index=?",
                (line["parent_page_id"], line["region_index"]),
            ).fetchone()
            if row is None or row[0] != line["reference_text"]:
                raise ValueError("Missing Berrutti crop or changed human reference")
            store.add(line, row[1], expected_rgb=row[2])
    store.seal()
    progress("lossless_crop_store_sealed", luisa_lines=len(lines), berrutti_lines=5212)
    pool = berrutti_lines + chosen
    sizes = [tuple(page["source_size"]) for page in cohort["pages"]]
    epochs, coverage = [], []
    for epoch in (1, 2, 3):
        synthetic = compose_epoch(
            pool,
            sizes,
            config["seed"],
            epoch,
            output,
            store,
            workers,
            progress,
            check_identity=epoch == 1,
        )
        seen = [source["line_id"] for item in synthetic for source in item["sources"]]
        if Counter(seen) != Counter(line["line_id"] for line in pool):
            raise ValueError("Synthetic pages lost or repeated source lines")
        assert all(tuple(item["size"]) in sizes for item in synthetic)
        luisa_buckets = distribute(chosen, 125, config["seed"] + epoch)
        synthetic_buckets = distribute(synthetic, 125, config["seed"] + epoch + 100)
        units = [
            {
                "page_id": page["page_id"],
                "luisa": luisa_buckets[index],
                "synthetic": synthetic_buckets[index],
            }
            for index, page in enumerate(
                sorted(cohort["pages"], key=lambda p: p["page_id"])
            )
        ]
        epochs.append(
            {
                "epoch": epoch,
                "stage": "mixed",
                "units": units,
                "samples": 125 + 5212 + len(chosen) + len(synthetic),
                "synthetic_pages": len(synthetic),
            }
        )
        coverage.append(
            {
                "epoch": epoch,
                "synthetic_pages": len(synthetic),
                "source_lines": len(seen),
                "every_source_line_exactly_once": True,
                "columns": dict(Counter(item["columns"] for item in synthetic)),
            }
        )
    epochs.append(
        {
            "epoch": 4,
            "stage": "berrutti_consolidation",
            "units": [
                {"page_id": page["page_id"], "luisa": [], "synthetic": []}
                for page in sorted(cohort["pages"], key=lambda p: p["page_id"])
            ],
            "samples": 5337,
            "synthetic_pages": 0,
        }
    )
    manifest = {
        "protocol": EXTENSION,
        "seed": config["seed"],
        "audit": audit,
        "records_sha256": file_sha256(records),
        "source_cohort_sha256": config["cohort_sha256"],
        "epochs": epochs,
        "synthetic_coverage": coverage,
        "render_identity_pages_verified": 2,
        "render_identity_columns": [1, 2],
        "optimizer_steps_per_epoch": 16,
        "total_optimizer_steps": 64,
        "family_weights_mixed": {
            "page": 0.25,
            "berrutti_lines": 0.25,
            "luisa": 0.25,
            "synthetic": 0.25,
        },
        "family_weights_consolidation": {"page": 0.5, "berrutti_lines": 0.5},
    }
    closed_json(output / "manifest.json", manifest)
    receipt = {
        "success": True,
        "protocol": EXTENSION,
        "manifest_sha256": file_sha256(output / "manifest.json"),
        "source_images_verified": True,
        "luisa_lines": len(chosen),
        "berrutti_pages": 125,
        "berrutti_lines": 5212,
        "synthetic_coverage": coverage,
        "render_identity_pages_verified": 2,
        "render_identity_columns": [1, 2],
        "source_document_independence_certified": False,
        "known_overlap_in_training": audit["known_overlap_in_training"],
    }
    closed_json(output / "completed.json", receipt)
    progress(
        "preparation_completed",
        success=True,
        synthetic_pages=sum(e["synthetic_pages"] for e in coverage),
    )
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--selection", choices=("include-all", "exclude-known-overlap"), required=True
    )
    args = parser.parse_args()
    os.umask(0o077)
    print(
        json.dumps(
            build(json.loads(args.config.read_text()), args.output, args.selection)
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

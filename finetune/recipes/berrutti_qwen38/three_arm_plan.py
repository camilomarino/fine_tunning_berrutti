"""Seal the requested three-epoch, full-125-page design without launching training."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from PIL import Image

from data.integrity import file_sha256
from data.berrutti import load_pages
from data.annotations import all_regions, aligned_regions, canonical_text  # noqa: F401
from finetune.recipes.berrutti_qwen38.generation_guard import (
    select_complete_validation_epoch,
)

from data.preparation.contracts import PROTOCOL  # noqa: F401

ARMS = ("pages", "lines", "combined")


def evaluation_schedule() -> list[dict]:
    """One initial baseline, nine full validations and three selected-model tests."""
    return [
        *[
            {"phase": "baseline", "arm": "shared", "split": split, "pages": 25}
            for split in ("validation", "test")
        ],
        *[
            {
                "phase": "after_epoch",
                "arm": arm,
                "epoch": epoch,
                "split": "validation",
                "pages": 25,
            }
            for arm in ARMS
            for epoch in (1, 2, 3)
        ],
        *[
            {"phase": "selected_checkpoint", "arm": arm, "split": "test", "pages": 25}
            for arm in ARMS
        ],
    ]


def select_test_epoch(evaluations: list[dict]) -> dict | None:
    """Wait for all epochs; ignore baseline/test CER and incomplete validation."""
    by_epoch = {}
    for row in evaluations:
        if (
            row["split"] != "validation"
            or row["epoch"] not in (1, 2, 3)
            or row.get("stage") == "baseline"
        ):
            continue
        if row["epoch"] in by_epoch:
            raise ValueError("Duplicate validation record for one epoch")
        by_epoch[row["epoch"]] = row
    if set(by_epoch) != {1, 2, 3}:
        return None
    eligible = [
        {"epoch": epoch, "validation": row}
        for epoch, row in by_epoch.items()
        if row.get("pages") == 25
        and row.get("all_pages_completed") is True
        and row.get("failed_pages") == 0
        and isinstance(row.get("cer"), (int, float))
        and row["cer"] >= 0
    ]
    selected = select_complete_validation_epoch(eligible)
    return None if selected is None else selected["validation"]


def make_plan(summary: dict) -> dict:
    """Fix exactly three arms/epochs and complete, identical Berrutti held-out sets."""
    if summary["train_pages"] != 125 or summary["train_lines"] <= 0:
        raise ValueError("The requested comparison requires all 125 training pages")
    if summary["validation_pages"] != 25 or summary["test_pages"] != 25:
        raise ValueError("Keep the complete 25-page validation and 25-page test")
    steps = math.ceil(125 / 8)
    arms = {}
    for arm in ARMS:
        full_pages = 125 if arm != "lines" else 0
        berrutti_lines = summary["train_lines"] if arm != "pages" else 0
        weights = (
            {"full_page": 1.0, "berrutti_lines_together": 0.0, "luisa_group": 0.0}
            if arm == "pages"
            else {
                "full_page": 0.0 if arm == "lines" else 0.5,
                "berrutti_lines_together": 1.0 if arm == "lines" else 0.5,
                "luisa_group": 0.0,
            }
        )
        arms[arm] = {
            "state": "PLANNED",
            "epochs": [
                {
                    "epoch": epoch,
                    "full_page_images": full_pages,
                    "berrutti_line_images": berrutti_lines,
                    "luisa_line_images": 0,
                    "image_presentations": full_pages + berrutti_lines,
                    "optimizer_steps": steps,
                    "all_examples_once": True,
                }
                for epoch in (1, 2, 3)
            ],
            "new_optimizer_steps": 3 * steps,
            "fresh_initialization": True,
            "inherits_trained_checkpoint": False,
            "weight_per_training_unit": weights,
        }
    return {
        "protocol": PROTOCOL,
        "status": "DESIGN_SEALED_NOT_LAUNCHED",
        "source_choice": "Berrutti only; explicitly confirmed by user",
        "arms": arms,
        "berrutti": summary,
        "optional_luisa_experiment": {
            "state": "PROPOSED_NOT_LAUNCHED",
            "primary_experiment": False,
            "source_image_type": "single annotated line crop",
            "published_test_lines_reserved": 2163,
            "requires_exclusion_audit_before_training": True,
            "source_document_independence_certified": False,
            "reference_policy": "original human transcripts; no Tesseract labels",
            "strategy": "extend the best Berrutti strategy after complete validation",
            "variants": [
                "add filtered LUISA single-line images",
                "add synthetic page images composed from filtered LUISA line images",
            ],
            "page_count": None,
            "epoch_count": None,
            "uses_berrutti_heldout_images": False,
        },
        "training": {
            "seed": 20261007,
            "epochs_per_arm": 3,
            "units_per_epoch": 125,
            "accumulation_units": 8,
            "steps_per_epoch": steps,
            "external_groups_per_epoch": 0,
            "external_groups_are_source_pages": False,
            "same_learning_rate_schedule_and_initial_weights": True,
            "combined_order": "page, ALL its Berrutti lines; repeat for every page",
            "line_order": "ALL Berrutti lines of unit; repeat for every page",
            "line_loss": "token-weighted mean within each source group",
            "all_lines_per_epoch": summary["train_lines"],
            "equal_gpu_time": False,
        },
        "evaluation": {
            "schedule_version": "shared-baseline-selected-test-v1",
            "dataset": "Berrutti",
            "image_type": "full page",
            "validation_pages": 25,
            "test_pages": 25,
            "baseline": "before epoch 1; zero optimizer updates",
            "baseline_shared_across_arms": True,
            "validation_after_each_epoch": True,
            "test_after_each_epoch": False,
            "test_policy": "once per arm, using its selected checkpoint after epoch 3",
            "selection": "minimum complete validation CER among epochs 1, 2, 3",
            "test_selects_checkpoint": False,
            "incomplete_cer_is_provisional": True,
            "generation_attempts_per_page_max": 2,
            "schedule": evaluation_schedule(),
            "scheduled_page_evaluations": sum(
                task["pages"] for task in evaluation_schedule()
            ),
            "page_evaluations_with_test_after_each_epoch": 500,
            "counts_exclude_retries": True,
            "baseline_reuse_requires": [
                "identical initial weights and initialized adapter",
                "identical held-out images, references and processor",
                "identical prompt, generation and scoring policy",
            ],
        },
        "execution": {
            "cluster": "Ruche",
            "gpu_count": 2,
            "gpu_type": "A100",
            "requires_new_small_gpu_assay_before_full_training": True,
            "long_epochs_require_checkpointed_slurm_continuations": True,
            "performance_profile_shared_across_arms": True,
            "performance_profile_state": "NOT_MEASURED",
        },
        "interpretation": {
            "hypothesis": "combined improves full-page validation CER",
            "improvement_guaranteed": False,
            "page_line_annotations_are_not_identical_on_all_pages": summary[
                "page_line_annotation_disagreements"
            ]
            > 0,
            "compares_complete_strategies_including_external_data": False,
        },
    }


def build(dataset: Path, split: Path, annotations: Path) -> dict:
    pages = load_pages(dataset, split)
    source = json.loads(annotations.read_text())["_via_img_metadata"]
    by_page = {}
    for value in source.values():
        page_id = Path(value["filename"]).stem
        if page_id in by_page:
            raise ValueError("Duplicate annotated source page")
        by_page[page_id] = value["regions"]
    rows = []
    for page in sorted(pages["train"], key=lambda row: row["page_id"]):
        regions = by_page[page["page_id"]]
        with Image.open(page["image"]) as image:
            size = image.size
        lines = all_regions(regions, size)
        rows.append(
            {
                "page_id": page["page_id"],
                "image_sha256": page["image_sha256"],
                "reference_sha256": page["reference_sha256"],
                "source_size": list(size),
                "page_and_lines_concordant": aligned_regions(
                    page["reference_text"], regions, size
                )
                is not None,
                "regions": lines,
            }
        )
    summary = {
        "train_pages": len(rows),
        "train_lines": sum(len(row["regions"]) for row in rows),
        "validation_pages": len(pages["validation"]),
        "test_pages": len(pages["test"]),
        "page_line_annotation_disagreements": sum(
            not row["page_and_lines_concordant"] for row in rows
        ),
        "excluded_training_pages": 0,
        "labels_modified": False,
    }
    return {
        "protocol": PROTOCOL,
        "dataset_manifest_sha256": file_sha256(dataset / "manifest.json"),
        "split_sha256": file_sha256(split),
        "annotations_sha256": file_sha256(annotations),
        "pages": rows,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "split", "annotations", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    cohort = build(args.dataset.resolve(), args.split, args.annotations)
    plan = make_plan(cohort["summary"])
    args.output.mkdir(parents=True, exist_ok=False)
    for name, payload in (("cohort.json", cohort), ("plan.json", plan)):
        with (args.output / name).open("x") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    print(json.dumps({"protocol": PROTOCOL, **cohort["summary"]}))


if __name__ == "__main__":
    main()

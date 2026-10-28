"""CPU checks for the original sealed Berrutti cohort and input hashes."""

import json
import sqlite3
from pathlib import Path

from data.berrutti import load_pages
from data.integrity import file_sha256

PROTOCOL = "three-arms-125-v1"
RUNTIME_PROFILE = "bf16-sdpa-fp32-projection-release-outputs-v2"


def verify_inputs(config, *, hash_file=file_sha256, page_loader=load_pages):
    """Validate the frozen operation before importing Torch or reserving GPUs."""
    if config["protocol"] != PROTOCOL or config["runtime_profile"] != RUNTIME_PROFILE:
        raise ValueError("Unexpected experiment protocol or unapproved runtime")
    code_root = Path(config["code_root"])
    for name, digest in config["source_hashes"].items():
        if hash_file(code_root / name) != digest:
            raise ValueError("Frozen runner source changed")
    for key in ("cohort", "split"):
        if hash_file(Path(config[f"{key}_path"])) != config[f"{key}_sha256"]:
            raise ValueError("Sealed experiment inputs changed")
    cohort = json.loads(Path(config["cohort_path"]).read_text())
    pages = page_loader(Path(config["dataset_root"]), Path(config["split_path"]))
    train = {p["page_id"]: p for p in pages["train"]}
    if (
        {split: len(rows) for split, rows in pages.items()}
        != {"train": 125, "validation": 25, "test": 25}
        or set(train) != {p["page_id"] for p in cohort["pages"]}
        or cohort["summary"]["train_lines"] != 5212
    ):
        raise ValueError("The complete 125-page/5212-line cohort is required")
    for page in cohort["pages"]:
        if any(
            page[key] != train[page["page_id"]][key]
            for key in ("image_sha256", "reference_sha256")
        ):
            raise ValueError("Training cohort differs from source images/references")
    with sqlite3.connect(f"file:{config['line_cache']}?mode=ro", uri=True) as cache:
        receipt = json.loads(
            cache.execute(
                "SELECT value FROM metadata WHERE key='closed_receipt'"
            ).fetchone()[0]
        )
    if (
        receipt.get("complete") is not True
        or receipt["cohort_sha256"] != config["cohort_sha256"]
        or receipt["line_images"] != 5212
    ):
        raise ValueError("Crop cache does not match the complete sealed cohort")
    return cohort, pages

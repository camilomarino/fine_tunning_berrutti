"""Build a closed static viewer from sealed data and existing OCR receipts.

This command performs no inference, changes no source data, and never exports
runtime configuration or raw operational events. Licensed images/transcripts
belong in a private output directory, outside version control.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import shutil
import sqlite3
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from data.integrity import file_sha256
from data.berrutti import load_pages
from evaluation.scoring import normalize_text

METHODS = {
    "pages": "Sólo páginas",
    "lines": "Sólo líneas",
    "combined": "Combinación intercalada",
    "experiment4": "Berrutti + LUISA + páginas sintéticas",
}
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
EVALUATION = re.compile(r"^epoch-(\d+)-(validation|test)$")
STATES = {
    "PLANNED",
    "PENDING",
    "RUNNING",
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "UNKNOWN",
}
COUNTS = ("char_errors", "char_count", "word_errors", "word_count")


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def identifier(value: str) -> str:
    if (
        not isinstance(value, str)
        or not IDENTIFIER.fullmatch(value)
        or value in (".", "..")
    ):
        raise ValueError("Unsafe public identifier")
    return value


def jailed_file(root: Path, relative: str | Path) -> Path:
    """Only existing regular inputs inside the declared source directory."""
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Input escapes its declared directory")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError("Missing input or symlink escaping its declared directory")
    return resolved


def finite(value, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"Invalid finite nonnegative {label}")
    return value


def count(value, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"Invalid nonnegative {label}")
    return value


def public_summary(row: dict) -> dict:
    """An explicit allow-list; scheduler success cannot close an OCR result."""
    pages = count(row["pages"], "pages")
    failed = count(row["failed_pages"], "failed_pages")
    retried = count(row["retried_pages"], "retried_pages")
    if not 0 <= failed <= pages or not 0 <= retried <= pages or pages > 25:
        raise ValueError("Invalid evaluation coverage")
    complete = row.get("all_pages_completed") is True
    if complete != (failed == 0):
        raise ValueError("Evaluation completion contradicts failed pages")
    return {
        "cer": finite(row["cer"], "CER"),
        "wer": finite(row["wer"], "WER") if "wer" in row else None,
        "pages": pages,
        "pages_completed": pages - failed,
        "failed_pages": failed,
        "retried_pages": retried,
        "all_pages_completed": complete and pages == 25,
        "provisional": not complete or pages != 25,
    }


class Bundle:
    """Content-addressed immutable payloads with a publication manifest."""

    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=False)
        self.files: dict[str, dict] = {}

    def payload(self, data: bytes, kind: str, suffix: str) -> str:
        digest = hashlib.sha256(data).hexdigest()
        relative = f"{kind}/{digest}.{suffix}"
        if relative not in self.files:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(data)
            self.files[relative] = {"sha256": digest, "size_bytes": len(data)}
        return relative

    def fragment(self, value: dict) -> str:
        return self.payload(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
            ).encode(),
            "data",
            "json",
        )

    def image(self, source: Path, expected_hash: str) -> str:
        suffix = source.suffix.lower().lstrip(".")
        if suffix not in {"png", "jpg", "jpeg", "gif", "webp", "tif", "tiff"}:
            raise ValueError("Unsupported image format")
        if file_sha256(source) != expected_hash:
            raise ValueError("Source image hash differs from sealed input")
        relative = f"images/{expected_hash}.{suffix}"
        if relative not in self.files:
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst)
            if file_sha256(target) != expected_hash:
                raise ValueError("Copied image changed during export")
            self.files[relative] = {
                "sha256": expected_hash,
                "size_bytes": target.stat().st_size,
            }
        return relative

    def named_json(self, relative: str, value: dict) -> None:
        data = (
            json.dumps(
                value, ensure_ascii=False, sort_keys=True, allow_nan=False
            ).encode()
            + b"\n"
        )
        with (self.root / relative).open("xb") as stream:
            stream.write(data)
        self.files[relative] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        }

    def finish(self, assets: Path) -> None:
        for name in ("index.html", "app.js", "style.css"):
            source = jailed_file(assets, name)
            target = self.root / name
            with source.open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst)
            self.files[name] = {
                "sha256": file_sha256(target),
                "size_bytes": target.stat().st_size,
            }
        # manifest itself is deliberately excluded from its closed file allow-list.
        manifest = {
            "schema_version": 1,
            "artifact_type": "berrutti-benchmark-viewer",
            "subtype": "qwen-finetuning-results",
            "normalization_policy": "ascii-text-v1",
            "files": self.files,
        }
        with (self.root / "manifest.json").open("x") as stream:
            json.dump(
                manifest, stream, ensure_ascii=False, allow_nan=False, sort_keys=True
            )
            stream.write("\n")


def text_fields(reference: str, prediction: str | None = None) -> dict:
    value = {
        "reference": reference,
        "reference_normalized": normalize_text(reference),
        "normalization_policy": "ascii-text-v1",
    }
    if prediction is not None:
        value.update(
            prediction=prediction, prediction_normalized=normalize_text(prediction)
        )
    return value


def validate_cohort(cohort: dict, pages: dict) -> dict:
    train = {row["page_id"]: row for row in pages["train"]}
    rows = cohort["pages"]
    if (
        {name: len(value) for name, value in pages.items()}
        != {"train": 125, "validation": 25, "test": 25}
        or len(rows) != 125
        or {row["page_id"] for row in rows} != set(train)
    ):
        raise ValueError("Full 125/25/25 Berrutti split and unique cohort required")
    if (
        sum(len(row["regions"]) for row in rows) != 5212
        or cohort["summary"]["train_lines"] != 5212
    ):
        raise ValueError("All 5212 Berrutti training lines are required")
    result = {}
    for row in rows:
        page = train[row["page_id"]]
        if any(row[key] != page[key] for key in ("image_sha256", "reference_sha256")):
            raise ValueError("Training cohort differs from sealed source")
        if len({region["region_index"] for region in row["regions"]}) != len(
            row["regions"]
        ):
            raise ValueError("Duplicate training line")
        for region in row["regions"]:
            if (
                hashlib.sha256(region["target"].encode()).hexdigest()
                != region["target_sha256"]
            ):
                raise ValueError("Training line reference changed")
        result[row["page_id"]] = row
    return result


def export_pages(bundle: Bundle, pages: dict, cohort: dict) -> tuple[list, dict]:
    catalog, line_sources = [], {}
    for split, rows in pages.items():
        for row in sorted(rows, key=lambda item: item["page_id"]):
            page_id = identifier(row["page_id"])
            image = bundle.image(row["image"], row["image_sha256"])
            lines = []
            if split == "train":
                with Image.open(row["image"]) as original:
                    rgb = original.convert("RGB")
                if tuple(cohort[page_id]["source_size"]) != rgb.size:
                    raise ValueError("Training page dimensions changed")
                try:
                    for region in cohort[page_id]["regions"]:
                        x, y, right, bottom = region["crop_xyxy"]
                        if any(
                            type(item) is not int for item in (x, y, right, bottom)
                        ) or not (
                            0 <= x < right <= rgb.width
                            and 0 <= y < bottom <= rgb.height
                        ):
                            raise ValueError(
                                "Training crop lies outside original pixels"
                            )
                        with rgb.crop((x, y, right, bottom)) as crop:
                            buffer = io.BytesIO()
                            crop.save(buffer, "PNG", compress_level=1)
                            png = buffer.getvalue()
                            with Image.open(io.BytesIO(png)) as restored:
                                if (
                                    restored.mode != "RGB"
                                    or restored.tobytes() != crop.tobytes()
                                ):
                                    raise ValueError("Training crop pixels changed")
                        line_id = f"berrutti-{page_id}-{region['region_index']}"
                        line = {
                            "id": line_id,
                            "image": bundle.payload(png, "images", "png"),
                            "box": [x, y, right - x, bottom - y],
                            **text_fields(region["target"]),
                        }
                        lines.append(line)
                        line_sources[line_id] = {
                            **line,
                            "source": "berrutti_line",
                            "image_sha256": row["image_sha256"],
                        }
                finally:
                    rgb.close()
            detail = bundle.fragment(
                {
                    "id": page_id,
                    "split": split,
                    "image": image,
                    "lines": lines,
                    **text_fields(row["reference_text"]),
                }
            )
            catalog.append(
                {
                    "id": page_id,
                    "split": split,
                    "image": image,
                    "detail": detail,
                    "line_count": len(lines),
                }
            )
    return catalog, line_sources


def export_evaluation(
    bundle: Bundle, root: Path, method: str, epoch: int, split: str, pages: list[dict]
) -> dict:
    closed = read_json(jailed_file(root, "closed.json"))
    if closed.get("epoch") != epoch or closed.get("split") != split:
        raise ValueError("Evaluation receipt belongs to another epoch/split")
    summary = public_summary(closed)
    if summary["pages"] != len(pages):
        raise ValueError("Closed evaluation must cover the whole held-out split")
    expected = {row["page_id"] for row in pages}
    page_root = root / "pages"
    observed = {path.name for path in page_root.iterdir() if path.is_dir()}
    if observed != expected:
        raise ValueError("Evaluation page directories do not cover the split exactly")
    documents, totals = [], Counter()
    keys = set()
    for page in sorted(pages, key=lambda item: item["page_id"]):
        page_id = identifier(page["page_id"])
        source = page_root / page_id
        row = read_json(jailed_file(root, f"pages/{page_id}/closed.json"))
        if row.get("page_id") != page_id or row.get("status") not in {
            "complete",
            "incomplete",
        }:
            raise ValueError("Invalid page evaluation identity/status")
        keys.add(row.get("evaluation_key"))
        attempts_count = row.get("attempts")
        if type(attempts_count) is not int or not 1 <= attempts_count <= 2:
            raise ValueError("At most two attempts per page are allowed")
        attempts = [
            read_json(jailed_file(root, f"pages/{page_id}/attempt-{n}.json"))
            for n in range(1, attempts_count + 1)
        ]
        persisted_attempts = {
            path.name
            for path in source.iterdir()
            if re.fullmatch(r"attempt-\d+\.json", path.name)
        }
        if persisted_attempts != {
            f"attempt-{number}.json" for number in range(1, attempts_count + 1)
        }:
            raise ValueError("Unexpected extra generation attempt")
        for index, attempt in enumerate(attempts, 1):
            if (
                attempt.get("attempt") != index
                or not isinstance(attempt.get("prediction"), str)
                or attempt.get("stop_reason")
                not in {
                    "eos",
                    "repetition",
                    "time_limit",
                    "max_new_tokens",
                    "token_limit",
                    "interrupted",
                    "unexpected_stop",
                    "unknown",
                }
            ):
                raise ValueError("Invalid persisted generation attempt")
            finite(attempt["seconds"], "attempt seconds")
            if index < len(attempts) and attempt["stop_reason"] == "eos":
                raise ValueError("A successful generation was retried")
        complete = row["status"] == "complete"
        if complete != (attempts[-1]["stop_reason"] == "eos"):
            raise ValueError("Page completion contradicts generation outcome")
        for key in COUNTS:
            totals[key] += count(row[key], key)
        if row["char_count"] <= 0 or row["word_count"] <= 0:
            raise ValueError("Invalid empty scored reference")
        # The scoring runner chooses the LAST attempt, including incomplete OCR.
        prediction = attempts[-1]["prediction"]
        if row["char_count"] != len(normalize_text(page["reference_text"])) or row[
            "word_count"
        ] != len(normalize_text(page["reference_text"]).split()):
            raise ValueError("Scored reference counts differ from sealed transcript")
        totals["failed_pages"] += not complete
        totals["retried_pages"] += len(attempts) == 2
        detail = {
            "page_id": page_id,
            **text_fields(page["reference_text"], prediction),
            "cer": row["char_errors"] / row["char_count"],
            "wer": row["word_errors"] / row["word_count"],
            "complete": complete,
            "provisional": not complete,
            "failed": not complete,
            "retried": len(attempts) == 2,
            "attempts": [
                {"stop_reason": a["stop_reason"], "seconds": a["seconds"]}
                for a in attempts
            ],
        }
        documents.append({"page_id": page_id, "detail": bundle.fragment(detail)})
    if len(keys) != 1 or None in keys or "" in keys:
        raise ValueError("Evaluation mixes checkpoints/policies")
    for key in COUNTS + ("failed_pages", "retried_pages"):
        if totals[key] != closed[key]:
            raise ValueError("Page receipts contradict aggregate evaluation counts")
    if not math.isclose(
        totals["char_errors"] / totals["char_count"],
        summary["cer"],
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise ValueError("Aggregate CER contradicts persisted counts")
    if summary["wer"] is not None and not math.isclose(
        totals["word_errors"] / totals["word_count"],
        summary["wer"],
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise ValueError("Aggregate WER contradicts persisted counts")
    return {
        "id": f"{method}-e{epoch}-{split}",
        "method": method,
        "stage": "baseline"
        if epoch == 0
        else "progress"
        if epoch > 4
        else "after_epoch",
        "epoch": epoch,
        "split": split,
        "summary": summary,
        "documents": documents,
    }


def export_evidence(bundle: Bundle, evidence: Path, pages: dict) -> list[dict]:
    evaluations = []
    for method, folder in (
        ("baseline", "shared_baseline"),
        *((name, name) for name in METHODS),
    ):
        source = evidence / folder / "evaluations"
        if not source.exists():
            continue
        if not source.resolve().is_relative_to(evidence.resolve()):
            raise ValueError("Evidence directory escapes its root")
        for directory in sorted(source.iterdir()):
            match = EVALUATION.fullmatch(directory.name)
            if (
                not match
                or not directory.is_dir()
                or not (directory / "closed.json").exists()
            ):
                continue  # In-flight partial evaluations are never advertised closed.
            epoch, split = int(match[1]), match[2]
            if (
                method == "baseline"
                and epoch != 0
                or method in {"pages", "lines", "combined"}
                and epoch not in (1, 2, 3)
            ):
                continue
            if not directory.resolve().is_relative_to(evidence.resolve()):
                raise ValueError("Evaluation escapes its evidence directory")
            export_method = (
                "experiment4_baseline"
                if method == "experiment4" and epoch == 0
                else method
            )
            evaluations.append(
                export_evaluation(
                    bundle, directory, export_method, epoch, split, pages[split]
                )
            )
    return evaluations


def training_history(evidence: Path, method: str) -> dict:
    """Display only observed finite scalar events; never claim unseen updates."""
    path = evidence / method / "events.jsonl"
    if not path.exists():
        return {"epochs": [], "updates": [], "updates_complete": False}
    path = jailed_file(evidence, f"{method}/events.jsonl")
    epochs, updates = {}, {}
    with path.open() as stream:
        for encoded in stream:
            event = json.loads(encoded)
            kind = event.get("event")
            if kind not in ("training_epoch", "optimizer_step"):
                continue
            epoch = count(event["epoch"], "epoch")
            if epoch not in range(1, 5 if method == "experiment4" else 4):
                raise ValueError("Training event belongs to an unexpected epoch")
            if kind == "training_epoch":
                public = {
                    "epoch": epoch,
                    "mean_loss": finite(event["mean_loss"], "mean_loss"),
                }
                for key in ("samples", "source_pages", "optimizer_steps"):
                    public[key] = count(event[key], key)
                if public["source_pages"] != 125 or public["optimizer_steps"] != 16:
                    raise ValueError(
                        "Training epoch does not cover the complete protocol"
                    )
                if epoch in epochs and epochs[epoch] != public:
                    raise ValueError("Contradictory training epoch events")
                epochs[epoch] = public
            else:
                step = count(event["optimizer_steps"], "optimizer_steps")
                if not (epoch - 1) * 16 < step <= epoch * 16:
                    raise ValueError("Training update falls outside its epoch")
                public = {
                    "epoch": epoch,
                    "step": step,
                    "loss": finite(event["loss"], "loss"),
                    "source_pages": count(event["page"], "source_pages"),
                }
                if step in updates and updates[step] != public:
                    raise ValueError("Contradictory training update events")
                updates[step] = public
    expected = 64 if method == "experiment4" else 48
    return {
        "epochs": [epochs[key] for key in sorted(epochs)],
        "updates": [updates[key] for key in sorted(updates)],
        "updates_complete": set(updates) == set(range(1, expected + 1)),
        "note": "La curva muestra únicamente actualizaciones presentes en los eventos exportados; puede omitir segmentos de recuperación anteriores.",
    }


def snapshot_method(snapshot: dict, method: str) -> dict:
    if method == "experiment4":
        return snapshot.get("additional_experiments", {}).get(method, {})
    return snapshot.get("primary_experiments", snapshot.get("runs", {})).get(method, {})


def method_status(
    method: str, snapshot: dict, evaluations: list[dict], evidence: Path
) -> dict:
    source = snapshot_method(snapshot, method)
    state = source.get("state", "PLANNED")
    if state not in STATES:
        state = "UNKNOWN"
    total = 4 if method == "experiment4" else 3
    own = [row for row in evaluations if row["method"] == method]
    finals = {
        row["epoch"]: row["summary"]
        for row in own
        if row["split"] == "validation" and 1 <= row["epoch"] <= total
    }
    tests = [row for row in own if row["split"] == "test"]
    if len(tests) > 1:
        raise ValueError("Test must be evaluated only once for the selected checkpoint")
    selection = None
    eligible = [
        (epoch, row) for epoch, row in finals.items() if row["all_pages_completed"]
    ]
    if len(finals) == total and eligible:
        epoch, row = min(eligible, key=lambda item: (item[1]["cer"], item[0]))
        selection = {"epoch": epoch, "validation_cer": row["cer"]}
    if tests and (selection is None or tests[0]["epoch"] != selection["epoch"]):
        raise ValueError(
            "Test differs from minimum complete final validation selection"
        )
    if tests and not tests[0]["summary"]["all_pages_completed"]:
        state = "INCOMPLETE_TEST"
    if method == "experiment4" and not own and not source:
        state = "PLANNED"
    completed_path = evidence / method / "completed.json"
    steps = None
    if completed_path.exists():
        closed = read_json(jailed_file(evidence, f"{method}/completed.json"))
        steps = count(closed["optimizer_steps"], "optimizer_steps")
        if (
            steps != total * 16
            or selection is None
            or closed.get("best_epoch") != selection["epoch"]
            or not math.isclose(
                finite(closed.get("best_validation_cer"), "selected CER"),
                selection["validation_cer"],
                rel_tol=1e-12,
            )
        ):
            raise ValueError(
                "Model completion contradicts full epochs or validation selection"
            )
    verification = {}
    audit_path = evidence / "result-verifications.json"
    if audit_path.exists():
        audit = read_json(jailed_file(evidence, "result-verifications.json")).get(
            method, {}
        )
        verification = {
            key: audit[key]
            for key in (
                "complete",
                "execution_verified",
                "training_complete",
                "validation_complete",
                "test_complete",
                "generation_attempts_exhausted",
                "committed_updates_contiguous_without_repetition",
            )
            if type(audit.get(key)) is bool
        }
    history = training_history(evidence, method)
    if steps is None and history["updates"]:
        steps = history["updates"][-1]["step"]
    return {
        "verification": verification,
        "id": method,
        "label": METHODS[method],
        "state": state,
        "epochs_completed": sum(row["all_pages_completed"] for row in finals.values()),
        "epochs_total": total,
        "optimizer_steps": steps,
        "selection": selection,
        "test": tests[0]["summary"] if tests else None,
        "exploratory": method == "experiment4",
        "test_contaminated": method == "experiment4",
        "training_history": history,
        "training_data": {
            "pages": 0 if method == "lines" else 125,
            "berrutti_lines": 0 if method == "pages" else 5212,
            "luisa_lines": 23774 if method == "experiment4" else 0,
        },
    }


def export_luisa(bundle: Bundle, root: Path, mixed: dict) -> tuple[str, dict]:
    """All original lines with their original source split, never silently filtered."""
    records = jailed_file(root, "records.jsonl")
    if mixed.get("records_sha256") != file_sha256(records):
        raise ValueError("LUISA records differ from sealed mixed curriculum")
    rows, sources = [], {}
    with records.open() as stream:
        for encoded in stream:
            row = json.loads(encoded)
            line_id = identifier("luisa-" + row["id"])
            if line_id in sources:
                raise ValueError("Duplicate LUISA line")
            image = bundle.image(jailed_file(root, row["image"]), row["image_sha256"])
            if row["source_split"] not in ("train", "test"):
                raise ValueError("Unexpected LUISA source split")
            detail = {
                "id": line_id,
                "source_split": row["source_split"],
                "image": image,
                **text_fields(row["reference_text"]),
            }
            rows.append(
                {"id": line_id, "image": image, "detail": bundle.fragment(detail)}
            )
            sources[line_id] = {
                **detail,
                "source": "luisa",
                "image_sha256": row["image_sha256"],
            }
    if len(rows) != 23774 or Counter(
        row["source_split"] for row in sources.values()
    ) != {"train": 21611, "test": 2163}:
        raise ValueError(
            "All 23774 original LUISA lines and source splits are required"
        )
    return bundle.fragment({"lines": rows}), sources


def export_luisa_cache(bundle: Bundle, mix: Path, mixed: dict) -> tuple[str, dict]:
    """Optional read-only verified preparation cache when originals are remote."""
    source = jailed_file(mix, "source-crops.sqlite")
    examples = {
        row["line_id"]: row
        for unit in mixed["epochs"][0]["units"]
        for row in unit["luisa"]
    }
    rows, sources = [], {}
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as database:
        for line_id, row in sorted(examples.items()):
            identifier(line_id)
            saved = database.execute(
                "SELECT reference_sha,png,rgb_sha FROM crops WHERE line_id=?",
                (line_id,),
            ).fetchone()
            if (
                saved is None
                or saved[0]
                != hashlib.sha256(row["reference_text"].encode()).hexdigest()
                or hashlib.sha256(saved[1]).hexdigest() != row["image_sha256"]
            ):
                raise ValueError(
                    "Cached LUISA image/reference differs from sealed corpus"
                )
            with Image.open(io.BytesIO(saved[1])) as image:
                if (
                    hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
                    != saved[2]
                ):
                    raise ValueError("Cached LUISA pixels differ")
                suffix = image.format.lower()
            image = bundle.payload(saved[1], "images", suffix)
            detail = {
                "id": line_id,
                "source_split": row["source_split"],
                "image": image,
                **text_fields(row["reference_text"]),
            }
            rows.append(
                {"id": line_id, "image": image, "detail": bundle.fragment(detail)}
            )
            sources[line_id] = {
                **detail,
                "source": "luisa",
                "image_sha256": row["image_sha256"],
            }
    if len(rows) != 23774 or Counter(
        row["source_split"] for row in sources.values()
    ) != {"train": 21611, "test": 2163}:
        raise ValueError("All original LUISA cache entries are required")
    return bundle.fragment({"lines": rows}), sources


def export_mixed(
    bundle: Bundle,
    mix: Path,
    cohort_path: Path,
    line_sources: dict,
    luisa_root: Path | None,
) -> dict:
    manifest = read_json(jailed_file(mix, "manifest.json"))
    receipt = read_json(jailed_file(mix, "completed.json"))
    if (
        receipt.get("success") is not True
        or receipt["manifest_sha256"] != file_sha256(mix / "manifest.json")
        or manifest["source_cohort_sha256"] != file_sha256(cohort_path)
        or manifest["audit"]["selection"] != "include-all"
        or manifest["audit"]["luisa_selected"] != 23774
    ):
        raise ValueError("Complete include-all mixed curriculum receipt required")
    luisa_index, luisa_sources = (
        export_luisa(bundle, luisa_root, manifest)
        if luisa_root
        else export_luisa_cache(bundle, mix, manifest)
    )
    sources = {**line_sources, **luisa_sources}
    pages, ids = [], set()
    for entry in manifest["epochs"]:
        epoch = entry["epoch"]
        if epoch not in (1, 2, 3, 4):
            raise ValueError("Unexpected synthetic curriculum epoch")
        coverage, individual_luisa = Counter(), Counter()
        unit_ids = [unit["page_id"] for unit in entry["units"]]
        expected_pages = {
            line_id.rsplit("-", 1)[0].removeprefix("berrutti-")
            for line_id in line_sources
        }
        if len(unit_ids) != 125 or set(unit_ids) != expected_pages:
            raise ValueError("Mixed epoch must contain each of the 125 source pages")
        for unit in entry["units"]:
            for row in unit["luisa"]:
                individual_luisa[row["line_id"]] += 1
                original = luisa_sources.get(row["line_id"])
                if (
                    original is None
                    or row["reference_text"] != original["reference"]
                    or row["image_sha256"] != original["image_sha256"]
                ):
                    raise ValueError(
                        "Mixed LUISA references differ from individual records"
                    )
            for page in unit["synthetic"]:
                page_id = identifier(page["page_id"])
                if page_id in ids or page["columns"] not in (1, 2):
                    raise ValueError("Duplicate synthetic page or invalid columns")
                ids.add(page_id)
                image = bundle.image(
                    jailed_file(mix, f"images/{page_id}.png"), page["image_sha256"]
                )
                lines = []
                for placement in page["sources"]:
                    original = sources.get(placement["line_id"])
                    if (
                        original is None
                        or placement["source"] != original["source"]
                        or placement["image_sha256"] != original["image_sha256"]
                        or placement["reference_sha256"]
                        != hashlib.sha256(original["reference"].encode()).hexdigest()
                    ):
                        raise ValueError("Synthetic line provenance changed")
                    coverage[placement["line_id"]] += 1
                    x, y, right, bottom = placement["box"]
                    lines.append(
                        {
                            key: original[key]
                            for key in (
                                "id",
                                "image",
                                "reference",
                                "reference_normalized",
                                "source",
                            )
                        }
                        | {
                            "box": [x, y, right - x, bottom - y],
                            "column": placement["column"],
                        }
                    )
                reference = "\n".join(line["reference"] for line in lines)
                if reference != page["reference_text"]:
                    raise ValueError(
                        "Synthetic page reference differs from all source lines"
                    )
                detail = bundle.fragment(
                    {
                        "id": page_id,
                        "epoch": epoch,
                        "columns": page["columns"],
                        "image": image,
                        "lines": lines,
                        **text_fields(reference),
                    }
                )
                pages.append(
                    {
                        "id": page_id,
                        "epoch": epoch,
                        "columns": page["columns"],
                        "image": image,
                        "detail": detail,
                        "line_count": len(lines),
                    }
                )
        if epoch <= 3 and individual_luisa != Counter(
            {line_id: 1 for line_id in luisa_sources}
        ):
            raise ValueError(
                "Mixed epoch must contain each of the 23774 individual LUISA lines"
            )
        if epoch == 4 and individual_luisa:
            raise ValueError("Consolidation must not contain individual LUISA lines")
        if epoch <= 3 and coverage != Counter({line_id: 1 for line_id in sources}):
            raise ValueError(
                "Synthetic epoch must cover every Berrutti/LUISA line exactly once"
            )
        if epoch == 4 and coverage:
            raise ValueError("Consolidation must not contain synthetic pages")
    if len(pages) != 1991 or {entry["epoch"] for entry in manifest["epochs"]} != {
        1,
        2,
        3,
        4,
    }:
        raise ValueError(
            "All 1991 sealed synthetic pages across the three mixed epochs are required"
        )
    return {
        "synthetic_index": bundle.fragment({"pages": pages}),
        "luisa_index": luisa_index,
        "synthetic_pages": len(pages),
        "luisa_lines": len(luisa_sources),
    }


def baseline_status(evaluations: list[dict], method: str) -> dict:
    rows = [row for row in evaluations if row["method"] == method]
    splits = {row["split"] for row in rows}
    if not rows:
        state = "PLANNED" if method == "experiment4_baseline" else "UNKNOWN"
    elif len(rows) == 2 and splits == {"validation", "test"}:
        state = (
            "COMPLETED"
            if all(row["summary"]["all_pages_completed"] for row in rows)
            else "INCOMPLETE"
        )
    else:
        state = "RUNNING"
    return {
        "state": state,
        "optimizer_steps": 0,
        "evaluations": [row["id"] for row in rows],
    }


def build(
    dataset: Path,
    split: Path,
    cohort: Path,
    evidence: Path,
    output: Path,
    *,
    mix_data: Path | None = None,
    luisa_root: Path | None = None,
    snapshot: Path | None = None,
    assets: Path | None = None,
) -> dict:
    pages = load_pages(dataset, split)
    sealed = validate_cohort(read_json(cohort), pages)
    observed = read_json(snapshot) if snapshot else {}
    observed = observed.get("snapshot", observed)
    bundle = Bundle(output)
    page_catalog, line_sources = export_pages(bundle, pages, sealed)
    evaluations = export_evidence(bundle, evidence, pages)
    mixed = (
        export_mixed(bundle, mix_data, cohort, line_sources, luisa_root)
        if mix_data
        else {}
    )
    methods = [method_status(name, observed, evaluations, evidence) for name in METHODS]
    catalog = {
        "schema_version": 1,
        "manifest": "manifest.json",
        "title": "Qwen–Berrutti · resultados de entrenamiento",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "observed_at": observed.get("checked_at_utc"),
        "normalization_policy": "ascii-text-v1",
        "protocol": {
            "id": "three-arms-125-v1",
            "epochs_per_arm": 3,
            "optimizer_steps_per_epoch": 16,
            "train_pages": 125,
            "train_lines": 5212,
            "validation_pages": 25,
            "test_pages": 25,
            "max_attempts_per_page": 2,
            "selection": "Mínimo CER de validación completa al final de cada época",
            "test_policy": "Una evaluación final descriptiva del checkpoint elegido",
            "page_line_reference_disagreements": 34,
            "shared_baseline": "Una evaluación previa compartida, cero actualizaciones",
            "exp4_interpretation": "Exploratorio; incluye todo LUISA, su test y solapamientos conocidos con test Berrutti. Independencia documental no demostrada.",
        },
        "methods": methods,
        "shared_baseline": baseline_status(evaluations, "baseline"),
        "experiment4_baseline": baseline_status(evaluations, "experiment4_baseline"),
        "pages": page_catalog,
        "evaluations": evaluations,
        "training_sources": {
            "berrutti_pages": 125,
            "berrutti_lines": 5212,
            "luisa_lines": mixed.get("luisa_lines", 0),
            "synthetic_pages": mixed.get("synthetic_pages", 0),
        },
        **{key: value for key, value in mixed.items() if key.endswith("_index")},
    }
    bundle.named_json("catalog.json", catalog)
    bundle.finish(assets or Path(__file__).with_name("web"))
    return {
        "schema_version": 1,
        "success": True,
        "pages": len(page_catalog),
        "berrutti_lines": len(line_sources),
        "evaluations": len(evaluations),
        "luisa_lines": mixed.get("luisa_lines", 0),
        "synthetic_pages": mixed.get("synthetic_pages", 0),
        "files": len(bundle.files),
        "manifest_sha256": file_sha256(output / "manifest.json"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "split", "cohort", "evidence", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("mix-data", "luisa-root", "snapshot", "assets"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(build(**vars(args)), ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

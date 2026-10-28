"""Publish audited, closed OCR through the benchmark's unchanged explorer.

No inference or training. The benchmark is an explicit local build dependency;
its shard builder and seven UI assets are pinned in the private build receipt.
Only held-out images, references and selected closed predictions are exported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from evaluation.scoring import normalize_text

MODELS = (
    ("base", "Qwen sin entrenamiento · 0 épocas", "run-1", 0, None),
    ("pages", "Berrutti: sólo páginas · 3 épocas", "run-0", 2, "base"),
    ("lines", "Berrutti: sólo líneas · 3 épocas", "run-2", 2, "base"),
    ("combined", "Berrutti: páginas + líneas · 3 épocas", "run-3", 2, "base"),
    ("luisa_base", "Qwen sin ajuste · 0 épocas (referencia LUISA)", "run-5", 0, None),
    ("luisa", "Berrutti + LUISA + sintéticas · 3+1 épocas", "run-4", 4, "luisa_base"),
)
PUBLISHED_MODELS = {"base", "pages", "lines", "combined", "luisa"}
TRAINING_DESCRIPTIONS = {
    "base": "Qwen sin fine-tuning; generación determinista sobre páginas Berrutti. 0 épocas.",
    "pages": "125 páginas completas Berrutti; 3 épocas.",
    "lines": "5.212 imágenes de líneas Berrutti; 3 épocas.",
    "combined": "125 páginas completas Berrutti y sus 5.212 líneas; 3 épocas.",
    "luisa_base": "Sin entrenamiento adicional; referencia de la prueba con LUISA.",
    "luisa": (
        "125 páginas y 5.212 líneas Berrutti, 23.774 líneas LUISA y páginas sintéticas; "
        "3 épocas con la mezcla + 1 sólo Berrutti. Exploratorio; test contaminado."
    ),
}
ASSETS = (
    "index.html",
    "app.js",
    "styles.css",
    "favicon.svg",
    "extensions.css",
    "extensions.js",
    "loader.js",
)
COUNTS = ("char_errors", "char_count", "word_errors", "word_count")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encoded(value) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def script(value) -> bytes:
    return (
        encoded(value)
        .replace(b"\xe2\x80\xa8", b"\\u2028")
        .replace(b"\xe2\x80\xa9", b"\\u2029")
    )


class Evidence:
    def __init__(self, pointer: Path):
        self.pointer = json.loads(pointer.read_bytes())
        if self.pointer["state"] != "VERIFIED":
            raise ValueError("Final audit must be verified")
        receipt = Path(self.pointer["verification_receipt"])
        self.root = receipt.parent
        self.receipt = json.loads(receipt.read_bytes())
        if self.receipt.get("success") is not True:
            raise ValueError("Final audit receipt is unsuccessful")
        manifest = (self.root / "evidence-manifest.json").read_bytes()
        if sha(manifest) != self.pointer["evidence_manifest_sha256"]:
            raise ValueError("Audit evidence manifest changed")
        self.files = json.loads(manifest)["files"]
        self.reads = {}

    def read(self, relative: str) -> bytes:
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("Evidence path escapes audit")
        data = path.read_bytes()
        expected = self.files[relative]
        if sha(data) != expected["sha256"]:
            raise ValueError(f"Sealed evidence changed: {relative}")
        self.reads[relative] = sha(data)
        return data

    def json(self, relative: str):
        return json.loads(self.read(relative))


def metrics(reference: str, prediction: str, closed: dict) -> dict:
    ref, pred = normalize_text(reference), normalize_text(prediction)
    values = {
        "char_errors": Levenshtein.distance(ref, pred),
        "char_count": len(ref),
        "word_errors": Levenshtein.distance(ref.split(), pred.split()),
        "word_count": len(ref.split()),
    }
    if any(values[k] != closed[k] for k in COUNTS):
        raise ValueError("Closed OCR score does not match original text")
    return values


def read_evaluation(evidence: Evidence, run: str, epoch: int, split: str, refs: dict):
    folder = f"closed-evaluations/{run}/evaluations/epoch-{epoch:02d}-{split}"
    summary = evidence.json(folder + "/closed.json")
    records = {}
    for path in sorted((evidence.root / folder / "pages").glob("*/closed.json")):
        closed = evidence.json(path.relative_to(evidence.root).as_posix())
        pid = path.parent.name
        if pid != closed["page_id"] or pid not in refs:
            raise ValueError("Evaluation page identity mismatch")
        attempts = closed["attempts"]
        if type(attempts) is not int or not 1 <= attempts <= 2:
            raise ValueError("Generation attempt limit violated")
        rows = [
            evidence.json(f"{folder}/pages/{pid}/attempt-{i}.json")
            for i in range(1, attempts + 1)
        ]
        first_eos = next(
            (i for i, row in enumerate(rows, 1) if row["stop_reason"] == "eos"), None
        )
        if first_eos is not None and first_eos != attempts:
            raise ValueError("Closed prediction is not the first successful attempt")
        if (closed["status"] == "complete") != (first_eos is not None):
            raise ValueError("Closed status contradicts attempts")
        if closed["status"] not in {"complete", "incomplete"}:
            raise ValueError("Unknown closed generation status")
        prediction = rows[-1]["prediction"]
        score = metrics(refs[pid], prediction, closed)
        records[pid] = {
            **score,
            "prediction": prediction,
            "status": closed["status"],
            "attempts": attempts,
            "stop_reasons": [r["stop_reason"] for r in rows],
        }
    totals = {k: sum(row[k] for row in records.values()) for k in COUNTS}
    failed = sum(row["status"] != "complete" for row in records.values())
    retried = sum(row["attempts"] > 1 for row in records.values())
    if len(records) != 25 or any(totals[k] != summary[k] for k in COUNTS):
        raise ValueError("Evaluation aggregate or page coverage mismatch")
    if failed != summary["failed_pages"] or retried != summary["retried_pages"]:
        raise ValueError("Evaluation retry/failure count mismatch")
    for field, numerator, denominator in (
        ("cer", "char_errors", "char_count"),
        ("wer", "word_errors", "word_count"),
    ):
        if not math.isclose(
            summary[field], totals[numerator] / totals[denominator], abs_tol=1e-12
        ):
            raise ValueError("Evaluation micro aggregate mismatch")
    if summary["all_pages_completed"] != (failed == 0):
        raise ValueError("Evaluation completion mismatch")
    return {
        **totals,
        "cer": summary["cer"],
        "wer": summary["wer"],
        "pages": 25,
        "completed": 25 - failed,
        "failed_pages": failed,
        "retried_pages": retried,
        "provisional": failed > 0,
    }, records


def result_projection(reference: str, row: dict, display_name: str = "") -> dict:
    ref, pred = normalize_text(reference), normalize_text(row["prediction"])

    def operations(left, right):
        counts = Counter(op.tag for op in Levenshtein.editops(left, right))
        return {
            "insertions": counts["insert"],
            "deletions": counts["delete"],
            "substitutions": counts["replace"],
        }

    return {
        "displayName": display_name
        + (" · INCOMPLETA" if row["status"] != "complete" else ""),
        "text": "",
        "textContent": row["prediction"],
        "scoringTextContent": pred,
        "cer": row["char_errors"] / row["char_count"],
        "wer": row["word_errors"] / row["word_count"],
        "gtChars": len(ref),
        "predChars": len(pred),
        "gtWords": len(ref.split()),
        "predWords": len(pred.split()),
        "latencySeconds": None,
        "ops": {
            **operations(ref, pred),
            "editDistance": row["char_errors"],
            "scoringGtChars": len(ref),
            "scoringPredChars": len(pred),
        },
        "wordOps": {
            **operations(ref.split(), pred.split()),
            "editDistance": row["word_errors"],
            "scoringGtChars": len(ref.split()),
            "scoringPredChars": len(pred.split()),
        },
    }


def verified_export(root: Path, expected_manifest_sha256: str) -> dict:
    """Read only exact, manifest-closed export bytes; never consume its UI."""
    manifest = (root / "manifest.json").read_bytes()
    if sha(manifest) != expected_manifest_sha256:
        raise ValueError("Audited export manifest changed")
    value = json.loads(manifest)
    expected = set(value["files"]) | {"manifest.json"}
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != expected:
        raise ValueError("Audited export file set changed")
    for relative, record in value["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Export path escapes root")
        data = path.read_bytes()
        if sha(data) != record["sha256"] or len(data) != record["size_bytes"]:
            raise ValueError("Audited export payload changed")
    return value


def read_shard(path: Path, method: str) -> dict:
    payload = path.read_bytes()
    prefix = ("window.OCR_DIFF_VIEWER_DATA." + method + "(").encode()
    if not payload.startswith(prefix) or not payload.endswith(b");\n"):
        raise ValueError("Invalid benchmark shard")
    return json.loads(payload[len(prefix) : -3])


V2_MANIFEST_SHA256 = "9994fe5dc725299d9f2f6e0f171d5569820412c94ec3267907facaa24cb65d6e"
MODEL_IDS = {
    "base": "qwen-base-sin-ajuste",
    "pages": "ajuste-paginas",
    "lines": "ajuste-lineas",
    "combined": "ajuste-combinado",
    "luisa_base": "base-propia-luisa",
    "luisa": "ajuste-luisa-exploratorio",
}


def build_v2(
    export: Path,
    export_sha256: str,
    audit_pointer: Path,
    viewer_source: Path,
    benchmark_root: Path,
    output: Path,
    shard_builder,
) -> dict:
    """Use the published v2 shell directly, with audited held-out OCR only."""
    verified_export(export, export_sha256)
    audit = Evidence(audit_pointer)
    public = json.loads((export / "results.json").read_bytes())
    if public["audit_manifest_sha256"] != audit.pointer["evidence_manifest_sha256"]:
        raise ValueError("Export and final audit differ")
    source_manifest = (viewer_source / "manifest.json").read_bytes()
    if sha(source_manifest) != V2_MANIFEST_SHA256:
        raise ValueError("Viewer source is not the published v2")
    source_files = json.loads(source_manifest)["files"]
    assets = {}
    for name in ASSETS:
        payload = (viewer_source / name).read_bytes()
        if (
            sha(payload) != source_files[name]["sha256"]
            or len(payload) != source_files[name]["size_bytes"]
        ):
            raise ValueError("Published v2 interface asset changed")
        assets[name] = payload
    # Preserve stable IDs for links and data; the context displays readable names.
    names = {entry[0]: entry[1] for entry in MODELS}
    public["models"] = [
        model for model in public["models"] if model["id"] in PUBLISHED_MODELS
    ]
    for model in public["models"]:
        model["name"] = names[model["id"]]
        model["training_description"] = TRAINING_DESCRIPTIONS[model["id"]]
        model["id"] = MODEL_IDS[model["id"]]
        if model["baseline"]:
            if model["baseline"] in PUBLISHED_MODELS:
                model["baseline"] = MODEL_IDS[model["baseline"]]
            else:
                model["baseline"] = None
                model["baseline_not_displayed"] = True
    public["page_status"] = {
        split: {
            MODEL_IDS[mid]: rows
            for mid, rows in models.items()
            if mid in PUBLISHED_MODELS
        }
        for split, models in public["page_status"].items()
    }
    output.mkdir(parents=True, exist_ok=False)

    def write(relative: str, data: bytes):
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)

    custom = Path(__file__).with_name("ocr_diff_web")
    context_js = (custom / "context.js").read_bytes()
    context_css = (custom / "context.css").read_bytes()
    index_hashes = {}
    for split, prefix in (("validation", ""), ("test", "test/")):
        catalog = json.loads((export / split / "data/catalog.json").read_bytes())
        models = []
        results = {}
        for entry in catalog["models"]:
            if entry["id"] not in PUBLISHED_MODELS:
                continue
            shard = read_shard(export / split / entry["path"], "registerModel")
            model = shard["model"]
            model["id"] = MODEL_IDS[entry["id"]]
            models.append(model)
            results[model["id"]] = {}
            for pid, compact in shard["results"].items():
                detail = read_shard(
                    export / split / compact["detailScript"], "registerDetail"
                )
                result = {k: v for k, v in compact.items() if k != "detailScript"}
                result.update(detail["result"])
                results[model["id"]][pid] = result
        pages = []
        for entry in catalog["pages"]:
            pid = entry["id"]
            detail = read_shard(
                export / split / entry["pageScript"], "registerPageDetail"
            )
            image = (export / split / entry["image"]).read_bytes()
            image_path = "images/" + pid + ".png"
            write(prefix + image_path, image)
            pages.append(
                {
                    **detail,
                    "image": image_path,
                    "imageMime": entry["imageMime"],
                    "models": {m["id"]: results[m["id"]][pid] for m in models},
                }
            )
        if len(pages) != 25 or len(models) != len(PUBLISHED_MODELS):
            raise ValueError("Held-out view coverage changed")
        view = {
            "dataset": "Berrutti · " + split,
            "source": catalog["source"],
            "counts": {
                "pages": 25,
                "models": len(models),
                "pageModelResults": 25 * len(models),
            },
            "models": models,
            "pages": pages,
        }
        shards = shard_builder(
            view,
            app_reference="app.js?sha256=" + sha(assets["app.js"]),
            extensions_reference="extensions.js?sha256=" + sha(assets["extensions.js"]),
        )
        for name, payload in {**assets, **shards.payloads}.items():
            if name != "index.html":
                write(prefix + name, payload)
        data_js = b"window.OCR_DIFF_VIEWER_CATALOG=" + script(shards.catalog) + b";\n"
        write(prefix + "data/catalog.json", encoded(shards.catalog))
        write(prefix + "data/catalog.js", data_js)
        metadata = {
            **public,
            "split": split,
            "publication_root": "../" if split == "test" else "./",
        }
        metadata_js = b"window.FINETUNING_RESULTS=" + script(metadata) + b";\n"
        write(prefix + "finetuning-data.js", metadata_js)
        write(prefix + "finetuning-context.js", context_js)
        write(prefix + "finetuning-context.css", context_css)
        html = assets["index.html"].decode()
        html = re.sub(
            r"data/catalog\.js\?sha256=[0-9a-f]{64}",
            "data/catalog.js?sha256=" + sha(data_js),
            html,
        )
        html = html.replace(
            "</head>",
            '<link rel="stylesheet" href="finetuning-context.css?sha256='
            + sha(context_css)
            + '">\n</head>',
        )
        html = html.replace(
            "</body>",
            '<script src="finetuning-data.js?sha256='
            + sha(metadata_js)
            + '"></script>\n<script src="finetuning-context.js?sha256='
            + sha(context_js)
            + '"></script>\n</body>',
        )
        write(prefix + "index.html", html.encode())
        index_hashes[split] = sha(html.encode())
    write("results.json", encoded(public))
    write("BENCHMARK_LICENSE.txt", (benchmark_root / "LICENSE").read_bytes())
    files = {
        p.relative_to(output).as_posix(): {
            "sha256": sha(p.read_bytes()),
            "size_bytes": p.stat().st_size,
        }
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }
    manifest = {
        "schema_version": 1,
        "artifact_type": "berrutti-benchmark-viewer",
        "subtype": "qwen-finetuning-ocr-diff-v2",
        "normalization_policy": "ascii-text-v1",
        "source_interface": {"manifest_sha256": V2_MANIFEST_SHA256},
        "counts": {
            "pages": 50,
            "models": len(PUBLISHED_MODELS),
            "selected_page_predictions": 50 * len(PUBLISHED_MODELS),
        },
        "files": files,
    }
    write("manifest.json", encoded(manifest))
    return {
        "manifest_sha256": sha(encoded(manifest)),
        "files": len(files) + 1,
        "bytes": sum(v["size_bytes"] for v in files.values()) + len(encoded(manifest)),
        "reference_manifest_sha256": V2_MANIFEST_SHA256,
        "reference_assets": {k: sha(v) for k, v in assets.items()},
        "source_export_sha256": export_sha256,
        "index_sha256": index_hashes,
        "no_iframes": True,
        "no_training_or_generations": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-pointer", type=Path, required=True)
    parser.add_argument("--audited-export", type=Path, required=True)
    parser.add_argument("--audited-export-sha256", required=True)
    parser.add_argument("--viewer-source", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.benchmark_root / "src"))
    from berrutti_bench.web.collection_view_shards import build_comparison_view_shards

    receipt = build_v2(
        args.audited_export,
        args.audited_export_sha256,
        args.audit_pointer,
        args.viewer_source,
        args.benchmark_root,
        args.output,
        build_comparison_view_shards,
    )
    with args.receipt.open("xb") as stream:
        stream.write(encoded(receipt))
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()

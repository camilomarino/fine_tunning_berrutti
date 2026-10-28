"""Read-only final fine-tuning audit. Run payload verification on Slurm CPU nodes."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

from reporting.berrutti.audit_statistics import paired_difference
from evaluation.scoring import score_pairs
from finetune.recipes.berrutti_qwen38.generation_guard import GenerationPolicy
from finetune.recipes.berrutti_qwen38.qwen_sampling import InstructSamplingPolicy


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def validate_updates(rows, *, epochs, stage, maximum, minimum):
    require(
        [row["optimizer_steps"] for row in rows] == list(range(1, 16 * epochs + 1)),
        "Missing, repeated or unordered committed update",
    )
    for row in rows:
        step = row["optimizer_steps"]
        require(
            row["stage"] == stage
            and row["epoch"] == (step - 1) // 16 + 1
            and row["page"] == min((((step - 1) % 16) + 1) * 8, 125)
            and row["total"] == 125,
            "Update has an invalid epoch or page cursor",
        )
        require(
            all(math.isfinite(row[k]) for k in ("loss", "lr", "gradient_norm")),
            "Nonfinite training update",
        )
        lr = minimum + 0.5 * (maximum - minimum) * (
            1 + math.cos(math.pi * (step - 1) / (16 * epochs - 1))
        )
        require(math.isclose(row["lr"], lr, rel_tol=1e-12), "Learning-rate drift")


def selected_epoch(validations, epochs):
    require(
        len(validations) == epochs
        and {v["epoch"] for v in validations} == set(range(1, epochs + 1)),
        "Only complete end-of-epoch validation may select a checkpoint",
    )
    eligible = [
        v
        for v in validations
        if v["split"] == "validation"
        and v["pages"] == 25
        and v["all_pages_completed"] is True
        and v["failed_pages"] == 0
        and math.isfinite(v["cer"])
        and v["cer"] >= 0
    ]
    require(bool(eligible), "No complete validation checkpoint")
    return min(eligible, key=lambda v: (v["cer"], v["epoch"]))


def validate_completion_header(completion, *, exploratory):
    required = {
        "best_epoch",
        "best_validation_cer",
        "test_cer_at_selected_epoch",
        "selected_test_failed_pages",
        "optimizer_steps",
        "selected_checkpoint",
    }
    require(required <= completion.keys(), "Incomplete final receipt")
    if exploratory:
        require(
            completion.get("success") is True
            and type(completion.get("result_complete")) is bool,
            "Mixed training receipt unsuccessful or missing completion status",
        )
    else:
        # The frozen three-arm runner predates explicit success/result_complete.
        # Its six-field receipt is checked against Slurm and the closed test.
        require(
            set(completion) == required,
            "Unexpected three-arm completion schema",
        )


def training_epoch_receipt(root, events, epoch, *, exploratory):
    rows = [e for e in events if e["event"] == "training_epoch" and e["epoch"] == epoch]
    require(len(rows) == 1, "Missing or repeated closed training epoch")
    event = {k: v for k, v in rows[0].items() if k not in ("event", "elapsed_segment")}
    if exploratory:
        receipt = read(root / "epochs" / f"epoch-{epoch:02d}-training.json")
        require(receipt == event, "Training epoch file differs from its event")
        return receipt
    return event


def check_attempts(page_root, row, reference, policy):
    attempts = sorted(Path(page_root).glob("attempt-[0-9]*.json"))
    actual = [p for p in attempts if not p.name.endswith(".started.json")]
    require(row["attempts"] in (1, 2), "Per-page generation budget exceeded")
    require(len(actual) == row["attempts"], "Attempt receipt coverage differs")
    require(len(attempts) == 2 * len(actual), "Attempt reservations differ")
    reason = None
    history = []
    prediction = None
    for number, path in enumerate(actual, 1):
        require(path.name == f"attempt-{number}.json", "Attempt sequence differs")
        attempt = read(path)
        reservation = read(Path(page_root) / f"attempt-{number}.started.json")
        expected = policy.attempt_parameters(reason)
        require(reason != "eos", "Retry after a successful generation")
        require(
            reservation == {"attempt": number, "parameters": expected}
            and attempt["attempt"] == number
            and attempt["parameters"] == expected,
            "Generation parameters or attempt reservation changed",
        )
        reason = attempt["stop_reason"]
        require(
            reason
            in {
                "eos",
                "repetition",
                "time_limit",
                "token_limit",
                "unexpected_stop",
                "interrupted",
            },
            "Unknown generation termination",
        )
        require(
            math.isfinite(attempt["seconds"])
            and attempt["seconds"] >= 0
            and attempt["generated_tokens"] >= 0,
            "Invalid generation timing or length",
        )
        if reason == "time_limit":
            require(attempt["seconds"] >= expected["max_seconds"], "Early time limit")
        if reason == "token_limit":
            require(
                attempt["generated_tokens"] >= expected["max_new_tokens"],
                "Early token limit",
            )
        prediction = attempt["prediction"]
        history.append({k: v for k, v in attempt.items() if k != "prediction"})
    require(
        (row["status"] == "complete") == (reason == "eos"),
        "Closed page status differs from its selected raw attempt",
    )
    require(
        row["status"] == "complete" or row["attempts"] == 2,
        "Incomplete page did not consume its allowed attempts",
    )
    require(
        row["generation_stop_counts"]
        == dict(Counter(a["stop_reason"] for a in history))
        and math.isclose(row["seconds"], sum(a["seconds"] for a in history)),
        "Closed page timing or stop counters differ",
    )
    metrics = score_pairs([(reference, prediction)])
    require(
        all(
            row[k] == metrics[k]
            for k in ("char_errors", "char_count", "word_errors", "word_count")
        ),
        "Independent CER/WER rescore disagrees with closed page",
    )
    return history


def audit_evaluation(
    root, summary, references, policy, adapter_digest=None, identity=None
):
    require(read(root / "closed.json") == summary, "Closed evaluation disagrees")
    closed = sorted((root / "pages").glob("*/closed.json"))
    rows = [read(p) for p in closed]
    require(
        len(rows) == 25
        and {r["page_id"] for r in rows} == set(references)
        and len({r["evaluation_key"] for r in rows}) == 1,
        "Evaluation page coverage or identity differs",
    )
    if adapter_digest is not None:
        encoded = json.dumps(
            {"runtime": identity, "adapter": adapter_digest, "split": summary["split"]},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
        require(
            all(
                r["evaluation_key"] == hashlib.sha256(encoded).hexdigest() for r in rows
            ),
            "Prediction evaluation key differs from the audited checkpoint",
        )
    diagnoses = []
    for path, row in zip(closed, rows, strict=True):
        history = check_attempts(path.parent, row, references[row["page_id"]], policy)
        if len(history) == 2 or row["status"] != "complete":
            diagnoses.append(
                {
                    "page_id": row["page_id"],
                    "status": row["status"],
                    "attempts": history,
                }
            )
    for key in ("char_errors", "char_count", "word_errors", "word_count"):
        require(summary[key] == sum(r[key] for r in rows), "Micro aggregation differs")
    failures = sum(r["status"] != "complete" for r in rows)
    require(
        summary["pages"] == 25
        and summary["failed_pages"] == failures
        and summary["all_pages_completed"] == (failures == 0)
        and summary["retried_pages"] == sum(r["attempts"] == 2 for r in rows),
        "Failure or retry aggregation differs",
    )
    require(
        summary["cer"] == summary["char_errors"] / summary["char_count"]
        and summary["wer"] == summary["word_errors"] / summary["word_count"],
        "Published CER or WER differs from edit counts",
    )
    stops = Counter()
    for row in rows:
        stops.update(row["generation_stop_counts"])
    require(
        summary["generation_stop_counts"] == dict(stops),
        "Termination aggregation differs",
    )
    return rows, diagnoses


def tensor_digest(torch, adapter):
    digest = hashlib.sha256()
    for name, tensor in sorted(adapter.items()):
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def check_rng(torch, rng):
    random.Random().setstate(rng["python"])
    generator = torch.Generator(device="cpu")
    generator.set_state(rng["torch"])
    require(
        torch.equal(generator.get_state(), rng["torch"]), "CPU RNG roundtrip differs"
    )
    require(
        len(rng["cuda"]) == 2
        and all(
            t.dtype == torch.uint8 and t.ndim == 1 and t.numel() > 0
            for t in rng["cuda"]
        ),
        "Saved CUDA RNG coverage differs",
    )


def audit_payload(torch, path, identity, config, expected_step):
    before = (path / "state.pt").stat()
    receipt = read(path / "closed.json")
    require(
        receipt["global_step"] == expected_step
        and receipt["runtime_identity"] == identity
        and sha(path / "state.pt") == receipt["state_sha256"],
        "Checkpoint hash or closed identity differs",
    )
    payload = torch.load(
        path / "state.pt", map_location="cpu", weights_only=False, mmap=True
    )
    require(payload["runtime_identity"] == identity, "Payload runtime differs")
    require(
        payload["scheduler"]
        == {
            "completed_steps": expected_step,
            "total_steps": config.get("total_optimizer_steps", 48),
            "version": "cosine-v1",
        },
        "Checkpoint scheduler differs",
    )
    adapter = payload["adapter"]
    require(bool(adapter), "Empty trained adapter")
    for tensor in adapter.values():
        require(
            tensor.is_floating_point() and bool(torch.isfinite(tensor).all()),
            "Nonfinite adapter",
        )
    optimizer = payload["optimizer"]
    indices = [
        index for group in optimizer["param_groups"] for index in group["params"]
    ]
    require(
        len(indices) == len(set(indices)) == len(adapter)
        and set(indices) == set(optimizer["state"]),
        "Incomplete AdamW parameter/moment coverage",
    )
    expected_lr = config["minimum_learning_rate"] + 0.5 * (
        config["learning_rate"] - config["minimum_learning_rate"]
    ) * (
        1
        + math.cos(
            math.pi
            * (expected_step - 1)
            / (config.get("total_optimizer_steps", 48) - 1)
        )
    )
    for group in optimizer["param_groups"]:
        require(
            math.isclose(group["lr"], expected_lr, rel_tol=1e-12),
            "Saved optimizer LR differs",
        )
        require(group["weight_decay"] == 0, "Saved AdamW weight decay differs")
    for moment in optimizer["state"].values():
        require(
            set(moment) == {"step", "exp_avg", "exp_avg_sq"}
            and float(moment["step"]) == expected_step,
            "Missing AdamW moments or invalid step",
        )
        require(
            moment["exp_avg"].shape == moment["exp_avg_sq"].shape,
            "Moment shapes differ",
        )
        require(
            bool(torch.isfinite(moment["exp_avg"]).all())
            and bool(torch.isfinite(moment["exp_avg_sq"]).all())
            and bool((moment["exp_avg_sq"] >= 0).all()),
            "Invalid AdamW moments",
        )
    require(
        Counter(tuple(t.shape) for t in adapter.values())
        == Counter(tuple(m["exp_avg"].shape) for m in optimizer["state"].values()),
        "Adapter and AdamW moment shape coverage differ",
    )
    check_rng(torch, payload["rng"])
    digest = tensor_digest(torch, adapter)
    after = (path / "state.pt").stat()
    require(
        (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
        "Checkpoint changed during audit",
    )
    result = {
        **receipt,
        "adapter_sha256": digest,
        "bytes": before.st_size,
        "adapter_tensors": len(adapter),
        "adamw_states": len(indices),
        "finite": True,
        "cpu_deserialization_checked": True,
        "scheduler_and_rng_checked": True,
        "new_gpu_restore_performed": False,
    }
    del payload, optimizer, adapter
    gc.collect()
    return result


def audit_case(spec, definition, out, torch):
    name = spec["name"]
    root = Path(spec["run_root"])
    config = read(spec["config"])
    state = read(root / "state.json")
    completion = read(root / "completed.json")
    validate_completion_header(completion, exploratory=spec["exploratory"])
    require(
        state["global_step"] == completion["optimizer_steps"] == spec["total_steps"],
        "Training step count differs",
    )
    require(
        state["epoch"] == spec["epochs"] + 1
        and state["group"] == 0
        and not state["stats"],
        "Final cursor differs",
    )
    identity = state["runtime_identity"]
    for filename, expected in config["source_hashes"].items():
        require(
            sha(Path(config["code_root"]) / filename) == expected,
            "Frozen source changed",
        )
    for key in ("cohort", "split"):
        require(
            sha(config[f"{key}_path"]) == config[f"{key}_sha256"],
            "Sealed input changed",
        )
    split = read(config["split_path"])["splits"]
    require(
        {key: len(value) for key, value in split.items()}
        == {"train": 125, "validation": 25, "test": 25},
        "Split coverage differs",
    )
    require(len(set(sum(split.values(), []))) == 175, "Split overlap detected")
    cohort = read(config["cohort_path"])
    require(
        cohort["summary"]["train_pages"] == 125
        and cohort["summary"]["train_lines"] == 5212,
        "Training cohort differs",
    )
    data_root = Path(config["dataset_root"])
    dataset = read(data_root / "manifest.json")
    references = {}
    for page in dataset["pages"]:
        reference = data_root / page["reference"]
        require(sha(reference) == page["reference_sha256"], "Reference changed")
        references[page["page_id"]] = reference.read_text()
    events = []
    for event_root in spec["event_roots"]:
        events.extend(
            json.loads(line)
            for line in (Path(event_root) / "events.jsonl").read_text().splitlines()
            if line.strip()
        )
    updates = [e for e in events if e["event"] == "optimizer_step"]
    validate_updates(
        updates,
        epochs=spec["epochs"],
        stage=spec["stage"],
        maximum=config["learning_rate"],
        minimum=config["minimum_learning_rate"],
    )
    initial_adapters = {
        e["initial_adapter_sha256"] for e in events if e["event"] == "runtime_loaded"
    }
    require(len(initial_adapters) == 1, "Initial adapter changed across continuations")
    checkpoints = {}
    for event_root in spec["event_roots"]:
        for path in (Path(event_root) / "checkpoints").glob("step-*/closed.json"):
            step = read(path)["global_step"]
            require(step not in checkpoints, "Duplicate closed checkpoint step")
            checkpoints[step] = path.parent
    require(
        set(checkpoints) == set(range(1, spec["total_steps"] + 1)),
        "Checkpoint coverage differs",
    )
    audited_checkpoints = []
    for step in sorted(checkpoints):
        row = audit_payload(torch, checkpoints[step], identity, config, step)
        row["path"] = str(checkpoints[step])
        audited_checkpoints.append(row)
        with (out / "progress.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "case": name,
                        "phase": "checkpoint",
                        "step": step,
                        "total": spec["total_steps"],
                    }
                )
                + "\n"
            )
    publish(out / f"{name}-checkpoints.json", audited_checkpoints)
    by_path = {r["path"]: r for r in audited_checkpoints}
    require(
        set(state["epoch_checkpoints"])
        == {str(e) for e in range(1, spec["epochs"] + 1)},
        "Epoch checkpoint coverage differs",
    )
    for epoch, checkpoint in state["epoch_checkpoints"].items():
        require(
            by_path[checkpoint]["global_step"] == 16 * int(epoch),
            "Epoch checkpoint step differs",
        )
    require(
        state["checkpoint"] == state["epoch_checkpoints"][str(spec["epochs"])],
        "Final checkpoint differs",
    )
    cursor = torch.load(state["rng_cursor"], map_location="cpu", weights_only=False)
    require(
        cursor["global_step"] == spec["total_steps"], "Final RNG cursor step differs"
    )
    check_rng(torch, cursor["rng"])
    cursor_sha = sha(state["rng_cursor"])
    baseline_root = Path(spec["baseline_root"])
    baseline = read(baseline_root / "baseline.json")
    require(
        read(root / "baseline.json") == baseline
        and baseline["runtime_identity"] == identity
        and baseline["optimizer_steps"] == 0,
        "Fresh baseline identity differs",
    )
    policy = InstructSamplingPolicy() if spec["exploratory"] else GenerationPolicy()
    per_page = {}
    all_diagnoses = []
    for partition in ("validation", "test"):
        summary = baseline[partition]
        rows, diagnoses = audit_evaluation(
            baseline_root / "evaluations" / f"epoch-00-{partition}",
            summary,
            {p: references[p] for p in split[partition]},
            policy,
        )
        per_page[f"baseline-{partition}"] = rows
        all_diagnoses.extend(
            {"evaluation": f"baseline-{partition}", **d} for d in diagnoses
        )
    training = []
    validations = []
    manifest = None
    if spec["exploratory"]:
        manifest_path = Path(config["mix_root"]) / "manifest.json"
        prepared = read(Path(config["mix_root"]) / "completed.json")
        require(
            prepared["success"] is True
            and sha(manifest_path) == prepared["manifest_sha256"],
            "Mixed manifest seal differs",
        )
        manifest = read(manifest_path)
        require(
            manifest["audit"]["selection"] == "include-all"
            and manifest["audit"]["luisa_selected"] == 23774,
            "Literal LUISA inclusion differs",
        )
        images = {}
        expected_luisa = None
        for entry in manifest["epochs"]:
            require(
                len(entry["units"]) == 125
                and {u["page_id"] for u in entry["units"]} == set(split["train"]),
                "Mixed source coverage differs",
            )
            luisa = [r["line_id"] for u in entry["units"] for r in u["luisa"]]
            if entry["epoch"] <= 3:
                require(
                    len(luisa) == len(set(luisa)) == 23774,
                    "LUISA coverage lost or duplicated",
                )
                if expected_luisa is None:
                    expected_luisa = set(luisa)
                require(set(luisa) == expected_luisa, "LUISA set differs across epochs")
            else:
                require(
                    not luisa and not any(u["synthetic"] for u in entry["units"]),
                    "External data in consolidation",
                )
            for unit in entry["units"]:
                for row in unit["luisa"] + unit["synthetic"]:
                    path = row["image"]
                    require(
                        path not in images or images[path] == row["image_sha256"],
                        "Conflicting image hashes",
                    )
                    images[path] = row["image_sha256"]

        def image_check(item):
            path, expected = item
            require(sha(path) == expected, "Mixed image changed")

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(image_check, images.items()))
        require(len(images) == 25765, "Unique mixed image coverage differs")
    for epoch in range(1, spec["epochs"] + 1):
        train = training_epoch_receipt(
            root, events, epoch, exploratory=spec["exploratory"]
        )
        require(
            train["epoch"] == epoch
            and train["optimizer_steps"] == 16
            and train["source_pages"] == 125
            and math.isfinite(train["mean_loss"]),
            "Invalid training epoch",
        )
        expected_samples = {"pages": 125, "lines": 5212, "combined": 5337}.get(name)
        if manifest is not None:
            entry = manifest["epochs"][epoch - 1]
            expected_samples = entry["samples"]
            require(
                train["family_counts"]
                == {
                    "count_page": 125,
                    "count_berrutti_lines": 5212,
                    "count_luisa": 23774 if epoch <= 3 else 0,
                    "count_synthetic": entry["synthetic_pages"],
                },
                "Training family coverage differs",
            )
        require(
            train["samples"] == expected_samples, "Incomplete training image coverage"
        )
        training.append(train)
        summary = read(root / "epochs" / f"epoch-{epoch:02d}.json")["validation"]
        require(
            summary["epoch"] == epoch
            and summary["stage"] == spec["stage"]
            and summary["split"] == "validation",
            "End-of-epoch evaluation differs",
        )
        validations.append(summary)
    best = selected_epoch(validations, spec["epochs"])
    require(
        completion["best_epoch"] == best["epoch"]
        and completion["best_validation_cer"] == best["cer"],
        "Checkpoint selection differs from full validation",
    )
    require(
        completion["selected_checkpoint"]
        == state["epoch_checkpoints"][str(best["epoch"])],
        "Selected checkpoint path differs",
    )
    test_files = list((root / "epochs").glob("*-selected-test.json"))
    require(len(test_files) == 1, "Test was evaluated for more than one checkpoint")
    test = read(test_files[0])["test"]
    require(
        test["epoch"] == best["epoch"]
        and test["split"] == "test"
        and completion["test_cer_at_selected_epoch"] == test["cer"]
        and completion["selected_test_failed_pages"] == test["failed_pages"]
        and (
            not spec["exploratory"]
            or completion["result_complete"] == test["all_pages_completed"]
        ),
        "Selected test receipt differs",
    )
    require(
        [p.name for p in (root / "evaluations").glob("*-test")]
        == [f"epoch-{best['epoch']:02d}-test"],
        "Unexpected test evaluation directory",
    )
    summaries = validations + [test]
    if spec["exploratory"]:
        markers = [100 * e + g for e in range(1, 5) for g in (4, 8, 12)]
        for marker in markers:
            row = read(root / "epochs" / f"epoch-{marker:02d}.json")
            require(
                row["optimizer_steps"] == (marker // 100 - 1) * 16 + marker % 100,
                "Progress checkpoint cursor differs",
            )
            require(
                row["validation"]["epoch"] == marker
                and row["validation"]["split"] == "validation",
                "Progress evaluation differs",
            )
            summaries.append(row["validation"])
    for summary in summaries:
        epoch, partition = summary["epoch"], summary["split"]
        checkpoint = (
            state["epoch_checkpoints"][str(epoch)]
            if epoch < 100
            else str(checkpoints[(epoch // 100 - 1) * 16 + epoch % 100])
        )
        key = f"epoch-{epoch:02d}-{partition}"
        rows, diagnoses = audit_evaluation(
            root / "evaluations" / key,
            summary,
            {p: references[p] for p in split[partition]},
            policy,
            by_path[checkpoint]["adapter_sha256"],
            identity,
        )
        per_page[key] = rows
        all_diagnoses.extend({"evaluation": key, **d} for d in diagnoses)
    require(
        read(root / "state.json") == state
        and read(root / "completed.json") == completion,
        "Training outputs changed during audit",
    )
    publish(out / f"{name}-per-page.json", per_page)
    publish(out / f"{name}-attempt-diagnosis.json", all_diagnoses)
    return {
        "name": name,
        "success": True,
        "execution_verified": True,
        "result_complete": test["all_pages_completed"],
        "exploratory": spec["exploratory"],
        "generation_policy": policy.metadata(),
        "initial_adapter_sha256": next(iter(initial_adapters)),
        "runtime_identity": identity,
        "model_revision": config["revision"],
        "seed": config["seed"],
        "epochs": spec["epochs"],
        "optimizer_steps": spec["total_steps"],
        "all_checkpoint_payloads_verified": len(audited_checkpoints),
        "baseline": baseline,
        "training": training,
        "validations": validations,
        "selected_epoch": best["epoch"],
        "test": test,
        "rng_cursor_sha256": cursor_sha,
        "mixed_images_rehashed": 25765 if spec["exploratory"] else 0,
        "evaluations_independently_rescored": len(per_page),
        "pages_independently_rescored": 25 * len(per_page),
        "no_training_or_generations": True,
        "per_page": per_page,
    }


def descriptive_comparisons(results, definition):
    comparisons = {}
    for split in ("validation", "test"):
        rows = {}
        for name, result in results.items():
            epoch = result["selected_epoch"]
            rows[name] = result["per_page"][f"epoch-{epoch:02d}-{split}"]
        rows["baseline"] = results["pages"]["per_page"][f"baseline-{split}"]
        rows["experiment4_baseline"] = results["experiment4"]["per_page"][
            f"baseline-{split}"
        ]
        pairs = [
            ("combined", "pages"),
            ("lines", "pages"),
            ("combined", "lines"),
            ("pages", "baseline"),
            ("combined", "baseline"),
            ("lines", "baseline"),
            ("experiment4", "experiment4_baseline"),
        ]
        for left, right in pairs:
            key = f"{split}:{left}-{right}"
            comparisons[key] = {}
            for by_roll in (False, True):
                comparisons[key]["roll" if by_roll else "page"] = paired_difference(
                    rows[left],
                    rows[right],
                    resamples=definition["bootstrap_resamples"],
                    seed=definition["statistics_seed"],
                    by_roll=by_roll,
                )
        if split == "test":
            common = set.intersection(
                *[
                    {r["page_id"] for r in rows[n] if r["status"] == "complete"}
                    for n in ("pages", "lines", "combined", "baseline")
                ]
            )
            comparisons["test:common_complete_pages_sensitivity"] = {
                "pages": len(common),
                "not_primary_endpoint": True,
                "models": {
                    n: {
                        "cer": sum(
                            r["char_errors"] for r in rows[n] if r["page_id"] in common
                        )
                        / sum(
                            r["char_count"] for r in rows[n] if r["page_id"] in common
                        ),
                        "char_errors": sum(
                            r["char_errors"] for r in rows[n] if r["page_id"] in common
                        ),
                    }
                    for n in ("pages", "lines", "combined", "baseline")
                },
            }
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--definition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(
        bool(os.environ.get("SLURM_JOB_ID")), "Checkpoint audit must run under Slurm"
    )
    os.umask(0o077)
    args.output.mkdir(exist_ok=False)
    started = time.monotonic()
    try:
        definition = read(args.definition)
        scheduler = [
            line.split("|")
            for line in (args.output.parent / "training-slurm.txt")
            .read_text()
            .splitlines()
        ]
        for spec in definition["cases"]:
            require(
                any(
                    row[:3] == [str(spec["final_job"]), "COMPLETED", "0:0"]
                    for row in scheduler
                ),
                "Final training segment must be Slurm COMPLETED with exit zero",
            )
            root = Path(spec["run_root"])
            validate_completion_header(
                read(root / "completed.json"), exploratory=spec["exploratory"]
            )
            events = [
                json.loads(line)
                for source in spec["event_roots"]
                for line in (Path(source) / "events.jsonl").read_text().splitlines()
                if line.strip()
            ]
            for epoch in range(1, spec["epochs"] + 1):
                training_epoch_receipt(
                    root, events, epoch, exploratory=spec["exploratory"]
                )
        import torch

        torch.set_num_threads(4)
        results = {}
        for spec in definition["cases"]:
            result = audit_case(spec, definition, args.output, torch)
            results[spec["name"]] = result
            publish(
                args.output / f"{spec['name']}-summary.json",
                {k: v for k, v in result.items() if k != "per_page"},
            )
        require(
            len({r["initial_adapter_sha256"] for r in results.values()}) == 1,
            "Common fresh initialization differs",
        )
        require(
            len(
                {
                    r["runtime_identity"]
                    for k, r in results.items()
                    if k != "experiment4"
                }
            )
            == 1,
            "Primary arms used different runtimes",
        )
        publish(
            args.output / "comparisons.json",
            descriptive_comparisons(results, definition),
        )
        publish(
            args.output / "completed.json",
            {
                "success": True,
                "receipt_kind": "independent_finetuning_audit",
                "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                "optimizer_updates_performed": 0,
                "generations_performed": 0,
                "cases_verified": list(results),
                "checkpoints_verified": sum(
                    r["all_checkpoint_payloads_verified"] for r in results.values()
                ),
                "pages_independently_rescored": sum(
                    r["pages_independently_rescored"] for r in results.values()
                ),
                "training_complete": True,
                "all_tests_complete": all(
                    r["result_complete"] for r in results.values()
                ),
                "lines_test_incomplete_preserved": not results["lines"][
                    "result_complete"
                ],
                "exploratory_luisa_kept_separate": True,
                "seconds": time.monotonic() - started,
            },
        )
    except Exception as exc:
        publish(
            args.output / "failed.json",
            {
                "success": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "seconds": time.monotonic() - started,
            },
        )
        raise


if __name__ == "__main__":
    main()

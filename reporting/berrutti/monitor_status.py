"""Bounded, read-only Slurm snapshots for the three Berrutti experiments."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

SSH = (
    "ssh",
    "-o",
    "IdentityAgent=none",
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=10",
    "ruche",
)


def configure_cluster(cluster: str) -> None:
    """Use a known SSH alias; scheduler IDs are local to each cluster."""
    global SSH
    aliases = {"Ruche": "ruche", "ClusterUY": "clusteruy"}
    if cluster not in aliases:
        raise ValueError("Monitor cluster must be Ruche or ClusterUY")
    SSH = (*SSH[:-1], aliases[cluster])


def remote(command: str) -> str:
    result = subprocess.run(
        [*SSH, command], capture_output=True, text=True, timeout=40, check=True
    )
    return result.stdout


def event_summary(payload: str) -> dict:
    """Read metrics and progress; exclude OCR, targets, paths and tracebacks."""
    if not payload.strip():
        events = []
    else:
        try:
            record = json.loads(payload)
        except json.JSONDecodeError:
            events = [
                json.loads(line)
                for line in payload.splitlines()
                if line.lstrip().startswith("{")
            ]
        else:
            if not isinstance(record, dict):
                raise ValueError("Expected an event object or JSONL event stream")
            events = [record]
    summary = {
        "evaluations": [],
        "latest_step": None,
        "latest_progress": None,
        "latest_training_epoch": None,
        "completed": False,
        "diagnosis": None,
        "selection": None,
    }
    for event in events:
        name = event.get("event")
        if name == "evaluation":
            summary["evaluations"].append(
                {
                    key: event[key]
                    for key in (
                        "epoch",
                        "split",
                        "cer",
                        "wer",
                        "pages",
                        "seconds",
                        "stage",
                        "failed_pages",
                        "retried_pages",
                        "all_pages_completed",
                        "generation_stop_counts",
                    )
                    if key in event
                }
            )
        elif name == "optimizer_step":
            summary["latest_step"] = {
                key: event.get(key)
                for key in ("stage", "epoch", "page", "total", "loss", "lr")
            }
            summary["latest_progress"] = {"phase": "training"}
        elif name == "sample_completed":
            summary["latest_progress"] = {
                "phase": "training",
                **{
                    key: event[key]
                    for key in (
                        "epoch",
                        "stage",
                        "samples_completed",
                        "source_page",
                        "representation",
                    )
                    if key in event
                },
            }
        elif name == "eval_progress":
            summary["latest_progress"] = {
                key: event.get(key) for key in ("epoch", "split", "completed", "total")
            }
        elif name == "training_epoch":
            summary["latest_training_epoch"] = {
                key: event[key]
                for key in (
                    "stage",
                    "epoch",
                    "mean_loss",
                    "seconds",
                    "source_pages",
                    "samples",
                    "supervised_tokens",
                    "visual_tokens",
                    "optimizer_steps",
                    "allocated_gpu_seconds",
                )
                if key in event
            }
        elif name == "completed":
            summary["completed"] = True
            if "result_complete" in event:
                summary["result_complete"] = event["result_complete"]
            if event.get("best_epoch") is not None:
                summary["selection"] = {
                    key: event[key]
                    for key in (
                        "best_epoch",
                        "best_validation_cer",
                        "test_cer_at_selected_epoch",
                        "selected_test_failed_pages",
                    )
                    if key in event
                }
        elif name in {"phase", "comparison_phase"}:
            summary["latest_progress"] = {"phase": event["phase"]}
        elif name in {"runtime_loaded", "projection", "training_memory"}:
            summary["latest_progress"] = {"phase": name}
        elif name in {"failed", "comparison_failed"}:
            failure = {
                key: event["diagnosis"][key]
                for key in ("phase", "error_type", "success", "gpu_runtime_success")
                if key in event["diagnosis"]
            }
            summary["diagnosis"] = {**(summary["diagnosis"] or {}), **failure}
            summary["latest_progress"] = {
                "phase": event["diagnosis"]["phase"],
                "failed": True,
            }
        elif name == "gpu_probe":
            summary["latest_progress"] = {
                "phase": "runtime" if event["runtime"] else "visibility",
                "success": event["success"],
            }
            summary["diagnosis"] = {
                "success": event["success"],
                "cuInit": event.get("cuInit"),
                "nvmlInit": event.get("nvmlInit"),
                "cuda_device_count": event.get("torch", {}).get("device_count"),
                "generation_guards": event.get("generation_guards"),
            }
    return summary


def scheduler_id(value):
    value = str(value)
    if not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", value):
        raise ValueError("Expected a numeric Slurm job or array element")
    return int(value) if value.isdigit() else value


def job_states(ids: list[int | str]) -> dict[int | str, str]:
    if not ids:
        return {}
    selector = ",".join(str(scheduler_id(job)) for job in ids)
    states = {}
    for line in remote(f"squeue -r -j {selector} -h -o '%i|%T'").splitlines():
        job_id, state = line.split("|", 1)
        states[scheduler_id(job_id)] = state
    accounting = remote(
        f"sacct -j {selector} -n -P -X --format=JobID%40,State 2>/dev/null || true"
    )
    for line in accounting.splitlines():
        parts = line.split("|")
        if len(parts) >= 2 and re.fullmatch(r"[0-9]+(?:_[0-9]+)?", parts[0].strip()):
            states.setdefault(scheduler_id(parts[0].strip()), parts[1].split()[0])
    return states


def pending_queue(ids: list[int | str]) -> dict[int | str, dict]:
    """Explain waiting without exposing scheduler IDs, nodes or free-form text."""
    if not ids:
        return {}
    selector = ",".join(str(scheduler_id(job)) for job in ids)
    payload = remote(
        "TZ=UTC SLURM_TIME_FORMAT='%Y-%m-%dT%H:%M:%S' "
        f"squeue -r -j {selector} -h -o '%i|%T|%r|%S|%Q'"
    )
    result = {}
    for line in payload.splitlines():
        job, state, reason, estimate, priority = line.split("|", 4)
        if state.strip() != "PENDING":
            continue
        reason = reason.strip().split(",", 1)[0]
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", reason):
            reason = "Other"
        try:
            start = datetime.fromisoformat(estimate.strip())
        except ValueError:
            start = None
        result[scheduler_id(job.strip())] = {
            "reason": reason,
            "estimated_start_utc": (
                start.replace(tzinfo=timezone.utc).isoformat() if start else None
            ),
            "start_is_estimate": True,
            "priority": int(priority.strip()),
        }
    return result


def performance_summary(payload: str, state: str) -> dict:
    """Whitelisted optimization receipts; never a training arm or complete CER."""
    result = {"state": state, "completed": False, "latest_progress": None}
    allowed = (
        "seconds",
        "reference_seconds",
        "cache_fill_seconds",
        "cache_hit_seconds",
        "identical_loss_and_gradients",
        "identical_tokens",
        "generated_tokens",
        "reference_status",
        "candidate_status",
        "failed_pages",
        "retried_pages",
        "source_pages",
        "line_images",
        "workers",
        "bytes",
        "optimizer_steps",
        "cuda_device_count",
        "pixels_preserved_exactly",
        "labels_modified",
        "full_generated_pages",
        "not_complete_validation",
        "success",
        "diagnostic_only",
        "optimization_accepted",
        "training_identity_passed",
        "generation_checked",
        "representation",
        "loss_identical",
        "loss_absolute_difference",
        "different_gradients",
        "maximum_gradient_absolute_difference",
        "maximum_gradient_relative_l2",
        "gpu_check",
        "berrutti_training",
        "representative_27b_assay",
        "same_pinned_weights_as_ruche",
        "shards",
        "checkpoint_reload",
        "optimizer_steps_before",
        "optimizer_steps_after",
        "training_committed",
    )
    try:
        rows = [json.loads(payload)] if payload.strip() else []
    except json.JSONDecodeError:
        rows = [
            json.loads(line)
            for line in payload.splitlines()
            if line.lstrip().startswith("{")
        ]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Expected performance receipt objects")
    for row in rows:
        if row.get("event") in {"failed", "comparison_failed"}:
            diagnosis = row["diagnosis"]
            result["latest_progress"] = {
                "phase": diagnosis["phase"],
                "error_type": diagnosis["error_type"],
                "success": False,
            }
            result["completed"] = False
            continue
        phase = row.get("event", row.get("phase"))
        if phase == "phase":
            phase = row.get("phase", phase)
        result["latest_progress"] = {
            "phase": phase,
            **{key: row[key] for key in allowed if key in row},
        }
        if phase == "training_identity":
            result.setdefault("training_checks", []).append(result["latest_progress"])
        elif phase == "identity_control":
            result.setdefault("identity_controls", []).append(result["latest_progress"])
        elif phase == "generation_identity":
            result["generation_check"] = result["latest_progress"]
        if phase in {
            "performance_assay_completed",
            "crop_cache_completed",
            "qwen_imported",
            "runtime_bundle_completed",
            "performance_identity_diagnostic_completed",
            "runner_smoke_completed",
            "preflight_completed",
            "host_cuda",
            "runtime_and_data_verified",
            "model_download_identity_verified",
            "synthetic_model_runtime_completed",
            "memory_recovery_completed",
        }:
            result["completed"] = (
                state == "COMPLETED"
                and row.get("success", row.get("complete", False)) is True
            )
            if phase == "memory_recovery_completed":
                result["completed"] = result["completed"] and (
                    row.get("checkpoint_reload") is True
                    and row.get("generation_checked") is True
                    and row.get("training_committed") is True
                    and type(row.get("optimizer_steps_before")) is int
                    and type(row.get("optimizer_steps_after")) is int
                    and row["optimizer_steps_before"] >= 0
                    and row.get("optimizer_steps_after")
                    == row["optimizer_steps_before"] + 1
                )
    return result


def arm_verification_summary(payload: str, state: str) -> dict:
    """Read an independent arm audit without treating it as training."""
    result = {"state": state, "completed": False, "latest_progress": None}
    if not payload.strip():
        return result
    row = json.loads(payload)
    required = (
        "success",
        "execution_verified",
        "diagnostic_only",
        "committed_updates_contiguous_without_repetition",
        "all_epoch_checkpoint_sha256_and_crc_valid",
        "source_hashes_match",
        "baseline_identical",
        "selection_uses_complete_validation",
        "test_only_selected_checkpoint",
        "generation_attempt_policy_preserved",
    )
    result["latest_progress"] = {
        "phase": "arm_verification_completed",
        **{
            key: row[key]
            for key in (
                "success",
                "diagnostic_only",
                "execution_verified",
                "result_complete",
                "test_cer_provisional",
                "epochs_verified",
                "optimizer_updates_performed",
            )
            if key in row
        },
    }
    result["completed"] = (
        state == "COMPLETED"
        and all(row.get(key) is True for key in required)
        and row.get("optimizer_updates_performed") == 0
    )
    return result


def read_performance_checks(checks: list[dict], default_cluster: str) -> dict:
    """Query each destination separately; identical Slurm IDs must never mix."""
    result = {}
    clusters = dict.fromkeys(spec.get("cluster", default_cluster) for spec in checks)
    try:
        for cluster in clusters:
            configure_cluster(cluster)
            group = [
                spec
                for spec in checks
                if spec.get("cluster", default_cluster) == cluster
            ]
            ids = [scheduler_id(spec["job_id"]) for spec in group]
            try:
                states = job_states(ids)
                queued = pending_queue(
                    [job for job in ids if states.get(job) == "PENDING"]
                )
                for spec in group:
                    path = PurePosixPath(spec["events_path"])
                    if not path.is_absolute():
                        raise ValueError("Performance receipt path must be absolute")
                    reader = (
                        "head -c 131072"
                        if spec.get("receipt_kind")
                        in {"arm_verification", "performance_diagnostic"}
                        or path.suffix == ".json"
                        else "tail -n 64"
                    )
                    payload = remote(
                        f"{reader} {shlex.quote(str(path))} 2>/dev/null || true"
                    )
                    job = scheduler_id(spec["job_id"])
                    summary = (
                        arm_verification_summary
                        if spec.get("receipt_kind") == "arm_verification"
                        else performance_summary
                    )
                    try:
                        check = summary(payload, states.get(job, "UNKNOWN"))
                    except (
                        json.JSONDecodeError,
                        ValueError,
                        TypeError,
                        KeyError,
                    ) as error:
                        # A partial or malformed receipt cannot approve its check
                        # or prevent fresh snapshots of the other running jobs.
                        check = {
                            "state": states.get(job, "UNKNOWN"),
                            "completed": False,
                            "latest_progress": None,
                            "monitor_error": type(error).__name__,
                        }
                    check["cluster"] = cluster
                    if job in queued:
                        check["queue"] = queued[job]
                    result[spec["name"]] = check
            except (subprocess.SubprocessError, OSError) as error:
                for spec in group:
                    result[spec["name"]] = {
                        "cluster": cluster,
                        "state": "UNKNOWN",
                        "completed": False,
                        "monitor_error": type(error).__name__,
                    }
    finally:
        configure_cluster(default_cluster)
    return result


def saved_evaluations(payload: str) -> list[dict]:
    """Recover closed metric files when a bounded event tail loses early epochs."""
    decoder = json.JSONDecoder()
    events = []
    remaining = payload.strip()
    while remaining:
        record, end = decoder.raw_decode(remaining)
        remaining = remaining[end:].strip()
        epoch = record.get("epoch", -1)
        for split in ("validation", "test"):
            if split in record:
                events.append(
                    {
                        "event": "evaluation",
                        "epoch": epoch,
                        "split": split,
                        **record[split],
                    }
                )
    return event_summary("\n".join(json.dumps(event) for event in events))[
        "evaluations"
    ]


def merge_evaluations(current: list[dict], saved: list[dict]) -> list[dict]:
    records = {(row["epoch"], row["split"]): row for row in current}
    for row in saved:
        key = row["epoch"], row["split"]
        if key in records and records[key] != row:
            raise ValueError("Closed metrics disagree with the event tail")
        records[key] = row
    return sorted(
        records.values(), key=lambda row: (row["epoch"], row["split"] != "validation")
    )


def write_private(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def experiment_focus(coordination: dict, runs: dict) -> dict:
    plan = coordination["experiment_plan"]
    primary = {
        name: {"definition": definition, "state": "PLANNED", "evaluations": []}
        for name, definition in plan["arms"].items()
    }
    for spec in coordination.get("runs", []):
        if spec.get("protocol") != plan["protocol"] or spec["arm"] not in primary:
            raise ValueError("Monitor accepts only the three current Berrutti arms")
        if spec["name"] in runs:
            primary[spec["arm"]].update(runs[spec["name"]])
    return {"experiment_plan": plan, "primary_experiments": primary}


def resolve_continuation(spec: dict) -> dict:
    """Follow only the runner's bounded, explicit next-job receipt."""
    path = PurePosixPath(spec["run_root"]) / "continuation.json"
    if not path.is_absolute():
        raise ValueError("Continuation root must be absolute")
    payload = remote(f"head -c 4096 {shlex.quote(str(path))} 2>/dev/null || true")
    if not payload.strip():
        return spec
    receipt = json.loads(payload)
    return {**spec, "job_id": scheduler_id(receipt["job_id"])}


def read_run(spec: dict, states: dict, queued: dict | None = None) -> dict:
    root = PurePosixPath(spec["run_root"])
    if not root.is_absolute():
        raise ValueError("Run root must be absolute")
    events = remote(
        f"tail -n 256 {shlex.quote(str(root / 'events.jsonl'))} 2>/dev/null || true"
    )
    run = {
        "state": states.get(scheduler_id(spec["job_id"]), "UNKNOWN"),
        **event_summary(events),
    }
    paths = [
        root / "baseline.json",
        *[
            root / "epochs" / f"epoch-{epoch:02d}.json"
            for epoch in (
                *range(1, spec.get("epochs", 3) + 1),
                *spec.get("progress_markers", []),
            )
        ],
        *[
            root / "epochs" / f"epoch-{epoch:02d}-selected-test.json"
            for epoch in range(1, spec.get("epochs", 3) + 1)
        ],
    ]
    quoted = " ".join(shlex.quote(str(path)) for path in paths)
    saved = remote(
        f'for metric in {quoted}; do if [ -f "$metric" ]; then head -c 16384 "$metric"; fi; done'
    )
    run["evaluations"] = merge_evaluations(run["evaluations"], saved_evaluations(saved))
    run["completed"] = run["state"] == "COMPLETED" and run["completed"]
    if queued and scheduler_id(spec["job_id"]) in queued:
        run["queue"] = queued[scheduler_id(spec["job_id"])]
    return run


def read_additional_experiments(specs: list[dict], default_cluster: str) -> dict:
    """Keep authorized extensions separate from the sealed three-arm comparison."""
    result = {}
    try:
        for spec in specs:
            cluster = spec.get("cluster", default_cluster)
            configure_cluster(cluster)
            entry = {
                "definition": spec["definition"],
                "cluster": cluster,
                "state": "PLANNED",
                "completed": False,
                "evaluations": [],
            }
            baseline = spec.get("baseline")
            launched = [spec] if spec.get("job_id") else []
            resolved = [resolve_continuation(run) for run in launched]
            jobs = resolved + ([baseline] if baseline else [])
            ids = [scheduler_id(run["job_id"]) for run in jobs]
            states = job_states(ids)
            queued = pending_queue([job for job in ids if states.get(job) == "PENDING"])
            if resolved:
                entry.update(read_run(resolved[0], states, queued))
            if baseline:
                entry["baseline"] = read_run(baseline, states, queued)
            result[spec["name"]] = entry
    finally:
        configure_cluster(default_cluster)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordination", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    coordination = json.loads(args.coordination.read_text())
    cluster = coordination.get("cluster", "Ruche")
    configure_cluster(cluster)
    experiment_focus(coordination, {})
    specs = [resolve_continuation(spec) for spec in coordination.get("runs", [])]
    shared = coordination.get("shared_baseline")
    if len({spec["arm"] for spec in specs}) != len(specs):
        raise ValueError("At most one run per current arm")
    checks = coordination.get("performance_checks", [])
    ids = [spec["job_id"] for spec in [*specs, *([shared] if shared else [])]]
    states = job_states(ids)
    queued = pending_queue([job for job in ids if states.get(job) == "PENDING"])
    runs = {spec["name"]: read_run(spec, states, queued) for spec in specs}
    snapshot = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "cluster": cluster,
        "runs": runs,
        **experiment_focus(coordination, runs),
    }
    if shared:
        snapshot["shared_baseline"] = read_run(shared, states, queued)
    if checks:
        snapshot["performance_checks"] = read_performance_checks(checks, cluster)
    if coordination.get("additional_experiments"):
        snapshot["additional_experiments"] = read_additional_experiments(
            coordination["additional_experiments"], cluster
        )
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    write_private(output / "latest.json", snapshot)
    print(
        json.dumps(
            {
                "checked_at_utc": snapshot["checked_at_utc"],
                "primary_states": {
                    name: run["state"]
                    for name, run in snapshot["primary_experiments"].items()
                },
            }
        )
    )


if __name__ == "__main__":
    main()

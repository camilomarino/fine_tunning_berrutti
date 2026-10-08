"""Read-only Ruche status snapshot for a local two-hour user timer.

Only bounded Slurm metadata and event tails cross SSH. No OCR text is read.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

RUNS = (
    ("full", "full_job", "full"),
    ("curriculum_sanity", "curriculum_sanity_job", "curriculum-sanity-v1"),
    ("curriculum", "curriculum_full_job", "curriculum-v1"),
)
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


def remote(command: str) -> str:
    result = subprocess.run(
        [*SSH, command], capture_output=True, text=True, timeout=40, check=True
    )
    return result.stdout


def event_summary(payload: str) -> dict:
    events = [json.loads(line) for line in payload.splitlines() if line.startswith("{")]
    evaluations = []
    latest_step = None
    latest_progress = None
    latest_training_epoch = None
    completed = False
    diagnosis = None
    for event in events:
        name = event.get("event")
        if name == "evaluation":
            evaluations.append(
                {
                    key: event[key]
                    for key in ("epoch", "split", "cer", "wer", "pages", "seconds")
                }
                | {
                    key: event[key]
                    for key in (
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
            latest_step = {
                key: event.get(key)
                for key in ("stage", "epoch", "page", "total", "loss", "lr")
            }
        elif name == "eval_progress":
            latest_progress = {
                key: event.get(key) for key in ("epoch", "split", "completed", "total")
            }
        elif name == "training_epoch":
            latest_training_epoch = {
                key: event.get(key)
                for key in ("stage", "epoch", "mean_loss", "seconds")
            }
        elif name == "completed":
            completed = True
        elif name == "legacy_probe_started":
            latest_progress = {"phase": "reproducing_initial_failure"}
        elif name == "legacy_probe_result":
            diagnosis = {
                key: event[key]
                for key in (
                    "generated_tokens",
                    "recognized_model_eos",
                    "legacy_would_raise",
                    "repeat_period_tokens",
                    "reference_tokens",
                    "seconds",
                )
            }
        elif name == "diagnostic_completed":
            completed = True
    return {
        "evaluations": evaluations,
        "latest_step": latest_step,
        "latest_progress": latest_progress,
        "latest_training_epoch": latest_training_epoch,
        "completed": completed,
        "diagnosis": diagnosis,
    }


def job_states(ids: list[int]) -> dict[int, str]:
    selector = ",".join(map(str, ids))
    queue = remote(f"squeue -j {selector} -h -o '%i|%T'")
    states = {}
    for line in queue.splitlines():
        job_id, state = line.split("|", 1)
        states[int(job_id)] = state
    accounting = remote(
        f"sacct -j {selector} -n -P -X --format=JobIDRaw,State 2>/dev/null || true"
    )
    for line in accounting.splitlines():
        parts = line.split("|", 1)
        if len(parts) == 2 and parts[0].isdigit():
            states.setdefault(int(parts[0]), parts[1].split()[0])
    return states


def alerts(previous: dict, current: dict) -> list[str]:
    messages = []
    for name, run in current["runs"].items():
        old = previous.get("runs", {}).get(name, {})
        seen = {(item["epoch"], item["split"]) for item in old.get("evaluations", [])}
        for item in run["evaluations"]:
            if (item["epoch"], item["split"]) not in seen:
                split = "validación" if item["split"] == "validation" else item["split"]
                messages.append(
                    f"{name}: {split} época {item['epoch']} "
                    f"CER {item['cer']:.2%} ({item['pages']} páginas)"
                )
        if run["state"] != old.get("state") and run["state"] in {
            "FAILED",
            "TIMEOUT",
            "CANCELLED",
            "COMPLETED",
        }:
            messages.append(f"{name}: Slurm {run['state']}")
    return messages


def write_private(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordination", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    coordination = json.loads(args.coordination.read_text(encoding="utf-8"))
    root = PurePosixPath(coordination["remote_root"])
    if not root.is_absolute():
        raise ValueError("Remote experiment root must be absolute")
    plans = [
        (name, int(coordination[key]), root / "runs" / folder)
        for name, key, folder in RUNS
    ]
    for diagnostic_name in (
        "generation_diagnostic",
        "generation_recovery",
        "training_sanity",
    ):
        diagnostic = coordination.get(diagnostic_name)
        if not diagnostic:
            continue
        diagnostic_root = PurePosixPath(diagnostic["run_root"])
        if not diagnostic_root.is_absolute():
            raise ValueError("Diagnostic root must be absolute")
        plans.append((diagnostic_name, int(diagnostic["job_id"]), diagnostic_root))
    states = job_states([job for _, job, _ in plans])
    runs = {}
    for name, job_id, run_root in plans:
        event_path = run_root / "events.jsonl"
        event_lines = remote(
            f"tail -n 256 {shlex.quote(str(event_path))} 2>/dev/null || true"
        )
        runs[name] = {
            "state": states.get(job_id, "UNKNOWN"),
            **event_summary(event_lines),
        }
        if (
            name in {"generation_diagnostic", "generation_recovery"}
            and runs[name]["state"] == "COMPLETED"
        ):
            runs[name]["completed"] = True
    snapshot = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs": runs,
    }
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output, 0o700)
    latest = output / "latest.json"
    previous = json.loads(latest.read_text()) if latest.exists() else {}
    with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(snapshot, ensure_ascii=False, sort_keys=True) + "\n")
    write_private(latest, snapshot)
    messages = alerts(previous, snapshot) if previous else []
    if messages:
        with (output / "alerts.jsonl").open("a", encoding="utf-8") as stream:
            for message in messages:
                stream.write(
                    json.dumps(
                        {
                            "checked_at_utc": snapshot["checked_at_utc"],
                            "message": message,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    print(json.dumps({"checked_at_utc": snapshot["checked_at_utc"], "runs": runs}))


if __name__ == "__main__":
    main()

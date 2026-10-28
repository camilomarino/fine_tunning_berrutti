"""Read live stage progress, including workers started before stage reporting."""

from __future__ import annotations

import json
import re
import time
import zipfile
from datetime import datetime
from pathlib import Path

from archive_berrutti import library, manage as core

STEPS = {"verify": 1, "download": 1, "extract": 2, "count": 2, "render": 3}


def recent_records(path):
    """Bounded read; tolerate a worker's incomplete last append."""
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 65536))
            lines = stream.read().splitlines(keepends=True)
    except FileNotFoundError:
        return []
    records = []
    for line in lines:
        if not line.endswith(b"\n"):
            continue
        try:
            records.append(json.loads(line))
        except (ValueError, UnicodeError):
            continue
    return records


def timestamp(value):
    try:
        return datetime.fromisoformat(value).timestamp()
    except (ValueError, TypeError):
        return 0


def estimate(samples, done, total, now):
    """Use recent work, excluding restarts, reused pages and stale samples."""
    window = []
    for instant, count in samples:
        if window and (count < window[-1][1] or instant < window[-1][0]):
            window = []
        if not window or count > window[-1][1]:
            window.append((instant, count))
    window = window[-50:]
    if len(window) < 3 or total is None or done >= total:
        return None, None
    elapsed = window[-1][0] - window[0][0]
    units = window[-1][1] - window[0][1]
    if elapsed < 2 or units <= 0 or done < window[-1][1]:
        return None, None
    interval = elapsed / (len(window) - 1)
    if now - window[-1][0] > max(30, 3 * interval):
        return None, None
    rate = units / elapsed
    return rate, (total - done) / rate


def locate(roll, local, final, job):
    stage = job.get("stage")
    if stage:
        for index, (_, row) in enumerate(roll["files"]):
            if row["key"] == stage["key"]:
                return stage["name"], index
    # Existing workers continue unmodified; their message and durable outputs
    # identify the stage and original being processed.
    transfer = job.get("transfer") or {}
    message = job.get("message", "")
    index = next(
        (
            i
            for i, (_, row) in enumerate(roll["files"])
            if row["key"] == transfer.get("key")
            or row["name"] in message
            or final[i]["output"]
            and final[i]["output"] in message
        ),
        0,
    )
    numbered = re.search(r"ZIP (\d+)/\d+", message)
    if numbered:
        index = min(len(local) - 1, int(numbered[1]) - 1)
    if transfer or "descargando" in message.lower():
        return "download", index
    if "convirtiendo" in message.lower():
        return "render", index
    if "extrayendo" in message.lower():
        return "extract", index
    if message.startswith("["):
        return (
            "download"
            if local[index]["download"] != "downloaded"
            else "extract"
            if local[index]["pdf"] != "extracted"
            else "count"
        ), index
    return "verify", index


def observe(progress, job_id, windows):
    """Fill an ETA from live observations when durable rate samples are absent."""
    if not progress or progress["stage"] not in {"download", "extract", "render"}:
        return
    key = (job_id, progress["stage"], progress["zip_index"])
    done = progress.get("bytes_done", progress["done"])
    total = progress.get("bytes_total", progress["total"])
    samples = windows.setdefault(key, [])
    now = time.time()
    if not samples or samples[-1][1] != done:
        samples.append((now, done))
        del samples[:-50]
    if progress["eta_seconds"] is None:
        rate, eta = estimate(samples, done, total, now)
        progress.update(rate=rate, eta_seconds=eta)


def snapshot(root, roll, local, final, job):
    if job["phase"] == "queued":
        return None
    stage, index = locate(roll, local, final, job)
    row, item, output = roll["files"][index][1], local[index], final[index]
    result = {
        "stage": stage,
        "step": STEPS[stage],
        "zip_index": index + 1,
        "zip_total": len(local),
        "done": 0,
        "total": None,
        "unit": None,
        "percent": None,
        "rate": None,
        "eta_seconds": None,
    }
    samples = []
    started = job.get("stage", {}).get("started", job.get("created", 0))
    if stage == "download":
        result.update(
            done=row["bytes"]
            if item["download"] == "downloaded"
            else item["partial_bytes"],
            total=row["bytes"],
            unit="bytes",
        )
        for record in recent_records(Path(job["job"]) / "progress.jsonl"):
            transfer = record.get("transfer") or {}
            instant = timestamp(record.get("updated"))
            if (
                transfer.get("key") == row["key"]
                and transfer.get("phase") == "downloading"
                and instant >= started
            ):
                samples.append((instant, transfer["bytes"]))
    elif stage == "extract":
        directory = library.entry_root(root, row)
        archive = core.zip_path(directory, library.local_entry(row))
        try:
            with zipfile.ZipFile(archive) as zipped:
                members = [
                    info
                    for info in zipped.infolist()
                    if not info.is_dir() and info.filename.lower().endswith(".pdf")
                ]
            pdf_dir = directory / "pdfs" / row["key"]
            paths = list(pdf_dir.glob("*.pdf"))
            sizes = [path.stat().st_size for path in paths]
            partials = []
            for path in pdf_dir.glob("*.pdf.part"):
                try:
                    partials.append(path.stat().st_size)
                except FileNotFoundError:
                    pass  # Published while the view was reading.
            result.update(
                done=len(paths),
                total=len(members),
                unit="PDF",
                bytes_done=sum(sizes) + sum(partials),
                bytes_total=sum(info.file_size for info in members),
            )
        except (FileNotFoundError, zipfile.BadZipFile):
            pass  # The worker records any extraction error in its job state.
    elif stage == "render":
        result.update(done=output["images"], total=output["expected"], unit="imágenes")
        if output["output"]:
            count = 0
            seen = set()
            for record in recent_records(Path(output["output"]) / "events.jsonl"):
                instant = timestamp(record.get("at_utc"))
                if (
                    record.get("event") == "page_complete"
                    and instant >= started
                    and record.get("page_id") not in seen
                ):
                    seen.add(record["page_id"])
                    count += 1
                    samples.append((instant, count))
            # Event counts are local to the bounded window, while 'done' also
            # includes earlier pages. Translate without treating them as new work.
            samples = [(at, output["images"] - count + value) for at, value in samples]
    if result["total"]:
        result["percent"] = min(100, 100 * result["done"] / result["total"])
    result["rate"], result["eta_seconds"] = estimate(
        samples, result["done"], result["total"], time.time()
    )
    return result

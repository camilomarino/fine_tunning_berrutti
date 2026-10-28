"""Small browser indexes produced on compute, read over SSH without allocations."""

from __future__ import annotations

import contextlib
import fcntl
import os
import sys
import threading
import time
import zipfile
from pathlib import Path

from archive_berrutti import agent, library, locations, manage as core, rolls, workflow


def roll_payload(root, roll):
    state = workflow.roll_status(root, roll)
    archives = [agent.artifacts(root, row) for _, row in roll["files"]]
    files, members = [], {}
    for entry in archives:
        entry_files = [entry["zip"], *entry["pdfs"]]
        if entry["partial_zip"]["present"]:
            entry_files.append(entry["partial_zip"])
        for conversion in entry["conversions"]:
            entry_files.extend(agent.image_files(conversion))
        files.extend({"key": entry["key"], **item} for item in entry_files)
        archive = Path(entry["zip"]["path"])
        if entry["zip"]["present"]:
            try:
                with zipfile.ZipFile(archive) as source:
                    members[str(archive)] = [
                        item.filename for item in source.infolist() if not item.is_dir()
                    ]
            except (OSError, zipfile.BadZipFile):
                pass
    resources = [
        {**item, "path": str(item["path"])} for item in locations.resources(root, roll)
    ]
    return {
        "schema_version": 1,
        "root": str(root),
        "roll": roll["roll"],
        "observed_at": core.now(),
        "state": state,
        "info": {**agent.roll_info(root, roll, status=state), "archives": archives},
        "resources": resources,
        "files": files,
        "zip_members": members,
    }


def publish(root, data, selected=None):
    """Atomic index; only selected roll details are rescanned during processing."""
    directory = root / "browse"
    directory.mkdir(parents=True, exist_ok=True)
    index_path = directory / "index.json"
    groups = (
        selected if selected is not None and index_path.exists() else rolls.group(data)
    )
    observed_ns = time.time_ns()
    payloads = [(roll, roll_payload(root, roll)) for roll in groups]
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        index = core.load(index_path) if index_path.exists() else {"items": {}}
        for roll, payload in payloads:
            previous = index["items"].get(roll["roll"], {})
            if previous.get("revision", 0) > observed_ns:
                continue
            relative = "rolls/" + core.digest(roll["roll"])[:20] + ".json"
            payload["revision"] = observed_ns
            library.replace_json(directory / relative, payload)
            index["items"][roll["roll"]] = {
                "revision": observed_ns,
                "path": "browse/" + relative,
                "state": payload["state"],
                "info": {
                    key: value
                    for key, value in payload["info"].items()
                    if key != "archives"
                },
                "file_counts": {},
            }
            for item in payload["files"]:
                key = "|".join(
                    (item["kind"], item.get("scope", ""), str(int(item["present"])))
                )
                counts = index["items"][roll["roll"]]["file_counts"]
                counts[key] = counts.get(key, 0) + 1
        valid = {roll["roll"] for roll in rolls.group(data)}
        index["items"] = {
            key: item for key, item in index["items"].items() if key in valid
        }
        index.update(
            schema_version=1,
            root=str(root),
            uid=os.getuid(),
            catalog=data,
            updated_at=core.now(),
            revision=time.time_ns(),
        )
        library.replace_json(index_path, index)
        library.replace_json(
            directory / "revision.json",
            {
                "schema_version": 1,
                "root": str(root),
                "revision": index["revision"],
                "updated_at": index["updated_at"],
            },
        )
    return {
        "rolls": len(index["items"]),
        "updated_at": index["updated_at"],
        "index": str(index_path),
    }


def maybe_publish(root, data, selected=None):
    from archive_berrutti import remote

    if not remote.runtime() and not (root / "browse/index.json").exists():
        return
    try:
        return publish(root, data, selected)
    except Exception as error:
        # A presentation index must not mask the outcome or fail the data job.
        print(
            f"No se pudo actualizar el índice del explorador: {error}",
            file=sys.stderr,
            flush=True,
        )


@contextlib.contextmanager
def reporting(root, data, roll):
    from archive_berrutti import remote

    enabled = bool(remote.runtime() or (root / "browse/index.json").exists())
    stopped = threading.Event()

    def report():
        while not stopped.wait(5):
            maybe_publish(root, data, [roll])

    thread = threading.Thread(target=report, daemon=True) if enabled else None
    if thread:
        thread.start()
    try:
        yield
    finally:
        if thread:
            stopped.set()
            thread.join()
            maybe_publish(root, data, [roll])

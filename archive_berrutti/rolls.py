"""Roll-level presentation of a library whose operations use individual ZIPs."""

from __future__ import annotations

from archive_berrutti import manage as core


def group(data):
    groups = {}
    for index, row in enumerate(data["files"], 1):
        key = core.roll_key(row["roll"]) if row["roll"] else row["key"]
        groups.setdefault(key, {"roll": row["roll"] or "sin número", "files": []})[
            "files"
        ].append((index, row))
    return list(groups.values())

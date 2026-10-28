"""One non-interactive JSON entry for datasets, examples, files and Archive jobs."""

from __future__ import annotations

import contextlib
import io
import json
import sqlite3
import subprocess
import sys

from archive_berrutti import agent as archive_agent, cli as archive_cli, manage as core
from data import access, manage

ARCHIVE_COMMANDS = {
    "list",
    "info",
    "files",
    "plan",
    "download",
    "complete",
    "remove",
    "job",
    "cancel",
    "refresh",
    "index",
}


def archive_arguments(arguments, where=None):
    """Use exactly the registered destination; do not silently fall back to local."""
    loc = access.location("archive_berrutti", where)
    if any(
        arg in {"--root", "--local", "--profile"}
        or arg.startswith(("--root=", "--profile="))
        for arg in arguments
    ):
        raise ValueError(
            "Elegir destino con --where; usar archive-berrutti para otras raíces/perfiles"
        )
    destination = ["--local"] if loc["host"] == "local" else ["--profile", loc["where"]]
    return [*destination, *arguments, "--root", loc["root"]]


def archive_result(command, arguments, where=None):
    # A common exact-ID spelling, while retaining Archive's multi-roll selector.
    if command in {"list", "info", "files", "plan", "download", "complete", "remove"}:
        arguments = [
            "--rolls"
            if arg == "--id"
            else "--rolls=" + arg[5:]
            if arg.startswith("--id=")
            else arg
            for arg in arguments
        ]
    output = io.StringIO()
    status = 0
    with contextlib.redirect_stdout(output):
        try:
            archive_cli.main(archive_arguments(["agent", command, *arguments], where))
        except SystemExit as error:
            status = error.code or 0
    try:
        result = json.loads(output.getvalue())
    except ValueError as error:
        raise ValueError("Archive no devolvió un documento JSON") from error
    result["dataset"] = "archive_berrutti"
    return result, status


def parser():
    cli = archive_agent.Parser(
        prog="berrutti-data agent",
        description="JSON v1, sin TTY. list sin conjunto enumera los tres. --where local|ruche elige destino.",
        epilog="Archive: list/info/files/plan/download/complete/remove/job/cancel/refresh/index archive_berrutti [opciones de archive-berrutti agent].",
    )
    commands = cli.add_subparsers(dest="command", required=True)
    for name in ("list", "info", "files", "text", "annotations", "refresh"):
        command = commands.add_parser(name)
        command.add_argument("dataset", nargs="?" if name == "list" else None)
        command.add_argument("--id", help="ID exacto de página o línea")
        if name in {"list", "files", "annotations"}:
            command.add_argument(
                "--offset", type=archive_agent.nonnegative_int, default=0
            )
            command.add_argument(
                "--limit", type=int, choices=range(1, 501), default=50, metavar="1..500"
            )
        if name in {"list", "files"}:
            command.add_argument(
                "--search", default="", help="ID; LUISA también texto humano del CSV"
            )
            command.add_argument("--split", choices=("train", "validation", "test"))
        if name == "files":
            command.add_argument(
                "--kind",
                choices=("all", "image", "reference", "ocr", "annotations"),
                default="all",
            )
        if name == "text":
            command.add_argument(
                "--kind", choices=("reference", "ocr"), default="reference"
            )
    return cli


def files(source, args):
    kinds = (
        ["image", "reference", "annotations"]
        if source.identifier == "berrutti_175"
        else ["image", "reference", "ocr"]
    )
    if args.kind != "all":
        kinds = [kind for kind in kinds if kind == args.kind]
    n = len(kinds)
    rows = source.records(
        offset=args.offset // max(n, 1),
        limit=min(500, (args.limit + max(n, 1) - 1) // max(n, 1) + 1),
        identifier=args.id,
        search=args.search,
        split=args.split,
    )
    items = (
        [item for row in rows["items"] for item in source.files(row, args.kind)]
        if n
        else []
    )
    start = args.offset % max(n, 1)
    items = items[start : start + args.limit]
    total = rows["total"] * n
    return {
        "items": items,
        "total": total,
        "offset": args.offset,
        "limit": args.limit,
        "next_offset": args.offset + len(items)
        if args.offset + len(items) < total
        else None,
        "status_source": "sealed_records_index_not_live_stat",
    }


def execute(args, where):
    if args.command == "list" and args.dataset is None:
        return {
            **manage.describe(),
            "selected_location": where or manage.locations()["default"],
        }, None
    item = access.dataset(args.dataset)
    if args.command == "info" and not args.id:
        return item, item["locations"].get(where or manage.locations()["default"])
    if args.command == "refresh":
        access.clear_cache(args.dataset, where)
    with contextlib.closing(access.Source(args.dataset, where)) as source:
        if args.command == "refresh":
            result = {
                "records": source.records(limit=1)["total"],
                "version": source.metadata["version"],
                "refreshed": "client_metadata_cache",
            }
        elif args.command == "list":
            result = source.records(
                offset=args.offset,
                limit=args.limit,
                identifier=args.id,
                search=args.search,
                split=args.split,
            )
        elif args.command == "files":
            result = files(source, args)
        else:
            if not args.id:
                raise archive_agent.UsageError("Este comando requiere --id")
            if args.command == "info":
                row = source.record(args.id)
                result = {
                    **row,
                    "files": source.files(row),
                    "contents": item.get("contents"),
                }
                if args.dataset == "berrutti_175":
                    annotations = source.annotations(args.id)
                    result["annotations"] = {
                        k: v for k, v in annotations.items() if k != "items"
                    }
            elif args.command == "text":
                result = source.text(args.id, args.kind)
            else:
                result = source.annotations(args.id)
                result["items"] = result["items"][
                    args.offset : args.offset + args.limit
                ]
                result.update(
                    offset=args.offset,
                    limit=args.limit,
                    next_offset=args.offset + len(result["items"])
                    if args.offset + len(result["items"]) < result["total"]
                    else None,
                )
        return result, {**source.location, "access": "sealed_index_cached_on_client"}


def main(argv=None, *, where=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    command = argv[0] if argv else None
    identifier = argv[1] if len(argv) > 1 and not argv[1].startswith("-") else None
    try:
        if (
            command in ARCHIVE_COMMANDS
            and identifier == "archive_berrutti"
            and not (command == "info" and len(argv) == 2)
        ):
            if "--help" in argv or "-h" in argv:
                archive_agent.parser().parse_args([command, "--help"])
            result, status = archive_result(command, argv[2:], where)
        else:
            args = parser().parse_args(argv)
            data, loc = execute(args, where)
            result = {
                "schema_version": 1,
                "ok": True,
                "command": args.command,
                "dataset": args.dataset,
                "location": loc,
                "data": data,
                "observed_at": core.now(),
            }
            status = 0
    except (
        ValueError,
        OSError,
        KeyError,
        sqlite3.Error,
        subprocess.SubprocessError,
    ) as error:
        invalid = isinstance(error, archive_agent.UsageError)
        result = {
            "schema_version": 1,
            "ok": False,
            "command": command,
            "dataset": identifier,
            "error": {
                "code": "invalid_arguments" if invalid else "data_error",
                "message": str(error),
            },
            "observed_at": core.now(),
        }
        status = 2 if invalid else 1
    print(json.dumps(result, ensure_ascii=False))
    if status:
        raise SystemExit(status)

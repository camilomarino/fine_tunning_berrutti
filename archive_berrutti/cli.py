"""Spanish interactive browser and scriptable commands for the Berrutti library."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

from archive_berrutti import library
from archive_berrutti import manage as core

DOWNLOAD_LABEL = {
    "pending": "pendiente",
    "downloaded": "descargado",
    "partial": "parcial",
    "failed": "fallido",
    "inconsistent": "revisar",
}
PDF_LABEL = {
    "none": "—",
    "extracted": "extraído",
    "partial": "parcial",
    "inconsistent": "revisar",
}


def print_json(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def options(**values):
    return SimpleNamespace(
        **{
            "all": False,
            "rolls": None,
            "roll_range": None,
            "match": None,
            "limit": None,
            "ids": None,
            "max_download_gib": None,
            "reserve_gib": 5,
            "quota_free_gib": None,
            "attempts": 3,
            "max_pdf_gib": 8,
            "pages": None,
            "first_pages": 5,
            "all_pages": False,
            "dpi": 200,
            "format": "jpeg",
            "quality": 95,
            "timeout": 180,
            "max_total_pages": 1000,
            **values,
        }
    )


def table(
    root: Path, indexed: list[tuple[int, dict]], selected: set[str] | None = None
):
    width = shutil.get_terminal_size((130, 24)).columns
    print(
        f"{'':3} {'ID':>4} {'Rollo':<8} {'MiB':>7} {'Descarga':<10} {'PDF':<8} {'Imágenes':<12} Título"
    )
    for index, row in indexed:
        state = library.status(root, row)
        mark = "[x]" if selected and row["key"] in selected else "[ ]"
        images = "—"
        if state["renders"]:
            complete = all(run["state"] == "complete" for run in state["renders"])
            images = f"{state['images']} {'OK' if complete else 'parcial'}"
        title = row["name"].split(" - Rollo ")[-1].removesuffix(".zip")
        prefix = (
            f"{mark} {index:4d} {row['roll'] or '?':<8} {row['bytes'] / 2**20:7.1f} "
            f"{DOWNLOAD_LABEL[state['download']]:<10} {PDF_LABEL[state['pdf']]:<8} {images:<12} "
        )
        print(prefix + title[: max(15, width - len(prefix))])


def filtered(
    root: Path,
    data: dict,
    query: str = "",
    downloaded: bool = False,
    states: str = "all",
) -> list[tuple[int, dict]]:
    result = []
    for index, row in enumerate(data["files"], 1):
        if query.casefold() not in row["name"].casefold():
            continue
        state = None
        if downloaded or states != "all":
            state = library.status(root, row)
        if downloaded and state["download"] != "downloaded":
            continue
        if states == "pending" and state["download"] == "downloaded":
            continue
        if states == "images" and not state["renders"]:
            continue
        if states == "partial" and not (
            state["download"] in {"partial", "failed", "inconsistent"}
            or state["pdf"] in {"partial", "inconsistent"}
            or any(run["state"] == "partial" for run in state["renders"])
        ):
            continue
        result.append((index, row))
    return result


def summary(root: Path, data: dict) -> dict:
    counts = {
        "files": len(data["files"]),
        "downloaded": 0,
        "partial": 0,
        "extracted": 0,
        "with_images": 0,
        "image_files": 0,
        "image_pages": 0,
    }
    for row in data["files"]:
        state = library.status(root, row)
        counts["downloaded"] += state["download"] == "downloaded"
        counts["partial"] += state["download"] in {"partial", "failed", "inconsistent"}
        counts["extracted"] += state["pdf"] == "extracted"
        counts["with_images"] += bool(state["renders"])
        counts["image_files"] += state["images"]
        counts["image_pages"] += state["image_pages"]
    return {
        "root": str(root),
        **counts,
        "catalog_gib": data["zip_bytes"] / core.GIB,
        "status_source": "Recibos y manifiestos; verify recalcula los hashes",
    }


def human_bytes(number: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if number < 1024 or unit == "TiB":
            return f"{number:.2f} {unit}"
        number /= 1024


def show_preview(root: Path, rows: list[dict]) -> None:
    estimate = library.preview(root, rows)
    print(
        f"\nSeleccionados: {estimate['selected']} archivos; "
        f"ya descargados: {estimate['already_downloaded']}"
    )
    print(f"ZIP seleccionados: {human_bytes(sum(row['bytes'] for row in rows))}")
    print(
        f"Descarga pendiente: {human_bytes(estimate['remaining_download_gib'] * core.GIB)}"
    )
    print(f"Destino: {root}")
    print("Los PDF extraídos y las imágenes necesitan espacio adicional.")


def detail(root: Path, data: dict, index: int, *, as_json=False) -> None:
    if not 1 <= index <= len(data["files"]):
        raise ValueError("ID fuera del inventario")
    row = data["files"][index - 1]
    state = library.status(root, row)
    if as_json:
        print_json({"id": index, **row, "local": state})
        return
    print(f"\nArchivo {index} · Rollo {row['roll']}")
    print(row["name"])
    print(f"ZIP: {human_bytes(row['bytes'])} · {DOWNLOAD_LABEL[state['download']]}")
    print(f"PDF: {PDF_LABEL[state['pdf']]} · {state['pdf_count']} archivos")
    print(
        f"Imágenes: {state['images']} archivos · Páginas distintas convertidas: {state['image_pages']}"
    )
    print(f"Carpeta: {state['path']}")
    for rendered in state["renders"]:
        config = rendered["config"]
        pages = (
            f"primeras {config['first_pages']} páginas"
            if config["first_pages"]
            else f"páginas {config['pages']}"
            if config["pages"]
            else "todas las páginas"
        )
        complete = "completo" if rendered["state"] == "complete" else "parcial"
        print(
            f"  {config['format'].upper()} · {config['dpi']} DPI · {pages}: "
            f"{rendered['images']}/{rendered['expected'] or '?'} imágenes ({complete})"
        )
        print(f"  {rendered['path']}")
    if not state["renders"]:
        print("Imágenes: todavía no convertidas")
    if state["problem"]:
        print(f"Revisar: {state['problem']}")
    print("El estado usa recibos y manifiestos; la opción verificar recalcula hashes.")


def save_session(root: Path, selected: set[str]) -> None:
    library.replace_json(root / "session.json", {"selected_keys": sorted(selected)})


def browse(args) -> None:
    from archive_berrutti import tui

    tui.run(args)


def parser():
    cli = argparse.ArgumentParser(
        prog="archive-berrutti",
        description="Explorar y preparar el Archivo SID por rollos.",
        epilog="Destino: --local o --profile NOMBRE (también después del comando). Sin opciones se usa profile use. Un --root explícito sin --profile selecciona local.",
    )
    cli.add_argument("--local", action="store_true", help="Usar la biblioteca local")
    cli.add_argument("--profile", help="Usar un perfil de servidor configurado")
    commands = cli.add_subparsers(dest="command")
    commands.add_parser(
        "profile", help="Configurar, desplegar y elegir servidores; profile --help"
    )
    commands.add_parser(
        "agent", help="API no interactiva: inventario, archivos y trabajos en JSON"
    )
    interactive = commands.add_parser(
        "browse", help="Explorador con flechas y Enter (comando por defecto)"
    )
    interactive.add_argument("--root", type=Path)
    interactive.add_argument("--metadata", type=Path)
    interactive.add_argument("--offline", action="store_true")
    view = commands.add_parser(
        "view", help="Abrir un documento en una terminal de cómputo"
    )
    view.add_argument("path", type=Path)
    view.add_argument("--root", type=Path)
    complete = commands.add_parser(
        "complete", help="Completar rollos: todas las páginas en PNG 200 DPI"
    )
    complete.add_argument("--root", type=Path)
    selection = complete.add_mutually_exclusive_group(required=True)
    selection.add_argument("--rolls")
    selection.add_argument("--all", action="store_true")
    roll_status = commands.add_parser(
        "roll-status", help="Estado final de cada rollo, en JSON"
    )
    roll_status.add_argument("--root", type=Path)
    roll_status.add_argument("--rolls")
    for name in (
        "refresh",
        "list",
        "status",
        "info",
        "plan",
        "download",
        "extract",
        "render",
        "prepare",
        "verify",
    ):
        command = commands.add_parser(
            name,
            help={
                "refresh": "Actualizar inventario; no descarga ZIP",
                "list": "Listar rollos y estados",
                "status": "Resumen de la biblioteca",
                "info": "Detalle de un archivo por ID",
                "plan": "Estimar selección sin descargar",
                "download": "Descargar selección",
                "extract": "Extraer PDF",
                "render": "Convertir páginas",
                "prepare": "Descargar, extraer y convertir",
                "verify": "Verificar hashes de ZIP, PDF e imágenes",
            }[name],
        )
        command.add_argument("--root", type=Path)
        if name == "refresh":
            command.add_argument("--metadata", type=Path)
        if name == "list":
            command.add_argument("--search", default="")
            command.add_argument(
                "--state",
                choices=("all", "downloaded", "pending", "images", "partial"),
                default="all",
            )
            command.add_argument("--offset", type=int, default=0)
            command.add_argument("--limit", type=core.positive, default=20)
            command.add_argument("--json", action="store_true")
        if name == "info":
            command.add_argument("id", type=core.positive)
            command.add_argument("--json", action="store_true")
        if name in {"plan", "download", "extract", "render", "prepare", "verify"}:
            for flag in ("rolls", "roll-range", "match", "ids"):
                command.add_argument(f"--{flag}")
            command.add_argument("--all", action="store_true")
            command.add_argument("--limit", type=core.positive)
            command.add_argument("--max-download-gib", type=core.nonnegative)
            if name == "plan":
                command.add_argument("--output", type=Path)
            else:
                command.add_argument("--reserve-gib", type=core.nonnegative, default=5)
                command.add_argument("--attempts", type=core.positive, default=3)
                command.add_argument("--quota-free-gib", type=core.nonnegative)
                command.add_argument("--max-pdf-gib", type=core.nonnegative, default=8)
        if name in {"render", "prepare"}:
            pages = command.add_mutually_exclusive_group()
            pages.add_argument("--pages")
            pages.add_argument("--first-pages", type=core.positive)
            pages.add_argument("--all-pages", action="store_true")
            command.add_argument("--dpi", type=core.positive, default=200)
            command.add_argument("--format", choices=("jpeg", "png"), default="jpeg")
            command.add_argument(
                "--quality", type=int, choices=range(1, 101), default=95
            )
            command.add_argument("--timeout", type=core.positive, default=180)
            command.add_argument("--max-total-pages", type=core.positive, default=1000)
    return cli


def main(argv=None):
    os.umask(0o077)
    cli = parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    from archive_berrutti import remote

    try:
        if arguments and arguments[0] == "profile":
            return remote.profile_command(arguments[1:])
        arguments, destination = remote.extract_destination(arguments)
        if destination:
            code = remote.launch(destination, arguments)
            if code:
                raise SystemExit(code)
            return
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        if "agent" in arguments:
            from archive_berrutti.agent import UsageError

            invalid = isinstance(error, UsageError)
            print(
                json.dumps(
                    {
                        "schema_version": 1,
                        "ok": False,
                        "command": arguments[arguments.index("agent") + 1]
                        if len(arguments) > arguments.index("agent") + 1
                        else None,
                        "root": None,
                        "error": {
                            "code": "invalid_arguments" if invalid else "remote_error",
                            "message": str(error),
                        },
                        "observed_at": core.now(),
                    },
                    ensure_ascii=False,
                )
            )
            raise SystemExit(2 if invalid else 1) from error
        cli.exit(1, f"Error: {error}\n")
    if arguments and arguments[0] == "agent":
        from archive_berrutti import agent

        return agent.main(arguments[1:])
    args = cli.parse_args(arguments or ["browse"])
    root, data, rows = None, None, None
    try:
        if socket.gethostname().split(".")[0] in {"ruche01", "ruche02"}:
            raise ValueError(
                "En Ruche, ejecutar dentro de Slurm; explorar localmente o con srun --pty"
            )
        if args.command == "browse":
            browse(args)
            return
        root = library.open_library(
            args.root or library.default_root(), create=args.command == "refresh"
        )
        if args.command == "view":
            import curses
            from archive_berrutti import tui, viewer

            curses.wrapper(
                lambda screen: viewer.show(tui.Terminal(screen), root, args.path)
            )
            return
        if args.command == "refresh":
            data = library.refresh(root, args.metadata)
            from archive_berrutti import published

            published.maybe_publish(root, data)
            print_json(
                {"files": len(data["files"]), "gib": data["zip_bytes"] / core.GIB}
            )
            return
        data = library.load_catalog(root)
        if args.command in {"complete", "roll-status"}:
            from archive_berrutti import rolls, workflow

            selected = library.choose(
                data, options(rolls=args.rolls, all=not args.rolls)
            )
            keys = {row["key"] for row in selected}
            groups = [
                roll
                for roll in rolls.group(data)
                if any(row["key"] in keys for _, row in roll["files"])
            ]
            if args.command == "complete":
                for roll in groups:
                    workflow.foreground(root, data, roll)
            print_json(
                [
                    {"roll": roll["roll"], **workflow.roll_status(root, roll)}
                    for roll in groups
                ]
            )
            return
        if args.command == "list":
            if args.offset < 0:
                raise ValueError("--offset debe ser no negativo")
            rows = filtered(
                root, data, args.search, args.state == "downloaded", args.state
            )
            displayed = rows[args.offset : args.offset + args.limit]
            if args.json:
                print_json(
                    [
                        {"id": index, **row, "local": library.status(root, row)}
                        for index, row in displayed
                    ]
                )
            else:
                table(root, displayed)
                print(
                    f"{len(rows)} resultados; mostrando {len(displayed)} desde {args.offset}. Usar --offset/--limit para paginar."
                )
        elif args.command == "status":
            print_json(summary(root, data))
        elif args.command == "info":
            detail(root, data, args.id, as_json=args.json)
        else:
            values = vars(args).copy()
            values.pop("command")
            settings = options(**values)
            if args.command in {"render", "prepare"} and not (
                args.pages or args.first_pages or args.all_pages
            ):
                settings.first_pages = 5
            rows = library.choose(data, settings)
            if args.command == "plan":
                if args.output:
                    core.save_new(args.output, library.plan(data, rows))
                print_json(library.preview(root, rows))
            elif args.command == "verify":
                print_json(library.verify(root, rows))
            elif args.command == "download":
                from archive_berrutti import rolls, workflow

                library.check_download_space(root, rows, settings)
                # Keep explicit file selections, including a subset of a roll.
                for roll in rolls.group({"files": rows}):
                    workflow.foreground(
                        root, data, roll, action="download", options=settings
                    )
                print_json(
                    {
                        "action": "download",
                        "files": len(rows),
                        "images": 0,
                        "outputs": [],
                    }
                )
            else:
                print_json(library.operate(root, data, rows, args.command, settings))
    except (EOFError, KeyboardInterrupt):
        print("\nSesión terminada. Los parciales se conservan para reanudar.")
        if args.command != "browse":
            cli.exit(130)
    except (
        OSError,
        ValueError,
        KeyError,
        argparse.ArgumentTypeError,
        zipfile.BadZipFile,
        subprocess.SubprocessError,
    ) as error:
        cli.exit(1, f"Error: {error}\n")
    finally:
        if (
            root is not None
            and data is not None
            and rows is not None
            and args.command in {"extract", "render", "prepare", "verify"}
        ):
            from archive_berrutti import published, rolls

            published.maybe_publish(root, data, rolls.group({"files": rows}))

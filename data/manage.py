"""Human/JSON catalog access and explicit dataset imports; reads need no Slurm."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from data import releases

BASE = Path(__file__).resolve().parent


def catalog():
    return json.loads((BASE / "catalog.json").read_text())


def locations():
    path = Path(
        os.environ.get("BERRUTTI_DATA_LOCATIONS", BASE / "locations.local.json")
    )
    return (
        json.loads(path.read_text())
        if path.exists()
        else {"schema_version": 1, "default": "local", "datasets": {}}
    )


def describe(identifier=None):
    settings = locations()
    items = []
    for item in catalog()["datasets"]:
        if identifier is None or item["id"] == identifier:
            items.append(
                {**item, "locations": settings["datasets"].get(item["id"], {})}
            )
    if identifier is not None and not items:
        raise ValueError(f"Conjunto desconocido: {identifier}")
    return {
        "schema_version": 1,
        "default_location": settings["default"],
        "datasets": items,
    }


def resolve(identifier, where=None, kind="raw"):
    settings = locations()
    where = where or settings["default"]
    item = describe(identifier)["datasets"][0]
    location = item["locations"].get(where)
    if not location or not location.get(kind):
        raise ValueError(
            f"{identifier}: {kind} no disponible en {where}; consultar show"
        )
    return location[kind]


def legacy_main(arguments=None):
    parser = argparse.ArgumentParser(
        prog="berrutti-data",
        description="Explorador de los tres conjuntos Berrutti. Sin argumentos abre el menú con flechas.",
        epilog="browse/agent/archive: --where local|ruche (o --local). Consultas rápidas: list/show/path. También python -m data.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "browse", help="Explorador común con flechas (entrada predeterminada)"
    )
    commands.add_parser(
        "agent", help="Consultas JSON de los tres conjuntos y operaciones Archive"
    )
    commands.add_parser(
        "archive", help="Todos los comandos de Archive desde la entrada común"
    )
    listing = commands.add_parser(
        "list", help="Conjuntos y ubicaciones registradas, sin conectarse"
    )
    listing.add_argument("--json", action="store_true")
    show = commands.add_parser("show", help="Metadatos y ubicaciones de un conjunto")
    show.add_argument("dataset")
    path = commands.add_parser(
        "path", help="Ruta en el host elegido, sin descargar ni reservar CPU"
    )
    path.add_argument("dataset")
    path.add_argument("--where", choices=("local", "ruche"))
    path.add_argument(
        "--kind", choices=("raw", "root", "records", "manifest"), default="raw"
    )
    pages = commands.add_parser(
        "import-berrutti",
        help="Copiar y verificar las 175 páginas, sin modificar la fuente",
    )
    pages.add_argument("--source", type=Path, required=True)
    pages.add_argument("--store", type=Path, required=True)
    pages.add_argument(
        "--split",
        type=Path,
        default=BASE / "berrutti_175/split-v1.json",
    )
    lines = commands.add_parser(
        "import-luisa",
        help="Importar el tar.gz de la revisión fijada; en Ruche usar Slurm",
    )
    lines.add_argument("--archive", type=Path, required=True)
    lines.add_argument("--store", type=Path, required=True)
    lines.add_argument("--revision", default=releases.LUISA_REVISION)
    check = commands.add_parser(
        "verify", help="Recalcular todos los hashes; en Ruche usar Slurm"
    )
    check.add_argument("root", type=Path)
    args = parser.parse_args(arguments)
    try:
        if args.command == "path":
            print(resolve(args.dataset, args.where, args.kind))
            return
        if args.command in {"list", "show"}:
            result = describe(args.dataset if args.command == "show" else None)
            if args.command == "list" and not args.json:
                print(f"Destino predeterminado: {result['default_location']}")
                for item in result["datasets"]:
                    print(f"\n{item['id']} · {item['name']} · {item['supervision']}")
                    for where, location in item["locations"].items():
                        print(
                            f"  {where}: {location.get('host', 'local')}:{location['raw']}"
                        )
                    if not item["locations"]:
                        print("  Sin ubicación registrada en esta máquina")
                return
        else:
            os.umask(0o077)
            if args.command == "import-berrutti":
                result = releases.import_berrutti(args.source, args.store, args.split)
            elif args.command == "import-luisa":
                result = releases.import_luisa(args.archive, args.store, args.revision)
            else:
                result = releases.verify(args.root)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")


def main(arguments=None):
    os.umask(0o077)
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    # Legacy import/verify/path contracts remain unchanged.
    legacy = {"list", "show", "path", "import-berrutti", "import-luisa", "verify"}
    if arguments and arguments[0] in legacy:
        return legacy_main(arguments)
    if arguments in (["--help"], ["-h"]):
        return legacy_main(arguments)
    from archive_berrutti.agent import Parser, UsageError

    destination = (
        Parser(add_help=False)
        if "agent" in arguments
        else argparse.ArgumentParser(add_help=False)
    )
    destinations = destination.add_mutually_exclusive_group()
    destinations.add_argument("--where", choices=("local", "ruche"))
    destinations.add_argument("--local", action="store_true")
    try:
        options, rest = destination.parse_known_args(arguments)
    except UsageError as error:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "ok": False,
                    "error": {"code": "invalid_arguments", "message": str(error)},
                }
            )
        )
        raise SystemExit(2) from error
    where = "local" if options.local else options.where
    command = rest[0] if rest else "browse"
    if command == "agent":
        from data import agent

        return agent.main(rest[1:], where=where)
    if command == "archive":
        from archive_berrutti import cli
        from data.agent import archive_arguments

        return cli.main(archive_arguments(rest[1:] or ["browse"], where))
    if command == "browse":
        from data import tui

        parser = argparse.ArgumentParser(prog="berrutti-data browse")
        parser.add_argument(
            "dataset", nargs="?", choices=[item["id"] for item in catalog()["datasets"]]
        )
        parser.add_argument(
            "--refresh-cache",
            action="store_true",
            help="Volver a leer índices; no toca datos ni trabajos",
        )
        args = parser.parse_args(rest[1:])
        try:
            return tui.run(args.dataset, where=where, refresh_cache=args.refresh_cache)
        except (OSError, ValueError, KeyError) as error:
            parser.exit(1, f"Error: {error}\n")
    legacy_main(arguments)

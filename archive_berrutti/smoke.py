"""A bounded real Archive.org smoke: two small ZIPs and three page images."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import platform
import socket
from pathlib import Path

from archive_berrutti import cli, library, manage as core


def smoke(root: Path, max_download_mib: float) -> dict:
    if socket.gethostname().split(".")[0] in {"ruche01", "ruche02"}:
        raise ValueError("Ejecutar el smoke de Ruche dentro de Slurm")
    if root.exists():
        raise ValueError("El smoke requiere una carpeta nueva")
    root = library.open_library(root, create=True)
    data = library.refresh(root)
    selected = sorted(data["files"], key=lambda row: row["bytes"])[:2]
    if (
        len(selected) != 2
        or sum(row["bytes"] for row in selected) > max_download_mib * 2**20
    ):
        raise ValueError("Los dos ZIP menores superan el presupuesto del smoke")
    core.save_new(
        root / "smoke_config.json",
        {
            "at_utc": core.now(),
            "python": platform.python_version(),
            "source_sha256": core.hashes(Path(core.__file__))["sha256"],
            "max_download_mib": max_download_mib,
            "selected_files": selected,
        },
    )
    transcript = root / "smoke_steps.jsonl"

    def step(action, rows, **changes):
        result = library.operate(
            root, data, rows, action, cli.options(reserve_gib=1, **changes)
        )
        with transcript.open("a") as stream:
            stream.write(json.dumps({"action": action, "result": result}) + "\n")
        return result

    estimate = library.preview(root, data["files"])
    if (root / "rolls").exists():
        raise AssertionError("El inventario descargó originales")
    with contextlib.redirect_stdout(io.StringIO()) as output:
        cli.main(["plan", "--root", str(root), "--all"])
    if json.loads(output.getvalue())["selected"] != len(data["files"]):
        raise AssertionError("La selección completa no coincide con el inventario")
    first, second = selected
    step("download", [first])
    original = core.zip_path(
        library.entry_root(root, first), library.local_entry(first)
    )
    original_time = original.stat().st_mtime_ns
    step("download", [second])
    step("download", selected)
    if original.stat().st_mtime_ns != original_time:
        raise AssertionError("Se reemplazó un original completado")
    step("extract", selected)
    rendered = step("render", selected, first_pages=1)
    image_times = {
        str(path): path.stat().st_mtime_ns
        for directory in rendered["outputs"]
        for path in (Path(directory) / "images").glob("*.jpg")
    }
    step("render", selected, first_pages=1)
    if any(
        Path(path).stat().st_mtime_ns != value for path, value in image_times.items()
    ):
        raise AssertionError("Se reemplazó una imagen completada")
    step("render", [first], first_pages=1, format="png")
    verified = library.verify(root, selected)
    states = [library.status(root, row) for row in selected]
    if any(
        state["download"] != "downloaded" or state["pdf"] != "extracted"
        for state in states
    ):
        raise AssertionError("Estado final incorrecto")
    if any(
        render["state"] != "complete" for state in states for render in state["renders"]
    ):
        raise AssertionError("Conversión incompleta")
    result = {
        "catalog_files": len(data["files"]),
        "whole_download_gib": estimate["zip_gib"],
        "downloaded_zips": 2,
        "downloaded_bytes": sum(row["bytes"] for row in selected),
        "rendered_images": sum(state["images"] for state in states),
        "verified": verified,
        "root": str(root),
        "states": states,
    }
    core.save_new(root / "smoke_result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--max-download-mib", type=core.nonnegative, default=2)
    args = parser.parse_args()
    cli.print_json(smoke(args.root.resolve(), args.max_download_mib))


if __name__ == "__main__":
    main()

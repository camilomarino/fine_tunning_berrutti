"""User-facing files and folders, without exposing internal manifests as data."""

from __future__ import annotations

from pathlib import Path

from archive_berrutti import library, manage as core


def resources(root, roll):
    images, pdfs, archives = [], [], []

    def add(group, label, path):
        group.append(
            {
                "id": str(path),
                "label": label,
                "path": path,
                "section": "Archivos",
                "disabled": not path.exists()
                or path.suffix == ".part"
                or "aún sin" in label,
                "directory": path.is_dir(),
            }
        )

    for part, (_, row) in enumerate(roll["files"], 1):
        directory = library.entry_root(root, row)
        suffix = f" · ZIP {part}/{len(roll['files'])}" if len(roll["files"]) > 1 else ""
        archive = core.zip_path(directory, library.local_entry(row))
        archive_label = (
            f"ZIP {part}/{len(roll['files'])}" if len(roll["files"]) > 1 else "ZIP"
        )
        add(
            archives,
            archive_label
            + (" · sin descargar" if not archive.exists() else "")
            + f" · {archive.name}",
            archive,
        )
        partial = archive.with_suffix(".zip.part")
        if partial.exists():
            add(archives, f"Descarga parcial{suffix}", partial)
        pdf_dir = directory / "pdfs" / row["key"]
        paths = sorted(pdf_dir.glob("*.pdf"))
        for path in paths:
            add(pdfs, f"PDF{suffix} · {path.name}", path)
        if not paths:
            add(pdfs, f"PDF{suffix} · aún sin extraer", pdf_dir)
        target = directory / "final_target.json"
        final = None
        if target.exists():
            try:
                relative = Path(core.load(target)["output"])
                if (
                    len(relative.parts) != 2
                    or relative.parts[0] != "renders"
                    or ".." in relative.parts
                ):
                    raise ValueError("Salida final fuera de renders")
                final = directory / relative / "images"
            except (OSError, ValueError, KeyError, TypeError):
                pass
        if final is not None:
            add(images, f"Carpeta de imágenes finales{suffix}", final)
        other = sorted((directory / "renders").glob("*/images"))
        for path in other:
            if path != final:
                add(
                    images,
                    f"Carpeta de muestras / otras imágenes{suffix} · {path.parent.name}",
                    path,
                )
        if final is None and not other:
            add(
                images,
                f"Carpeta de imágenes{suffix} · aún sin generar",
                directory / "renders",
            )
    # Ready-to-use data first. Missing targets remain visible with their location.
    available, missing = [], []
    for item in [*images, *pdfs, *archives]:
        (missing if item["disabled"] else available).append(item)
    return available + missing

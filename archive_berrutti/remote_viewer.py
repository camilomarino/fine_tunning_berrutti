"""Fetch only selected documents into a bounded local viewing cache."""

from __future__ import annotations

import curses
import shutil
from pathlib import Path

from archive_berrutti import remote, remote_library, viewer


class Document:
    def __init__(self, client, path, files):
        self.client = client
        self.pdf = path.suffix.lower() == ".pdf"
        self.path = path
        self.inner = None
        self.loaded = None
        self.files = sorted(
            (
                file
                for file in files
                if file["present"]
                and (
                    file["path"] == str(path)
                    if self.pdf
                    else Path(file["path"]).parent == path.parent
                    and Path(file["path"]).suffix.lower() in viewer.IMAGE_SUFFIXES
                )
            ),
            key=lambda file: Path(file["path"]).name.casefold(),
        )
        self.index = next(
            i for i, file in enumerate(self.files) if file["path"] == str(path)
        )
        self.count = len(self.files)
        if self.pdf:
            cached = self.client.cached_file(self.files[0])
            self.inner = viewer.Document(cached.parent, cached)
            self.count = self.inner.count

    @property
    def current_path(self):
        return self.path if self.pdf else Path(self.files[self.index]["path"])

    @property
    def external_path(self):
        self.load()
        return self.inner.current_path

    def load(self):
        if self.pdf:
            self.inner.index = self.index
        elif self.loaded != self.index:
            cached = self.client.cached_file(self.files[self.index])
            if self.inner:
                self.inner.close()
            self.inner = viewer.Document(cached.parent, cached)
        image = self.inner.load()
        self.loaded = self.index
        return image

    def close(self):
        if self.inner:
            self.inner.close()


def show(view, client, path, files):
    item = next(
        (file for file in files if file["path"] == str(path) and file["present"]), None
    )
    if item is None:
        raise ValueError("Archivo no disponible en el índice remoto")
    large = item["bytes"] > remote_library.MAX_FILE
    pdf_without_tools = path.suffix.lower() == ".pdf" and not (
        shutil.which("pdfinfo") and shutil.which("pdftoppm")
    )
    if large or pdf_without_tools:
        if (
            view.menu(
                "Visualizar en Ruche",
                ["Abrir visor con CPU de Slurm", "Volver"],
                lines=[
                    "El archivo supera la caché local de 64 MiB."
                    if large
                    else "Falta Poppler en esta máquina.",
                    "La vista remota usa una asignación solo mientras el documento está abierto.",
                ],
            )
            != 0
        ):
            return
        curses.def_prog_mode()
        curses.endwin()
        try:
            code = remote.launch_compute(
                client.profile, ["view", str(path), "--root", str(client.root)]
            )
            if code:
                raise ValueError(f"El visor remoto terminó con código {code}")
        finally:
            curses.reset_prog_mode()
            view.screen.refresh()
        return
    view.progress("Leyendo solo el documento elegido por SSH…")
    document = Document(client, path, files)
    viewer.show(view, client.root, path, document=document)

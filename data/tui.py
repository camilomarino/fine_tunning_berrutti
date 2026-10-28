"""Unified dataset browser, sharing Archive navigation and image/PDF viewer."""

from __future__ import annotations

import contextlib
import curses
import json
import os
import sqlite3
import sys

from PIL import Image, ImageDraw, ImageOps

from archive_berrutti import cli as archive_cli, remote, remote_library, tui, viewer
from data import access, manage

ERRORS = (*tui.ERRORS, sqlite3.Error)

LABELS = {
    "berrutti_175": "Berrutti 175 · páginas, texto humano y cajas de líneas",
    "luisa_tesseract_lines": "LUISA · líneas, texto humano y OCR Tesseract",
    "archive_berrutti": "Archive Berrutti · rollos, PDF e imágenes",
}


class Document:
    """Fetch/decode only the selected image; next/previous uses the record index."""

    pdf = False

    def __init__(self, source, row, *, search="", split=None, boxes=False, crop=None):
        self.source, self.search, self.split = source, search, split
        self.boxes, self.crop = boxes, crop
        self.count = source.records(search=search, split=split, limit=1)["total"]
        # IDs are immutable and sorted in the SQL index, never filesystem enumeration.
        clauses, values = ["id < ?"], [row["id"]]
        if search:
            clauses.append("instr(search, ?) > 0")
            values.append(search.casefold())
        if split:
            field = (
                "project_split"
                if source.identifier == "berrutti_175"
                else "source_split"
            )
            clauses.append(field + " = ?")
            values.append(split)
        self.index = (
            source.db()
            .execute(
                "SELECT count(*) FROM records WHERE " + " AND ".join(clauses), values
            )
            .fetchone()[0]
        )
        if crop is not None:
            self.index, self.count = 0, 1
        self.row = row
        self.loaded = None
        self.image = None

    def current(self):
        if self.crop is None:
            self.row = self.source.records(
                offset=self.index, limit=1, search=self.search, split=self.split
            )["items"][0]
        return self.row

    @property
    def current_path(self):
        return self.source.path(self.current()["image"])

    @property
    def external_path(self):
        return self.source.image_path(self.current())

    def load(self):
        if self.loaded == self.index:
            return self.image
        row = self.current()
        path = self.source.image_path(row)
        with Image.open(path) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
        if self.crop is not None:
            cropped = image.crop(self.crop)
            image.close()
            image = cropped
        elif self.boxes:
            draw = ImageDraw.Draw(image)
            for line in self.source.annotations(row["id"])["items"]:
                draw.rectangle(
                    line["crop_xyxy"], outline="red", width=max(2, image.width // 700)
                )
        if self.image:
            self.image.close()
        self.image, self.loaded = image, self.index
        return image

    def close(self):
        if self.image:
            self.image.close()


class Application:
    def __init__(self, view, where=None):
        self.view = view
        self.where = where or manage.locations()["default"]

    def run(self):
        current = 0
        while True:
            catalog = manage.describe()["datasets"]
            entries = sorted(catalog, key=lambda item: list(LABELS).index(item["id"]))
            labels = [
                LABELS[item["id"]]
                + (
                    " · disponible"
                    if self.where in item["locations"]
                    else " · sin copia aquí"
                )
                for item in entries
            ]
            choice = self.view.menu(
                "Conjuntos de datos",
                [*labels, "Cambiar ubicación · " + self.where, "Salir"],
                subtitle="Destino: " + self.where,
                lines=[
                    "Elegí un conjunto para explorar imágenes, textos y archivos.",
                    "Archive permite además descargar y procesar rollos por partes.",
                ],
                initial=current,
            )
            if choice is None or choice == len(labels) + 1:
                return
            current = choice
            if choice == len(labels):
                choices = ["local", "ruche"]
                selected = self.view.menu(
                    "Dónde están los datos",
                    ["Esta máquina", "Ruche · SSH"],
                    initial=choices.index(self.where),
                )
                if selected is not None:
                    self.where = choices[selected]
                continue
            self.open(entries[choice]["id"])

    def open(self, identifier):
        try:
            if identifier == "archive_berrutti":
                self.archive()
            else:
                self.view.progress(
                    "Leyendo índice del conjunto; las imágenes se abren una por una…"
                )
                with contextlib.closing(
                    access.Source(identifier, self.where)
                ) as source:
                    self.records(source)
        except ERRORS as error:
            self.view.message(
                "No se pudo abrir el conjunto",
                [str(error), "Podés cambiar la ubicación desde el menú de entrada."],
            )

    def archive(self):
        loc = access.location("archive_berrutti", self.where)
        args = archive_cli.parser().parse_args(["browse", "--root", loc["root"]])
        backend = None
        try:
            if loc["host"] != "local":
                self.view.progress("Leyendo el estado de los rollos por SSH…")
                profile = remote.validate(
                    remote.configuration()["profiles"][self.where], deployed=True
                )
                backend = remote_library.Client(profile, loc["root"])
            tui.Application(self.view, args, backend=backend).run()
        finally:
            if backend:
                backend.close()

    def records(self, source):
        offset, current, search, split = 0, 0, "", None
        while True:
            height, _ = self.view.screen.getmaxyx()
            capacity = max(1, height - 10)
            result = source.records(
                offset=offset, limit=min(500, capacity), search=search, split=split
            )
            rows = result["items"]
            current = min(current, max(0, len(rows) - 1))
            unit = "Páginas" if source.identifier == "berrutti_175" else "Líneas"
            ready = self.view.frame(
                f"{unit} · {result['total']} ejemplos · {self.where}",
                f"Buscar: {search or '(todos)'} · Grupo: {split or 'todos'}",
                "↑ ↓ mover · → / Enter abrir · ← volver · / buscar · f filtrar",
            )
            if ready:
                self.view.put(
                    3, "ID                      Grupo          Contenido", curses.A_BOLD
                )
                for index, row in enumerate(rows):
                    group = row.get("project_split") or row.get("source_split") or "—"
                    brief = " ".join(
                        row.get(
                            "reference_text", "Imagen + texto humano + cajas de líneas"
                        ).split()
                    )
                    self.view.put(
                        4 + index,
                        f"{row['id']:<24}{group:<15}{brief}",
                        self.view.accent if index == current else 0,
                    )
                if not rows:
                    self.view.put(5, "No hay ejemplos con este filtro.")
                self.view.put(height - 5, str(source.root), curses.A_DIM)
                self.view.put(
                    height - 4,
                    f"{offset + 1 if rows else 0}–{offset + len(rows)} de {result['total']} · PgUp/PgDn página · g ID · i información",
                    curses.A_DIM,
                )
            key = self.view.key()
            if key in tui.BACK or key == "q":
                return
            if not ready:
                continue
            try:
                if key in tui.OPEN and rows:
                    self.detail(source, rows[current], search, split)
                elif key == curses.KEY_DOWN:
                    if current == len(rows) - 1 and result["next_offset"] is not None:
                        offset, current = result["next_offset"], 0
                    else:
                        current = min(max(0, len(rows) - 1), current + 1)
                elif key == curses.KEY_UP:
                    if current == 0 and offset:
                        offset, current = max(0, offset - capacity), capacity - 1
                    else:
                        current = max(0, current - 1)
                elif key == curses.KEY_NPAGE and result["next_offset"] is not None:
                    offset, current = result["next_offset"], 0
                elif key == curses.KEY_PPAGE:
                    offset, current = max(0, offset - capacity), 0
                elif key == curses.KEY_HOME:
                    offset, current = 0, 0
                elif key == curses.KEY_END:
                    offset, current = max(0, result["total"] - capacity), capacity - 1
                elif key == "/":
                    query = self.view.edit(
                        "Buscar por ID"
                        + (
                            " o texto humano"
                            if source.identifier == "luisa_tesseract_lines"
                            else ""
                        ),
                        search,
                    )
                    if query is not None:
                        search, offset, current = query, 0, 0
                elif key == "f":
                    splits = (
                        [None, "train", "validation", "test"]
                        if source.identifier == "berrutti_175"
                        else [None, "train", "test"]
                    )
                    choice = self.view.menu(
                        "Grupo del proyecto"
                        if source.identifier == "berrutti_175"
                        else "Grupo publicado por LUISA",
                        [s or "Todos" for s in splits],
                    )
                    if choice is not None:
                        split, offset, current = splits[choice], 0, 0
                elif key == "g":
                    value = self.view.edit("Abrir ID exacto")
                    if value:
                        self.detail(source, source.record(value), "", None)
                elif key == "i":
                    self.view.message(
                        "Contenido y procedencia",
                        [
                            source.description["name"],
                            source.description.get("description", ""),
                            "",
                            f"Ejemplos: {source.metadata['record_count']}",
                            f"Versión: {source.metadata['version']}",
                            f"Ubicación: {self.where} · {source.root}",
                            "",
                            "Berrutti: los grupos train/validation/test pertenecen al proyecto. Las cajas conservan su orden de anotación, no un orden de lectura validado."
                            if source.identifier == "berrutti_175"
                            else "LUISA: train/test son los grupos de origen. No hay correspondencia con páginas ni cajas. Texto humano y Tesseract están separados; no se mezcla con el entrenamiento actual.",
                            "",
                            "Guía: data/README.md · Uso: data/CLI.md",
                        ],
                    )
            except ERRORS as error:
                self.view.message("No se pudo abrir", [str(error)])

    def image(self, source, row, search="", split=None, **kwargs):
        document = Document(source, row, search=search, split=split, **kwargs)
        viewer.show(
            self.view, source.root, source.path(row["image"]), document=document
        )

    def detail(self, source, row, search, split):
        entries = [
            {
                "label": "Imagen · abrir visor",
                "section": "Archivos",
                "path": row["paths"]["image"],
                "action": "image",
            },
            {
                "label": "Transcripción humana · texto completo"
                if source.identifier == "berrutti_175"
                else "Transcripción humana de la línea",
                "section": "Archivos",
                "path": row["paths"]["reference"],
                "action": "reference",
            },
        ]
        if source.identifier == "berrutti_175":
            annotation = source.annotations(row["id"])
            entries.extend(
                [
                    {
                        "label": f"Líneas con cajas y transcripción · {annotation['total']}",
                        "section": "Anotaciones",
                        "path": annotation["path"],
                        "action": "lines",
                    },
                    {
                        "label": "Ver cajas sobre la imagen",
                        "section": "Anotaciones",
                        "path": row["paths"]["image"],
                        "action": "boxes",
                    },
                ]
            )
        else:
            entries.append(
                {
                    "label": "OCR de Tesseract · texto automático",
                    "section": "Archivos",
                    "path": row["paths"]["ocr_prediction"],
                    "action": "ocr",
                }
            )
        entries.append(
            {
                "label": "Información del ejemplo y procedencia",
                "section": "Información",
                "action": "info",
            }
        )
        current, top = 0, 0
        while True:
            current, top, capacity, ready = self.view.listing(
                row["id"],
                entries,
                current,
                top,
                subtitle=f"{self.where} · {source.metadata['version']}",
                status="Las cajas conservan su orden original; no garantiza orden de lectura."
                if source.identifier == "berrutti_175"
                else "Texto humano y OCR son fuentes distintas. LUISA queda fuera del entrenamiento actual.",
            )
            key = self.view.key()
            if key in tui.BACK or key == "q":
                return
            current = tui.navigate(key, current, len(entries), capacity)
            if not ready:
                continue
            try:
                if key == "p" and entries[current].get("path"):
                    self.view.message(
                        "Ruta completa · " + self.where, [entries[current]["path"]]
                    )
                elif key in tui.OPEN:
                    action = entries[current]["action"]
                    if action in {"image", "boxes"}:
                        self.image(source, row, search, split, boxes=action == "boxes")
                    elif action in {"reference", "ocr"}:
                        result = source.text(row["id"], action)
                        self.view.message(
                            entries[current]["label"],
                            [result["path"], "", *result["text"].splitlines()],
                        )
                    elif action == "lines":
                        self.lines(source, row)
                    else:
                        self.view.message(
                            "Información del ejemplo",
                            json.dumps(row, ensure_ascii=False, indent=2).splitlines(),
                        )
            except ERRORS as error:
                self.view.message("No se pudo mostrar", [str(error)])

    def lines(self, source, row):
        annotations = source.annotations(row["id"])
        lines = annotations["items"]
        if not lines:
            self.view.message("Líneas", ["Esta página no tiene regiones anotadas."])
            return
        labels = [
            f"{line['region_index'] + 1:03d} · {' '.join(line['text'].split())}"
            for line in lines
        ]
        current = 0
        while True:
            choice = self.view.menu(
                "Líneas · " + row["id"],
                labels,
                "Orden original de anotación; coordenadas en píxeles",
                initial=current,
            )
            if choice is None:
                return
            current = choice
            line = lines[choice]
            action = self.view.menu(
                f"Región {choice + 1} · {row['id']}",
                ["Ver recorte de la línea", "Leer transcripción completa"],
                lines=[
                    "Caja x, y, ancho, alto: " + str(line["bbox_xywh"]),
                    line["text"],
                ],
            )
            if action == 0:
                self.image(source, row, crop=line["crop_xyxy"])
            elif action == 1:
                self.view.message(
                    "Transcripción humana de la región", line["text"].splitlines()
                )


def run(identifier=None, *, where=None, refresh_cache=False):
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise ValueError(
            "El explorador necesita TTY; usar berrutti-data agent list para agentes"
        )
    if os.environ.get("TERM", "dumb") == "dumb":
        raise ValueError("Configurar TERM para una terminal interactiva")
    if refresh_cache:
        for item in manage.describe()["datasets"]:
            if (
                item["id"] != "archive_berrutti"
                and (identifier is None or identifier == item["id"])
                and (where or manage.locations()["default"]) in item["locations"]
            ):
                access.clear_cache(item["id"], where)

    def start(screen):
        app = Application(tui.Terminal(screen, brand="DATOS BERRUTTI"), where)
        if identifier:
            app.open(identifier)
        else:
            app.run()

    try:
        curses.wrapper(start)
    except curses.error as error:
        raise ValueError(f"No se pudo abrir la terminal: {error}") from error

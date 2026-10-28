"""Arrow-key terminal interface; reading and selecting never fetch payloads."""

from __future__ import annotations

import argparse
import contextlib
import curses
import io
import os
import re
import subprocess
import sys
import textwrap
import time
import zipfile
from pathlib import Path

from archive_berrutti import (
    cli,
    library,
    locations,
    manage as core,
    monitor,
    remote,
    rolls,
    workflow,
    viewer,
)

ENTER = ("\n", "\r", curses.KEY_ENTER)
OPEN = (*ENTER, curses.KEY_RIGHT)
BACK = ("\x1b", curses.KEY_LEFT)
ERRORS = (
    OSError,
    ValueError,
    KeyError,
    argparse.ArgumentTypeError,
    zipfile.BadZipFile,
    subprocess.SubprocessError,
)
FILTERS = (
    ("all", "Todos"),
    ("incomplete", "Incompletos"),
    ("complete", "Completos"),
    ("error", "Con error"),
    ("active", "Procesando / en cola"),
)
STAGE_GUIDE = "Por ZIP: 1 Descarga → 2 Extraer PDF → 3 Crear imágenes"


def duration(seconds):
    if seconds < 60:
        return f"{max(1, round(seconds))} s"
    minutes = max(1, round(seconds / 60))
    return f"{minutes} min" if minutes < 60 else f"{minutes // 60} h {minutes % 60} min"


def image_count(state):
    if state["expected"] is not None:
        return f"{state['images']}/{state['expected']}"
    return f"{state['images']}/?" if state["images"] else "0"


def progress_line(state):
    progress = state.get("progress")
    if not progress:
        return (state["job"] or {}).get("message") or "Esperando turno"
    percent = progress["percent"]
    if progress.get("bytes_total"):
        percent = min(100, 100 * progress["bytes_done"] / progress["bytes_total"])
    filled = int((percent or 0) / 10)
    bar = "#" * filled + "-" * (10 - filled)
    load = f"[{bar}] {percent:.1f}%" if percent is not None else "Calculando total…"
    eta = progress["eta_seconds"]
    remaining = (
        f"~{duration(eta)} restantes" if eta is not None else "Estimando tiempo…"
    )
    if percent == 100:
        remaining = "Finalizando etapa"
    return f"{load} · {remaining} · ZIP {progress['zip_index']}/{progress['zip_total']}"


def roll_order(pair):
    roll, state = pair
    if state["complete"]:
        priority = 0
    elif state["active"]:
        priority = 1
    elif state["label"] == "Sin descargar":
        priority = 3
    else:
        priority = 2
    return priority, roll["roll"].casefold()


def state_label(state):
    current = state.get("progress")
    if state["active"] and current:
        stage = current["stage"]
        if stage == "download":
            phase = (state["job"].get("transfer") or {}).get("phase")
            label = {
                "connecting": "Conectando",
                "verifying": "Verificando ZIP",
                "retrying": "Reintentando descarga",
                "failed": "Descarga fallida",
                "done": "ZIP verificado",
            }.get(phase, f"Descarga {current['percent'] or 0:.1f}%")
        elif stage == "extract":
            label = f"Extrayendo PDF {current['done']}/{current['total'] if current['total'] is not None else '?'}"
        elif stage == "render":
            label = "Creando imágenes"
            if current["total"] is not None and current["done"] >= current["total"]:
                label = "Verificando imágenes"
            elif current["zip_total"] > 1:
                label = f"Imágenes {current['done']}/{current['total'] or '?'}"
        else:
            label = "Contando páginas" if stage == "count" else "Verificando originales"
        eta = (
            f" · ~{duration(current['eta_seconds'])}"
            if current["eta_seconds"] is not None
            else ""
        )
        return f"{current['step']}/3 {label}{eta}"
    transfer = (state["job"] or {}).get("transfer")
    if state["active"] and transfer:
        if transfer["phase"] == "downloading":
            return f"Descarga {state['download']['percent']:.1f}%"
        return {
            "connecting": "Conectando",
            "verifying": "Verificando ZIP",
            "retrying": "Reintentando",
            "failed": "Descarga fallida",
            "done": "ZIP verificado",
        }[transfer["phase"]]
    if state["active"]:
        return state["label"]
    if state["complete"]:
        return "Completo"
    if state["errors"]:
        return "Error · reintentar"
    if state["label"] == "Sin descargar":
        return "Sin descargar"
    return {
        "download": "Falta descargar",
        "extract": "Falta extraer PDF",
        "render": "Falta convertir a imágenes",
    }[state["next_step"]]


def download_line(state):
    download = state["download"]
    filled = min(10, int(download["percent"] / 10))
    bar = "#" * filled + "-" * (10 - filled)
    return (
        f"Descarga [{bar}] {download['percent']:.1f}% · "
        f"{cli.human_bytes(download['bytes'])} / {cli.human_bytes(download['total_bytes'])}"
    )


def wrapped(lines, width):
    return [
        part
        for line in lines
        for part in textwrap.wrap(str(line), max(1, width)) or [""]
    ]


def navigate(key, current, length, capacity):
    delta = {
        curses.KEY_UP: -1,
        curses.KEY_DOWN: 1,
        curses.KEY_PPAGE: -capacity,
        curses.KEY_NPAGE: capacity,
        curses.KEY_HOME: -length,
        curses.KEY_END: length,
    }.get(key, 0)
    return max(0, min(max(0, length - 1), current + delta))


class Terminal:
    """Small curses view, also usable with a scripted screen in integration tests."""

    def __init__(self, screen, brand="ARCHIVE BERRUTTI"):
        self.screen = screen
        self.brand = brand
        self.accent = curses.A_REVERSE
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_CYAN, -1)
            self.accent |= curses.color_pair(1)
        self.cursor(False)
        curses.set_escdelay(100)
        screen.keypad(True)

    def cursor(self, visible):
        try:
            curses.curs_set(int(visible))
        except curses.error:
            pass

    def put(self, y, text, attr=0, x=2):
        height, width = self.screen.getmaxyx()
        if 0 <= y < height and x < width - 1:
            # Replace control characters from remote filenames and progress output.
            clean = "".join(c if c.isprintable() else " " for c in str(text))
            try:
                if hasattr(self.screen, "clrtoeol"):
                    self.screen.move(y, x)
                    self.screen.clrtoeol()
                self.screen.addnstr(y, x, clean, max(0, width - x - 2), attr)
            except curses.error:
                pass

    def frame(
        self, title, subtitle="", hints="↑ ↓ mover · → / Enter elegir · ← / Esc volver"
    ):
        self.screen.erase()
        height, width = self.screen.getmaxyx()
        self.put(0, self.brand, curses.A_BOLD)
        self.put(1, title, curses.A_BOLD)
        self.put(2, subtitle, curses.A_DIM)
        self.put(height - 2, hints, curses.A_DIM)
        self.put(
            height - 1, "Ctrl+C interrumpir; se conservan los parciales", curses.A_DIM
        )
        if height < 16 or width < 70:
            self.put(4, "Amplía la terminal a 70 columnas × 16 filas.")
            self.put(5, "Esc vuelve; q sale del explorador.")
            return False
        return True

    def key(self, timeout=-1):
        self.screen.refresh()
        if hasattr(self.screen, "timeout"):
            self.screen.timeout(timeout)
        try:
            return self.screen.get_wch()
        except curses.error:
            return None

    def menu(self, title, items, subtitle="", lines=(), initial=0):
        """Return an item index, or None when cancelled; no numbered input."""
        current = initial
        while True:
            height, width = self.screen.getmaxyx()
            ready = self.frame(title, subtitle)
            if ready:
                intro = wrapped(lines, width - 6)
                # Keep all choices reachable even for long paths or small terminals.
                limit = min(len(intro), max(0, height - 10))
                for y, line in enumerate(intro[:limit], 4):
                    self.put(y, line)
                start = 5 + limit
                capacity = max(1, height - start - 3)
                offset = max(0, current - capacity + 1)
                for index in range(offset, min(len(items), offset + capacity)):
                    attr = self.accent if index == current else 0
                    self.put(
                        start + index - offset,
                        ("› " if attr else "  ") + items[index],
                        attr,
                    )
            key = self.key()
            if key in BACK:
                return None
            if key == curses.KEY_UP:
                current = (current - 1) % len(items)
            elif key == curses.KEY_DOWN:
                current = (current + 1) % len(items)
            elif key == curses.KEY_HOME:
                current = 0
            elif key == curses.KEY_END:
                current = len(items) - 1
            elif key in OPEN and ready:
                return current

    def listing(self, title, items, current, top, subtitle="", status="", shortcuts=""):
        """A selectable list with an always-visible full path for the focused item."""
        height, width = self.screen.getmaxyx()
        current = min(current, max(0, len(items) - 1))
        path = items[current].get("path") if items else None
        lines = wrapped([str(path)], width - 6) if path else []
        limit = max(1, height - 12)
        clipped = len(lines) > limit
        lines = lines[:limit]
        if clipped:
            lines[-1] = "… p: ver ruta completa"
        footer = height - 3 - len(lines)
        capacity = max(1, footer - 6)
        # Headings occupy rows, never keyboard stops. Keep selection indexed by
        # the original items so actions and file shortcuts retain their targets.
        rows, selected_row, section = [], 0, None
        for index, item in enumerate(items):
            if item.get("section") != section:
                if rows:
                    rows.append((None, ""))
                section = item.get("section")
                rows.append(
                    (
                        None,
                        f"── {section.upper()} "
                        + "─" * max(0, width - len(section) - 12),
                    )
                )
            if index == current:
                selected_row = len(rows)
            rows.append((index, item["label"]))
        top = max(min(top, selected_row), selected_row - capacity + 1)
        ready = self.frame(
            title, subtitle, "↑ ↓ mover · → / Enter abrir · ← / Esc volver"
        )
        if ready:
            self.put(3, status, curses.A_DIM)
            if top and items and items[current].get("section"):
                self.put(4, items[current]["section"].upper(), curses.A_BOLD)
            for position in range(top, min(len(rows), top + capacity)):
                index, label = rows[position]
                self.put(
                    5 + position - top,
                    ("› " if index == current else "  ") + label,
                    curses.A_DIM
                    if index is not None and items[index].get("disabled")
                    else self.accent
                    if index == current
                    else curses.A_DIM
                    if index is None
                    else 0,
                )
            if not items:
                self.put(5, "Esta carpeta está vacía.")
            if path:
                self.put(
                    footer - 1,
                    "Ruta completa:" if not clipped else "Ruta:",
                    curses.A_BOLD,
                )
                for line, y in zip(lines, range(footer, height - 3)):
                    self.put(y, line)
            self.put(
                height - 1,
                shortcuts or "p ruta completa · PgUp/PgDn desplazar",
                curses.A_DIM,
            )
        return current, top, capacity, ready

    def edit(self, title, default="", hint=""):
        """Editable text with arrows/Home/End; Escape cancels instead of accepting."""
        value, position = default, len(default)
        self.cursor(True)
        try:
            while True:
                _, width = self.screen.getmaxyx()
                self.frame(
                    title, hint, "← → mover cursor · Enter aceptar · Esc cancelar"
                )
                capacity = max(1, width - 8)
                offset = max(0, position - capacity + 1)
                self.put(5, value[offset : offset + capacity], curses.A_UNDERLINE, 4)
                try:
                    self.screen.move(5, 4 + position - offset)
                except curses.error:
                    pass
                key = self.key()
                if key == "\x1b":
                    return None
                if key in ENTER:
                    return value.strip()
                if key == curses.KEY_LEFT:
                    position = max(0, position - 1)
                elif key == curses.KEY_RIGHT:
                    position = min(len(value), position + 1)
                elif key in (curses.KEY_HOME, "\x01"):
                    position = 0
                elif key in (curses.KEY_END, "\x05"):
                    position = len(value)
                elif key in (curses.KEY_BACKSPACE, "\x7f", "\b") and position:
                    value = value[: position - 1] + value[position:]
                    position -= 1
                elif key == curses.KEY_DC:
                    value = value[:position] + value[position + 1 :]
                elif key == "\x15":
                    value, position = "", 0
                elif isinstance(key, str) and key.isprintable():
                    value = value[:position] + key + value[position:]
                    position += len(key)
        finally:
            self.cursor(False)

    def message(self, title, lines):
        top = 0
        while True:
            height, width = self.screen.getmaxyx()
            content = wrapped(lines() if callable(lines) else lines, width - 6)
            capacity = max(1, height - 8)
            top = min(top, max(0, len(content) - capacity))
            if self.frame(
                title, hints="↑ ↓ / PgUp PgDn desplazar · Enter / Esc volver"
            ):
                for y, line in enumerate(content[top : top + capacity], 4):
                    self.put(y, line)
            key = self.key(timeout=1000 if callable(lines) else -1)
            if key in OPEN + BACK:
                return
            if key == curses.KEY_DOWN:
                top += 1
            elif key == curses.KEY_UP:
                top = max(0, top - 1)
            elif key == curses.KEY_NPAGE:
                top += capacity
            elif key == curses.KEY_PPAGE:
                top = max(0, top - capacity)
            elif key == curses.KEY_HOME:
                top = 0
            elif key == curses.KEY_END:
                top = len(content)

    def progress(self, text):
        self.frame(
            "Operación en curso",
            hints="Ctrl+C interrumpir; el trabajo completado se conserva",
        )
        _, width = self.screen.getmaxyx()
        for y, line in enumerate(wrapped([text], width - 6)[:8], 4):
            self.put(y, line)
        self.screen.refresh()


class ProgressStream(io.TextIOBase):
    """Keep engine stdout inside the screen, including Poppler completion records."""

    def __init__(self, progress):
        self.progress = progress
        self.pending = ""

    def write(self, text):
        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            if line.strip():
                self.progress(line)
        return len(text)

    def flush(self):
        if self.pending.strip():
            self.progress(self.pending)
        self.pending = ""


class Application:
    def __init__(self, view, args, backend=None):
        self.view, self.args = view, args
        self.backend = backend
        if backend:
            self.root = backend.root
            self.load()
            return
        self.root = library.open_library(
            args.root or library.default_root(), create=True
        )
        if not (self.root / "active_catalog.json").exists():
            if args.offline and not args.metadata:
                raise ValueError(
                    "Falta inventario local; importar con refresh --metadata"
                )
            view.progress("Inventariando Archive.org; solo se obtiene metadata.")
            library.refresh(self.root, args.metadata)
        self.load()
        library.remember(self.root)

    def load(self):
        self.data = (
            self.backend.index["catalog"]
            if self.backend
            else library.load_catalog(self.root)
        )
        self.rolls = rolls.group(self.data)
        self.states = {}
        self.progress_windows = {}
        self.last_poll = 0

    def status(self, roll):
        if roll["roll"] not in self.states:
            self.states[roll["roll"]] = (
                self.backend.state(roll)
                if self.backend
                else workflow.roll_status(self.root, roll)
            )
            state = self.states[roll["roll"]]
            if state.get("progress"):
                monitor.observe(
                    state["progress"], state["job"]["job"], self.progress_windows
                )
        return self.states[roll["roll"]]

    def root_label(self):
        if self.backend:
            return (
                self.backend.warning
                or f"{self.backend.profile['name']} · {self.root} · índice {self.backend.index['updated_at']}"
            )
        return remote.label(self.root)

    def poll_states(self):
        if self.backend:
            if self.backend.poll():
                self.load()
        elif time.monotonic() - self.last_poll > 1:
            for roll in self.rolls:
                state = self.states.get(roll["roll"])
                if state and (
                    state["active"]
                    or workflow.job_status(self.root, roll) != state["job"]
                ):
                    del self.states[roll["roll"]]
            self.last_poll = time.monotonic()

    def preview(self, roll):
        if self.backend:
            plan = self.backend.plan([roll])
            return {
                "remaining_download_gib": plan["remaining_download_bytes"] / core.GIB
            }
        return library.preview(self.root, [row for _, row in roll["files"]])

    def guarded(self, callback, *args):
        try:
            return callback(*args)
        except ERRORS as error:
            self.states.clear()
            self.view.message("No se completó la acción", [str(error)])

    def run(self):
        query, current, top, mode = "", 0, 0, "all"
        focused = None
        while True:
            self.poll_states()
            all_rows = sorted(
                [(roll, self.status(roll)) for roll in self.rolls], key=roll_order
            )
            rows = [
                (roll, state)
                for roll, state in all_rows
                if any(
                    query.casefold() in row["name"].casefold()
                    for _, row in roll["files"]
                )
                and (
                    mode == "all"
                    or mode == "complete"
                    and state["complete"]
                    or mode == "incomplete"
                    and not state["complete"]
                    or mode == "error"
                    and state["errors"]
                    or mode == "active"
                    and state["active"]
                )
            ]
            # A completed job moves up the list. Keep actions on the same roll.
            current = next(
                (
                    index
                    for index, (roll, _) in enumerate(rows)
                    if roll["files"][0][1]["key"] == focused
                ),
                min(current, max(0, len(rows) - 1)),
            )
            focused = rows[current][0]["files"][0][1]["key"] if rows else None
            height, _ = self.view.screen.getmaxyx()
            capacity = max(1, height - 12)
            top = max(min(top, current), current - capacity + 1)
            ready = self.view.frame(
                f"Rollos · {sum(state['complete'] for _, state in all_rows)}/{len(all_rows)} completos",
                STAGE_GUIDE,
                "↑ ↓ explorar · → / Enter abrir rollo · / buscar · Esc salir",
            )
            if ready:
                label = dict(FILTERS)[mode]
                self.view.put(
                    3,
                    f"{label} · Buscar: {query or '(todos)'} · {len(rows)} resultados",
                    curses.A_DIM,
                )
                self.view.put(
                    4,
                    f"{'Rollo':<9} {'Imágenes':>13}  Estado",
                    curses.A_BOLD,
                )
                for position in range(top, min(len(rows), top + capacity)):
                    roll, state = rows[position]
                    self.view.put(
                        5 + position - top,
                        f"{roll['roll']:<9} {image_count(state):>13}  {state_label(state)}",
                        self.view.accent if position == current else 0,
                    )
                if rows:
                    roll, state = rows[current]
                    self.view.put(
                        height - 6,
                        "e → Ver imágenes y carpetas"
                        if state["complete"]
                        else "e → Ver avance en vivo"
                        if state["active"]
                        else "Enter → Reintentar rollo"
                        if state["errors"]
                        else "Enter → Completar rollo",
                        curses.A_BOLD,
                    )
                    message = (
                        state["errors"][0]
                        if state["errors"]
                        else progress_line(state)
                        if state["active"]
                        else "Todas las imágenes están listas."
                        if state["complete"]
                        else {
                            "download": "Completar descarga lo que falta y genera todas las imágenes.",
                            "extract": "Completar extrae los PDF y genera las imágenes.",
                            "render": "Completar genera las imágenes que faltan.",
                        }[state["next_step"]]
                    )
                    self.view.put(height - 5, message)
                else:
                    self.view.put(6, "No hay rollos con esta búsqueda o filtro.")
                self.view.put(height - 4, f"Carpeta: {self.root_label()}", curses.A_DIM)
                self.view.put(
                    height - 1,
                    "c completar · e abrir rollo · d eliminar · f filtrar · F2 biblioteca",
                    curses.A_DIM,
                )
            key = self.view.key(timeout=1000)
            if key in ("q", *BACK):
                return
            if not ready:
                continue
            if key in (
                curses.KEY_UP,
                curses.KEY_DOWN,
                curses.KEY_PPAGE,
                curses.KEY_NPAGE,
                curses.KEY_HOME,
                curses.KEY_END,
            ):
                delta = {
                    curses.KEY_UP: -1,
                    curses.KEY_DOWN: 1,
                    curses.KEY_PPAGE: -capacity,
                    curses.KEY_NPAGE: capacity,
                    curses.KEY_HOME: -len(rows),
                    curses.KEY_END: len(rows),
                }[key]
                current = max(0, min(max(0, len(rows) - 1), current + delta))
                focused = rows[current][0]["files"][0][1]["key"] if rows else None
            elif key in OPEN and rows:
                self.guarded(self.roll_menu, rows[current][0])
            elif key == "c" and rows:
                self.guarded(self.complete, rows[current][0])
            elif key == "d" and rows:
                self.guarded(self.remove, rows[current][0])
            elif key == "e" and rows:
                self.guarded(self.details, rows[current][0])
            elif key == "/":
                answer = self.view.edit(
                    "Buscar un rollo", query, "Número o título · Ctrl+U vacía"
                )
                if answer is not None:
                    query, current, top = answer, 0, 0
                    focused = None
            elif key == "f":
                choice = self.view.menu(
                    "Mostrar rollos",
                    [label for _, label in FILTERS],
                    initial=next(
                        i for i, pair in enumerate(FILTERS) if pair[0] == mode
                    ),
                )
                if choice is not None:
                    mode, current, top = FILTERS[choice][0], 0, 0
                    focused = None
            elif key in ("m", curses.KEY_F2):
                self.guarded(self.library_menu)
            elif key == "r":
                self.states.clear()

    def roll_menu(self, roll):
        current, top, focused = 0, 0, None
        while True:
            self.poll_states()
            self.states.pop(roll["roll"], None)
            state = self.status(roll)
            items = (
                self.backend.resources(roll)
                if self.backend
                else locations.resources(self.root, roll)
            )
            actions = []
            if not state["active"] and not state["complete"]:
                actions.append(
                    (
                        "complete",
                        "Reintentar procesamiento"
                        if state["errors"]
                        else "Descargar y procesar"
                        if state["zip_done"] < state["zip_total"]
                        else "Continuar procesamiento",
                    )
                )
            if state["label"] != "Sin descargar":
                actions.append(("remove", "Eliminar datos del rollo"))
            actions.append(("advanced", "Herramientas y registros"))
            items += [
                {"id": action, "label": label, "action": action, "section": "Acciones"}
                for action, label in actions
            ]
            if focused is None and not any(
                item.get("path") and not item.get("disabled") for item in items
            ):
                current = next(
                    (i for i, item in enumerate(items) if item.get("action")), 0
                )
            else:
                current = next(
                    (i for i, item in enumerate(items) if item["id"] == focused),
                    min(current, len(items) - 1),
                )
            enabled = [i for i, item in enumerate(items) if not item.get("disabled")]
            if current not in enabled:
                current = enabled[0]
            status = (
                state["errors"][0]
                if state["errors"]
                else progress_line(state)
                if state["active"]
                else "No hay archivos disponibles. Descargar y procesar prepara el rollo."
                if not any(
                    item.get("path") and not item.get("disabled") for item in items
                )
                else "Elegí imágenes, PDF o ZIP; su ruta aparece abajo."
            )
            current, top, capacity, ready = self.view.listing(
                f"Rollo {roll['roll']}",
                items,
                current,
                top,
                f"{state_label(state)} · {image_count(state)} imágenes finales",
                status,
                "c procesar · d borrar · p ruta · a herramientas",
            )
            focused = items[current]["id"]
            key = self.view.key(timeout=1000)
            if key in BACK or key == "q":
                return
            if not ready:
                continue
            enabled = [i for i, item in enumerate(items) if not item.get("disabled")]
            if current not in enabled:
                current = enabled[0]
            moved = enabled[
                navigate(key, enabled.index(current), len(enabled), capacity)
            ]
            if moved != current:
                current, focused = moved, items[moved]["id"]
            elif key in OPEN:
                item = items[current]
                if item.get("disabled"):
                    continue
                if item.get("action"):
                    self.guarded(getattr(self, item["action"]), roll)
                else:
                    self.guarded(self.open_location, item["path"])
            elif key == "p" and items[current].get("path"):
                self.show_path(items[current]["path"])
            elif key == "c" and not state["active"] and not state["complete"]:
                self.guarded(self.complete, roll)
            elif key == "d":
                self.guarded(self.remove, roll)
            elif key == "a":
                self.guarded(self.advanced, roll)

    def show_path(self, path):
        if self.backend:
            self.view.message("Ruta remota", [f"{self.backend.profile['host']}:{path}"])
            return
        self.view.message(
            "Ruta completa",
            [
                str(path),
                "",
                "Existe"
                if path.exists()
                else "Todavía no existe; es la ubicación prevista.",
            ],
        )

    def open_location(self, path):
        if self.backend:
            from archive_berrutti import remote_viewer

            for roll in self.rolls:
                if any(
                    path.is_relative_to(library.entry_root(self.root, row))
                    for _, row in roll["files"]
                ):
                    body = self.backend.detail(roll)
                    resource = next(
                        (
                            item
                            for item in body["resources"]
                            if item["path"] == str(path)
                        ),
                        None,
                    )
                    if resource and resource.get("disabled"):
                        return
                    if resource and resource["directory"]:
                        self.folder(path, remote_files=body["files"])
                        return
                    if path.suffix.lower() == ".zip":
                        self.view.message(
                            "ZIP · contenido",
                            [str(path), "", *body["zip_members"].get(str(path), [])],
                        )
                        return
                    remote_viewer.show(self.view, self.backend, path, body["files"])
                    return
            raise ValueError("Archivo fuera del inventario remoto")
        workflow.confined(self.root, str(path.relative_to(self.root)))
        if not path.exists():
            self.view.message(
                "Todavía no está disponible",
                [
                    str(path),
                    "",
                    "Completar lo que falta genera los archivos de este rollo.",
                ],
            )
        elif path.is_dir():
            self.folder(path)
        elif path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                members = [
                    info.filename for info in archive.infolist() if not info.is_dir()
                ]
            self.view.message("ZIP · contenido", [str(path), "", *members])
        elif path.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg"}:
            viewer.show(self.view, self.root, path)
        else:
            self.view.message(
                "Archivo",
                [
                    str(path),
                    "",
                    "Esta terminal no tiene visor gráfico. La ruta corresponde al archivo en esta máquina.",
                ],
            )

    def folder(self, path, remote_files=None):
        current, top, focused = 0, 0, None
        while True:
            paths = (
                sorted(
                    (
                        Path(item["path"])
                        for item in remote_files
                        if item["present"] and Path(item["path"]).parent == path
                    ),
                    key=lambda item: item.name.casefold(),
                )
                if remote_files is not None
                else sorted(
                    path.iterdir(),
                    key=lambda item: (not item.is_dir(), item.name.casefold()),
                )
            )
            items = [
                {
                    "id": str(item),
                    "label": (
                        "Carpeta · " if remote_files is None and item.is_dir() else ""
                    )
                    + item.name,
                    "path": item,
                }
                for item in paths
            ]
            current = next(
                (i for i, item in enumerate(items) if item["id"] == focused),
                min(current, max(0, len(items) - 1)),
            )
            current, top, capacity, ready = self.view.listing(
                f"Carpeta · {path.name}",
                items,
                current,
                top,
                f"{len(items)} archivos y carpetas",
                str(path),
            )
            key = self.view.key(timeout=1000)
            if key in BACK or key == "q":
                return
            if not ready:
                continue
            current = navigate(key, current, len(items), capacity)
            focused = items[current]["id"] if items else None
            if key in OPEN and items:
                self.guarded(self.open_location, items[current]["path"])
            elif key == "p":
                self.show_path(items[current]["path"] if items else path)

    def complete(self, roll):
        rows = [row for _, row in roll["files"]]
        state = self.status(roll)
        if state["active"]:
            self.details(roll)
            return
        estimate = self.preview(roll)
        choice = self.view.menu(
            f"Completar Rollo {roll['roll']}",
            ["Completar en segundo plano", "Cancelar"],
            lines=[
                workflow.PROFILE_LABEL,
                f"{len(rows)} ZIP · Descarga pendiente: {cli.human_bytes(estimate['remaining_download_gib'] * core.GIB)}",
                "Descarga, extrae todos los PDF y convierte todas las páginas.",
                "Reutiliza lo ya terminado. Puedes seguir explorando y cerrar la interfaz.",
                f"Destino: {self.root_label()}",
            ],
        )
        if choice == 0:
            if self.backend:
                self.view.progress("Enviando procesamiento a Slurm…")
                self.backend.action(
                    ["agent", "complete", "--rolls", roll["roll"], "--background"]
                )
            else:
                workflow.start(self.root, self.data, roll)
            self.states.pop(roll["roll"], None)

    def remove(self, roll):
        state = self.status(roll)
        choice = self.view.menu(
            f"Eliminar Rollo {roll['roll']}",
            ["Cancelar", f"Eliminar datos del Rollo {roll['roll']}"],
            lines=[
                "Borra los ZIP, PDF e imágenes de este rollo en esta biblioteca.",
                "Detiene el trabajo de este rollo antes de borrar."
                if state["active"]
                else "Incluye copias de recuperación de estos archivos.",
                "Se conserva el catálogo para poder descargarlo otra vez.",
                f"Destino: {self.root_label()}",
            ],
        )
        if choice == 1:
            if self.backend:
                self.view.progress("Eliminando los datos seleccionados en el servidor…")
                self.backend.action(
                    [
                        "agent",
                        "remove",
                        "--rolls",
                        roll["roll"],
                        "--yes",
                        "--stop-active",
                    ]
                )
            else:
                workflow.remove(self.root, roll, stop_active=True)
            self.states.pop(roll["roll"], None)

    def details(self, roll):
        self.roll_menu(roll)

    def detail_lines(self, roll):
        self.poll_states()
        self.states.pop(roll["roll"], None)
        state = self.status(roll)
        lines = [
            f"Estado: {state_label(state)}",
            STAGE_GUIDE,
        ]
        if state.get("progress"):
            current = state["progress"]
            lines += [progress_line(state)]
            if current["total"] is not None:
                lines += [
                    f"ZIP actual: {current['done']}/{current['total']} {current['unit']}"
                ]
            if current.get("bytes_total"):
                lines += [
                    f"Extracción: {cli.human_bytes(current['bytes_done'])} / {cli.human_bytes(current['bytes_total'])}"
                ]
            lines += ["El tiempo estimado corresponde a esta etapa del ZIP actual."]
        lines += [
            download_line(state),
            f"Imágenes finales: {state['images']} / {state['expected'] if state['expected'] is not None else 'total todavía desconocido'}",
            f"ZIP: {state['zip_done']}/{state['zip_total']} · PDF extraídos: {state['pdfs']}",
            workflow.PROFILE_LABEL,
        ]
        lines += [f"Error: {error}" for error in state["errors"]]
        if state["job"]:
            lines += [
                state["job"].get("message", ""),
                f"Registro: {state['job']['job']}",
                "state.json: estado actual · progress.jsonl: historial · worker.log: salida del proceso",
            ]
            if state["job"].get("scheduler") == "slurm":
                lines += [
                    f"Trabajo Slurm: {state['job'].get('slurm_job_id') or 'confirmando envío'}",
                    state["job"].get("scheduler_warning", ""),
                ]
            log = Path(state["job"]["job"]) / "worker.log"
            if state["errors"] and not self.backend and log.exists():
                with log.open("rb") as stream:
                    stream.seek(max(0, log.stat().st_size - 6000))
                    lines += (
                        stream.read()
                        .decode("utf-8", errors="replace")
                        .splitlines()[-30:]
                    )
        lines += ["", "Carpetas de imágenes finales:", *state["outputs"]]
        lines += [
            "",
            "La ficha del rollo muestra las carpetas de imágenes finales y de muestra.",
        ]
        return lines

    def library_menu(self):
        choice = self.view.menu(
            "Biblioteca",
            [
                "Cambiar carpeta",
                "Actualizar inventario",
                "Releer estados",
                "Ayuda",
                "Volver",
            ],
            self.root_label(),
        )
        if choice == 0:
            self.destination()
        elif choice == 1:
            if self.args.offline:
                raise ValueError(
                    "Modo offline: importar otro snapshot con refresh --metadata"
                )
            if self.backend:
                self.view.progress("Actualizando inventario e índice bajo Slurm…")
                self.backend.action(["agent", "refresh"])
            else:
                library.refresh(self.root)
            self.load()
        elif choice == 2:
            if self.backend:
                self.backend.poll(force=True)
            self.states.clear()
        elif choice == 3:
            self.view.message(
                "Uso básico",
                [
                    "El listado muestra rollo, imágenes finales y un único estado. Enter o → abre la ficha del rollo; ← o Esc vuelve.",
                    "Orden: completos, procesando/en cola, incompletos/con error y sin descargar; alfabético dentro de cada grupo.",
                    "Sin descargar no implica un trabajo pendiente. En cola sí indica un trabajo solicitado esperando su turno.",
                    "Completar descarga todos sus ZIP, extrae todos los PDF y guarda cada página en PNG a 200 DPI.",
                    "Completo exige que estén todas las imágenes finales. Las muestras de pocas páginas y otras conversiones no cuentan como finales.",
                    "Falta descargar / extraer PDF / convertir a imágenes indica el próximo paso. Enter → Completar realiza todos los pasos restantes.",
                    "La ficha muestra los ZIP, PDF y carpetas de imágenes. Al mover el cursor se ve la ruta completa; → abre la carpeta o archivo. p muestra la ruta en una pantalla propia.",
                    "La columna Imágenes cuenta la salida final una sola vez, sin sumar otras versiones.",
                    "Imágenes muestra hechas/total cuando se conoce el total. Las etapas 1 Descarga → 2 Extraer PDF → 3 Crear imágenes se repiten por ZIP.",
                    "El tiempo restante se estima con avance reciente para la etapa del ZIP actual; aparece al reunir suficientes muestras.",
                    "Una sola acción descarga y procesa lo que falta; pasa a Continuar o Reintentar según el estado. Los archivos ausentes están atenuados y las flechas los saltean.",
                    "Enter → Eliminar borra los datos locales de ese rollo después de confirmarlo.",
                    "Los trabajos se ejecutan de uno en uno en segundo plano y continúan al cerrar la interfaz. En Slurm duran como máximo lo que permita la asignación.",
                    "PDF e imágenes se abren con el visor del sistema si hay escritorio gráfico; por SSH se muestra la ruta en la máquina remota.",
                ],
            )

    def advanced(self, roll):
        rows = [row for _, row in roll["files"]]
        labels = [
            "Extraer solo PDF",
            "Conversión personalizada",
            "Verificar integridad de los archivos",
            "Ver registros y detalle técnico",
            "Descargar solo ZIP (sin procesar)",
            "Volver",
        ]
        choice = self.view.menu(f"Herramientas · Rollo {roll['roll']}", labels)
        if choice is None or choice == 5:
            return
        if choice == 4:
            self.download_only(roll)
            return
        if choice == 3:
            self.view.message(
                f"Registros · Rollo {roll['roll']}", lambda: self.detail_lines(roll)
            )
            return
        opts = self.render_settings() if choice == 1 else cli.options()
        if opts is None:
            return
        if (
            self.view.menu(
                labels[choice],
                ["Cancelar", "Ejecutar"],
                lines=[
                    f"Rollo {roll['roll']} · {len(rows)} ZIP",
                    f"Destino: {self.root}",
                ],
            )
            != 1
        ):
            return
        self.view.progress(labels[choice])
        try:
            if self.backend:
                arguments = [
                    {0: "extract", 1: "render", 2: "verify"}[choice],
                    "--rolls",
                    roll["roll"],
                ]
                if choice == 1:
                    for name in (
                        "pages",
                        "first_pages",
                        "dpi",
                        "format",
                        "quality",
                        "timeout",
                        "max_total_pages",
                    ):
                        value = getattr(opts, name, None)
                        if value is not None:
                            arguments += ["--" + name.replace("_", "-"), str(value)]
                    if opts.all_pages:
                        arguments.append("--all-pages")
                self.backend.action(arguments)
                return
            with contextlib.redirect_stdout(ProgressStream(self.view.progress)):
                if choice == 2:
                    library.verify(self.root, rows)
                else:
                    library.operate(
                        self.root,
                        self.data,
                        rows,
                        {0: "extract", 1: "render"}[choice],
                        opts,
                        progress=self.view.progress,
                    )
        finally:
            self.states.clear()
        self.view.message(
            "Operación terminada", ["El listado actualiza el estado final del rollo."]
        )

    def download_only(self, roll):
        state = self.status(roll)
        if state["active"]:
            return
        rows = [row for _, row in roll["files"]]
        estimate = self.preview(roll)
        confirmed = self.view.menu(
            f"Descargar ZIP · Rollo {roll['roll']}",
            ["Descargar en segundo plano", "Cancelar"],
            lines=[
                f"{len(rows)} ZIP · Faltan {cli.human_bytes(estimate['remaining_download_gib'] * core.GIB)}",
                "La ficha del rollo muestra el avance en vivo.",
                f"Destino: {self.root}",
            ],
        )
        if confirmed == 0:
            if self.backend:
                self.view.progress("Enviando descarga a Slurm…")
                self.backend.action(
                    ["agent", "download", "--rolls", roll["roll"], "--background"]
                )
            else:
                workflow.start(self.root, self.data, roll, action="download")
            self.states.pop(roll["roll"], None)

    def render_settings(self):
        settings = cli.options()
        current = 0
        while True:
            pages = (
                f"primeras {settings.first_pages}"
                if settings.first_pages
                else settings.pages or "todas"
            )
            choice = self.view.menu(
                "Configurar imágenes",
                [
                    f"Páginas: {pages}",
                    f"Resolución: {settings.dpi} DPI",
                    f"Formato: {settings.format.upper()}",
                    f"Máximo global: {settings.max_total_pages} páginas",
                    "Continuar a la confirmación",
                    "Cancelar",
                ],
                initial=current,
            )
            if choice is None or choice == 5:
                return None
            if choice == 4:
                return settings
            current = choice
            try:
                if choice == 0:
                    mode = self.view.menu(
                        "Páginas por PDF",
                        [
                            "Primeras N páginas",
                            "Páginas concretas",
                            "Todas las páginas",
                        ],
                    )
                    if mode is None:
                        continue
                    if mode == 2:
                        settings.first_pages, settings.pages, settings.all_pages = (
                            None,
                            None,
                            True,
                        )
                        continue
                    answer = self.view.edit(
                        "Primeras páginas" if mode == 0 else "Números de página",
                        str(settings.first_pages or 5)
                        if mode == 0
                        else settings.pages or "1-5,10",
                    )
                    if answer is not None:
                        if mode == 0:
                            settings.first_pages = core.positive(answer)
                            settings.pages, settings.all_pages = None, False
                        else:
                            # Syntax is validated before an operation starts; PDF limits are checked by the engine.
                            for token in answer.split(","):
                                match = re.fullmatch(r"(\d+)(?:-(\d+))?", token.strip())
                                if not match or not 1 <= int(match[1]) <= int(
                                    match[2] or match[1]
                                ):
                                    raise ValueError(
                                        "Usar páginas positivas o rangos: 1-5,10"
                                    )
                            settings.first_pages, settings.pages, settings.all_pages = (
                                None,
                                answer,
                                False,
                            )
                elif choice == 2:
                    fmt = self.view.menu(
                        "Formato de imagen", ["JPEG (calidad 95)", "PNG (sin pérdida)"]
                    )
                    if fmt is not None:
                        settings.format = ("jpeg", "png")[fmt]
                else:
                    field = "dpi" if choice == 1 else "max_total_pages"
                    answer = self.view.edit(
                        "Resolución DPI" if choice == 1 else "Límite total de páginas",
                        str(getattr(settings, field)),
                    )
                    if answer is not None:
                        setattr(settings, field, core.positive(answer))
            except ERRORS as error:
                self.view.message("Valor inválido", [str(error)])

    def destination(self):
        profile = self.backend.profile if self.backend else remote.runtime()
        answer = self.view.edit(
            "Carpeta de biblioteca"
            + (" en " + profile["name"] if profile else " local"),
            str(self.root),
            "Ctrl+U vacía · abre una biblioteca remota con índice publicado"
            if self.backend
            else "Ctrl+U vacía · abre una biblioteca existente o crea otra; los datos permanecen en su carpeta",
        )
        if answer is None or not answer:
            return
        if self.backend:
            from archive_berrutti.remote_library import Client

            candidate = Client(self.backend.profile, answer)
            self.backend.close()
            self.backend = candidate
            self.root = candidate.root
            self.load()
            return
        candidate = library.open_library(Path(answer), create=True)
        if not (candidate / "active_catalog.json").exists():
            with core.locked(candidate):
                snapshot = Path("catalogs") / f"{library.identifier()}.json"
                core.save_new(candidate / snapshot, self.data)
                library.replace_json(
                    candidate / "active_catalog.json",
                    {"path": str(snapshot), "sha256": core.digest(self.data)},
                )
        # Validate the target before changing the live session or remembered path.
        library.load_catalog(candidate)
        self.root = candidate
        self.load()
        library.remember(self.root)


def run(args, backend=None):
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise ValueError(
            "El explorador necesita una terminal interactiva. Para scripts usa list, status o plan; por SSH usa ssh -t."
        )
    if os.environ.get("TERM", "dumb") == "dumb":
        raise ValueError(
            "La terminal necesita TERM válido (por ejemplo xterm-256color)."
        )
    try:

        def start(screen):
            app = Application(Terminal(screen), args, backend=backend)
            try:
                app.run()
            finally:
                if app.backend:
                    app.backend.close()

        curses.wrapper(start)
    except curses.error as error:
        raise ValueError(f"No se pudo abrir la terminal: {error}") from error

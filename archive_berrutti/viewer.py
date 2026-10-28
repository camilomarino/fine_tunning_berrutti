"""Local image/PDF viewer: native terminal graphics or a portable block preview.

Protocol references:
https://sw.kovidgoyal.net/kitty/graphics-protocol/
https://iterm2.com/documentation-images.html
"""

from __future__ import annotations

import base64
import curses
import fcntl
import io
import math
import os
import re
import secrets
import shutil
import struct
import subprocess
import sys
import tempfile
import termios
import threading
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from archive_berrutti import manage as core, workflow

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
MAX_EDGE = 2400
MODES = ("auto", "blocks", "kitty", "iterm2")
MODE_LABELS = (
    "Automática según la terminal",
    "Vista aproximada por caracteres · compatible por SSH",
    "Imagen gráfica · terminal con protocolo Kitty",
    "Imagen gráfica · terminal con protocolo iTerm2",
)


def backend(mode="auto", environ=None):
    if mode not in MODES:
        raise ValueError("Vista inválida: usar auto, blocks, kitty o iterm2")
    if mode != "auto":
        return mode
    env = os.environ if environ is None else environ
    # Multiplexers can block or transform graphics sequences. Never change their
    # configuration; users can select a native mode when passthrough is configured.
    if env.get("TMUX") or env.get("STY"):
        return "blocks"
    if env.get("TERM") == "xterm-kitty" or env.get("KITTY_WINDOW_ID"):
        return "kitty"
    if env.get("TERM_PROGRAM") == "iTerm.app" or env.get("ITERM_SESSION_ID"):
        return "iterm2"
    return "blocks"


class Document:
    """Keep one decoded page and one temporary raster; never generate a corpus."""

    def __init__(self, root, path):
        self.root = root
        self.path = workflow.confined(root, str(path.relative_to(root)))
        self.pdf = path.suffix.lower() == ".pdf"
        self.index = 0
        self.loaded = None
        self.image = None
        self.temporary = None
        if self.pdf:
            if not all(shutil.which(tool) for tool in ("pdfinfo", "pdftoppm")):
                raise ValueError(
                    "Para ver PDF se necesitan pdfinfo y pdftoppm (Poppler)"
                )
            info = core.run(["pdfinfo", str(path)], 20)
            match = re.search(r"^Pages:\s+(\d+)$", info, re.MULTILINE)
            if not match or int(match[1]) < 1:
                raise ValueError("No se pudo contar las páginas del PDF")
            self.count = int(match[1])
            self.paths = [path]
        else:
            self.paths = sorted(
                (
                    p
                    for p in path.parent.iterdir()
                    if p.is_file()
                    and p.suffix.lower() in IMAGE_SUFFIXES
                    and ".part." not in p.name.casefold()
                ),
                key=lambda p: p.name.casefold(),
            )
            self.index = self.paths.index(path)
            self.count = len(self.paths)

    @property
    def current_path(self):
        return self.path if self.pdf else self.paths[self.index]

    def load(self):
        if self.loaded == self.index:
            return self.image
        source = workflow.confined(
            self.root, str(self.current_path.relative_to(self.root))
        )
        if self.pdf:
            if self.temporary is None:
                self.temporary = tempfile.TemporaryDirectory(prefix="berrutti-viewer-")
            prefix = Path(self.temporary.name) / "page"
            core.run(
                [
                    "pdftoppm",
                    "-f",
                    str(self.index + 1),
                    "-l",
                    str(self.index + 1),
                    "-singlefile",
                    "-scale-to",
                    str(MAX_EDGE),
                    "-png",
                    str(source),
                    str(prefix),
                ],
                60,
            )
            source = prefix.with_suffix(".png")
        with Image.open(source) as original:
            oriented = ImageOps.exif_transpose(original)
            rgba = oriented.convert("RGBA")
            canvas = Image.new("RGB", rgba.size, "white")
            canvas.paste(rgba, mask=rgba.getchannel("A"))
            rgba.close()
            oriented.close()
        if self.image is not None:
            self.image.close()
        self.image, self.loaded = canvas, self.index
        return self.image

    def close(self):
        if self.image is not None:
            self.image.close()
        if self.temporary is not None:
            self.temporary.cleanup()


@dataclass
class Viewport:
    zoom: float = 1.0
    x: float = 0.5
    y: float = 0.5

    def render(self, source, width, height):
        scale = min(width / source.width, height / source.height) * self.zoom
        visible_w, visible_h = width / scale, height / scale
        left = (source.width - visible_w) * (
            self.x if visible_w < source.width else 0.5
        )
        top = (source.height - visible_h) * (
            self.y if visible_h < source.height else 0.5
        )
        # EXTENT transforms point-sample when shrinking; fine text can alias or
        # disappear. Resize the visible source box with a downsampling filter,
        # and compose only the viewport, never a huge zoomed intermediate image.
        right, bottom = left + visible_w, top + visible_h
        box = (
            max(0, left),
            max(0, top),
            min(source.width, right),
            min(source.height, bottom),
        )
        x0, y0 = round((box[0] - left) * scale), round((box[1] - top) * scale)
        x1, y1 = round((box[2] - left) * scale), round((box[3] - top) * scale)
        canvas = Image.new("RGB", (width, height), (32, 32, 32))
        if x1 > x0 and y1 > y0:
            with source.resize(
                (x1 - x0, y1 - y0), Image.Resampling.LANCZOS, box=box
            ) as visible:
                canvas.paste(visible, (x0, y0))
        return canvas


def cell_pixels():
    try:
        rows, columns, width, height = struct.unpack(
            "HHHH", fcntl.ioctl(sys.stdout.fileno(), termios.TIOCGWINSZ, b"\0" * 8)
        )
        if all((rows, columns, width, height)):
            return width / columns, height / rows
    except (OSError, ValueError, io.UnsupportedOperation):
        pass
    return 8, 16  # Many SSH sessions report cells but no pixel dimensions.


def png_bytes(image):
    with io.BytesIO() as stream:
        image.save(stream, "PNG")
        return stream.getvalue()


def graphic_bytes(image, mode, image_id, columns, rows):
    payload = base64.b64encode(png_bytes(image))
    if mode == "kitty":
        chunks = []
        for offset in range(0, len(payload), 4096):
            header = (
                f"a=T,f=100,t=d,i={image_id},q=2,C=1,c={columns},r={rows},"
                if offset == 0
                else ""
            )
            more = int(offset + 4096 < len(payload))
            chunks.append(
                f"\x1b_G{header}m={more};".encode()
                + payload[offset : offset + 4096]
                + b"\x1b\\"
            )
        return b"".join(chunks)
    # Always inline; never use iTerm2's file-download mode. Large payloads use
    # the documented multipart transfer to stay below the OSC size limit.
    settings = f"inline=1;width={columns};height={rows};preserveAspectRatio=0".encode()
    if len(payload) < 512 * 1024:
        return b"\x1b]1337;File=" + settings + b":" + payload + b"\a"
    return (
        b"\x1b]1337;MultipartFile="
        + settings
        + b"\a"
        + b"".join(
            b"\x1b]1337;FilePart=" + payload[i : i + 4096] + b"\a"
            for i in range(0, len(payload), 4096)
        )
        + b"\x1b]1337;FileEnd\a"
    )


class Painter:
    def __init__(self, view, mode="auto"):
        self.view = view
        self.mode = mode
        self.image_id = secrets.randbelow(2**31 - 1) + 1
        self.visible = None
        self.shades = []
        if curses.has_colors() and curses.COLOR_PAIRS >= 6:
            if curses.COLORS >= 256:
                # Stay below color-pair 256, including on ncurses builds that
                # advertise more pairs but expose only 8-bit color attributes.
                levels = min(15, math.isqrt(min(256, curses.COLOR_PAIRS) - 2))
                self.shades = [
                    16
                    if i == 0
                    else 231
                    if i == levels - 1
                    else 232 + round((255 * i / (levels - 1) - 8) / 10)
                    for i in range(levels)
                ]
            else:
                self.shades = [curses.COLOR_BLACK, curses.COLOR_WHITE]
            for top, fg in enumerate(self.shades):
                for bottom, bg in enumerate(self.shades):
                    curses.init_pair(2 + top * len(self.shades) + bottom, fg, bg)

    def write(self, data):
        sys.stdout.flush()
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()

    def clear(self):
        if self.visible == "kitty":
            self.write(f"\x1b_Ga=d,d=I,i={self.image_id},q=2;\x1b\\".encode())
        elif self.visible == "iterm2":
            self.write(b"\x1b[2J")
        self.visible = None
        if hasattr(self.view.screen, "clearok"):
            self.view.screen.clearok(True)

    def paint(self, source, viewport, columns, rows):
        mode = backend(self.mode)
        cell_w, cell_h = cell_pixels()
        width, height = max(1, round(columns * cell_w)), max(1, round(rows * cell_h))
        ratio = min(1, MAX_EDGE / max(width, height))
        with viewport.render(
            source, max(1, round(width * ratio)), max(1, round(height * ratio))
        ) as image:
            if mode != "blocks":
                self.view.screen.refresh()
                # Native graphics bypass curses; save/restore cursor and repaint
                # the text buffer fully on the next view to avoid stale regions.
                self.visible = mode
                self.write(
                    b"\x1b7\x1b[6;3H"
                    + graphic_bytes(image, mode, self.image_id, columns, rows)
                    + b"\x1b8"
                )
                return
            with image.convert("L").resize(
                (columns, rows * 2), Image.Resampling.LANCZOS
            ) as small:
                pixels = small.load()
                levels = len(self.shades)
                for y in range(rows):
                    for x in range(columns):
                        if levels:
                            top = round(pixels[x, 2 * y] * (levels - 1) / 255)
                            bottom = round(pixels[x, 2 * y + 1] * (levels - 1) / 255)
                            char, attr = (
                                "▀",
                                curses.color_pair(2 + top * levels + bottom),
                            )
                        else:
                            top, bottom = (
                                pixels[x, 2 * y] >= 128,
                                pixels[x, 2 * y + 1] >= 128,
                            )
                            char, attr = (" ", "▄", "▀", "█")[2 * top + bottom], 0
                        try:
                            self.view.screen.addnstr(y + 5, x + 2, char, 1, attr)
                        except curses.error:
                            pass


def external(path):
    if not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ) or not shutil.which("xdg-open"):
        raise ValueError(
            "No hay escritorio gráfico disponible en esta sesión. Usá la vista de terminal o la ruta del archivo."
        )
    process = subprocess.Popen(
        ["xdg-open", str(path)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    threading.Thread(target=process.wait, daemon=True).start()


def show(view, root, path, *, document=None):
    mode = os.environ.get("ARCHIVE_BERRUTTI_VIEWER", "auto")
    backend(mode)  # Validate before allocating a PDF raster or altering the screen.
    if path.suffix.lower() == ".pdf":
        view.progress("Abriendo PDF · contando páginas…")
    document = document or Document(root, path)
    painter = None
    try:
        painter = Painter(view, mode)
        viewport = Viewport()
        while True:
            painter.clear()
            height, width = view.screen.getmaxyx()
            kind = "PDF · página" if document.pdf else "Imagen"
            ready = view.frame(
                f"{kind} {document.index + 1}/{document.count} · {document.current_path.name}",
                f"Zoom {viewport.zoom:.1f}× · "
                + (
                    "Vista aproximada por caracteres; v cambia la vista"
                    if backend(painter.mode) == "blocks"
                    else "Vista gráfica en la terminal"
                ),
                "↑ ↓ anterior/siguiente · + − zoom · 0 ajustar · ← / Esc volver",
            )
            if ready:
                view.put(
                    3,
                    "WASD desplazar al ampliar · g ir a página/imagen",
                    curses.A_DIM,
                )
                view.put(
                    height - 1,
                    "p ruta completa · v tipo de vista · o visor externo · ? ayuda",
                    curses.A_DIM,
                )
                if document.pdf and document.loaded != document.index:
                    view.put(4, "Preparando esta página del PDF…", curses.A_DIM)
                    view.screen.refresh()
                try:
                    image = document.load()
                    view.put(4, f"{image.width} × {image.height} píxeles", curses.A_DIM)
                    painter.paint(
                        image, viewport, max(1, width - 4), max(1, height - 8)
                    )
                except (
                    OSError,
                    ValueError,
                    Image.DecompressionBombError,
                    subprocess.SubprocessError,
                ) as error:
                    view.put(4, f"No se pudo mostrar esta página: {error}")
            key = view.key()
            if isinstance(key, str):
                key = key.lower()
            if key in (curses.KEY_LEFT, "\x1b", "q"):
                return
            if not ready:
                continue
            previous = document.index
            if key in (curses.KEY_DOWN, curses.KEY_NPAGE, "n", " "):
                document.index = min(document.count - 1, document.index + 1)
            elif key in (curses.KEY_UP, curses.KEY_PPAGE, "b"):
                document.index = max(0, document.index - 1)
            elif key == curses.KEY_HOME:
                document.index = 0
            elif key == curses.KEY_END:
                document.index = document.count - 1
            elif key in ("+", "="):
                viewport.zoom = min(16, viewport.zoom * 1.5)
            elif key == "-":
                viewport.zoom = max(1, viewport.zoom / 1.5)
            elif key == "0":
                viewport = Viewport()
            elif key in ("w", "s", "k", "j"):
                viewport.y = min(
                    1, max(0, viewport.y + (0.1 if key in ("s", "j") else -0.1))
                )
            elif key in ("a", "d", "h", "l"):
                viewport.x = min(
                    1, max(0, viewport.x + (0.1 if key in ("d", "l") else -0.1))
                )
            elif key in ("g", "p", "v", "o", "?"):
                painter.clear()
                if key == "g":
                    value = view.edit(
                        f"Ir a {'página' if document.pdf else 'imagen'} (1–{document.count})"
                    )
                    if value is not None:
                        if value.isdigit() and 1 <= int(value) <= document.count:
                            document.index = int(value) - 1
                        else:
                            view.message(
                                "Número fuera de rango",
                                [f"Ingresá un número entre 1 y {document.count}."],
                            )
                elif key == "p":
                    view.message("Ruta completa", [str(document.current_path)])
                elif key == "v":
                    choice = view.menu(
                        "Tipo de vista",
                        MODE_LABELS,
                        initial=MODES.index(painter.mode),
                        lines=[
                            "La vista gráfica requiere soporte de tu terminal.",
                            "Si no aparece la imagen, volvé con v y elegí caracteres.",
                        ],
                    )
                    if choice is not None:
                        painter.mode = MODES[choice]
                elif key == "o":
                    try:
                        external(
                            getattr(document, "external_path", document.current_path)
                        )
                    except (OSError, ValueError, subprocess.SubprocessError) as error:
                        view.message("Visor externo", [str(error)])
                else:
                    view.message(
                        "Visor · ayuda",
                        [
                            "↑ / ↓ y PgUp / PgDn: página o imagen anterior/siguiente.",
                            "Home / End: primera/última. g: ir a un número.",
                            "+ / −: ampliar/reducir. 0: ajustar a la ventana.",
                            "W / S: desplazar arriba/abajo al ampliar. A / D: desplazar a los lados.",
                            "También se puede desplazar con h / j / k / l.",
                            "p: ruta completa. o: abrir en visor de escritorio, si está disponible.",
                            "v: elegir gráficos Kitty/iTerm2 o vista por caracteres.",
                            "La vista por caracteres permite reconocer la página; no ofrece la nitidez de una imagen gráfica para leer texto pequeño.",
                            "Solo se visualiza la página elegida. Los PDF originales, imágenes finales y estados del rollo no se modifican.",
                            "← / Esc / q: volver a la carpeta o ficha del rollo.",
                        ],
                    )
            if previous != document.index:
                viewport = Viewport()
    finally:
        try:
            if painter is not None:
                painter.clear()
        finally:
            document.close()

"""Keyboard, read-only PDF preview, zoom and native graphics transport."""

from __future__ import annotations

import base64
import curses
import io
import os
import re
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageStat

from tests.data.test_archive_library import LibraryFixture, ScriptedScreen
from archive_berrutti import library, manage as core, tui, viewer


class ViewerTests(LibraryFixture):
    def images(self):
        directory = self.root / "preview-test"
        directory.mkdir()
        paths = []
        for i, color in enumerate(("white", "gray", "black"), 1):
            path = directory / f"page-{i:03d}.png"
            Image.new("RGB", (60, 80), color).save(path)
            paths.append(path)
        return paths

    def show(self, path, keys):
        screen = ScriptedScreen(keys)
        with (
            patch.object(curses, "has_colors", return_value=False),
            patch.object(curses, "curs_set"),
            patch.dict(os.environ, {"ARCHIVE_BERRUTTI_VIEWER": "blocks"}),
        ):
            viewer.show(tui.Terminal(screen), self.root, path)
        with self.assertRaises(StopIteration):
            next(screen.keys)
        return "\n".join(screen.frames)

    def test_image_viewer_navigates_zooms_pans_and_returns_without_writes(self):
        paths = self.images()
        Image.new("RGB", (10, 10)).save(paths[0].parent / "page.part.png")
        before = {path: path.stat().st_mtime_ns for path in self.root.rglob("*")}
        frames = self.show(
            paths[0],
            [
                curses.KEY_DOWN,
                "+",
                "s",
                "d",
                "p",
                "\x1b",
                "g",
                "3",
                "\n",
                curses.KEY_UP,
                "0",
                curses.KEY_LEFT,
            ],
        )
        self.assertIn("Imagen 1/3", frames)
        self.assertIn("Imagen 2/3", frames)
        self.assertIn("Imagen 3/3", frames)
        self.assertIn("Zoom 1.5×", frames)
        self.assertIn(str(paths[1]), frames)
        self.assertEqual(
            before, {path: path.stat().st_mtime_ns for path in self.root.rglob("*")}
        )

    @unittest.skipUnless(
        shutil.which("pdfinfo") and shutil.which("pdftoppm"), "Poppler missing"
    )
    def test_pdf_renders_only_requested_pages_and_cleans_private_temporary_files(self):
        row = self.data["files"][0]
        self.operate([row], "download")
        self.operate([row], "extract")
        directory = library.entry_root(self.root, row)
        pdf = next((directory / "pdfs" / row["key"]).glob("*.pdf"))
        before = {path: path.stat().st_mtime_ns for path in self.root.rglob("*")}
        with patch.object(core, "run", wraps=core.run) as run:
            frames = self.show(pdf, [curses.KEY_DOWN, "+", "s", curses.KEY_LEFT])
        calls = [
            call.args[0] for call in run.call_args_list if call.args[0][0] == "pdftoppm"
        ]
        self.assertEqual(len(calls), 2)  # Zoom/pan reuse the current raster.
        for page, command in enumerate(calls, 1):
            self.assertEqual(command[command.index("-f") + 1], str(page))
            self.assertEqual(command[command.index("-l") + 1], str(page))
            self.assertIn("-singlefile", command)
            self.assertIn("-scale-to", command)
            self.assertFalse(Path(command[-1]).parent.exists())
        self.assertIn("PDF · página 1/3", frames)
        self.assertIn("PDF · página 2/3", frames)
        self.assertEqual(
            before, {path: path.stat().st_mtime_ns for path in self.root.rglob("*")}
        )
        self.assertFalse((directory / "renders").exists())

    def test_unreadable_page_stays_in_viewer_and_next_image_can_be_opened(self):
        paths = self.images()
        paths[0].write_bytes(b"not an image")
        frames = self.show(paths[0], [curses.KEY_DOWN, curses.KEY_LEFT])
        self.assertIn("No se pudo mostrar esta página", frames)
        self.assertIn("Imagen 2/3", frames)

    def test_native_image_is_cleared_for_path_dialog_and_when_returning(self):
        path = self.images()[0]
        for mode in ("kitty", "iterm2"):
            screen = ScriptedScreen(["p", "\x1b", curses.KEY_LEFT])
            writes = []
            with (
                self.subTest(mode=mode),
                patch.object(curses, "has_colors", return_value=False),
                patch.object(curses, "curs_set"),
                patch.dict(os.environ, {"ARCHIVE_BERRUTTI_VIEWER": mode}),
                patch.object(viewer.Painter, "write", side_effect=writes.append),
            ):
                viewer.show(tui.Terminal(screen), self.root, path)
            self.assertEqual(len(writes), 4)  # Display, clear, display, clear.
            self.assertEqual(writes[1], writes[3])
            self.assertIn(str(path), "\n".join(screen.frames))
            if mode == "kitty":
                image_id = re.search(rb"i=(\d+)", writes[0])[1]
                self.assertIn(b"a=d,d=I,i=" + image_id + b",q=2", writes[-1])
            else:
                self.assertEqual(writes[-1], b"\x1b[2J")

    def test_mode_selection_and_invalid_page_keep_viewer_open(self):
        path = self.images()[0]
        frames = self.show(
            path,
            [
                "g",
                "9",
                "\n",
                "\x1b",
                "v",
                curses.KEY_HOME,
                curses.KEY_DOWN,
                "\n",
                "?",
                "\x1b",
                "\x1b",
            ],
        )
        self.assertIn("Número fuera de rango", frames)
        self.assertIn("Tipo de vista", frames)
        self.assertIn("Visor · ayuda", frames)

    def test_up_down_keep_navigating_pages_when_zoomed_and_left_always_returns(self):
        path = self.images()[0]
        frames = self.show(
            path,
            [
                curses.KEY_UP,
                "+",
                curses.KEY_DOWN,
                "+",
                curses.KEY_UP,
                "+",
                curses.KEY_LEFT,
            ],
        )
        titles = [line for line in frames.splitlines() if line.startswith("Imagen ")]
        self.assertEqual(
            [title.split()[1] for title in titles],
            [
                "1/3",
                "1/3",
                "1/3",
                "2/3",
                "2/3",
                "1/3",
                "1/3",
            ],
        )
        self.assertIn("↑ ↓ anterior/siguiente", frames)
        self.assertIn("← / Esc volver", frames)

    def test_section_headings_are_skipped_and_actions_remain_reachable_when_small(self):
        items = [
            {
                "label": "Imágenes",
                "section": "Archivos",
                "path": self.root / ("long-path-" * 20),
            },
            {"label": "PDF", "section": "Archivos"},
            {"label": "ZIP", "section": "Archivos"},
            {"label": "Eliminar", "section": "Acciones"},
            {"label": "Herramientas", "section": "Acciones"},
        ]
        with (
            patch.object(curses, "has_colors", return_value=False),
            patch.object(curses, "curs_set"),
        ):
            screen = ScriptedScreen([], size=(16, 70))
            view = tui.Terminal(screen)
            current, top, capacity, ready = view.listing("Rollo", items, 0, 0)
            self.assertTrue(ready)
            self.assertIn("ARCHIVOS", "\n".join(screen.lines.values()))
            for _ in range(3):
                current = tui.navigate(curses.KEY_DOWN, current, len(items), capacity)
                current, top, capacity, _ = view.listing("Rollo", items, current, top)
            self.assertEqual(items[current]["label"], "Eliminar")
            self.assertIn("ACCIONES", "\n".join(screen.lines.values()))
            self.assertIn("› Eliminar", "\n".join(screen.lines.values()))
            current = tui.navigate(curses.KEY_END, current, len(items), capacity)
            view.listing("Rollo", items, current, top)
            self.assertIn("› Herramientas", "\n".join(screen.lines.values()))


class GraphicsTests(unittest.TestCase):
    def test_downsampling_preserves_fine_line_density_without_false_bands(self):
        # A quarter of each four-row period is ink. Shrinking must preserve its
        # average coverage instead of sampling individual white/black rows.
        with Image.new("RGB", (2048, 2048), "white") as source:
            for y in range(0, 2048, 4):
                source.paste("black", (0, y, 2048, y + 1))
            for size in (32, 257):
                with (
                    self.subTest(size=size),
                    viewer.Viewport().render(source, size, size) as small,
                ):
                    stats = ImageStat.Stat(small)
                    self.assertAlmostEqual(stats.mean[0], 255 * 0.75, delta=2)
                    self.assertLess(stats.stddev[0], 2)

    def test_grayscale_uses_true_black_and_white_with_portable_color_pairs(self):
        for pairs in (64, 256, 65536):
            with (
                self.subTest(pairs=pairs),
                patch.object(curses, "has_colors", return_value=True),
                patch.object(curses, "COLORS", 256, create=True),
                patch.object(curses, "COLOR_PAIRS", pairs, create=True),
                patch.object(curses, "init_pair") as init,
            ):
                painter = viewer.Painter(None)
            self.assertEqual((painter.shades[0], painter.shades[-1]), (16, 231))
            self.assertTrue(
                all(call.args[0] < min(pairs, 256) for call in init.call_args_list)
            )
            if pairs >= 256:
                self.assertGreater(len(painter.shades), 8)

    def test_backend_is_portable_and_multiplexers_require_explicit_choice(self):
        self.assertEqual(viewer.backend(environ={"TERM": "xterm-256color"}), "blocks")
        self.assertEqual(viewer.backend(environ={"TERM": "xterm-kitty"}), "kitty")
        self.assertEqual(
            viewer.backend(environ={"TERM_PROGRAM": "iTerm.app"}), "iterm2"
        )
        self.assertEqual(
            viewer.backend(environ={"TERM": "xterm-kitty", "TMUX": "test"}), "blocks"
        )
        self.assertEqual(viewer.backend("kitty", {"TMUX": "test"}), "kitty")

    def test_kitty_transport_contains_bounded_png_chunks_and_suppresses_replies(self):
        with Image.frombytes("RGB", (128, 128), os.urandom(128 * 128 * 3)) as image:
            stream = viewer.graphic_bytes(image, "kitty", 123, 80, 20)
            chunks = re.findall(rb"\x1b_G([^;]+);(.*?)\x1b\\", stream)
            self.assertGreater(len(chunks), 1)
            self.assertIn(b"i=123,q=2,C=1,c=80,r=20", chunks[0][0])
            self.assertEqual(chunks[-1][0], b"m=0")
            self.assertTrue(all(len(payload) <= 4096 for _, payload in chunks))
            decoded = base64.b64decode(b"".join(payload for _, payload in chunks))
            with Image.open(io.BytesIO(decoded)) as restored:
                self.assertEqual(restored.tobytes(), image.tobytes())

    def test_iterm_transports_inline_png_and_large_payloads_in_chunks(self):
        with Image.new("RGB", (10, 20), "white") as image:
            stream = viewer.graphic_bytes(image, "iterm2", 1, 20, 10)
            self.assertTrue(stream.startswith(b"\x1b]1337;File=inline=1;"))
            payload = stream.split(b":", 1)[1].removesuffix(b"\a")
            with Image.open(io.BytesIO(base64.b64decode(payload))) as restored:
                self.assertEqual(restored.size, image.size)
        with Image.frombytes("RGB", (512, 512), os.urandom(512 * 512 * 3)) as image:
            stream = viewer.graphic_bytes(image, "iterm2", 1, 20, 10)
            self.assertTrue(stream.startswith(b"\x1b]1337;MultipartFile=inline=1;"))
            parts = re.findall(rb"\x1b]1337;FilePart=([^\a]+)\a", stream)
            self.assertTrue(all(len(part) <= 4096 for part in parts))
            self.assertTrue(stream.endswith(b"\x1b]1337;FileEnd\a"))
            with Image.open(io.BytesIO(base64.b64decode(b"".join(parts)))) as restored:
                self.assertEqual(restored.tobytes(), image.tobytes())

    def test_zoom_pan_crop_preserves_aspect_and_changes_visible_region(self):
        with Image.new("RGB", (100, 200), "white") as source:
            source.paste("black", (0, 0, 100, 100))
            with viewer.Viewport().render(source, 100, 100) as fit:
                self.assertEqual(fit.getpixel((0, 50)), (32, 32, 32))
                self.assertEqual(fit.getpixel((50, 75)), (255, 255, 255))
            with viewer.Viewport(zoom=2, y=0).render(source, 100, 100) as top:
                self.assertEqual(top.getpixel((50, 50)), (0, 0, 0))
            with viewer.Viewport(zoom=2, y=1).render(source, 100, 100) as bottom:
                self.assertEqual(bottom.getpixel((50, 50)), (255, 255, 255))

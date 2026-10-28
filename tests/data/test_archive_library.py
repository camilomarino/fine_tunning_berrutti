"""Test incremental libraries, interactive selections and global conversion limits."""

from __future__ import annotations

import contextlib
import curses
import hashlib
import io
import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from archive_berrutti import cli, library, manage as core, tui


class ScriptedScreen:
    """Exercise the real terminal view using keyboard events, without a tty."""

    def __init__(self, keys, size=(24, 110)):
        self.keys = iter(keys)
        self.size = size
        self.lines = {}
        self.frames = []

    def getmaxyx(self):
        return self.size

    def erase(self):
        self.lines = {}

    def addnstr(self, y, x, text, count, attr):
        self.lines[y] = text[:count]

    def move(self, y, x):
        pass

    def keypad(self, enabled):
        pass

    def refresh(self):
        self.frames.append("\n".join(text for _, text in sorted(self.lines.items())))

    def get_wch(self):
        try:
            return next(self.keys)
        except StopIteration:
            raise AssertionError("The interface requested an unexpected key") from None


class LibraryFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "library"
        self.payloads = {}
        files = []
        for roll in ("099", "100"):
            pdf = self.base / f"{roll}.pdf"
            pages = [
                Image.new("RGB", (20, 25), color)
                for color in ("white", "gray", "black")
            ]
            pages[0].save(pdf, "PDF", save_all=True, append_images=pages[1:])
            zipped = io.BytesIO()
            with zipfile.ZipFile(zipped, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.write(pdf, f"Original - Rollo {roll}.pdf")
            payload = zipped.getvalue()
            name = f"Archivo - Rollo {roll} - Fichas.zip"
            self.payloads[name] = payload
            files.append(
                {
                    "name": name,
                    "size": str(len(payload)),
                    "format": "ZIP",
                    "md5": hashlib.md5(payload).hexdigest(),
                    "sha1": hashlib.sha1(payload).hexdigest(),
                }
            )
        self.metadata = self.base / "metadata.json"
        self.metadata.write_text(
            json.dumps({"metadata": {"identifier": core.ITEM}, "files": files})
        )
        self.data = library.refresh(self.root, self.metadata)

    def fake_fetch(self, url, destination, entry, attempts, reserve, *, progress=None):
        if destination.exists():
            return core.verify_zip(destination, entry)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.payloads[entry["name"]])
        return core.verify_zip(destination, entry)

    def operate(self, rows, action, **kwargs):
        with patch.object(core, "fetch", side_effect=self.fake_fetch):
            with contextlib.redirect_stdout(io.StringIO()):
                return library.operate(
                    self.root,
                    self.data,
                    rows,
                    action,
                    cli.options(reserve_gib=0, **kwargs),
                    progress=lambda text: None,
                )

    def interface(self, keys, *, run=True):
        args = SimpleNamespace(root=self.root, metadata=None, offline=True)
        screen = ScriptedScreen(keys)
        with (
            patch.object(library, "PREFERENCES", self.base / "preferences.json"),
            patch.object(curses, "has_colors", return_value=False),
            patch.object(curses, "curs_set"),
        ):
            app = tui.Application(tui.Terminal(screen), args)
            if run:
                app.run()
                with self.assertRaises(StopIteration):
                    next(screen.keys)
        return app, screen


class LibraryTests(LibraryFixture):
    def test_refresh_and_selection_never_download_payloads(self):
        with patch.object(core.urllib.request, "urlopen") as network:
            new = library.refresh(self.root, self.metadata)
            chosen = library.choose(new, cli.options(rolls="99"))
            estimate = library.preview(self.root, chosen)
        network.assert_not_called()
        self.assertEqual(estimate["selected"], 1)
        self.assertFalse((self.root / "rolls").exists())
        self.assertEqual(len(list((self.root / "catalogs").glob("*.json"))), 2)

    def test_ids_and_range_selection_and_budget(self):
        self.assertEqual(
            library.choose(self.data, cli.options(ids="1-2")), self.data["files"]
        )
        with self.assertRaises(ValueError):
            library.choose(self.data, cli.options(ids="3"))
        with self.assertRaises(ValueError):
            library.choose(self.data, cli.options(ids="1", all=True))
        with self.assertRaises(ValueError):
            library.choose(self.data, cli.options(ids="1", max_download_gib=0))

    def test_additional_roll_and_refresh_reuse_original_without_duplication(self):
        first, second = self.data["files"]
        self.operate([first], "download")
        directory = library.entry_root(self.root, first)
        archive = core.zip_path(directory, library.local_entry(first))
        previous = archive.stat().st_mtime_ns
        self.assertEqual(archive.name, first["name"])
        self.operate([second], "download")
        self.data = library.refresh(self.root, self.metadata)
        self.operate(self.data["files"], "download")
        self.assertEqual(archive.stat().st_mtime_ns, previous)
        self.assertEqual(len(list(self.root.glob("rolls/*/*/zips/*.zip"))), 2)
        self.assertEqual(
            library.preview(self.root, self.data["files"])["remaining_download_gib"], 0
        )
        self.assertEqual(cli.summary(self.root, self.data)["downloaded"], 2)

    @unittest.skipUnless(
        shutil.which("pdftoppm") and shutil.which("pdfinfo"), "Poppler missing"
    )
    def test_prepare_conversion_status_resume_and_multiple_configs(self):
        row = self.data["files"][0]
        self.operate([row], "prepare", first_pages=2, dpi=72)
        state = library.status(self.root, row)
        self.assertEqual(
            (state["download"], state["pdf"], state["images"]),
            ("downloaded", "extracted", 2),
        )
        self.assertEqual(state["renders"][0]["state"], "complete")
        directory = Path(state["renders"][0]["path"])
        image = next((directory / "images").glob("*.jpg"))
        previous = image.stat().st_mtime_ns
        self.operate([row], "render", first_pages=2, dpi=72)
        self.assertEqual(image.stat().st_mtime_ns, previous)
        self.operate([row], "render", first_pages=1, dpi=100, format="png")
        self.assertEqual(len(library.status(self.root, row)["renders"]), 2)
        self.assertEqual(library.verify(self.root, [row])["verified_files"], 1)
        image.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "imagen cambió"):
            library.verify(self.root, [row])

    @unittest.skipUnless(
        shutil.which("pdftoppm") and shutil.which("pdfinfo"), "Poppler missing"
    )
    def test_global_page_limit_is_checked_before_any_images(self):
        self.operate(self.data["files"], "download")
        self.operate(self.data["files"], "extract")
        with self.assertRaisesRegex(ValueError, "presupuesto global"):
            self.operate(
                self.data["files"],
                "render",
                first_pages=None,
                all_pages=True,
                max_total_pages=5,
            )
        self.assertFalse(list(self.root.glob("rolls/*/*/renders/*/images/*")))

    def test_partial_zip_and_corrupt_receipt_are_explorable(self):
        row = self.data["files"][0]
        directory = library.entry_root(self.root, row)
        archive = core.zip_path(directory, library.local_entry(row))
        archive.parent.mkdir(parents=True)
        archive.with_suffix(".zip.part").write_bytes(b"partial")
        self.assertEqual(library.status(self.root, row)["download"], "partial")
        self.operate([row], "download")
        (directory / "receipts" / f"{row['key']}.json").write_text("corrupt")
        state = library.status(self.root, row)
        self.assertEqual(state["download"], "inconsistent")
        self.assertTrue(state["problem"])

    def test_render_settings_validate_and_support_arrow_choices(self):
        down = curses.KEY_DOWN
        keys = [
            "\n",
            down,
            down,
            "\n",
            down,
            "\n",
            "\x15",
            *"invalid",
            "\n",
            "\n",
            "\n",
            "\x15",
            *"72",
            "\n",
            down,
            "\n",
            down,
            "\n",
            down,
            "\n",
            "\x15",
            "2",
            "\n",
            down,
            "\n",
        ]
        app, screen = self.interface(keys, run=False)
        settings = app.render_settings()
        self.assertTrue(settings.all_pages)
        self.assertIsNone(settings.first_pages)
        self.assertEqual(
            (settings.dpi, settings.format, settings.max_total_pages), (72, "png", 2)
        )
        self.assertIn("Valor inválido", "\n".join(screen.frames))

    def test_cli_list_and_plan_work_without_network_and_json_is_machine_readable(self):
        with patch.object(core.urllib.request, "urlopen") as network:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                cli.main(["list", "--root", str(self.root), "--json"])
            rows = json.loads(output.getvalue())
            self.assertEqual(len(rows), 2)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                cli.main(["plan", "--root", str(self.root), "--all"])
            self.assertEqual(json.loads(output.getvalue())["selected"], 2)
        network.assert_not_called()

    def test_catalog_pointer_tampering_is_detected(self):
        pointer = core.load(self.root / "active_catalog.json")
        (self.root / pointer["path"]).write_text("{}")
        with self.assertRaisesRegex(ValueError, "snapshot"):
            library.load_catalog(self.root)

    def test_arrow_destination_change_imports_catalog_without_download(self):
        candidate = self.base / "another-library"
        keys = [curses.KEY_F2, "\n", "\x15", *str(candidate), "\n", "\x1b"]
        with patch.object(core.urllib.request, "urlopen") as network:
            app, _ = self.interface(keys)
        network.assert_not_called()
        self.assertEqual(app.root, candidate)
        self.assertEqual(
            core.load(self.base / "preferences.json")["root"], str(candidate)
        )
        self.assertEqual(library.load_catalog(candidate)["files"], self.data["files"])
        self.assertFalse((candidate / "rolls").exists())

    def test_right_arrow_in_text_field_moves_cursor_instead_of_accepting(self):
        app, _ = self.interface(
            [curses.KEY_HOME, curses.KEY_RIGHT, "X", "\n"], run=False
        )
        self.assertEqual(app.view.edit("Editar", "ab"), "aXb")

    def test_small_terminal_cannot_accept_hidden_operation(self):
        screen = ScriptedScreen(["\n", "\x1b"], size=(12, 50))
        with (
            patch.object(curses, "has_colors", return_value=False),
            patch.object(curses, "curs_set"),
        ):
            view = tui.Terminal(screen)
            self.assertIsNone(
                view.menu("Descargar", ["Cancelar", "Ejecutar"], initial=1)
            )
        self.assertIn("Amplía la terminal", "\n".join(screen.frames))

    def test_no_tty_reports_batch_commands_without_opening_library(self):
        with (
            patch.object(tui.sys.stdin, "isatty", return_value=False),
            patch.object(library, "open_library") as opened,
        ):
            with self.assertRaisesRegex(ValueError, "terminal interactiva"):
                cli.browse(SimpleNamespace(root=self.root, offline=True, metadata=None))
        opened.assert_not_called()

    def test_human_detail_and_json_detail_and_downloaded_filter(self):
        row = self.data["files"][0]
        self.operate([row], "download")
        self.assertEqual(len(cli.filtered(self.root, self.data, downloaded=True)), 1)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            cli.detail(self.root, self.data, 1)
        self.assertIn("descargado", output.getvalue())
        self.assertIn("todavía no convertidas", output.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as output:
            cli.main(["info", "--root", str(self.root), "1", "--json"])
        self.assertEqual(
            json.loads(output.getvalue())["local"]["download"], "downloaded"
        )


if __name__ == "__main__":
    unittest.main()

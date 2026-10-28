"""Unified contracts, indexed selection, bounded remote previews and keyboard flow."""

from __future__ import annotations

import contextlib
import curses
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from archive_berrutti import tui as archive_tui
from data import access, agent, manage, tui
from tests.data.test_archive_library import LibraryFixture, ScriptedScreen


class BrowserTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.settings = {"schema_version": 1, "default": "local", "datasets": {}}
        for name in ("berrutti_175", "luisa_tesseract_lines"):
            root = self.base / name
            (root / "raw/images").mkdir(parents=True)
            rows = []
            pages = {}
            for i in range(1, 4):
                identifier = f"p{i}" if name == "berrutti_175" else f"{i:08d}"
                image = root / "raw/images" / (identifier + ".png")
                Image.new("RGB", (60, 80), "white").save(image)
                reference = image.with_suffix(".gt.txt")
                reference.write_text(f"Referencia humana {i}\n")
                row = {
                    "id": identifier,
                    "dataset": name,
                    "unit": "page" if name == "berrutti_175" else "line",
                    "image": str(image.relative_to(root)),
                    "reference": str(reference.relative_to(root)),
                    "image_sha256": access.digest(image.read_bytes()),
                    "reference_sha256": access.digest(reference.read_bytes()),
                    "project_split": "test" if i == 3 else "train",
                }
                pages[identifier] = {
                    "filename": image.name,
                    "regions": [
                        {
                            "shape_attributes": {
                                "name": "rect",
                                "x": 2,
                                "y": 3,
                                "width": 35,
                                "height": 12,
                            },
                            "region_attributes": {"text": f"Texto de caja {i}"},
                        }
                    ],
                }
                if name == "luisa_tesseract_lines":
                    predicted = image.with_suffix(".pred.txt")
                    predicted.write_text(f"OCR {i}\n")
                    row.update(
                        source_split=row["project_split"],
                        project_split=None,
                        reference_text=f"Referencia humana {i}",
                        ocr_text=f"OCR {i}",
                        ocr_prediction=str(predicted.relative_to(root)),
                    )
                rows.append(row)
            (root / "records.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            (root / "raw/annotations.json").write_text(
                json.dumps({"_via_img_metadata": pages})
            )
            annotations = root / "raw/annotations.json"
            (root / "raw/manifest.json").write_text(
                json.dumps(
                    {
                        "annotations": {
                            "path": "annotations.json",
                            "format": "via2",
                            "sha256": access.digest(annotations.read_bytes()),
                        }
                    }
                )
            )
            (root / "release.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "dataset": name,
                        "version": "test-v1",
                        "record_count": 3,
                        "records": "records.jsonl",
                    }
                )
            )
            self.settings["datasets"][name] = {
                "local": {
                    "host": "local",
                    "root": str(root),
                    "raw": str(root / "raw"),
                    "version": "test-v1",
                }
            }
        mock = patch.object(manage, "locations", return_value=self.settings)
        mock.start()
        self.addCleanup(mock.stop)
        env = patch.dict(
            os.environ,
            {
                "BERRUTTI_DATA_CACHE": str(self.base / "cache"),
                "ARCHIVE_BERRUTTI_VIEWER": "blocks",
            },
        )
        env.start()
        self.addCleanup(env.stop)

    def call(self, *args):
        output = io.StringIO()
        code = 0
        with contextlib.redirect_stdout(output):
            try:
                manage.main(["agent", *args])
            except SystemExit as error:
                code = error.code
        result = json.loads(output.getvalue())
        self.assertEqual(result["schema_version"], 1)
        return result, code

    def test_agent_pagination_exact_id_and_split_keep_source_separate(self):
        result, code = self.call(
            "list", "luisa_tesseract_lines", "--split", "test", "--limit", "1"
        )
        self.assertEqual(code, 0)
        self.assertEqual(result["data"]["total"], 1)
        row = result["data"]["items"][0]
        self.assertIsNone(row["project_split"])
        self.assertEqual(row["source_split"], "test")
        self.assertTrue(Path(row["paths"]["image"]).is_absolute())
        result, _ = self.call("list", "luisa_tesseract_lines", "--search", "HUMANA 2")
        self.assertEqual(result["data"]["items"][0]["id"], "00000002")
        first, _ = self.call("list", "berrutti_175", "--limit", "2")
        last, _ = self.call(
            "list",
            "berrutti_175",
            "--offset",
            str(first["data"]["next_offset"]),
            "--limit",
            "2",
        )
        self.assertEqual([r["id"] for r in last["data"]["items"]], ["p3"])
        self.assertIsNone(last["data"]["next_offset"])

    def test_file_pagination_counts_files_not_examples(self):
        found = []
        for offset in range(0, 9, 2):
            result, code = self.call(
                "files", "berrutti_175", "--offset", str(offset), "--limit", "2"
            )
            self.assertEqual(code, 0)
            self.assertEqual(result["data"]["total"], 9)
            found.extend((r["id"], r["kind"]) for r in result["data"]["items"])
        self.assertEqual(len(found), len(set(found)))
        self.assertEqual(len(found), 9)
        absent, _ = self.call("files", "luisa_tesseract_lines", "--kind", "annotations")
        self.assertEqual(absent["data"]["total"], 0)

    def test_human_text_ocr_and_line_geometry_are_distinct(self):
        human, _ = self.call("text", "luisa_tesseract_lines", "--id", "00000001")
        ocr, _ = self.call(
            "text", "luisa_tesseract_lines", "--id", "00000001", "--kind", "ocr"
        )
        self.assertEqual(human["data"]["source"], "human_transcription")
        self.assertEqual(ocr["data"]["source"], "tesseract")
        self.assertNotEqual(human["data"]["text"], ocr["data"]["text"])
        annotation, _ = self.call("annotations", "berrutti_175", "--id", "p1")
        self.assertEqual(annotation["data"]["items"][0]["crop_xyxy"], [2, 3, 37, 15])
        self.assertIn("not_verified", annotation["data"]["order"])

    def test_json_errors_and_missing_location_do_not_fall_back(self):
        for args in [
            ("list", "berrutti_175", "--where", "ruche"),
            ("list", "berrutti_175", "--limit", "0"),
            ("list", "berrutti_175", "--where", "nowhere"),
            ("text", "berrutti_175", "--id", "missing"),
            ("annotations", "luisa_tesseract_lines", "--id", "00000001"),
        ]:
            with self.subTest(args=args):
                result, code = self.call(*args)
                self.assertFalse(result["ok"])
                self.assertNotEqual(code, 0)

    def test_repeated_queries_use_index_and_never_scan_payloads(self):
        with contextlib.closing(access.Source("berrutti_175")) as source:
            source.records(limit=1)
        with (
            patch.object(access.Source, "read", side_effect=AssertionError("reread")),
            patch.object(Path, "rglob", side_effect=AssertionError("scan")),
        ):
            result, code = self.call("list", "berrutti_175", "--id", "p2")
            self.assertEqual(code, 0)
            self.assertEqual(result["data"]["items"][0]["id"], "p2")

    def test_remote_reads_only_metadata_and_selected_image_and_cleans_preview(self):
        loc = dict(self.settings["datasets"]["berrutti_175"]["local"], host="ruche")
        self.settings["datasets"]["berrutti_175"]["ruche"] = loc
        commands = []

        def transfer(command, **kwargs):
            import shlex
            from types import SimpleNamespace

            commands.append(command)
            arguments = shlex.split(command[-1])
            self.assertEqual(arguments[:2], ["head", "-c"])
            return SimpleNamespace(
                returncode=0, stdout=Path(arguments[-1]).read_bytes(), stderr=b""
            )

        with patch.object(access.subprocess, "run", side_effect=transfer):
            source = access.Source("berrutti_175", "ruche")
            rows = source.records(limit=2)["items"]
            self.assertEqual(len(commands), 2)
            preview = source.image_path(rows[0])
            self.assertTrue(preview.exists())
            source.image_path(rows[1])
            self.assertFalse(preview.exists())
            folder = Path(source._preview.name)
            self.assertEqual(len(list(folder.iterdir())), 1)
            source.close()
            self.assertFalse(folder.exists())
            self.assertEqual(len(commands), 4)
            self.assertTrue(
                all("srun" not in str(c) and "python" not in str(c) for c in commands)
            )

    def test_bad_path_and_annotation_hash_fail(self):
        with contextlib.closing(access.Source("berrutti_175")) as source:
            for value in ("/etc/passwd", "../outside"):
                with self.assertRaises(ValueError):
                    source.path(value)
            path = source.root / "raw/annotations.json"
            path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "Hash"):
                source.annotations("p1")

    def test_refresh_only_invalidates_client_cache_and_keeps_originals(self):
        original = self.base / "berrutti_175"
        before = {str(p): p.read_bytes() for p in original.rglob("*") if p.is_file()}
        self.call("list", "berrutti_175")
        result, code = self.call("refresh", "berrutti_175")
        self.assertEqual(code, 0)
        self.assertEqual(result["data"]["records"], 3)
        after = {str(p): p.read_bytes() for p in original.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_indexed_images_are_viewed_in_filtered_order_without_scanning(self):
        with contextlib.closing(access.Source("berrutti_175")) as source:
            row = source.record("p3")
            with patch.object(Path, "iterdir", side_effect=AssertionError("scan")):
                document = tui.Document(source, row, split="test")
                try:
                    self.assertEqual(document.count, 1)
                    self.assertEqual(document.index, 0)
                    self.assertEqual(document.load().size, (60, 80))
                finally:
                    document.close()

    def interface(self, keys):
        screen = ScriptedScreen(keys)
        with (
            patch.object(curses, "has_colors", return_value=False),
            patch.object(curses, "curs_set"),
        ):
            tui.Application(archive_tui.Terminal(screen, brand="DATOS BERRUTTI")).run()
        with self.assertRaises(StopIteration):
            next(screen.keys)
        return "\n".join(screen.frames)

    def test_human_page_image_text_line_crop_and_back_navigation(self):
        right, left, down = curses.KEY_RIGHT, curses.KEY_LEFT, curses.KEY_DOWN
        frames = self.interface(
            [
                right,
                right,
                right,
                down,
                left,
                down,
                right,
                left,
                down,
                right,
                right,
                right,
                left,
                left,
                left,
                left,
                left,
            ]
        )
        self.assertIn("Conjuntos de datos", frames)
        self.assertIn("Imagen 2/3", frames)
        self.assertIn("Referencia humana 1", frames)
        self.assertIn("Texto de caja 1", frames)
        self.assertIn("── ANOTACIONES", frames)

    def test_human_luisa_filters_searches_and_reads_ocr(self):
        right, left, down = curses.KEY_RIGHT, curses.KEY_LEFT, curses.KEY_DOWN
        frames = self.interface(
            [
                down,
                right,
                "/",
                "2",
                "\n",
                right,
                down,
                down,
                right,
                left,
                left,
                left,
                left,
            ]
        )
        self.assertIn("1 ejemplos", frames)
        self.assertIn("OCR 2", frames)


class ArchiveFacadeTests(LibraryFixture):
    def test_common_facade_reuses_archive_states_and_explicit_root(self):
        config = {
            "default": "local",
            "datasets": {
                "archive_berrutti": {
                    "local": {
                        "host": "local",
                        "root": str(self.root),
                        "raw": str(self.root),
                    }
                }
            },
        }
        with patch.object(manage, "locations", return_value=config):
            result, code = agent.archive_result("list", ["--limit", "1"], "local")
            self.assertEqual(code, 0)
            self.assertEqual(result["dataset"], "archive_berrutti")
            self.assertEqual(result["root"], str(self.root))
            self.assertEqual(result["data"]["total"], 2)
            plan, code = agent.archive_result(
                "complete", ["--rolls", "099", "--dry-run"], "local"
            )
            self.assertEqual(code, 0)
            self.assertTrue(plan["ok"])
            self.assertFalse((self.root / "jobs").exists())
            detail, code = agent.archive_result("info", ["--id", "099"], "local")
            self.assertEqual(code, 0)
            self.assertEqual(detail["data"]["items"][0]["roll"], "099")

    def test_archive_browser_returns_left_to_common_menu(self):
        config = {
            "default": "local",
            "datasets": {
                "archive_berrutti": {
                    "local": {
                        "host": "local",
                        "root": str(self.root),
                        "raw": str(self.root),
                    }
                }
            },
        }
        screen = ScriptedScreen(
            [
                curses.KEY_DOWN,
                curses.KEY_DOWN,
                curses.KEY_RIGHT,
                curses.KEY_LEFT,
                curses.KEY_LEFT,
            ]
        )
        with (
            patch.object(manage, "locations", return_value=config),
            patch.object(curses, "has_colors", return_value=False),
            patch.object(curses, "curs_set"),
            patch.object(archive_tui.library, "remember"),
        ):
            tui.Application(archive_tui.Terminal(screen, brand="DATOS BERRUTTI")).run()
        self.assertIn("Rollos", "\n".join(screen.frames))
        self.assertIn("Conjuntos de datos", screen.frames[-1])

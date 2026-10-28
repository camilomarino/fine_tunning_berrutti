"""Exercise selection, interrupted transfers, immutable sources and PDF rendering."""

from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
import tempfile
import threading
import unittest
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from archive_berrutti import manage


def entry(payload: bytes, name: str = "Rollo 099.zip") -> dict:
    return {
        "key": hashlib.sha256(name.encode()).hexdigest()[:20],
        "name": name,
        "roll": "099",
        "bytes": len(payload),
        "md5": hashlib.md5(payload).hexdigest(),
        "sha1": hashlib.sha1(payload).hexdigest(),
    }


@contextlib.contextmanager
def server(payload: bytes, *, ignore_range=False, wrong_range=False):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requested = self.headers.get("Range")
            requests.append(requested)
            offset = (
                int(requested.removeprefix("bytes=").removesuffix("-"))
                if requested
                else 0
            )
            if ignore_range:
                offset = 0
            status = 206 if requested and not ignore_range else 200
            self.send_response(status)
            self.send_header("Content-Length", str(len(payload) - offset))
            if status == 206:
                start = offset + 1 if wrong_range else offset
                self.send_header(
                    "Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}"
                )
            self.end_headers()
            self.wfile.write(payload[offset:])

        def log_message(self, *args):
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{http.server_port}/file.zip", requests
    finally:
        http.shutdown()
        http.server_close()
        thread.join()


class SelectionTests(unittest.TestCase):
    def setUp(self):
        files = [
            {**entry(b"sample", name), "roll": roll}
            for roll, name in [
                ("099", "Rollo 099 - fichas.zip"),
                ("099r", "Rollo 099r - fichas.zip"),
                ("100", "Rollo 100 - periodicos.zip"),
            ]
        ]
        metadata = {
            "metadata": {"identifier": "synthetic"},
            "files": [{**row, "format": "ZIP", "size": row["bytes"]} for row in files],
        }
        self.data = manage.catalog(metadata)

    def options(self, **changes):
        return SimpleNamespace(
            **{
                "all": False,
                "rolls": None,
                "roll_range": None,
                "match": None,
                "limit": None,
                "max_download_gib": None,
                **changes,
            }
        )

    def test_exact_roll_normalizes_padding_without_including_other_versions(self):
        chosen = manage.select(self.data, self.options(rolls="99"))
        self.assertEqual([row["roll"] for row in chosen], ["099"])

    def test_range_includes_variants_and_intersects_title_filter(self):
        chosen = manage.select(
            self.data, self.options(roll_range="99:100", match="FICHAS")
        )
        self.assertEqual([row["roll"] for row in chosen], ["099", "099r"])

    def test_empty_unknown_ambiguous_or_overbudget_selection_fails(self):
        for args in [
            self.options(),
            self.options(rolls="101"),
            self.options(all=True, rolls="99"),
            self.options(match="absent"),
            self.options(all=True, max_download_gib=0),
        ]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                manage.select(self.data, args)

    def test_all_includes_every_zip_and_limit_is_deterministic(self):
        self.assertEqual(len(manage.select(self.data, self.options(all=True))), 3)
        self.assertEqual(
            len(manage.select(self.data, self.options(all=True, limit=1))), 1
        )

    def test_repository_data_cannot_be_written_to_tracked_directories(self):
        with self.assertRaises(ValueError):
            manage.private_path(manage.REPO / "experiments" / "unwanted.zip")

    def test_exact_pages_reject_out_of_bounds_first_pages_clamps(self):
        self.assertEqual(manage.page_selection("1-2,2,4", None, False, 4), [1, 2, 4])
        self.assertEqual(list(manage.page_selection(None, 5, False, 2)), [1, 2])
        for spec in ["0", "3-1", "1-5", "no"]:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                manage.page_selection(spec, None, False, 4)


class TransferTests(unittest.TestCase):
    def test_resume_uses_range_and_only_publishes_correct_checksum(self):
        payload = b"some source bytes" * 30
        with (
            tempfile.TemporaryDirectory() as temporary,
            server(payload) as (url, requests),
        ):
            destination = Path(temporary) / "source.zip"
            destination.with_suffix(".zip.part").write_bytes(payload[:7])
            updates = []
            actual = manage.fetch(
                url, destination, entry(payload), 1, 0, progress=updates.append
            )
            self.assertEqual(requests, ["bytes=7-"])
            self.assertEqual(destination.read_bytes(), payload)
            self.assertEqual(actual["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(
                (updates[0]["phase"], updates[0]["bytes"]), ("connecting", 7)
            )
            self.assertEqual(updates[-2]["phase"], "verifying")
            self.assertEqual(
                (updates[-1]["phase"], updates[-1]["bytes"]), ("done", len(payload))
            )
            manage.fetch(url, destination, entry(payload), 1, 0)
            self.assertEqual(len(requests), 1)

    def test_ignored_range_restarts_instead_of_appending_duplicate_bytes(self):
        payload = b"complete source"
        with (
            tempfile.TemporaryDirectory() as temporary,
            server(payload, ignore_range=True) as (url, _),
        ):
            destination = Path(temporary) / "source.zip"
            destination.with_suffix(".zip.part").write_bytes(payload[:3])
            updates = []
            manage.fetch(
                url, destination, entry(payload), 1, 0, progress=updates.append
            )
            self.assertEqual(destination.read_bytes(), payload)
            downloading = next(
                item for item in updates if item["phase"] == "downloading"
            )
            self.assertEqual(downloading["bytes"], 0)

    def test_invalid_range_keeps_partial_without_publishing(self):
        payload = b"complete source"
        with (
            tempfile.TemporaryDirectory() as temporary,
            server(payload, wrong_range=True) as (url, _),
        ):
            destination = Path(temporary) / "source.zip"
            partial = destination.with_suffix(".zip.part")
            partial.write_bytes(payload[:3])
            with self.assertRaisesRegex(ValueError, "Content-Range"):
                manage.fetch(url, destination, entry(payload), 1, 0)
            self.assertFalse(destination.exists())
            self.assertEqual(partial.read_bytes(), payload[:3])

    def test_wrong_checksum_never_becomes_a_completed_download(self):
        payload = b"corrupt source"
        with tempfile.TemporaryDirectory() as temporary, server(payload) as (url, _):
            destination = Path(temporary) / "source.zip"
            expected = entry(b"correct source")
            updates = []
            with self.assertRaisesRegex(ValueError, "Checksum"):
                manage.fetch(url, destination, expected, 1, 0, progress=updates.append)
            self.assertFalse(destination.exists())
            self.assertTrue(destination.with_suffix(".zip.part").exists())
            self.assertEqual(updates[-1]["phase"], "failed")
            self.assertIn("Checksum", updates[-1]["error"])
            self.assertNotIn("done", [item["phase"] for item in updates])

    def test_retry_preserves_bad_payload_then_downloads_again(self):
        payload = b"source bytes"
        with (
            tempfile.TemporaryDirectory() as temporary,
            server(payload) as (url, requests),
        ):
            destination = Path(temporary) / "source.zip"
            destination.with_suffix(".zip.part").write_bytes(b"x" * len(payload))
            with patch.object(manage.time, "sleep"):
                manage.fetch(url, destination, entry(payload), 2, 0)
            self.assertEqual(requests, [None])
            self.assertEqual(len(list(Path(temporary).glob("*.rejected-*"))), 1)

    def test_full_download_rejects_insufficient_space_before_http(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(
                root=Path(temporary) / "data",
                plan=Path(temporary) / "plan.json",
                attempts=1,
                reserve_gib=0,
                quota_free_gib=0,
            )
            manage.save_new(
                args.plan, {"identifier": "synthetic", "files": [entry(b"x")]}
            )
            with patch.object(manage.urllib.request, "urlopen") as network:
                with self.assertRaisesRegex(ValueError, "cuota"):
                    manage.download(args)
                network.assert_not_called()

    def test_live_ruche_soft_quota_is_checked_not_the_hard_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(manage.os.environ, {"WORKDIR": "/gpfs/workdir/synthetic"}):
                with patch.object(manage.Path, "is_relative_to", return_value=True):
                    with patch.object(
                        manage,
                        "run",
                        return_value=(
                            "gpfs workdir USR 100 101 200 0 none | 1 10 11 0 none\n"
                        ),
                    ):
                        with self.assertRaisesRegex(ValueError, "Espacio insuficiente"):
                            manage.space(Path(temporary), 2048, 0, quota=True)

    def test_compute_quota_uses_numeric_uid_and_checks_file_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(manage.os.environ, {"WORKDIR": "/gpfs/workdir/synthetic"}):
                with patch.object(manage.Path, "is_relative_to", return_value=True):
                    with patch.object(manage.os, "getuid", return_value=1234):
                        with patch.object(
                            manage,
                            "run",
                            return_value=(
                                "gpfs workdir USR 100 100000 110000 0 none | 8 10 11 0 none\n"
                            ),
                        ) as probe:
                            with self.assertRaisesRegex(
                                ValueError, "cantidad de archivos"
                            ):
                                manage.space(Path(temporary), 0, 0, quota=True, files=3)
                            probe.assert_called_once_with(
                                ["mmlsquota", "-u", "1234", "gpfs"], 30
                            )


class PreparationTests(unittest.TestCase):
    def fixture(self, directory: Path, *, pdf=False):
        root = directory / "data"
        (root / "zips").mkdir(parents=True)
        source = directory / "source.pdf"
        if pdf:
            images = [
                Image.new("RGB", (24, 30), color)
                for color in ("white", "gray", "black")
            ]
            images[0].save(source, "PDF", save_all=True, append_images=images[1:])
        else:
            source.write_bytes(b"synthetic pdf payload")
        archive = directory / "source.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
            zipped.write(source, "../../original.pdf")
        row = entry(archive.read_bytes())
        archive.rename(root / "zips" / f"{row['key']}.zip")
        manage.save_new(
            root / "selection.json", {"identifier": "synthetic", "files": [row]}
        )
        return root, row

    def test_extraction_flattens_paths_and_preserves_completed_pdfs(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root, row = self.fixture(directory)
            args = SimpleNamespace(root=root, reserve_gib=0, max_pdf_gib=1)
            manage.extract(args)
            pdf = root / "pdfs" / row["key"] / "0001-original.pdf"
            previous = pdf.stat().st_mtime_ns
            manage.extract(args)
            self.assertEqual(pdf.stat().st_mtime_ns, previous)
            self.assertFalse((directory / "original.pdf").exists())
            pdf.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "cambió"):
                manage.extract(args)

    @unittest.skipUnless(
        shutil.which("pdftoppm") and shutil.which("pdfinfo"), "Poppler missing"
    )
    def test_render_subset_resume_and_configuration_and_corruption_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root, _ = self.fixture(directory, pdf=True)
            manage.extract(SimpleNamespace(root=root, reserve_gib=0, max_pdf_gib=1))
            args = SimpleNamespace(
                root=root,
                output=directory / "render",
                dpi=72,
                format="png",
                quality=95,
                pages="1,3",
                first_pages=None,
                all_pages=False,
                timeout=30,
                max_total_pages=2,
                reserve_gib=0,
            )
            manage.render(args)
            records = manage.read_journal(args.output / "pages.jsonl")
            self.assertEqual([row["page_number"] for row in records.values()], [1, 3])
            self.assertTrue(
                all(row["label_status"] == "unlabeled" for row in records.values())
            )
            images = sorted((args.output / "images").glob("*.png"))
            times = [image.stat().st_mtime_ns for image in images]
            with (args.output / "pages.jsonl").open("ab") as journal:
                journal.write(b'{"page_id":')
            manage.render(args)
            self.assertEqual([image.stat().st_mtime_ns for image in images], times)
            self.assertEqual(len(list(args.output.glob("pages.jsonl.incomplete-*"))), 1)
            args.dpi = 150
            with self.assertRaisesRegex(ValueError, "Configuración distinta"):
                manage.render(args)
            args.dpi = 72
            images[0].write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "Imagen completada modificada"):
                manage.render(args)

    def test_json_publication_does_not_overwrite_old_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            manage.save_new(path, {"value": 1})
            with self.assertRaises(FileExistsError):
                manage.save_new(path, {"value": 2})
            self.assertEqual(json.loads(path.read_text()), {"value": 1})


if __name__ == "__main__":
    unittest.main()

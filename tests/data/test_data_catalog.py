"""Dataset import integrity, split separation, source provenance and private paths."""

from __future__ import annotations

import csv
import functools
import io
import json
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from data import manage, releases


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)

    def lines(self):
        raw = self.base / "raw"
        (raw / "images").mkdir(parents=True)
        for split, suffix, ids in (("train", "21611", [2, 1]), ("test", "2163", [3])):
            for prefix in ("gts", "preds"):
                with (raw / f"{prefix}{suffix}.csv").open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["id", "content"])
                    for identifier in ids if prefix == "gts" else reversed(ids):
                        writer.writerow(
                            [identifier, f"{prefix} synthetic, line {identifier}"]
                        )
            for identifier in ids:
                image = raw / "images" / f"{identifier:08d}.gif"
                # Upstream uses .gif names for PNG bytes.
                Image.new("RGB", (35, 12), "white").save(image, format="PNG")
                image.with_suffix(".gt.txt").write_text(
                    f"gts synthetic, line {identifier}\n"
                )
                image.with_suffix(".pred.txt").write_text(
                    f"preds synthetic, line {identifier}\n"
                )
        return raw

    def test_line_pairs_join_by_id_keep_gt_and_ocr_distinct_and_do_not_assign_project_split(
        self,
    ):
        raw = self.lines()
        output = self.base / "records.jsonl"
        counts = releases.index_luisa(raw, output, expected={"train": 2, "test": 1})
        self.assertEqual(counts, {"train": 2, "test": 1})
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(
            [row["id"] for row in rows], ["00000001", "00000002", "00000003"]
        )
        for row in rows:
            self.assertTrue(row["reference_text"].startswith("gts"))
            self.assertTrue(row["ocr_text"].startswith("preds"))
            self.assertIsNone(row["project_split"])
            self.assertIsNone(row["source_document"])
            self.assertTrue(row["requires_overlap_audit"])
            self.assertEqual(row["width"], 35)
            self.assertEqual(row["image_format"], "PNG")

    def test_mismatched_sidecar_or_missing_image_never_gets_a_sealed_release(self):
        raw = self.lines()
        (raw / "images/00000001.gt.txt").write_text("other text")
        with self.assertRaisesRegex(ValueError, "No coincide"):
            releases.index_luisa(
                raw, self.base / "records.jsonl", expected={"train": 2, "test": 1}
            )
        self.assertFalse((self.base / "release.json").exists())

    def test_whitespace_variants_are_reported_and_originals_are_preserved(self):
        raw = self.lines()
        sidecar = raw / "images/00000003.gt.txt"
        sidecar.write_text("gts  synthetic,   line 3\n")
        before = sidecar.read_bytes()
        output = self.base / "records.jsonl"
        releases.index_luisa(raw, output, expected={"train": 2, "test": 1})
        row = json.loads(output.read_text().splitlines()[-1])
        self.assertEqual(row["csv_sidecar_agreement"]["reference"], "whitespace_only")
        self.assertEqual(row["reference_text"], "gts synthetic, line 3")
        self.assertEqual(sidecar.read_bytes(), before)

    def test_duplicate_ids_fail_instead_of_silently_overwriting(self):
        path = self.base / "duplicate.csv"
        path.write_text("id,content\n1,first\n00000001,second\n")
        with self.assertRaisesRegex(ValueError, "duplicado"):
            releases.csv_records(path)

    def test_tar_traversal_links_and_wrong_subtree_are_rejected(self):
        for index, (name, kind) in enumerate(
            [
                ("root/luisa_tesseract_lines/../../outside", tarfile.REGTYPE),
                ("root/luisa_tesseract_lines/link", tarfile.SYMTYPE),
                ("root/other_dataset/file", tarfile.REGTYPE),
            ]
        ):
            with self.subTest(name=name):
                archive = self.base / f"{index}.tar.gz"
                with tarfile.open(archive, "w:gz") as tar:
                    member = tarfile.TarInfo(name)
                    member.type = kind
                    member.linkname = "/tmp/outside"
                    member.size = 1 if kind == tarfile.REGTYPE else 0
                    tar.addfile(member, io.BytesIO(b"x"))
                with self.assertRaises(ValueError):
                    releases.extract_luisa(archive, self.base / str(index))

    def test_extraction_budget_and_duplicate_members_fail(self):
        archive = self.base / "data.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for _ in range(2):
                member = tarfile.TarInfo("root/luisa_tesseract_lines/same")
                member.size = 4
                tar.addfile(member, io.BytesIO(b"data"))
        for index, maximum in enumerate((3, 30)):
            with self.assertRaises(ValueError):
                releases.extract_luisa(archive, self.base / str(index), maximum=maximum)

    def test_tree_digest_matches_git_for_entire_source_subtree(self):
        raw = self.lines()
        expected = releases.git_tree(raw)
        subprocess.run(["git", "init", "-q", str(raw)], check=True)
        subprocess.run(["git", "-C", str(raw), "add", "."], check=True)
        actual = subprocess.check_output(
            ["git", "-C", str(raw), "write-tree"], text=True
        ).strip()
        self.assertEqual(expected, actual)

    def test_streamed_import_seals_exact_files_without_reopening_images(self):
        raw = self.lines()
        tree = releases.git_tree(raw)
        archive = self.base / "source.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(raw, arcname="repository/luisa_tesseract_lines")
        original_sha = releases.sha256

        def no_image_reads(path):
            if path.suffix == ".gif":
                raise AssertionError(
                    "Images must be checked from the extraction stream"
                )
            return original_sha(path)

        with (
            patch.object(releases, "LUISA_TREE", tree),
            patch.object(
                releases,
                "index_luisa",
                functools.partial(
                    releases.index_luisa, expected={"train": 2, "test": 1}
                ),
            ),
            patch.object(releases, "sha256", side_effect=no_image_reads),
        ):
            result = releases.import_luisa(archive, self.base / "store")
        target = Path(result["root"])
        self.assertEqual(result["record_count"], 3)
        self.assertEqual(result["source_tree_sha1"], tree)
        self.assertEqual(result["image_formats"], {"PNG": 3})
        self.assertTrue(releases.verify(target)["ok"])
        self.assertEqual(releases.git_tree(target / "raw"), tree)

    def test_seal_verifies_payloads_and_does_not_replace_an_existing_release(self):
        stage, target = releases.staging(self.base, "test", "v1")
        (stage / "raw").mkdir()
        (stage / "raw/sample.txt").write_text("synthetic")
        (stage / "records.jsonl").write_text("{}\n")
        releases.seal(stage, target, {"dataset": "test", "record_count": 1})
        self.assertEqual((target.parent.parent / "current").resolve(), target)
        self.assertTrue(releases.verify(target)["ok"])
        with self.assertRaises(ValueError):
            releases.staging(self.base, "test", "v1")
        (target / "raw/sample.txt").write_text("modified")
        with self.assertRaisesRegex(ValueError, "modificado"):
            releases.verify(target)

    def test_login_and_tracked_repository_destinations_are_rejected(self):
        with patch.object(releases.socket, "gethostname", return_value="ruche01"):
            with self.assertRaisesRegex(ValueError, "Slurm"):
                releases.staging(self.base, "test", "v1")
        with self.assertRaises(ValueError):
            releases.staging(
                Path(__file__).resolve().parents[2] / "experiments", "test", "v1"
            )

    def test_catalog_paths_never_fall_back_to_a_different_host(self):
        settings = {
            "default": "ruche",
            "datasets": {
                "berrutti_175": {
                    "local": {"raw": "/local/data"},
                    "ruche": {"raw": "/remote/data"},
                }
            },
        }
        with patch.object(manage, "locations", return_value=settings):
            self.assertEqual(manage.resolve("berrutti_175"), "/remote/data")
            self.assertEqual(manage.resolve("berrutti_175", "local"), "/local/data")
            with self.assertRaises(ValueError):
                manage.resolve("luisa_tesseract_lines", "local")

    def test_payload_paths_are_gitignored_but_catalog_and_docs_are_not(self):
        root = Path(__file__).resolve().parents[2]
        for path in [
            "data/berrutti_175/releases/v1/raw/image.png",
            "data/luisa_tesseract_lines/.staging/x/records.jsonl",
            "data/locations.local.json",
            "data/archive_berrutti/local",
        ]:
            self.assertEqual(
                subprocess.run(
                    ["git", "check-ignore", "-q", path], cwd=root
                ).returncode,
                0,
            )
        self.assertEqual(
            subprocess.run(
                ["git", "check-ignore", "-q", "data/catalog.json"], cwd=root
            ).returncode,
            1,
        )

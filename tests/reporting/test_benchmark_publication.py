"""Evidence-boundary regressions for the audited-results publication."""

import json
import tempfile
import unittest
from pathlib import Path

from reporting.berrutti.benchmark_publication import (
    Evidence,
    encoded,
    metrics,
    read_evaluation,
    read_shard,
    sha,
    verified_export,
)


class BenchmarkPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.files = {}
        self.refs = {f"page-{i:02}": "abc" for i in range(25)}
        self.folder = "closed-evaluations/run-2/evaluations/epoch-02-test"
        for pid in self.refs:
            row = {
                "page_id": pid,
                "status": "complete",
                "attempts": 1,
                "char_errors": 0,
                "char_count": 3,
                "word_errors": 0,
                "word_count": 1,
            }
            self.write(f"{self.folder}/pages/{pid}/closed.json", row)
            self.write(
                f"{self.folder}/pages/{pid}/attempt-1.json",
                {"stop_reason": "eos", "prediction": "abc"},
            )
        self.summary = {
            "char_errors": 0,
            "char_count": 75,
            "word_errors": 0,
            "word_count": 25,
            "cer": 0,
            "wer": 0,
            "failed_pages": 0,
            "retried_pages": 0,
            "all_pages_completed": True,
        }
        self.write(self.folder + "/closed.json", self.summary)

    def write(self, relative, value):
        data = encoded(value)
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self.files[relative] = {"sha256": sha(data), "size_bytes": len(data)}

    def evidence(self):
        manifest = encoded({"files": self.files})
        (self.root / "evidence-manifest.json").write_bytes(manifest)
        (self.root / "verification.json").write_bytes(encoded({"success": True}))
        pointer = self.root / "pointer.json"
        pointer.write_bytes(
            encoded(
                {
                    "state": "VERIFIED",
                    "verification_receipt": str(self.root / "verification.json"),
                    "evidence_manifest_sha256": sha(manifest),
                }
            )
        )
        return Evidence(pointer)

    def test_changed_closed_prediction_is_rejected_before_export(self):
        evidence = self.evidence()
        path = self.root / f"{self.folder}/pages/page-00/attempt-1.json"
        path.write_text('{"prediction":"changed"}')
        with self.assertRaisesRegex(ValueError, "Sealed evidence changed"):
            read_evaluation(evidence, "run-2", 2, "test", self.refs)

    def test_incomplete_output_preserves_its_score_and_attempts(self):
        pid = "page-00"
        self.write(
            f"{self.folder}/pages/{pid}/closed.json",
            {
                "page_id": pid,
                "status": "incomplete",
                "attempts": 2,
                "char_errors": 3,
                "char_count": 3,
                "word_errors": 1,
                "word_count": 1,
            },
        )
        for i in (1, 2):
            self.write(
                f"{self.folder}/pages/{pid}/attempt-{i}.json",
                {"stop_reason": "repetition", "prediction": "***"},
            )
        self.summary.update(
            char_errors=3,
            word_errors=1,
            cer=3 / 75,
            wer=1 / 25,
            failed_pages=1,
            retried_pages=1,
            all_pages_completed=False,
        )
        self.write(self.folder + "/closed.json", self.summary)
        summary, records = read_evaluation(
            self.evidence(), "run-2", 2, "test", self.refs
        )
        self.assertTrue(summary["provisional"])
        self.assertEqual(summary["completed"], 24)
        self.assertEqual(summary["char_errors"], 3)
        self.assertEqual(records[pid]["prediction"], "***")
        self.assertEqual(records[pid]["attempts"], 2)

    def test_first_success_cannot_be_replaced_by_a_later_attempt(self):
        relative = f"{self.folder}/pages/page-00/closed.json"
        closed = json.loads((self.root / relative).read_bytes())
        closed["attempts"] = 2
        self.write(relative, closed)
        self.write(
            f"{self.folder}/pages/page-00/attempt-2.json",
            {"stop_reason": "eos", "prediction": "abc"},
        )
        with self.assertRaisesRegex(ValueError, "first successful attempt"):
            read_evaluation(self.evidence(), "run-2", 2, "test", self.refs)

    def test_aggregate_cannot_drop_failed_page_errors(self):
        self.summary["char_errors"] = 1
        self.write(self.folder + "/closed.json", self.summary)
        with self.assertRaisesRegex(ValueError, "aggregate"):
            read_evaluation(self.evidence(), "run-2", 2, "test", self.refs)

    def test_original_and_normalized_text_use_official_policy(self):
        score = {"char_errors": 0, "char_count": 3, "word_errors": 0, "word_count": 1}
        self.assertEqual(metrics("ábc\n", "abc", score), score)
        score["char_errors"] = 1
        with self.assertRaises(ValueError):
            metrics("ábc\n", "abc", score)

    def export_fixture(self):
        root = self.root / "export"
        root.mkdir()
        payload = b"sealed OCR"
        (root / "text.txt").write_bytes(payload)
        manifest = encoded(
            {
                "files": {
                    "text.txt": {"sha256": sha(payload), "size_bytes": len(payload)}
                }
            }
        )
        (root / "manifest.json").write_bytes(manifest)
        return root, sha(manifest)

    def test_export_rejects_changed_closed_payload(self):
        root, digest = self.export_fixture()
        verified_export(root, digest)
        (root / "text.txt").write_text("changed OCR")
        with self.assertRaisesRegex(ValueError, "payload changed"):
            verified_export(root, digest)

    def test_export_rejects_unlisted_files(self):
        root, digest = self.export_fixture()
        (root / "extra.txt").write_text("unsealed")
        with self.assertRaisesRegex(ValueError, "file set changed"):
            verified_export(root, digest)

    def test_shard_cannot_execute_arbitrary_script(self):
        path = self.root / "unsafe.js"
        path.write_text(
            'alert("unexpected");window.OCR_DIFF_VIEWER_DATA.registerModel({});\n'
        )
        with self.assertRaisesRegex(ValueError, "Invalid benchmark shard"):
            read_shard(path, "registerModel")


if __name__ == "__main__":
    unittest.main()

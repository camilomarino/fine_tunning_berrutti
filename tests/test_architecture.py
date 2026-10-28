"""Keep dataset preparation independent from recipes and result publication."""

import ast
import importlib
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def imports(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


class LayerBoundaries(unittest.TestCase):
    def test_dataset_code_does_not_import_training_or_reporting(self):
        forbidden = ("finetune", "reporting", "experiments", "torch", "transformers")
        for folder in ("data", "data/preparation", "archive_berrutti"):
            for path in (ROOT / folder).glob("*.py"):
                with self.subTest(path=path.relative_to(ROOT)):
                    self.assertFalse(
                        [
                            name
                            for name in imports(path)
                            if name.split(".")[0] in forbidden
                        ]
                    )

    def test_dataset_preparation_can_be_imported_without_training_or_ml(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import data.preparation.luisa; import data.agent; "
                "import archive_berrutti.agent; "
                "assert not any(n.split('.')[0] in "
                "{'finetune','reporting','experiments','torch','transformers'} "
                "for n in sys.modules)",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_current_training_and_reporting_do_not_depend_on_legacy_modules(self):
        for folder in ("finetune", "reporting", "evaluation"):
            for path in (ROOT / folder).rglob("*.py"):
                with self.subTest(path=path.relative_to(ROOT)):
                    names = list(imports(path))
                    self.assertFalse(any(n.startswith("experiments") for n in names))
                    if folder == "finetune":
                        self.assertFalse(any(n.startswith("reporting") for n in names))

    def test_legacy_imports_alias_the_implementation_instead_of_copying_state(self):
        for name, target in {
            "runner": "finetune.recipes.berrutti_qwen38.runner",
            "luisa_prepare": "data.preparation.luisa",
            "benchmark_publication": "reporting.berrutti.benchmark_publication",
            "scoring": "evaluation.scoring",
        }.items():
            with self.subTest(module=name):
                self.assertIs(
                    importlib.import_module("experiments.berrutti_qwen38." + name),
                    importlib.import_module(target),
                )

    def test_legacy_cli_and_canonical_cli_have_the_same_help(self):
        for name, target in {
            "runner": "finetune.recipes.berrutti_qwen38.runner",
            "results_viewer": "reporting.berrutti.results_viewer",
            "luisa_prepare": "data.preparation.luisa",
        }.items():
            results = [
                subprocess.run(
                    [sys.executable, "-m", module, "--help"],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                for module in ("experiments.berrutti_qwen38." + name, target)
            ]
            with self.subTest(module=name):
                self.assertEqual(results[0].stdout, results[1].stdout)
                self.assertEqual(results[0].stderr, "")


if __name__ == "__main__":
    unittest.main()

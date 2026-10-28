"""Execute rendered wrappers with a stand-in scheduler, never a local trainer."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from finetune.jobs import prepare, render


class FineTuneJobs(unittest.TestCase):
    def profile(self, root, cluster="ruche"):
        return {
            "cluster": cluster,
            "partition": "gpua100" if cluster == "ruche" else "normal",
            "account": "authorized_project",
            "gpu_type": "a100" if cluster == "ruche" else "a40",
            "gpus": 2,
            "cpus": 8,
            "memory_gb": 96,
            "scratch_gb": 30,
            "time": "02:00:00",
            "workdir": str(root),
        }

    def test_site_specific_gres_and_container_use_same_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = {"engine": "command", "argv": ["python", "-m", "any_recipe"]}
            for site, gres, engine in (
                ("ruche", "gpu:2", "apptainer"),
                ("clusteruy", "gpu:a40:2", "singularity"),
            ):
                profile = self.profile(root, site)
                profile["account"] = "authorized.project"
                profile["container"] = {
                    "image": "/private/runtime.sif",
                    "binds": ["/private/data"],
                }
                script = render(profile, recipe, str(root / "run"))
                self.assertIn(f"#SBATCH --gres={gres}\n", script)
                self.assertIn(f"srun {engine} exec --nv --cleanenv", script)
                self.assertIn("python -m any_recipe", script)
                self.assertIn("#SBATCH --nodes=1", script)
                subprocess.run(["bash", "-n"], input=script, text=True, check=True)

    def test_literal_arguments_failure_receipt_and_no_second_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, bins, scratch = (
                root / "run with spaces",
                root / "bin",
                root / "scratch",
            )
            bins.mkdir()
            scheduler = bins / "srun"
            scheduler.write_text('#!/bin/bash\nexec "$@"\n')
            scheduler.chmod(0o700)
            recipe = {
                "engine": "command",
                "argv": [
                    "bash",
                    "-c",
                    'printf "%s" "$1" > "$FINETUNE_RUN_DIR/received"; exit 7',
                    "worker",
                    "literal $(touch PWNED) `touch ALSO_PWNED` ' quotes",
                ],
            }
            prepare(self.profile(root), recipe, str(bundle), bundle)
            environment = {
                **os.environ,
                "PATH": str(bins) + os.pathsep + os.environ["PATH"],
                "SLURM_JOB_ID": "123",
                "TMPDIR": str(scratch),
            }
            result = subprocess.run(
                ["bash", str(bundle / "job.sh")],
                env=environment,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 7, result.stderr)
            self.assertEqual((bundle / "received").read_text(), recipe["argv"][-1])
            self.assertFalse((root / "PWNED").exists())
            self.assertFalse((root / "ALSO_PWNED").exists())
            self.assertEqual(
                json.loads((bundle / "process.json").read_text()), {"exit_code": 7}
            )
            again = subprocess.run(
                ["bash", str(bundle / "job.sh")], env=environment, capture_output=True
            )
            self.assertNotEqual(again.returncode, 0)
            self.assertEqual(
                json.loads((bundle / "process.json").read_text()), {"exit_code": 7}
            )

    def test_trl_recipe_is_frozen_without_loading_ml_or_rewriting_training_options(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "training.yaml"
            original = (
                b"model_name_or_path: some/model\nmodel_revision: pinned\nseed: 17\n"
            )
            config.write_bytes(original)
            recipe = {
                "engine": "trl-sft",
                "executable": "/env/bin/trl",
                "config": str(config),
            }
            bundle = root / "run"
            manifest = prepare(self.profile(root), recipe, str(bundle), bundle)
            config.write_text("modified after preparation")
            self.assertEqual((bundle / "recipe.yaml").read_bytes(), original)
            self.assertIn(
                f"/env/bin/trl sft --config {bundle}/recipe.yaml --output_dir {bundle}/checkpoints",
                (bundle / "job.sh").read_text(),
            )
            self.assertEqual(manifest["state"], "PREPARED_NOT_SUBMITTED")
            self.assertFalse(manifest["gpu_smoke_passed"])
            with self.assertRaises(FileExistsError):
                prepare(self.profile(root), recipe, str(bundle), bundle)
            subprocess.run(
                ["sha256sum", "--check", "SHA256SUMS"],
                cwd=bundle,
                check=True,
                capture_output=True,
            )

    def test_changed_bundle_and_execution_outside_slurm_fail_before_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = {"engine": "command", "argv": ["touch", str(root / "ran")]}
            bundle = root / "run"
            prepare(self.profile(root), recipe, str(bundle), bundle)
            environment = {k: v for k, v in os.environ.items() if k != "SLURM_JOB_ID"}
            result = subprocess.run(
                ["bash", str(bundle / "job.sh")], env=environment, capture_output=True
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((bundle / "started.json").exists())
            (bundle / "recipe.json").write_text("changed")
            result = subprocess.run(
                ["bash", str(bundle / "job.sh")],
                env={**environment, "SLURM_JOB_ID": "123"},
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((bundle / "started.json").exists())
            self.assertFalse((root / "ran").exists())

    def test_invalid_resources_directive_injection_and_reserved_variables_rejected(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = {"engine": "command", "argv": ["true"]}
            for changes in (
                {"account": "project\n#SBATCH --exclusive"},
                {"gpus": True},
                {"gpus": 0},
                {"time": "24:99:00"},
                {"partition": "gpu", "gpu_type": "a100"},
                {"nodes": 2},
                {"workdir": "relative"},
                {"environment": {"CUDA_VISIBLE_DEVICES": "0"}},
            ):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    render({**self.profile(root), **changes}, recipe, str(root / "run"))

    def test_container_receives_explicit_environment_and_allocated_scratch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bins, bundle = root / "bin", root / "run"
            bins.mkdir()
            scripts = {
                "srun": 'exec "$@"',
                "apptainer": (
                    'test "$1 $2 $3" = "exec --nv --cleanenv" || exit 9\n'
                    'shift 3\nwhile [ "$1" = --bind ]; do shift 2; done\n'
                    'test "$1" = /private/runtime.sif || exit 9\n'
                    'shift\nexec "$@"'
                ),
            }
            for name, body in scripts.items():
                executable = bins / name
                executable.write_text("#!/bin/bash\nset -eu\n" + body + "\n")
                executable.chmod(0o700)
            profile = self.profile(root)
            profile["container"] = {"image": "/private/runtime.sif", "binds": []}
            profile["environment"] = {"OMP_NUM_THREADS": "4"}
            recipe = {
                "engine": "command",
                "argv": [
                    "bash",
                    "-c",
                    'printf "%s\\n" "$OMP_NUM_THREADS" "$TMPDIR" "$XDG_CACHE_HOME" "$CUDA_VISIBLE_DEVICES" > "$FINETUNE_RUN_DIR/environment"',
                ],
            }
            prepare(profile, recipe, str(bundle), bundle)
            scratch = str(root / "scratch with spaces")
            result = subprocess.run(
                ["bash", str(bundle / "job.sh")],
                env={
                    **os.environ,
                    "PATH": str(bins) + os.pathsep + os.environ["PATH"],
                    "SLURM_JOB_ID": "123",
                    "TMPDIR": scratch,
                    "CUDA_VISIBLE_DEVICES": "1,3",
                },
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                (bundle / "environment").read_text().splitlines(),
                ["4", scratch, scratch + "/finetune-cache", "1,3"],
            )


if __name__ == "__main__":
    unittest.main()

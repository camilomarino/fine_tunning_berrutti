"""Prepare private, reproducible Slurm jobs without submitting or importing ML."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from finetune.jobs import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument(
        "--run-dir", required=True, help="New absolute directory on cluster"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    identifier = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    )
    output = args.output or Path(".private/finetune") / identifier
    try:
        recipe = json.loads(args.recipe.read_text())
        if isinstance(recipe, dict) and recipe.get("engine") == "trl-sft":
            recipe["config"] = str((args.recipe.parent / recipe["config"]).resolve())
        manifest = prepare(
            json.loads(args.profile.read_text()), recipe, args.run_dir, output
        )
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({**manifest, "bundle": str(output.resolve())}))


if __name__ == "__main__":
    main()

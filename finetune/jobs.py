"""One Slurm wrapper for Ruche and ClusterUY, independent of model architecture."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from pathlib import Path


def text(value):
    if not isinstance(value, str) or not value or any(c in value for c in "\n\r\0"):
        raise ValueError("Se esperaba texto no vacío en una sola línea")
    return value


def absolute(value):
    value = text(value)
    if not value.startswith("/") or ".." in Path(value).parts:
        raise ValueError("Las rutas del cluster deben ser absolutas, sin '..'")
    return value


def fields(value, required, optional=()):
    if not isinstance(value, dict) or set(value) - set(required) - set(optional):
        raise ValueError("Campos desconocidos en la configuración")
    if set(required) - set(value):
        raise ValueError("Faltan campos obligatorios en la configuración")


def validate(profile, recipe):
    fields(
        profile,
        (
            "cluster",
            "partition",
            "account",
            "gpu_type",
            "gpus",
            "cpus",
            "memory_gb",
            "scratch_gb",
            "time",
            "workdir",
        ),
        ("qos", "container", "environment"),
    )
    if profile["cluster"] not in ("ruche", "clusteruy"):
        raise ValueError("Cluster esperado: ruche o clusteruy")
    for key in ("partition", "account", "gpu_type", "qos"):
        if key in profile and not re.fullmatch(r"[A-Za-z0-9_.-]+", text(profile[key])):
            raise ValueError(f"Valor Slurm inválido: {key}")
    for key in ("gpus", "cpus", "memory_gb", "scratch_gb"):
        if type(profile[key]) is not int or profile[key] < 1:
            raise ValueError(f"Recurso positivo requerido: {key}")
    if not re.fullmatch(r"(?:\d+-)?\d{1,2}:[0-5]\d:[0-5]\d", text(profile["time"])):
        raise ValueError("Walltime esperado: [días-]hh:mm:ss")
    absolute(profile["workdir"])
    if profile["cluster"] == "ruche":
        expected = {"gpua100": "a100", "gpu": "v100", "gpu_test": "v100"}
        if expected.get(profile["partition"]) != profile["gpu_type"]:
            raise ValueError("Partición y GPU de Ruche incompatibles con este wrapper")
    if "container" in profile:
        fields(profile["container"], ("image", "binds"))
        absolute(profile["container"]["image"])
        if not isinstance(profile["container"]["binds"], list):
            raise ValueError("container.binds debe ser una lista de directorios")
        for path in profile["container"]["binds"]:
            absolute(path)
            if ":" in path or "," in path:
                raise ValueError("Directorio de bind ambiguo")
    environment = profile.get("environment", {})
    if not isinstance(environment, dict):
        raise ValueError("environment debe ser un mapa de variables explícitas")
    for key, value in environment.items():
        if (
            not isinstance(key, str)
            or not re.fullmatch(r"[A-Z_][A-Z0-9_]*", key)
            or key
            in {
                "HOME",
                "CODEX_HOME",
                "TMPDIR",
                "XDG_CACHE_HOME",
                "CUDA_VISIBLE_DEVICES",
            }
            or key.startswith(("SLURM_", "FINETUNE_"))
        ):
            raise ValueError("Variable reservada o nombre inválido")
        text(value)
    fields(recipe, ("engine",), ("executable", "config", "argv"))
    if recipe["engine"] == "trl-sft":
        fields(recipe, ("engine", "executable", "config"))
        absolute(recipe["executable"])
        if not Path(recipe["config"]).is_file():
            raise ValueError("Falta el archivo local de configuración TRL")
    elif recipe["engine"] == "command":
        fields(recipe, ("engine", "argv"))
        if not isinstance(recipe["argv"], list) or not recipe["argv"]:
            raise ValueError("argv debe ser una lista no vacía, sin shell implícito")
        for token in recipe["argv"]:
            text(token)
    else:
        raise ValueError("Motor esperado: trl-sft o command")


def render(profile, recipe, run_dir):
    """Render only. This never queries SSH, submits work or imports GPU libraries."""
    validate(profile, recipe)
    absolute(run_dir)
    if not re.fullmatch(r"[/A-Za-z0-9_. -]+", run_dir):
        raise ValueError("Ruta de operación incompatible con Slurm/binds")
    if profile.get("container") and any(c in profile["workdir"] for c in ":,"):
        raise ValueError("Directorio de trabajo incompatible con los binds")
    quote = shlex.quote
    gres = f"gpu:{profile['gpus']}"
    if profile["cluster"] == "clusteruy":
        gres = f"gpu:{profile['gpu_type']}:{profile['gpus']}"
    lines = [
        "#!/bin/bash",
        f"#SBATCH --partition={profile['partition']}",
        f"#SBATCH --account={profile['account']}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --gres={gres}",
        f"#SBATCH --cpus-per-task={profile['cpus']}",
        f"#SBATCH --mem={profile['memory_gb']}G",
        f"#SBATCH --tmp={profile['scratch_gb']}G",
        f"#SBATCH --time={profile['time']}",
        f'#SBATCH --output="{run_dir}/slurm-%j.out"',
    ]
    if profile.get("qos"):
        lines.append(f"#SBATCH --qos={profile['qos']}")
    lines += [
        "set -euo pipefail",
        "umask 077",
        ': "${SLURM_JOB_ID:?Ejecutar mediante sbatch, nunca en login}"',
        f"export FINETUNE_RUN_DIR={quote(run_dir)}",
        'cd "$FINETUNE_RUN_DIR"',
        "sha256sum --check SHA256SUMS",
        '(set -o noclobber; printf \'{"job_id":"%s"}\\n\' "$SLURM_JOB_ID" > started.json)',
    ]
    if profile["cluster"] == "ruche":
        lines += [': "${TMPDIR:?Slurm debe proporcionar el SSD solicitado}"']
    else:
        lines += ['export TMPDIR="/scratch/$USER/$SLURM_JOB_ID"']
    environment = {
        "OMP_NUM_THREADS": str(profile["cpus"]),
        "OPENBLAS_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        **profile.get("environment", {}),
    }
    lines += [
        'export XDG_CACHE_HOME="$TMPDIR/finetune-cache"',
        'mkdir -p "$XDG_CACHE_HOME"',
        *(f"export {key}={quote(value)}" for key, value in environment.items()),
        f"cd {quote(profile['workdir'])}",
    ]
    if recipe["engine"] == "trl-sft":
        command = [
            recipe["executable"],
            "sft",
            "--config",
            f"{run_dir}/recipe.yaml",
            "--output_dir",
            f"{run_dir}/checkpoints",
        ]
    else:
        command = [token.replace("{run_dir}", run_dir) for token in recipe["argv"]]
    prefix = ["srun"]
    if profile.get("container"):
        container = profile["container"]
        prefix += [
            "apptainer" if profile["cluster"] == "ruche" else "singularity",
            "exec",
            "--nv",
            "--cleanenv",
        ]
        for path in dict.fromkeys([run_dir, profile["workdir"], *container["binds"]]):
            prefix += ["--bind", f"{path}:{path}"]
        # TMPDIR is assigned inside the allocation; expand it only in this shell.
        launch = shlex.join(prefix) + ' --bind "$TMPDIR:$TMPDIR" '
        launch += shlex.join(
            [
                container["image"],
                "env",
                *[f"{key}={value}" for key, value in environment.items()],
            ]
        )
        launch += ' "TMPDIR=$TMPDIR" "XDG_CACHE_HOME=$XDG_CACHE_HOME" '
        launch += ' "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:?Slurm debe asignar las GPU}" '
        launch += ' "FINETUNE_RUN_DIR=$FINETUNE_RUN_DIR" ' + shlex.join(command)
    else:
        launch = shlex.join(prefix + command)
    lines += [
        "set +e",
        launch,
        "result=$?",
        "set -e",
        'printf \'{"exit_code":%d}\\n\' "$result" > "$FINETUNE_RUN_DIR/process.json"',
        'exit "$result"',
        "",
    ]
    return "\n".join(lines)


def prepare(profile, recipe, run_dir, output):
    """Seal a small private bundle. Inputs and deployed environments are not copied."""
    script = render(profile, recipe, run_dir)
    payloads = {
        "job.sh": script.encode(),
        "profile.json": (json.dumps(profile, indent=2) + "\n").encode(),
        "recipe.json": (json.dumps(recipe, indent=2) + "\n").encode(),
    }
    if recipe["engine"] == "trl-sft":
        payloads["recipe.yaml"] = Path(recipe["config"]).read_bytes()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    for name, payload in payloads.items():
        path = output / name
        path.write_bytes(payload)
        path.chmod(0o700 if name == "job.sh" else 0o600)
    hashes = {
        name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()
    }
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha}  {name}\n" for name, sha in hashes.items())
    )
    manifest = {
        "schema_version": 1,
        "state": "PREPARED_NOT_SUBMITTED",
        "cluster": profile["cluster"],
        "engine": recipe["engine"],
        "run_dir": run_dir,
        "sha256": hashes,
        "gpu_smoke_passed": False,
        "environment_and_data_staged": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest

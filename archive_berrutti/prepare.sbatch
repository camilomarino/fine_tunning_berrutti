#!/bin/bash
#SBATCH --job-name=berrutti-archive
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit through Slurm; do not execute on login}"
: "${ARCHIVE_CODE_ROOT:?Set the staged code directory in WORKDIR}"
: "${ARCHIVE_PYTHON:?Set an existing Python >=3.11 with Pillow}"
cd "$ARCHIVE_CODE_ROOT"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONDONTWRITEBYTECODE=1
ruche-quota --minimal
srun "$ARCHIVE_PYTHON" -m archive_berrutti "$@"

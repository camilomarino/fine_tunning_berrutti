#!/bin/bash
#SBATCH --job-name=berrutti-data-import
#SBATCH --partition=cpu_med
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
set -euo pipefail
umask 077
: "${BERRUTTI_DATA_CODE:?Ruta nueva al código del repositorio}"
: "${BERRUTTI_DATA_PYTHON:?Python existente con Pillow, anyascii y rapidfuzz}"
cd "$BERRUTTI_DATA_CODE"
export PYTHONPATH="$BERRUTTI_DATA_CODE"
export PYTHONDONTWRITEBYTECODE=1
exec "$BERRUTTI_DATA_PYTHON" -u -m data "$@"

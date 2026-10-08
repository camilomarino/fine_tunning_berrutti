#!/bin/bash
#SBATCH --job-name=berrutti-qwen38-prep
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
umask 077
: "${EXPERIMENT_ROOT:?Set the private experiment root in WORKDIR}"
FEASIBILITY_ROOT="$WORKDIR/berrutti-finetune-feasibility/20261007-r1"
"$FEASIBILITY_ROOT/bootstrap/bin/uv" pip install --python "$FEASIBILITY_ROOT/env/bin/python" \
  'anyascii==0.3.3' 'rapidfuzz>=3.13,<4'
export PYTHONPATH="$EXPERIMENT_ROOT/code:$EXPERIMENT_ROOT/code/src"
"$FEASIBILITY_ROOT/env/bin/python" - <<'PY'
from pathlib import Path
from experiments.berrutti_qwen38.common import load_pages, score_pairs

root = Path(__import__('os').environ['EXPERIMENT_ROOT'])
pages = load_pages(root / 'dataset', root / 'code/experiments/berrutti_qwen38/split-v1.json')
assert {name: len(rows) for name, rows in pages.items()} == {
    'train': 125, 'validation': 25, 'test': 25,
}
assert score_pairs([('é', 'e')])['cer'] == 0
print('DATASET_AND_METRICS_OK')
PY

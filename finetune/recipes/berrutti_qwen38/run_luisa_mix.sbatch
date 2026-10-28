#!/bin/bash
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --tmp=30G
#SBATCH --time=24:00:00
set -euo pipefail
umask 077
MODE="$1"
RUN_ROOT="$2"
OPERATION_ROOT="$3"
RUNTIME_PYTHON="$4"
RUNTIME_BUNDLE="$5"
export PYTHONPATH="$OPERATION_ROOT/code"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TORCHDYNAMO_DISABLE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="$RUN_ROOT/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$RUN_ROOT/cache/inductor"
export HF_MODULES_CACHE="$RUN_ROOT/cache/hf-modules"
mkdir -p "$RUN_ROOT/jobs/$SLURM_JOB_ID"
printf '%s\n' '{"event":"phase","phase":"local_runtime_staging"}' >> "$RUN_ROOT/events.jsonl"
srun tar -xf "$RUNTIME_BUNDLE" -C "$TMPDIR"
export PYTHONPATH="$OPERATION_ROOT/code:$TMPDIR/site-packages"
cd "$OPERATION_ROOT/code"
set +e
srun "$RUNTIME_PYTHON" -u -m finetune.recipes.berrutti_qwen38.luisa_mix \
  --config "$OPERATION_ROOT/config.json" --mode "$MODE" --root "$RUN_ROOT"
result=$?
set -e
if [ "$result" -eq 85 ] && [ "$MODE" = train ]; then
  continuation=$(sbatch --parsable --account="$SLURM_JOB_ACCOUNT" \
    --dependency="afterany:$SLURM_JOB_ID" --kill-on-invalid-dep=yes \
    --output="$RUN_ROOT/jobs/%j.out" --error="$RUN_ROOT/jobs/%j.out" \
    "$OPERATION_ROOT/code/finetune/recipes/berrutti_qwen38/run_luisa_mix.sbatch" \
    "$MODE" "$RUN_ROOT" "$OPERATION_ROOT" "$RUNTIME_PYTHON" "$RUNTIME_BUNDLE")
  continuation="${continuation%%;*}"
  printf '{"job_id":%s,"previous_job_id":%s}\n' "$continuation" "$SLURM_JOB_ID" > "$RUN_ROOT/continuation.tmp"
  mv "$RUN_ROOT/continuation.tmp" "$RUN_ROOT/continuation.json"
  exit 0
fi
exit "$result"

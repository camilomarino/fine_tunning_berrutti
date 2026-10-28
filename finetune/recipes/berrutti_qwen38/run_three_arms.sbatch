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
RUNTIME_CONTAINER="${6:-}"
if [ -n "$RUNTIME_CONTAINER" ]; then
  export TMPDIR="/scratch/$USER/$SLURM_JOB_ID"
  mkdir -p "$TMPDIR"
fi
export PYTHONPATH="$OPERATION_ROOT/code"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TORCHDYNAMO_DISABLE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="$RUN_ROOT/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$RUN_ROOT/cache/inductor"
export HF_MODULES_CACHE="$RUN_ROOT/cache/hf-modules"
mkdir -p "$RUN_ROOT/jobs/$SLURM_JOB_ID"
if [ -n "$RUNTIME_CONTAINER" ]; then
  # Keep host CUDA initialized for the ClusterUY Singularity/A40 integration.
  python3 -u - "$RUN_ROOT/jobs/$SLURM_JOB_ID/host-cuda.json" <<'PY' &
import ctypes, json, signal, sys
cuda = ctypes.CDLL("libcuda.so.1")
result = cuda.cuInit(0)
print(json.dumps({"phase": "host_cuda_holder", "cuInit": result}), flush=True)
if result:
    raise SystemExit(1)
with open(sys.argv[1], "x") as stream:
    json.dump({"success": True, "cuInit": result}, stream)
signal.pause()
PY
  cuda_holder_pid=$!
  trap 'kill "$cuda_holder_pid" 2>/dev/null || true' EXIT
  for attempt in {1..10}; do
    if [ -f "$RUN_ROOT/jobs/$SLURM_JOB_ID/host-cuda.json" ]; then break; fi
    kill -0 "$cuda_holder_pid"
    sleep 1
  done
  test -f "$RUN_ROOT/jobs/$SLURM_JOB_ID/host-cuda.json"
fi
if [ -n "$RUNTIME_BUNDLE" ]; then
  printf '%s\n' '{"event":"phase","phase":"local_runtime_staging"}' >> "$RUN_ROOT/events.jsonl"
  srun tar -xf "$RUNTIME_BUNDLE" -C "$TMPDIR"
  export PYTHONPATH="$OPERATION_ROOT/code:$TMPDIR/site-packages"
fi
cd "$OPERATION_ROOT/code"
launch=(srun)
if [ -n "$RUNTIME_CONTAINER" ]; then
  # Preserve the allocation mask and runtime settings across --cleanenv.
  for variable in PYTHONPATH OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS \
    TORCHDYNAMO_DISABLE CUDA_DEVICE_ORDER CUDA_VISIBLE_DEVICES HF_HUB_OFFLINE \
    TRANSFORMERS_OFFLINE TOKENIZERS_PARALLELISM TRITON_CACHE_DIR \
    TORCHINDUCTOR_CACHE_DIR HF_MODULES_CACHE TMPDIR; do
    export "SINGULARITYENV_${variable}=${!variable}"
  done
  launch+=(singularity exec --nv --cleanenv --bind "$OPERATION_ROOT:$OPERATION_ROOT" \
    --bind "$TMPDIR:$TMPDIR" "$RUNTIME_CONTAINER")
fi
set +e
"${launch[@]}" "$RUNTIME_PYTHON" -u \
  -m finetune.recipes.berrutti_qwen38.runner --config "$OPERATION_ROOT/config.json" \
  --mode "$MODE" --root "$RUN_ROOT"
result=$?
set -e
if [ "$result" -eq 85 ]; then
  if [ "$MODE" = baseline ] || [ "$MODE" = smoke ]; then exit 85; fi
  site_options=()
  if [ -n "$RUNTIME_CONTAINER" ]; then
    site_options=(--partition=normal --qos=gpu --gres=gpu:a40:2)
  fi
  continuation=$(sbatch --parsable "${site_options[@]}" --account="$SLURM_JOB_ACCOUNT" \
    --dependency="afterany:$SLURM_JOB_ID" --kill-on-invalid-dep=yes \
    --output="$RUN_ROOT/jobs/%j.out" --error="$RUN_ROOT/jobs/%j.out" \
    "$OPERATION_ROOT/code/finetune/recipes/berrutti_qwen38/run_three_arms.sbatch" \
    "$MODE" "$RUN_ROOT" "$OPERATION_ROOT" "$RUNTIME_PYTHON" "$RUNTIME_BUNDLE" "$RUNTIME_CONTAINER")
  continuation="${continuation%%;*}"
  printf '{"job_id":%s,"previous_job_id":%s}\n' "$continuation" "$SLURM_JOB_ID" > "$RUN_ROOT/continuation.tmp"
  mv "$RUN_ROOT/continuation.tmp" "$RUN_ROOT/continuation.json"
  exit 0
fi
exit "$result"

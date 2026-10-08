#!/bin/bash
#SBATCH --job-name=berrutti-qwen38-curriculum
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem=110G
#SBATCH --time=12:00:00
set -euo pipefail
umask 077
: "${EXPERIMENT_ROOT:?Set the private experiment root in WORKDIR}"
: "${CURRICULUM_MODE:?Set CURRICULUM_MODE=sanity or full}"
FEASIBILITY_ROOT="$WORKDIR/berrutti-finetune-feasibility/20261007-r1"
export HF_HOME="$FEASIBILITY_ROOT/hf-cache"
KERNEL_PATH="$(cat "$FEASIBILITY_ROOT/kernel-hub-path.txt")"
export PYTHONPATH="$KERNEL_PATH:$EXPERIMENT_ROOT/code:$EXPERIMENT_ROOT/code/src"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
RUN_NAME="curriculum-v1"
if [[ "$CURRICULUM_MODE" == sanity ]]; then
  RUN_NAME="curriculum-sanity-v1"
elif [[ "$CURRICULUM_MODE" != full ]]; then
  echo "Unsupported CURRICULUM_MODE" >&2
  exit 2
fi
"$FEASIBILITY_ROOT/env/bin/python" -m experiments.berrutti_qwen38.train_curriculum \
  --dataset "$EXPERIMENT_ROOT/dataset" \
  --split "$EXPERIMENT_ROOT/code/experiments/berrutti_qwen38/split-v1.json" \
  --crops "$EXPERIMENT_ROOT/inputs/curriculum-v1.jsonl" \
  --baseline "$EXPERIMENT_ROOT/runs/full" \
  --output "$EXPERIMENT_ROOT/runs/$RUN_NAME" \
  --mode "$CURRICULUM_MODE"

#!/bin/bash
#SBATCH --job-name=berrutti-4b-v100-smoke
#SBATCH --partition=gpu_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=70G
#SBATCH --time=01:00:00
set -euo pipefail
umask 077
: "${EXPERIMENT_ROOT:?Set the private experiment root in WORKDIR}"
FEASIBILITY_ROOT="$WORKDIR/berrutti-finetune-feasibility/20261007-r1"
export HF_HOME="$FEASIBILITY_ROOT/hf-cache"
KERNEL_PATH="$(cat "$FEASIBILITY_ROOT/kernel-hub-path.txt")"
export PYTHONPATH="$KERNEL_PATH:$EXPERIMENT_ROOT/code:$EXPERIMENT_ROOT/code/src"
export TOKENIZERS_PARALLELISM=false
"$FEASIBILITY_ROOT/env/bin/python" -m experiments.berrutti_qwen38.smoke_v100

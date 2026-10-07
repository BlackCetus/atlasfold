#!/bin/bash
#SBATCH --job-name=af-prott5
#SBATCH --output=slurm/log/af-prott5-%j.txt
#SBATCH --account=crescendo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=0-12:00:00
#SBATCH --chdir=/e/project1/crescendo/reim1/atlasfold

set -euo pipefail

module purge
module use /e/project1/crescendo/hoffbauer1/easybuild/easybuild/jupiter/modules/all/Core
module load CUDA/12.8.0 GCC

export UV_CACHE_DIR=/e/project1/crescendo/reim1/uv-cache
export TRITON_HOME=/e/project1/crescendo/reim1/triton
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=WARN
# ProtT5 weights must be pre-downloaded into HF_HUB_CACHE (compute nodes are offline).
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1   # stream the tqdm progress bar / loss live to the log (DDP non-TTY)
# cueq fast kernels need Blackwell (sm_100+); GH200 is sm_90 -> torch triangle
# backend. Activation checkpointing + expandable segments keep it within 95 GB.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Compute nodes are offline -> wandb logs locally; sync from a login node with:
#   wandb sync experiments/stage1-prott5xl/wandb/offline-run-*
export WANDB_MODE=offline
mkdir -p "$UV_CACHE_DIR" "$TRITON_HOME" slurm/log

source /e/project1/crescendo/reim1/atlasfold/.venv/bin/activate

RUN_NAME=stage1-prott5xl
OUT_DIR=/e/project1/crescendo/reim1/atlasfold/experiments
CKPT="$OUT_DIR/$RUN_NAME/checkpoints/last.ckpt"
RESUME=""
if [ -f "$CKPT" ]; then RESUME="--resume_from_checkpoint $CKPT"; echo "Resuming from $CKPT"; fi

# ProtT5 backbone: override lm_name for BOTH the model and the data tokenizer so
# featurization matches the frozen ProtT5 encoder.
# --gpu-bind=none: expose all 4 GPUs to each srun task (Lightning DDP picks local_rank).
srun --gpu-bind=none python /e/project1/crescendo/reim1/atlasfold/scripts/train_monomer.py \
  configs/monomer/train_stage1.yaml \
  --experiment_name "$RUN_NAME" \
  --out_dir "$OUT_DIR" \
  --num_nodes "$SLURM_NNODES" \
  --num_gpus "${SLURM_GPUS_ON_NODE:-4}" \
  --wandb \
  $RESUME \
  --override model.lm_name=prott5-xl train.data.lm_name=prott5-xl \
    model.trunk.blocks_per_ckpt=1 \
    model.diffusion_head.blocks_per_ckpt=1 \
    model.confidence_head.blocks_per_ckpt=1 \
    train.wandb.entity=ge28jer \
    train.wandb.project=strucPred-bench \
    train.trainer.val_check_interval=0.1 \
    train.trainer.ckpt_every_n_train_steps=100 \
    train.trainer.ckpt_save_top_k=1

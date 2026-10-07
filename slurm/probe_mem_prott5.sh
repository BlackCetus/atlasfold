#!/bin/bash
#SBATCH --job-name=mem-prott5
#SBATCH --output=slurm/log/mem-prott5-%j.txt
#SBATCH --account=crescendo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:25:00
#SBATCH --chdir=/e/project1/crescendo/reim1/atlasfold

set -euo pipefail
module purge
module use /e/project1/crescendo/hoffbauer1/easybuild/easybuild/jupiter/modules/all/Core
module load CUDA/12.8.0 GCC
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p slurm/log
source /e/project1/crescendo/reim1/atlasfold/.venv/bin/activate

# 1-GPU, a few steps, activation checkpointing ON. Memory proxy for the DDP run.
srun --gpu-bind=none python scripts/train_monomer.py configs/monomer/train_stage1.yaml \
  --experiment_name probe-prott5-mem \
  --out_dir /e/project1/crescendo/reim1/atlasfold/experiments \
  --num_nodes 1 --num_gpus 1 \
  --override \
    model.lm_name=prott5-xl train.data.lm_name=prott5-xl \
    model.trunk.blocks_per_ckpt=1 \
    model.diffusion_head.blocks_per_ckpt=1 \
    model.confidence_head.blocks_per_ckpt=1 \
    train.trainer.limit_train_batches=3 \
    train.trainer.limit_val_batches=1 \
    train.data.num_workers=4 \
    train.wandb.use=false
echo "=== PROBE EXIT $? ==="

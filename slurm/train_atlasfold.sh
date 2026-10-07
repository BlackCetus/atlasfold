#!/bin/bash
#SBATCH --job-name=atlasfold
#SBATCH --output=slurm/log/atlasfold-%j.txt
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
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=1
mkdir -p "$UV_CACHE_DIR" "$TRITON_HOME"

source /e/project1/crescendo/reim1/atlasfold/.venv/bin/activate

srun python /e/project1/crescendo/reim1/atlasfold/scripts/train_monomer.py configs/monomer/train_stage1.yaml \
 --out_dir /e/project1/crescendo/reim1/atlasfold/experiments \
 --num_nodes "$SLURM_NNODES" \
 --num_gpus "$SLURM_GPUS" \
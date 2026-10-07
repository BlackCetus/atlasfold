#!/bin/bash
#SBATCH --job-name=gpu-vis
#SBATCH --output=slurm/log/gpu-vis-%j.txt
#SBATCH --account=crescendo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=0-00:05:00
#SBATCH --chdir=/e/project1/crescendo/reim1/atlasfold

set -uo pipefail
module purge
module use /e/project1/crescendo/hoffbauer1/easybuild/easybuild/jupiter/modules/all/Core
module load CUDA/12.8.0 GCC
source /e/project1/crescendo/reim1/atlasfold/.venv/bin/activate

probe='echo "task=$SLURM_PROCID local=$SLURM_LOCALID CVD=[$CUDA_VISIBLE_DEVICES] devcount=$(python -c "import torch;print(torch.cuda.device_count())")"'

echo "===== default srun binding ====="
srun bash -c "$probe"
echo "===== srun --gpu-bind=none ====="
srun --gpu-bind=none bash -c "$probe"
echo "===== DONE ====="

#!/bin/bash
#SBATCH --job-name=cueq-probe
#SBATCH --output=slurm/log/cueq-probe-%j.txt
#SBATCH --account=crescendo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:05:00
#SBATCH --chdir=/e/project1/crescendo/reim1/atlasfold

set -euo pipefail
module purge
module use /e/project1/crescendo/hoffbauer1/easybuild/easybuild/jupiter/modules/all/Core
module load CUDA/12.8.0 GCC
source /e/project1/crescendo/reim1/atlasfold/.venv/bin/activate

python - <<'PY'
import importlib.util as u
import torch
print("HOST", __import__("socket").gethostname())
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available(),
      "device_count", torch.cuda.device_count())
for m in ["cuequivariance_torch", "cuequivariance", "cuequivariance_ops_torch", "triton"]:
    print("findable", m, u.find_spec(m) is not None)
try:
    from cuequivariance_torch.primitives.triangle import triangle_attention  # repo's path
    print("REPO cueq triangle import: OK")
except Exception as e:
    print("REPO cueq triangle import: FAIL", type(e).__name__, str(e)[:160])
PY

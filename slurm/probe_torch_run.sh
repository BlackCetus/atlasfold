#!/bin/bash
#SBATCH --job-name=torch-run
#SBATCH --output=slurm/log/torch-run-%j.txt
#SBATCH --account=crescendo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:10:00
#SBATCH --chdir=/e/project1/crescendo/reim1/atlasfold

set -euo pipefail
module purge
module use /e/project1/crescendo/hoffbauer1/easybuild/easybuild/jupiter/modules/all/Core
module load CUDA/12.8.0 GCC
source /e/project1/crescendo/reim1/atlasfold/.venv/bin/activate

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
import cuequivariance_torch as ct
print("cuequivariance_torch", getattr(ct, "__version__", "?"))
from atlasfold.model.network import trunk

dev = "cuda"
stack = trunk.LMStack(channel_s=32, channel_z=16, num_heads=2, num_tri_heads=1,
                      dropout_z=0.0, num_blocks=2, blocks_per_ckpt=None).to(dev)
B, L = 1, 16
s = torch.randn(B, L, 32, device=dev, requires_grad=True)
z = torch.randn(B, L, L, 16, device=dev, requires_grad=True)
mask = torch.ones(B, L, dtype=torch.bool, device=dev)
try:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        so, zo = stack(s, z, mask, "torch")   # the exact training backend
    (so.float().sum() + zo.float().sum()).backward()
    ok = s.grad is not None and torch.isfinite(s.grad).all()
    print(f"CUEQ LMStack fwd+bwd OK: s_out={tuple(so.shape)} grad_finite={bool(ok)}")
except Exception as e:
    import traceback; traceback.print_exc()
    print("CUEQ LMStack FAIL:", type(e).__name__, str(e)[:160])
PY

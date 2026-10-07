#!/bin/bash
#SBATCH --job-name=af-tput
#SBATCH --output=slurm/log/af-tput-%j.txt
#SBATCH --account=crescendo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=0-01:00:00
#SBATCH --chdir=/e/project1/crescendo/reim1/atlasfold

set -uo pipefail   # NOT -e: a config may OOM; we continue to the next
module purge
module use /e/project1/crescendo/hoffbauer1/easybuild/easybuild/jupiter/modules/all/Core
module load CUDA/12.8.0 GCC
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p slurm/log /tmp/af-tput
source /e/project1/crescendo/reim1/atlasfold/.venv/bin/activate

OUT=/e/project1/crescendo/reim1/atlasfold
# Baseline AtlasLM-3B (default lm in _model.yaml). B = per-GPU batch, C = blocks_per_ckpt.
# global_batch = B*4 -> accumulate_grad_batches = 1 (clean per-micro-batch timing).
for spec in "1:1" "1:2" "1:4" "1:null" "2:1" "2:2" "4:1"; do
  B=${spec%:*}; C=${spec#*:}
  LOG="slurm/log/tput_b${B}_c${C}.txt"
  echo "######## CONFIG batch_size=$B blocks_per_ckpt=$C ########"
  export MASTER_PORT=$((20000 + RANDOM % 20000))   # fresh port each config
  srun --gpu-bind=none --kill-on-bad-exit=0 python scripts/train_monomer.py configs/monomer/train_stage1.yaml \
    --experiment_name "tput-b${B}-c${C}" --out_dir "$OUT/experiments_probe" \
    --num_nodes 1 --num_gpus 4 \
    --override \
      model.trunk.blocks_per_ckpt=$C \
      model.diffusion_head.blocks_per_ckpt=$C \
      model.confidence_head.blocks_per_ckpt=$C \
      train.data.batch_size=$B \
      train.data.global_batch_size=$((B*4)) \
      train.trainer.limit_train_batches=25 \
      train.trainer.limit_val_batches=0 \
      train.wandb.use=false \
    > "$LOG" 2>&1
  RC=$?
  if [ $RC -ne 0 ]; then
    if grep -qiE "out of memory|OutOfMemory" "$LOG"; then echo "  -> OOM (does not fit)"; else echo "  -> FAILED rc=$RC"; grep -iE "error|assert" "$LOG" | tail -3; fi
  else
    # last steady-state it/s from the tqdm bar
    ITS=$(tr '\r' '\n' < "$LOG" | grep -oE "[0-9.]+it/s" | tail -1)
    SPS=$(tr '\r' '\n' < "$LOG" | grep -oE "[0-9.]+it/s" | tail -1 | grep -oE "^[0-9.]+")
    echo "  -> OK  micro-batch rate=${ITS:-?}  (samples/s/GPU = rate * $B)"
  fi
done
echo "===== THROUGHPUT SWEEP DONE ====="
# cleanup probe experiment dir
rm -rf "$OUT/experiments_probe" 2>/dev/null || true

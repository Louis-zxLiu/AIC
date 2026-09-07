#!/bin/bash
#SBATCH -J aic_smoke2
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH -o /public/home/jqfan/AIC/automation/slurm_%x_%j.out
#SBATCH -e /public/home/jqfan/AIC/automation/slurm_%x_%j.err

set -euo pipefail
cd /public/home/jqfan/AIC
export PYTHONPATH=/public/home/jqfan/AIC
export TORCH_HOME=/public/home/jqfan/AIC/cache/torch
export XDG_CACHE_HOME=/public/home/jqfan/AIC/cache/xdg
export TMPDIR=/public/home/jqfan/AIC/tmp
nvidia-smi
/public/home/jqfan/AIC/envs/aic128/bin/python - <<'PY'
from pathlib import Path
import torch
from aic.data.dataset import scan_samples
import aic.tools.train
root = Path("/public/home/jqfan/AIC/dataset")
train_root = next(
    p for p in root.iterdir()
    if p.is_dir() and (p / "labels").is_dir() and all((p / name).is_dir() for name in ("visible", "infrared", "depth"))
)
print("cuda_available", torch.cuda.is_available())
print("cuda_count", torch.cuda.device_count())
print("device_names", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("train_root", train_root)
print("train_samples", len(scan_samples(str(train_root), require_labels=True)))
PY

#!/bin/bash
#SBATCH -J aic_ir128
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=72:00:00
#SBATCH -o /public/home/jqfan/AIC/automation/slurm_%x_%j.out
#SBATCH -e /public/home/jqfan/AIC/automation/slurm_%x_%j.err

set -euo pipefail
cd /public/home/jqfan/AIC
export CONDA_PKGS_DIRS=/public/home/jqfan/AIC/cache/conda_pkgs
export PIP_CACHE_DIR=/public/home/jqfan/AIC/cache/pip
export TORCH_HOME=/public/home/jqfan/AIC/cache/torch
export XDG_CACHE_HOME=/public/home/jqfan/AIC/cache/xdg
export TMPDIR=/public/home/jqfan/AIC/tmp
export PYTHONPATH=/public/home/jqfan/AIC
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TMPDIR"
/public/home/jqfan/AIC/envs/aic128/bin/python automation/run_deiminit_teachers_student_fold0_sugon_128.py --stage ir

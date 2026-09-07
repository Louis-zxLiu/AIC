#!/usr/bin/env python3
"""703: rebuild full-data teachers, then run full-data DEIM-MAL.

The teacher durations are tied to the converged epochs measured on the fixed
seed-42 1600/400 split: IR epoch 57 and Depth epoch 63.  Full-data runs use
58 and 64 epochs respectively (zero-based checkpoint names), then the student
is trained for 64 epochs.  All subprocesses use GPUs 2,3 and seed 42.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import yaml

ROOT = Path("/home/u703/AIC_Project_sy/AIC")
OUT = ROOT / "outputs"
AUTO = ROOT / "automation"
PY = "/home/u703/miniconda3/envs/AIC/bin/python"
TORCHRUN = "/home/u703/miniconda3/envs/AIC/bin/torchrun"
TRAIN_ROOT = "/home/u703/AIC_Project_sy/dataset/训练集"
TEST_ROOT = "/home/u703/AIC_Project_sy/dataset/测试集"
ENV = os.environ.copy()
ENV.update({
    "CUDA_VISIBLE_DEVICES": "2,3",
    "PYTHONPATH": str(ROOT),
    "NCCL_P2P_DISABLE": "1",
    "NCCL_IB_DISABLE": "1",
    "TORCH_HOME": str(ROOT / "cache/torch"),
    "XDG_CACHE_HOME": str(ROOT / "cache/xdg"),
    "TMPDIR": str(ROOT / "tmp"),
    "PYTHONUNBUFFERED": "1",
})


def run(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n===== START {time.strftime('%F %T')} =====\n")
        f.flush()
        p = subprocess.run(cmd, cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT)
        f.write(f"\n===== EXIT {p.returncode} {time.strftime('%F %T')} =====\n")
    if p.returncode:
        raise RuntimeError(f"failed ({p.returncode}): {' '.join(cmd)}; see {log_path}")


def write_cfg(src: Path, dst: Path, *, save_dir: str, epochs: int,
              full_data: bool, batch_size: int, grad_accum_steps: int,
              lr: float, ir_ckpt: str | None = None,
              depth_ckpt: str | None = None) -> None:
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    cfg["train"]["seed"] = 42
    cfg["train"]["data_root"] = TRAIN_ROOT
    cfg["train"]["full_data"] = full_data
    cfg["train"]["epochs"] = epochs
    cfg["train"]["save_dir"] = save_dir
    cfg["train"]["batch_size"] = batch_size
    cfg["train"]["grad_accum_steps"] = grad_accum_steps
    cfg["train"]["lr"] = lr
    if ir_ckpt is not None:
        cfg["distill"]["ir_teacher_ckpt"] = ir_ckpt
    if depth_ckpt is not None:
        cfg["distill"]["depth_teacher_ckpt"] = depth_ckpt
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


def ensure_epoch(path: Path, epoch: int) -> Path:
    ckpt = path / f"epoch{epoch}.pt"
    if not ckpt.is_file():
        raise RuntimeError(f"missing checkpoint: {ckpt}")
    return ckpt


def infer(cfg: Path, ckpt: Path, tag: str, epoch: int) -> str:
    pred = OUT / f"{tag}_pred_epoch{epoch}"
    zip_path = ROOT / "submissions" / f"{tag}_epoch{epoch}.zip"
    pred.mkdir(parents=True, exist_ok=True)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "aic/tools/infer.py", "--config", str(cfg), "--checkpoint", str(ckpt),
           "--test_root", TEST_ROOT, "--out_dir", str(pred), "--zip_path", str(zip_path)]
    run(cmd, AUTO / f"{tag}_epoch{epoch}_infer.log")
    txts = list(pred.glob("*.txt"))
    if len(txts) != 1000:
        raise RuntimeError(f"{tag} epoch{epoch}: expected 1000 txt, got {len(txts)}")
    return str(zip_path)


def main() -> int:
    for p in (ROOT / "cache/torch", ROOT / "cache/xdg", ROOT / "tmp"):
        p.mkdir(parents=True, exist_ok=True)
    # Full-data teachers, with durations fixed by fold-0 convergence.
    ir_cfg = ROOT / "configs/_auto_teacher_ir_x_full_seed42_ep58.yml"
    depth_cfg = ROOT / "configs/_auto_teacher_depth_x_full_seed42_ep64.yml"
    write_cfg(ROOT / "configs/dfine_teacher_ir_x_full.yml", ir_cfg,
              save_dir=str(OUT / "auto_teacher_ir_x_full_seed42_bs6"), epochs=58,
              full_data=True, batch_size=6, grad_accum_steps=2, lr=1.8666667e-4)
    write_cfg(ROOT / "configs/dfine_teacher_depth_x_full.yml", depth_cfg,
              save_dir=str(OUT / "auto_teacher_depth_x_full_seed42_bs6"), epochs=64,
              full_data=True, batch_size=6, grad_accum_steps=2, lr=1.8666667e-4)
    run([TORCHRUN, "--standalone", "--nproc_per_node=2", "--master_port=29630",
         "aic/tools/train.py", "--config", str(ir_cfg), "--mono_modality", "ir"],
        AUTO / "auto_teacher_ir_x_full_seed42.log")
    run([TORCHRUN, "--standalone", "--nproc_per_node=2", "--master_port=29631",
         "aic/tools/train.py", "--config", str(depth_cfg), "--mono_modality", "depth"],
        AUTO / "auto_teacher_depth_x_full_seed42.log")
    ir_ckpt = ensure_epoch(OUT / "auto_teacher_ir_x_full_seed42_bs6", 57)
    depth_ckpt = ensure_epoch(OUT / "auto_teacher_depth_x_full_seed42_bs6", 63)

    student_cfg = ROOT / "configs/_auto_deim_mal_full2000_seed42_64.yml"
    write_cfg(ROOT / "configs/dfine_tristream_x_m2d_deim_mal_fold0_64.yml", student_cfg,
              save_dir=str(OUT / "run_x_m2d_deim_mal_full2000_seed42_bs2_4gpu_64"), epochs=64,
              # Four ranks keep the same effective global batch (16) as the
              # previous two-rank bs4 configuration, while halving steps.
              full_data=True, batch_size=2, grad_accum_steps=2, lr=1.8666667e-4,
              ir_ckpt=str(ir_ckpt), depth_ckpt=str(depth_ckpt))
    student_dir = OUT / "run_x_m2d_deim_mal_full2000_seed42_bs2_4gpu_64"
    # GPU 0/1 are acquired only after the two-card teacher stage has exited;
    # train the expensive tri-modal student on four cards.
    ENV["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    run([TORCHRUN, "--standalone", "--nproc_per_node=4", "--master_port=29632",
         "aic/tools/train.py", "--config", str(student_cfg)],
        AUTO / "run_x_m2d_deim_mal_full2000_seed42_bs2_4gpu_64.log")
    results = []
    for epoch in (47, 63):
        results.append({"epoch": epoch, "zip": infer(student_cfg,
                       ensure_epoch(student_dir, epoch),
                       "run_x_m2d_deim_mal_full2000_seed42_bs2_4gpu_64", epoch)})
    (AUTO / "run_x_m2d_deim_mal_full2000_seed42_results.json").write_text(
        json.dumps({"seed": 42, "ir_teacher_epoch": 57, "depth_teacher_epoch": 63,
                    "results": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (AUTO / "run_x_m2d_deim_mal_full2000_seed42.done").write_text(time.strftime("%F %T"), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

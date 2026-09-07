#!/usr/bin/env python3
"""Retrain the full-data 5-channel Depth teacher, then DEIM-MAL student."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import torch
import yaml

ROOT = Path("/home/u703/AIC_Project_sy/AIC")
OUT = ROOT / "outputs"
AUTO = ROOT / "automation"
PY = "/home/u703/miniconda3/envs/AIC/bin/python"
TORCHRUN = "/home/u703/miniconda3/envs/AIC/bin/torchrun"
TRAIN_ROOT = "/home/u703/AIC_Project_sy/dataset/训练集"
TEST_ROOT = "/home/u703/AIC_Project_sy/dataset/测试集"
IR_CKPT = OUT / "auto_teacher_ir_x_full_seed42_bs6/epoch57.pt"
DEPTH_DIR = OUT / "auto_teacher_depth_x_full_seed42_d5_bs6"
STUDENT_DIR = OUT / "run_x_m2d_deim_mal_full2000_seed42_d5teachers_4gpu_64"


def env(gpus: str) -> dict[str, str]:
    value = os.environ.copy()
    value.update({
        "CUDA_VISIBLE_DEVICES": gpus,
        "PYTHONPATH": str(ROOT),
        "NCCL_P2P_DISABLE": "1",
        "NCCL_IB_DISABLE": "1",
        "TORCH_HOME": str(ROOT / "cache/torch"),
        "XDG_CACHE_HOME": str(ROOT / "cache/xdg"),
        "TMPDIR": str(ROOT / "tmp"),
        "PYTHONUNBUFFERED": "1",
    })
    return value


def run(cmd: list[str], log_path: Path, gpus: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n===== START {time.strftime('%F %T')} GPUs={gpus} =====\n")
        log.flush()
        result = subprocess.run(cmd, cwd=ROOT, env=env(gpus), stdout=log, stderr=subprocess.STDOUT)
        log.write(f"\n===== EXIT {result.returncode} {time.strftime('%F %T')} =====\n")
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log_path}")


def write_depth_config() -> Path:
    # Use the known-good fold0 Depth config because it explicitly defines the
    # repaired five-channel input. Only data scope, batch and output change.
    source = ROOT / "configs/dfine_teacher_depth_x_fold0.yml"
    cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
    assert int(cfg["backbone"]["depth_in_chs"]) == 5
    cfg["train"].update({
        "data_root": TRAIN_ROOT,
        "seed": 42,
        "full_data": True,
        "batch_size": 6,
        "grad_accum_steps": 2,
        "epochs": 64,
        "lr": 1.8666667e-4,
        "save_dir": str(DEPTH_DIR),
        "keep_last_n": 3,
    })
    path = ROOT / "configs/_auto_teacher_depth_x_full_seed42_d5_bs6.yml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def write_student_config(depth_ckpt: Path) -> Path:
    source = ROOT / "configs/dfine_tristream_x_m2d_deim_mal_fold0_64.yml"
    cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
    assert int(cfg["backbone"]["depth_in_chs"]) == 5
    cfg["train"].update({
        "data_root": TRAIN_ROOT,
        "seed": 42,
        "full_data": True,
        # Match the successful four-GPU fold0 DEIM-MAL optimization recipe.
        "batch_size": 2,
        "grad_accum_steps": 3,
        "epochs": 64,
        "lr": 1.4e-4,
        "save_dir": str(STUDENT_DIR),
        "keep_last_n": 24,
    })
    cfg["distill"]["ir_teacher_ckpt"] = str(IR_CKPT)
    cfg["distill"]["depth_teacher_ckpt"] = str(depth_ckpt)
    path = ROOT / "configs/_auto_deim_mal_full2000_seed42_d5teachers_4gpu_64.yml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def assert_teacher_shapes(depth_ckpt: Path) -> None:
    if not IR_CKPT.is_file() or not depth_ckpt.is_file():
        raise RuntimeError("required teacher checkpoint is missing")
    depth = torch.load(depth_ckpt, map_location="cpu", weights_only=False)
    depth_state = depth["ema"] if depth.get("ema") is not None else depth["model"]
    key = "backbone.stem_depth.stem1.conv.weight"
    if tuple(depth_state[key].shape) != (32, 5, 3, 3):
        raise RuntimeError(f"Depth teacher is not five-channel: {depth_state[key].shape}")


def wait_for_main_gpus() -> None:
    while True:
        raw = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ], text=True)
        rows = {int(a): (int(b), int(c)) for a, b, c in
                ([x.strip() for x in line.split(",")] for line in raw.splitlines())}
        if all(rows[i][0] < 4000 and rows[i][1] <= 20 for i in (0, 1, 2, 3)):
            return
        print(f"[wait] GPUs 0-3 not all free: {rows}", flush=True)
        time.sleep(30)


def infer(cfg: Path, epoch: int) -> None:
    ckpt = STUDENT_DIR / f"epoch{epoch}.pt"
    if not ckpt.is_file():
        raise RuntimeError(f"missing student checkpoint {ckpt}")
    tag = f"run_x_m2d_deim_mal_full2000_seed42_d5teachers_4gpu_epoch{epoch}"
    pred = OUT / f"{tag}_pred"
    zip_path = ROOT / "submissions" / f"{tag}.zip"
    pred.mkdir(parents=True, exist_ok=True)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    run([PY, "aic/tools/infer.py", "--config", str(cfg), "--checkpoint", str(ckpt),
         "--test_root", TEST_ROOT, "--out_dir", str(pred), "--zip_path", str(zip_path)],
        AUTO / f"{tag}_infer.log", "0")
    if len(list(pred.glob("*.txt"))) != 1000:
        raise RuntimeError(f"invalid inference output in {pred}")


def main() -> None:
    for path in (ROOT / "cache/torch", ROOT / "cache/xdg", ROOT / "tmp"):
        path.mkdir(parents=True, exist_ok=True)
    depth_cfg = write_depth_config()
    run([TORCHRUN, "--standalone", "--nproc_per_node=2", "--master_port=29640",
         "aic/tools/train.py", "--config", str(depth_cfg), "--mono_modality", "depth"],
        AUTO / "teacher_depth_full_seed42_d5_bs6.log", "2,3")
    depth_ckpt = DEPTH_DIR / "epoch63.pt"
    assert_teacher_shapes(depth_ckpt)
    student_cfg = write_student_config(depth_ckpt)
    wait_for_main_gpus()
    run([TORCHRUN, "--standalone", "--nproc_per_node=4", "--master_port=29641",
         "aic/tools/train.py", "--config", str(student_cfg)],
        AUTO / "deim_mal_full_seed42_d5teachers_4gpu.log", "0,1,2,3")
    infer(student_cfg, 47)
    infer(student_cfg, 63)
    (AUTO / "depth5_then_deim_mal_main.done").write_text(time.strftime("%F %T"), encoding="utf-8")


if __name__ == "__main__":
    main()

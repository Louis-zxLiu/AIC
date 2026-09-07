#!/usr/bin/env python3
"""703 contrast run for full2000 DEIM-MAL with DDP repeat-factor sampling enabled."""
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
IR_CKPT = OUT / "auto_teacher_ir_x_full_seed42_bs6" / "epoch57.pt"
DEPTH_CKPT = OUT / "auto_teacher_depth_x_full_seed42_d5_bs6" / "epoch63.pt"
SAVE_DIR = OUT / "run_x_m2d_deim_mal_full2000_seed42_d5teachers_gpu23_bs2_64_ddprfs"

ENV = os.environ.copy()
ENV.update(
    {
        "CUDA_VISIBLE_DEVICES": "2,3",
        "PYTHONPATH": str(ROOT),
        "NCCL_P2P_DISABLE": "1",
        "NCCL_IB_DISABLE": "1",
        "TORCH_HOME": str(ROOT / "cache" / "torch"),
        "XDG_CACHE_HOME": str(ROOT / "cache" / "xdg"),
        "TMPDIR": str(ROOT / "tmp"),
        "PYTHONUNBUFFERED": "1",
        "PYTHONUTF8": "1",
    }
)


def run(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n===== START {time.strftime('%F %T')} =====\n")
        handle.flush()
        result = subprocess.run(cmd, cwd=ROOT, env=ENV, stdout=handle, stderr=subprocess.STDOUT)
        handle.write(f"\n===== EXIT {result.returncode} {time.strftime('%F %T')} =====\n")
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log_path}")


def write_cfg() -> Path:
    src = ROOT / "configs" / "dfine_tristream_x_m2d_deim_mal_fold0_64.yml"
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    cfg["train"].update(
        {
            "data_root": TRAIN_ROOT,
            "seed": 42,
            "full_data": True,
            "batch_size": 2,
            "grad_accum_steps": 3,
            "epochs": 64,
            "lr": 1.4e-4,
            "save_dir": str(SAVE_DIR),
            "keep_last_n": 1,
            "keep_best_n": 1,
        }
    )
    cfg["distill"]["ir_teacher_ckpt"] = str(IR_CKPT)
    cfg["distill"]["depth_teacher_ckpt"] = str(DEPTH_CKPT)
    path = ROOT / "configs" / "_auto_deim_mal_full2000_seed42_d5teachers_gpu23_bs2_64_ddprfs.yml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def best_epoch(out_dir: Path) -> tuple[int, float]:
    curves = json.loads((out_dir / "curves.json").read_text(encoding="utf-8"))
    epochs = curves.get("eval_epochs", [])
    values = curves.get("eval_metrics", {}).get("mAP@50-95", [])
    pairs = [(int(epoch), float(value)) for epoch, value in zip(epochs, values) if value is not None]
    if not pairs:
        raise RuntimeError(f"no validation results in {out_dir / 'curves.json'}")
    epoch, value = max(pairs, key=lambda item: item[1])
    ckpt = out_dir / f"epoch{epoch}.pt"
    if not ckpt.is_file():
        raise RuntimeError(f"missing best checkpoint {ckpt}")
    return epoch, value


def infer(cfg: Path, epoch: int) -> str:
    ckpt = SAVE_DIR / f"epoch{epoch}.pt"
    pred_dir = OUT / f"run_x_m2d_deim_mal_full2000_seed42_d5teachers_gpu23_bs2_64_ddprfs_pred_epoch{epoch}"
    zip_path = ROOT / "submissions" / f"run_x_m2d_deim_mal_full2000_seed42_d5teachers_gpu23_bs2_64_ddprfs_epoch{epoch}.zip"
    pred_dir.mkdir(parents=True, exist_ok=True)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            PY,
            "aic/tools/infer.py",
            "--config",
            str(cfg),
            "--checkpoint",
            str(ckpt),
            "--test_root",
            TEST_ROOT,
            "--out_dir",
            str(pred_dir),
            "--zip_path",
            str(zip_path),
        ],
        AUTO / f"run_x_m2d_deim_mal_full2000_seed42_d5teachers_gpu23_bs2_64_ddprfs_infer_epoch{epoch}.log",
    )
    txt_count = len(list(pred_dir.glob("*.txt")))
    if txt_count != 1000:
        raise RuntimeError(f"expected 1000 txt files, got {txt_count}")
    return str(zip_path)


def main() -> int:
    for path in (ROOT / "cache" / "torch", ROOT / "cache" / "xdg", ROOT / "tmp"):
        path.mkdir(parents=True, exist_ok=True)
    if not IR_CKPT.is_file():
        raise FileNotFoundError(IR_CKPT)
    if not DEPTH_CKPT.is_file():
        raise FileNotFoundError(DEPTH_CKPT)

    cfg = write_cfg()
    train_cmd = [TORCHRUN, "--standalone", "--nproc_per_node=2", "--master_port=29641", "aic/tools/train.py", "--config", str(cfg)]
    run(
        train_cmd,
        AUTO / "run_x_m2d_deim_mal_full2000_seed42_d5teachers_gpu23_bs2_64_ddprfs.log",
    )
    epoch, val_map = best_epoch(SAVE_DIR)
    zip_path = infer(cfg, epoch)
    zip_epoch63 = infer(cfg, 63)
    result = {
        "seed": 42,
        "ir_teacher": str(IR_CKPT),
        "depth_teacher": str(DEPTH_CKPT),
        "student_dir": str(SAVE_DIR),
        "best_epoch": epoch,
        "val_map_50_95": val_map,
        "zip": zip_path,
        "epoch63_zip": zip_epoch63,
    }
    (AUTO / "run_x_m2d_deim_mal_full2000_seed42_d5teachers_gpu23_bs2_64_ddprfs_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (AUTO / "run_x_m2d_deim_mal_full2000_seed42_d5teachers_gpu23_bs2_64_ddprfs.done").write_text(
        time.strftime("%F %T") + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

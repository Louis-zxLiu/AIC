#!/usr/bin/env python3
"""Train the fixed-fold GCL ablation, then infer its best validation epoch."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path("/home/u703/AIC_Project_sy/AIC")
PY = "/home/u703/miniconda3/envs/AIC/bin/python"
TORCHRUN = "/home/u703/miniconda3/envs/AIC/bin/torchrun"
CONFIG = ROOT / "configs/dfine_tristream_x_m2d_pkd_nocross_gcl_fold0_64.yml"
OUT = ROOT / "outputs/run_x_m2d_pkd_nocross_gcl_fold0_64"
AUTO = ROOT / "automation"


def env_for(devices: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": devices,
            "PYTHONPATH": str(ROOT),
            "TORCH_HOME": str(ROOT / "cache/torch"),
            "XDG_CACHE_HOME": str(ROOT / "cache/xdg"),
            "TMPDIR": str(ROOT / "tmp"),
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def best_epoch() -> tuple[int, float]:
    data = json.loads((OUT / "curves.json").read_text(encoding="utf-8"))
    epochs = data["eval_epochs"]
    scores = data["eval_metrics"]["mAP@50-95"]
    i = max(range(len(scores)), key=lambda j: scores[j])
    return int(epochs[i]), float(scores[i])


def test_root() -> Path:
    dataset = Path("/home/u703/AIC_Project_sy/dataset")
    candidates = [
        p
        for p in dataset.iterdir()
        if (p / "visible").is_dir() and not (p / "labels").is_dir()
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"could not identify unique test root: {candidates}")
    return candidates[0]


def main() -> None:
    AUTO.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    train_log = AUTO / "gcl_fold0_64_train.log"
    train_cmd = [
        TORCHRUN,
        "--nproc_per_node=2",
        "--master_port=29571",
        "aic/tools/train.py",
        "--config",
        str(CONFIG),
    ]
    with train_log.open("a", encoding="utf-8") as log:
        result = subprocess.run(train_cmd, cwd=ROOT, env=env_for("2,3"), stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"GCL training failed with code {result.returncode}; see {train_log}")

    epoch, score = best_epoch()
    ckpt = OUT / f"epoch{epoch}.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)

    root = test_root()
    pred = ROOT / f"outputs/run_x_m2d_pkd_nocross_gcl_fold0_64_pred_epoch{epoch}"
    zip_path = ROOT / f"submissions/run_x_m2d_pkd_nocross_gcl_fold0_64_epoch{epoch}.zip"
    infer_log = AUTO / f"gcl_fold0_64_infer_epoch{epoch}.log"
    pred.mkdir(parents=True, exist_ok=True)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    infer_cmd = [
        PY,
        "aic/tools/infer.py",
        "--config",
        str(CONFIG),
        "--checkpoint",
        str(ckpt),
        "--test_root",
        str(root),
        "--out_dir",
        str(pred),
        "--zip_path",
        str(zip_path),
    ]
    with infer_log.open("a", encoding="utf-8") as log:
        result = subprocess.run(infer_cmd, cwd=ROOT, env=env_for("2"), stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"GCL inference failed with code {result.returncode}; see {infer_log}")
    count = len(list(pred.glob("*.txt")))
    if count != 1000:
        raise RuntimeError(f"expected 1000 prediction files, got {count}")

    result_data = {
        "best_epoch": epoch,
        "val_map_50_95": score,
        "checkpoint": str(ckpt),
        "prediction_dir": str(pred),
        "zip": str(zip_path),
        "completed_at": time.strftime("%F %T"),
    }
    (AUTO / "gcl_fold0_64_result.json").write_text(
        json.dumps(result_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (AUTO / "gcl_fold0_64.done").write_text(time.strftime("%F %T"), encoding="utf-8")
    print(json.dumps(result_data, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

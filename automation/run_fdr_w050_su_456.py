#!/usr/bin/env python3
"""Run the pending FDR=0.5 fold-0 ablation on su GPUs 4,5,6."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


PROJECT = Path("/sdd1/AIC/project")
OUT = Path("/sdd/lzx")
PY = "/sdd1/AIC/miniconda3/envs/AIC/bin/python"
TORCHRUN = "/sdd1/AIC/miniconda3/envs/AIC/bin/torchrun"
CONFIG = PROJECT / "configs/dfine_tristream_x_m2d_fdr_fold0_64_w050.yml"
SAVE_DIR = OUT / "run_x_m2d_fdr_fold0_64_w050"
AUTO = OUT / "cache"


def env_for(devices: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": devices,
            "PYTHONPATH": str(PROJECT),
            "TORCH_HOME": str(OUT / "cache" / "torch"),
            "XDG_CACHE_HOME": str(OUT / "cache"),
            "TMPDIR": str(OUT / "cache" / "tmp"),
            "PYTHONUNBUFFERED": "1",
            "NCCL_P2P_DISABLE": "1",
            "NCCL_IB_DISABLE": "1",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
        }
    )
    return env


def best_epoch() -> tuple[int, float]:
    data = json.loads((SAVE_DIR / "curves.json").read_text(encoding="utf-8"))
    epochs = data["eval_epochs"]
    scores = data["eval_metrics"]["mAP@50-95"]
    i = max(range(len(scores)), key=lambda j: scores[j])
    return int(epochs[i]), float(scores[i])


def find_test_root() -> Path:
    dataset = Path("/sdd1/AIC/dataset")
    candidates = [
        p for p in dataset.iterdir() if (p / "visible").is_dir() and not (p / "labels").is_dir()
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"ambiguous test root: {candidates}")
    return candidates[0]


def main() -> None:
    AUTO.mkdir(parents=True, exist_ok=True)
    (OUT / "submissions").mkdir(parents=True, exist_ok=True)
    (OUT / "cache" / "tmp").mkdir(parents=True, exist_ok=True)
    lock = AUTO / "fdr_w050_su_456.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()} {time.strftime('%F %T')}\n".encode())
        os.close(fd)
    except FileExistsError:
        raise RuntimeError(f"already running or locked: {lock}")

    train_log = AUTO / "fdr_w050_su_456_train.log"
    train_cmd = [
        TORCHRUN,
        "--standalone",
        "--nproc_per_node=3",
        "--master_port=29622",
        "aic/tools/train.py",
        "--config",
        str(CONFIG),
    ]
    with train_log.open("a", encoding="utf-8") as log:
        result = subprocess.run(train_cmd, cwd=PROJECT, env=env_for("4,5,6"), stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"w050 training failed ({result.returncode}); see {train_log}")

    epoch, val_map = best_epoch()
    ckpt = SAVE_DIR / f"epoch{epoch}.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    pred = OUT / "fdr_inference" / f"w050_epoch{epoch}"
    zip_path = OUT / "submissions" / f"submit_x_m2d_fdr_w050_epoch{epoch}.zip"
    pred.mkdir(parents=True, exist_ok=True)
    infer_log = AUTO / f"fdr_w050_su_456_infer_epoch{epoch}.log"
    infer_cmd = [
        PY,
        "aic/tools/infer.py",
        "--config",
        str(CONFIG),
        "--checkpoint",
        str(ckpt),
        "--test_root",
        str(find_test_root()),
        "--out_dir",
        str(pred),
        "--zip_path",
        str(zip_path),
    ]
    with infer_log.open("a", encoding="utf-8") as log:
        result = subprocess.run(infer_cmd, cwd=PROJECT, env=env_for("4"), stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"w050 inference failed ({result.returncode}); see {infer_log}")
    count = len(list(pred.glob("*.txt")))
    if count != 1000:
        raise RuntimeError(f"expected 1000 txt files, got {count}")
    result_data = {
        "best_epoch": epoch,
        "val_map_50_95": val_map,
        "checkpoint": str(ckpt),
        "prediction_dir": str(pred),
        "zip": str(zip_path),
        "completed_at": time.strftime("%F %T"),
    }
    (AUTO / "fdr_w050_su_456_result.json").write_text(
        json.dumps(result_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (AUTO / "fdr_w050_su_456.done").write_text(time.strftime("%F %T"), encoding="utf-8")
    print(json.dumps(result_data, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

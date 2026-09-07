#!/usr/bin/env python3
"""Train DEIM-D-FINE-X initialized MAL+M2D fold0, then infer its best EMA epoch."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


PROJECT = Path("/sdd1/AIC/project")
OUT = Path("/sdd/lzx")
PY = "/sdd1/AIC/miniconda3/envs/AIC/bin/python3.11"
TORCHRUN = "/sdd1/AIC/miniconda3/envs/AIC/bin/torchrun"
CONFIG = PROJECT / "configs/dfine_tristream_x_m2d_deim_mal_deiminit_fold0_64.yml"
SAVE_DIR = OUT / "run_x_m2d_deim_mal_deiminit_fold0_64"
AUTO = OUT / "cache"
DEVICES = "0,1,2,4,5,6"


def env_for(devices: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
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
    })
    return env


def best_epoch() -> tuple[int, float]:
    data = json.loads((SAVE_DIR / "curves.json").read_text(encoding="utf-8"))
    epochs = data["eval_epochs"]
    scores = data["eval_metrics"]["mAP@50-95"]
    i = max(range(len(scores)), key=lambda j: scores[j])
    return int(epochs[i]), float(scores[i])


def find_test_root() -> Path:
    dataset = Path("/sdd1/AIC/dataset")
    candidates = [p for p in dataset.iterdir() if (p / "visible").is_dir() and not (p / "labels").is_dir()]
    if len(candidates) != 1:
        raise RuntimeError(f"ambiguous test root: {candidates}")
    return candidates[0]


def main() -> None:
    for d in (AUTO / "tmp", AUTO / "torch", OUT / "submissions"):
        d.mkdir(parents=True, exist_ok=True)
    lock = AUTO / "deiminit_mal_fold0_6gpu_su.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()} {time.strftime('%F %T')}\n".encode())
        os.close(fd)
    except FileExistsError:
        raise RuntimeError(f"already running or locked: {lock}")

    train_log = AUTO / "deiminit_mal_fold0_6gpu_train.log"
    train_cmd = [TORCHRUN, "--standalone", "--nproc_per_node=6", "--master_port=29632", "aic/tools/train.py", "--config", str(CONFIG)]
    # Resume from the latest complete epoch after an external SIGHUP or node
    # interruption. The training code restores optimizer, scheduler, EMA and
    # history from this checkpoint.
    checkpoints = sorted(SAVE_DIR.glob("epoch*.pt"), key=lambda p: int(p.stem[5:]))
    if checkpoints:
        train_cmd += ["--resume", str(checkpoints[-1])]
    with train_log.open("a", encoding="utf-8") as log:
        result = subprocess.run(train_cmd, cwd=PROJECT, env=env_for(DEVICES), stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"training failed ({result.returncode}); see {train_log}")

    epoch, val_map = best_epoch()
    ckpt = SAVE_DIR / f"epoch{epoch}.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    pred = OUT / "deiminit_mal_inference" / f"epoch{epoch}"
    zip_path = OUT / "submissions" / f"submit_x_m2d_deim_mal_deiminit_fold0_epoch{epoch}.zip"
    pred.mkdir(parents=True, exist_ok=True)
    infer_log = AUTO / f"deiminit_mal_fold0_6gpu_infer_epoch{epoch}.log"
    infer_cmd = [PY, "aic/tools/infer.py", "--config", str(CONFIG), "--checkpoint", str(ckpt), "--test_root", str(find_test_root()), "--out_dir", str(pred), "--zip_path", str(zip_path)]
    with infer_log.open("a", encoding="utf-8") as log:
        result = subprocess.run(infer_cmd, cwd=PROJECT, env=env_for("0"), stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"inference failed ({result.returncode}); see {infer_log}")
    count = len(list(pred.glob("*.txt")))
    if count != 1000:
        raise RuntimeError(f"expected 1000 txt files, got {count}")
    result_data = {"best_epoch": epoch, "val_map_50_95": val_map, "checkpoint": str(ckpt), "prediction_dir": str(pred), "zip": str(zip_path), "devices": DEVICES, "completed_at": time.strftime('%F %T')}
    (AUTO / "deiminit_mal_fold0_6gpu_result.json").write_text(json.dumps(result_data, ensure_ascii=False, indent=2), encoding="utf-8")
    (AUTO / "deiminit_mal_fold0_6gpu.done").write_text(time.strftime('%F %T'), encoding="utf-8")
    print(json.dumps(result_data, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

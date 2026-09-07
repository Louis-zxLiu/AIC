#!/usr/bin/env python3
"""Wait for the ddprfs full2000 run to finish, then infer epoch63."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/u703/AIC_Project_sy/AIC")
OUT = ROOT / "outputs"
AUTO = ROOT / "automation"
PY = "/home/u703/miniconda3/envs/AIC/bin/python"
DONE = AUTO / "run_x_m2d_deim_mal_full2000_seed42_d5teachers_gpu23_bs2_64_ddprfs.done"
CFG = ROOT / "configs/_auto_deim_mal_full2000_seed42_d5teachers_gpu23_bs2_64_ddprfs.yml"
CKPT = OUT / "run_x_m2d_deim_mal_full2000_seed42_d5teachers_gpu23_bs2_64_ddprfs" / "epoch63.pt"
PRED = OUT / "run_x_m2d_deim_mal_full2000_seed42_d5teachers_gpu23_bs2_64_ddprfs_pred_epoch63"
ZIP = ROOT / "submissions" / "run_x_m2d_deim_mal_full2000_seed42_d5teachers_gpu23_bs2_64_ddprfs_epoch63.zip"

ENV = os.environ.copy()
ENV.update(
    {
        "PYTHONPATH": str(ROOT),
        "PYTHONUNBUFFERED": "1",
        "PYTHONUTF8": "1",
    }
)


def wait_for_done() -> None:
    while not DONE.is_file():
        time.sleep(60)
    time.sleep(15)


def main() -> int:
    wait_for_done()
    if not CFG.is_file():
        raise FileNotFoundError(CFG)
    if not CKPT.is_file():
        raise FileNotFoundError(CKPT)
    PRED.mkdir(parents=True, exist_ok=True)
    ZIP.parent.mkdir(parents=True, exist_ok=True)
    log = AUTO / "run_x_m2d_deim_mal_full2000_seed42_d5teachers_gpu23_bs2_64_ddprfs_epoch63.log"
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n===== START {time.strftime('%F %T')} =====\n")
        handle.flush()
        result = subprocess.run(
            [
                PY,
                "aic/tools/infer.py",
                "--config",
                str(CFG),
                "--checkpoint",
                str(CKPT),
                "--test_root",
                "/home/u703/AIC_Project_sy/dataset/测试集",
                "--out_dir",
                str(PRED),
                "--zip_path",
                str(ZIP),
            ],
            cwd=ROOT,
            env=ENV,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        handle.write(f"\n===== EXIT {result.returncode} {time.strftime('%F %T')} =====\n")
    if result.returncode:
        raise RuntimeError(f"inference failed; see {log}")
    if len(list(PRED.glob("*.txt"))) != 1000:
        raise RuntimeError(f"expected 1000 txt files under {PRED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

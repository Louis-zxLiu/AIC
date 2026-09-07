#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path("/home/u703/AIC_Project_sy/AIC")
OUT = ROOT / "outputs"
AUTO = ROOT / "automation"
PY = "/home/u703/miniconda3/envs/AIC/bin/python"
CONFIG = ROOT / "configs/_auto_deim_mal_full2000_seed42_d5teachers_gpu23_bs2_64.yml"
MODEL = OUT / "run_x_m2d_deim_mal_full2000_seed42_d5teachers_gpu23_bs2_64"
TEST_ROOT = "/home/u703/AIC_Project_sy/dataset/测试集"


def main() -> None:
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONPATH": str(ROOT),
        "TORCH_HOME": str(ROOT / "cache/torch"),
        "XDG_CACHE_HOME": str(ROOT / "cache/xdg"),
        "TMPDIR": str(ROOT / "tmp"),
        "PYTHONUNBUFFERED": "1",
    })
    AUTO.mkdir(parents=True, exist_ok=True)
    for epoch in (39, 41):
        ckpt = MODEL / f"epoch{epoch}.pt"
        pred = OUT / f"run_x_m2d_deim_mal_full2000_seed42_d5teachers_gpu23_bs2_pred_epoch{epoch}"
        zip_path = ROOT / "submissions" / f"run_x_m2d_deim_mal_full2000_seed42_d5teachers_gpu23_bs2_epoch{epoch}.zip"
        pred.mkdir(parents=True, exist_ok=True)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [PY, "aic/tools/infer.py", "--config", str(CONFIG),
               "--checkpoint", str(ckpt), "--test_root", TEST_ROOT,
               "--out_dir", str(pred), "--zip_path", str(zip_path)]
        log = AUTO / f"infer_deim_mal_full_gpu0_epoch{epoch}.log"
        with log.open("a", encoding="utf-8") as handle:
            result = subprocess.run(cmd, cwd=ROOT, env=env,
                                    stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"inference epoch{epoch} failed; see {log}")
        count = len(list(pred.glob("*.txt")))
        if count != 1000:
            raise RuntimeError(f"epoch{epoch}: expected 1000 txt, got {count}")
        print(f"epoch{epoch}: {zip_path} ({count} txt)", flush=True)


if __name__ == "__main__":
    main()

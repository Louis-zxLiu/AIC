#!/usr/bin/env python3
"""Run MAL+FDR weights 0.25 and 0.50 serially on GPUs 2/3."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path("/home/u703/AIC_Project_sy/AIC")
PY = "/home/u703/miniconda3/envs/AIC/bin/python"
TORCHRUN = "/home/u703/miniconda3/envs/AIC/bin/torchrun"
AUTO = ROOT / "automation"
LOCK = AUTO / "mal_fdr_fold0_sequence.lock"
DONE = AUTO / "mal_fdr_fold0_sequence.done"
JOBS = (
    (
        "w025",
        ROOT / "configs/dfine_tristream_x_m2d_deim_mal_fdr_fold0_64_w025.yml",
        ROOT / "outputs/run_x_m2d_deim_mal_fdr_fold0_64_w025",
        29573,
    ),
    (
        "w050",
        ROOT / "configs/dfine_tristream_x_m2d_deim_mal_fdr_fold0_64_w050.yml",
        ROOT / "outputs/run_x_m2d_deim_mal_fdr_fold0_64_w050",
        29574,
    ),
)


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
            "NCCL_P2P_DISABLE": "1",
            "NCCL_IB_DISABLE": "1",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
        }
    )
    return env


def run(command: list[str], log_path: Path, devices: str) -> None:
    with log_path.open("a", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env_for(devices),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log_path}")


def best_epoch(save_dir: Path) -> tuple[int, float]:
    data = json.loads((save_dir / "curves.json").read_text(encoding="utf-8"))
    epochs = data.get("eval_epochs", [])
    scores = data.get("eval_metrics", {}).get("mAP@50-95", [])
    pairs = [(int(epoch), float(score)) for epoch, score in zip(epochs, scores) if score is not None]
    if not pairs:
        raise RuntimeError(f"no validation metrics in {save_dir / 'curves.json'}")
    return max(pairs, key=lambda pair: pair[1])


def test_root() -> Path:
    dataset = Path("/home/u703/AIC_Project_sy/dataset")
    candidates = [
        path
        for path in dataset.iterdir()
        if (path / "visible").is_dir() and not (path / "labels").is_dir()
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"could not identify unique test root: {candidates}")
    return candidates[0]


def main() -> None:
    AUTO.mkdir(parents=True, exist_ok=True)
    if DONE.exists():
        print(DONE.read_text(encoding="utf-8"), flush=True)
        return
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()} {time.strftime('%F %T')}\n".encode())
        os.close(fd)
    except FileExistsError as exc:
        raise RuntimeError(f"MAL+FDR sequence is already locked: {LOCK}") from exc

    results = []
    for tag, config, save_dir, port in JOBS:
        train_log = AUTO / f"mal_fdr_fold0_{tag}_train.log"
        run(
            [
                TORCHRUN,
                "--nproc_per_node=2",
                f"--master_port={port}",
                "aic/tools/train.py",
                "--config",
                str(config),
            ],
            train_log,
            "2,3",
        )
        epoch, score = best_epoch(save_dir)
        checkpoint = save_dir / f"epoch{epoch}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

        prediction_dir = ROOT / f"outputs/run_x_m2d_deim_mal_fdr_fold0_64_{tag}_pred_epoch{epoch}"
        zip_path = ROOT / f"submissions/run_x_m2d_deim_mal_fdr_fold0_64_{tag}_epoch{epoch}.zip"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                PY,
                "aic/tools/infer.py",
                "--config",
                str(config),
                "--checkpoint",
                str(checkpoint),
                "--test_root",
                str(test_root()),
                "--out_dir",
                str(prediction_dir),
                "--zip_path",
                str(zip_path),
            ],
            AUTO / f"mal_fdr_fold0_{tag}_infer_epoch{epoch}.log",
            "2",
        )
        count = len(list(prediction_dir.glob("*.txt")))
        if count != 1000:
            raise RuntimeError(f"{tag}: expected 1000 prediction txt files, got {count}")
        record = {
            "tag": tag,
            "best_epoch": epoch,
            "val_map_50_95": score,
            "checkpoint": str(checkpoint),
            "zip": str(zip_path),
        }
        results.append(record)
        (AUTO / f"mal_fdr_fold0_{tag}_result.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    summary = {"results": results, "completed_at": time.strftime("%F %T")}
    (AUTO / "mal_fdr_fold0_sequence_result.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    DONE.write_text(time.strftime("%F %T"), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

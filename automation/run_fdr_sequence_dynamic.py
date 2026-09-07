#!/usr/bin/env python3
"""Run the two FDR ablations serially on the first two sufficiently free GPUs.

The script is intended to run on ``su``.  It does not kill or reconfigure other
users' jobs: it waits for two GPUs with at least ``MIN_FREE_MIB`` free memory,
then keeps those GPUs occupied through the two torchrun jobs and their inference.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path("/sdd1/AIC/project")
OUT = Path("/sdd/lzx")
LOG = OUT / "fdr_sequence.log"
MIN_FREE_MIB = int(os.environ.get("FDR_MIN_FREE_MIB", "24000"))
POLL_SEC = int(os.environ.get("FDR_POLL_SEC", "30"))


def log(msg: str) -> None:
    line = time.strftime("[%Y-%m-%d %H:%M:%S] ") + msg
    print(line, flush=True)


def gpu_snapshot() -> list[tuple[int, int, int, int]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    rows = []
    for raw in out.splitlines():
        if not raw.strip():
            continue
        idx, used, total, util = [int(x.strip()) for x in raw.split(",")]
        rows.append((idx, used, total, util))
    return rows


def choose_gpus() -> tuple[int, int]:
    while True:
        rows = gpu_snapshot()
        candidates = [r for r in rows if r[2] - r[1] >= MIN_FREE_MIB and r[3] <= 20]
        candidates.sort(key=lambda r: (r[2] - r[1], -r[1]), reverse=True)
        if len(candidates) >= 2:
            chosen = (candidates[0][0], candidates[1][0])
            log(f"selected GPUs {chosen}; snapshot={rows}")
            return chosen
        log(f"waiting for 2 free GPUs (need {MIN_FREE_MIB} MiB each); snapshot={rows}")
        time.sleep(POLL_SEC)


def run(cmd: list[str], log_path: Path, env: dict[str, str]) -> None:
    log(f"exec: {' '.join(cmd)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fp:
        proc = subprocess.Popen(cmd, cwd=PROJECT, env=env, stdout=fp, stderr=subprocess.STDOUT)
        rc = proc.wait()
    if rc:
        raise RuntimeError(f"command failed rc={rc}: {' '.join(cmd)}; see {log_path}")


def best_epoch(save_dir: Path) -> tuple[int, float]:
    data = json.loads((save_dir / "curves.json").read_text(encoding="utf-8"))
    epochs = data.get("eval_epochs", [])
    maps = data.get("eval_metrics", {}).get("mAP@50-95", [])
    pairs = [(int(e), float(m)) for e, m in zip(epochs, maps) if m is not None]
    if not pairs:
        raise RuntimeError(f"no validation mAP in {save_dir / 'curves.json'}")
    return max(pairs, key=lambda x: x[1])


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cache").mkdir(exist_ok=True)
    os.environ.update(
        {
            "TORCH_HOME": str(OUT / "cache" / "torch"),
            "XDG_CACHE_HOME": str(OUT / "cache"),
            "TMPDIR": str(OUT / "cache" / "tmp"),
            "PYTHONUNBUFFERED": "1",
        }
    )
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    g0, g1 = choose_gpus()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = f"{g0},{g1}"
    # train.py imports the package from the project root; when launched by
    # torchrun Python otherwise only adds ``aic/tools`` to sys.path.
    env["PYTHONPATH"] = str(PROJECT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env.setdefault("NCCL_IB_DISABLE", "1")
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")

    # First retrain the IR teacher with the repaired five-channel structure.
    teacher_cfg = PROJECT / "configs/dfine_teacher_ir_x_fold0_d5.yml"
    teacher_dir = OUT / "teacher_ir_x_fold0_d5"
    teacher_cmd = [
        "torchrun", "--standalone", "--nproc_per_node=2", "--master_port=29619",
        "aic/tools/train.py", "--config", str(teacher_cfg), "--mono_modality", "ir",
    ]
    run(teacher_cmd, OUT / "teacher_ir_x_fold0_d5_train.log", env)
    teacher_ep, teacher_score = best_epoch(teacher_dir)
    teacher_ckpt = teacher_dir / f"epoch{teacher_ep}.pt"
    if not teacher_ckpt.exists():
        raise RuntimeError(f"retrained IR teacher checkpoint missing: {teacher_ckpt}")
    log(f"IR teacher d5: best epoch={teacher_ep}, mAP@50-95={teacher_score:.6f}")

    jobs = [
        ("w025", PROJECT / "configs/dfine_tristream_x_m2d_fdr_fold0_64_w025.yml"),
        ("w050", PROJECT / "configs/dfine_tristream_x_m2d_fdr_fold0_64_w050.yml"),
    ]
    for tag, cfg in jobs:
        # Make a run-local config so the selected teacher epoch is explicit,
        # without modifying the checked-in project configuration.
        run_cfg = OUT / "cache" / f"dfine_fdr_{tag}_ir{teacher_ep}.yml"
        text = cfg.read_text(encoding="utf-8")
        text = re.sub(r"ir_teacher_ckpt: .*", f"ir_teacher_ckpt: {teacher_ckpt}", text)
        run_cfg.write_text(text, encoding="utf-8")
        save_dir = OUT / f"run_x_m2d_fdr_fold0_64_{tag}"
        train_log = OUT / f"fdr_{tag}_train.log"
        cmd = [
            "torchrun",
            "--standalone",
            "--nproc_per_node=2",
            "--master_port=29620" if tag == "w025" else "--master_port=29621",
            "aic/tools/train.py",
            "--config",
            str(run_cfg),
        ]
        run(cmd, train_log, env)
        ep, score = best_epoch(save_dir)
        ckpt = save_dir / f"epoch{ep}.pt"
        if not ckpt.exists():
            raise RuntimeError(f"best checkpoint missing: {ckpt}")
        log(f"{tag}: best epoch={ep}, mAP@50-95={score:.6f}")
        infer_dir = OUT / "fdr_inference" / f"{tag}_epoch{ep}"
        zip_path = OUT / "submissions" / f"submit_x_m2d_fdr_{tag}_epoch{ep}.zip"
        infer_dir.mkdir(parents=True, exist_ok=True)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        infer_cmd = [
            "python",
            "aic/tools/infer.py",
            "--config",
            str(run_cfg),
            "--checkpoint",
            str(ckpt),
            "--test_root",
            "/sdd1/AIC/dataset/测试集",
            "--out_dir",
            str(infer_dir),
            "--zip_path",
            str(zip_path),
        ]
        run(infer_cmd, OUT / f"fdr_{tag}_infer.log", env)
        txts = list(infer_dir.glob("*.txt"))
        log(f"{tag}: inference complete, txt={len(txts)}, zip={zip_path}")
    log("FDR sequence complete")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"FATAL: {exc!r}")
        raise

#!/usr/bin/env python3
"""Queue the next GPU-2/3 sequence after the current ddprfs run finishes.

Sequence:
1. Wait for the current fold0 64-epoch ddprfs chain to complete.
2. Train IRFS teachers + student on GPUs 2/3.
3. Infer the student's best checkpoint with EMA + hflip TTA.
4. Build a SWA checkpoint from the tail checkpoints and infer it.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import zipfile
from pathlib import Path

import yaml


ROOT = Path("/home/u703/AIC_Project_sy/AIC")
OUT = ROOT / "outputs"
AUTO = ROOT / "automation"
SUBMISSIONS = ROOT / "submissions"
PY = Path("/home/u703/miniconda3/envs/AIC/bin/python")
TORCHRUN = Path("/home/u703/miniconda3/envs/AIC/bin/torchrun")
TRAIN_ROOT = Path("/home/u703/AIC_Project_sy/dataset/训练集")
TEST_ROOT = Path("/home/u703/AIC_Project_sy/dataset/测试集")
DEIM_INIT = ROOT / "weights/deim_dfine_hgnetv2_x_coco_50e.pth"

WAIT_DONE = AUTO / "run_x_m2d_deim_mal_deiminit_fold0_64_ddprfs.done"
WAIT_LOG = AUTO / "run_deiminit_fold0_64_ddprfs_then_irfs_tta_swa_gpu23.log"
LOCK = AUTO / "run_deiminit_fold0_64_ddprfs_then_irfs_tta_swa_gpu23.lock"
DONE = AUTO / "run_deiminit_fold0_64_ddprfs_then_irfs_tta_swa_gpu23.done"

EPOCHS = 64
NPROC = 2
SWA_LAST_K = 6

IR_TAG = "teacher_ir_x_deiminit_fold0_64_irfs"
DEPTH_TAG = "teacher_depth_x_deiminit_fold0_64_irfs"
STUDENT_TAG = "run_x_m2d_deim_mal_irfs_deiminit_fold0_64"

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


def env_for() -> dict[str, str]:
    env = ENV.copy()
    env.setdefault("OMP_NUM_THREADS", "6")
    env.setdefault("MKL_NUM_THREADS", "6")
    return env


def run(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n===== START {time.strftime('%F %T')} =====\n")
        handle.write(" ".join(cmd) + "\n")
        handle.flush()
        result = subprocess.run(cmd, cwd=ROOT, env=env_for(), stdout=handle, stderr=subprocess.STDOUT)
        handle.write(f"\n===== EXIT {result.returncode} {time.strftime('%F %T')} =====\n")
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log_path}")


def latest_checkpoint(out_dir: Path) -> Path | None:
    pairs: list[tuple[int, Path]] = []
    for path in out_dir.glob("epoch*.pt"):
        m = re.fullmatch(r"epoch(\d+)", path.stem)
        if m:
            pairs.append((int(m.group(1)), path))
    return max(pairs, default=(None, None))[1]


def completed(out_dir: Path) -> bool:
    curves = out_dir / "curves.json"
    if not curves.exists():
        return False
    rec = json.loads(curves.read_text(encoding="utf-8"))
    epochs = [int(e) for e in rec.get("eval_epochs", [])]
    return bool(epochs) and max(epochs) >= EPOCHS - 1


def best_epoch(out_dir: Path) -> tuple[int, float]:
    rec = json.loads((out_dir / "curves.json").read_text(encoding="utf-8"))
    pairs = [
        (int(epoch), float(value))
        for epoch, value in zip(
            rec.get("eval_epochs", []),
            rec.get("eval_metrics", {}).get("mAP@50-95", []),
        )
        if value is not None
    ]
    if not pairs:
        raise RuntimeError(f"no validation mAP in {out_dir / 'curves.json'}")
    epoch, value = max(pairs, key=lambda item: item[1])
    ckpt = out_dir / f"epoch{epoch}.pt"
    if not ckpt.is_file():
        raise RuntimeError(f"best checkpoint was pruned or missing: {ckpt}")
    return epoch, value


def write_cfg(
    src: Path,
    dst: Path,
    *,
    save_dir: Path,
    train_root: Path,
    detector: Path,
    mono_teacher: bool = False,
    ir_ckpt: Path | None = None,
    depth_ckpt: Path | None = None,
    class_balance_mode: str = "irfs",
    keep_last_n: int = 1,
    keep_best_n: int = 1,
) -> None:
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    cfg["pretrained_detector"] = str(detector)
    cfg["load_pretrained_detector"] = True
    train = cfg["train"]
    train["seed"] = 42
    train["data_root"] = str(train_root)
    train["full_data"] = False
    train["epochs"] = EPOCHS
    train["save_dir"] = str(save_dir)
    train["keep_last_n"] = keep_last_n
    train["keep_best_n"] = keep_best_n
    cb = train.setdefault("class_balance", {})
    cb["enabled"] = True
    cb["oversample_thr"] = float(cb.get("oversample_thr", 0.2))
    cb["mode"] = class_balance_mode
    if mono_teacher:
        train["batch_size"] = 6
        train["grad_accum_steps"] = 2
        train["lr"] = 1.8666667e-4
    else:
        train["batch_size"] = 2
        train["grad_accum_steps"] = 3
        train["lr"] = 1.4e-4
    if ir_ckpt is not None:
        cfg.setdefault("distill", {})["ir_teacher_ckpt"] = str(ir_ckpt)
    if depth_ckpt is not None:
        cfg.setdefault("distill", {})["depth_teacher_ckpt"] = str(depth_ckpt)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


def run_train(cfg: Path, out_dir: Path, log: Path, *, mono: str | None, port: int) -> None:
    if completed(out_dir):
        print(f"[skip] {out_dir.name} already reached epoch {EPOCHS - 1}", flush=True)
        return
    cmd = [
        str(TORCHRUN),
        "--standalone",
        f"--nproc_per_node={NPROC}",
        f"--master_port={port}",
        "aic/tools/train.py",
        "--config",
        str(cfg),
    ]
    latest = latest_checkpoint(out_dir)
    if latest is not None:
        cmd += ["--resume", str(latest)]
    if mono:
        cmd += ["--mono_modality", mono]
    run(cmd, log)


def validate_zip(zip_path: Path, pred_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
    txts = [name for name in names if name.endswith(".txt") and "/" not in name]
    if len(txts) != 1000 or len(names) != 1000:
        raise RuntimeError(f"zip root must contain exactly 1000 txt files: {zip_path}")
    for txt in pred_dir.glob("*.txt"):
        with txt.open("r", encoding="utf-8") as handle:
            if sum(1 for _ in handle) > 100:
                raise RuntimeError(f"{txt} has more than 100 lines")


def infer(cfg: Path, checkpoint: Path, out_dir: Path, zip_path: Path, *, tta_hflip: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        PY,
        "aic/tools/infer.py",
        "--config",
        str(cfg),
        "--checkpoint",
        str(checkpoint),
        "--test_root",
        str(TEST_ROOT),
        "--out_dir",
        str(out_dir),
        "--zip_path",
        str(zip_path),
    ]
    if tta_hflip:
        cmd.append("--tta_hflip")
    run(cmd, AUTO / f"{zip_path.stem}.log")
    txt_count = len(list(out_dir.glob("*.txt")))
    if txt_count != 1000:
        raise RuntimeError(f"expected 1000 txt files, got {txt_count}: {out_dir}")
    validate_zip(zip_path, out_dir)


def build_swa_checkpoint(save_dir: Path, output: Path) -> None:
    run(
        [
            PY,
            "aic/tools/build_swa_checkpoint.py",
            "--save_dir",
            str(save_dir),
            "--output",
            str(output),
            "--last_k",
            str(SWA_LAST_K),
        ],
        AUTO / f"{output.stem}.log",
    )


def wait_for_prerequisite() -> None:
    while not WAIT_DONE.is_file():
        print(f"[wait] current ddprfs run not finished yet: {WAIT_DONE.name}", flush=True)
        time.sleep(60)
    time.sleep(15)


def ensure_common() -> None:
    for path in (
        OUT,
        AUTO,
        SUBMISSIONS,
        ROOT / "cache/torch",
        ROOT / "cache/xdg",
        ROOT / "tmp",
    ):
        path.mkdir(parents=True, exist_ok=True)
    for path in (TRAIN_ROOT, TEST_ROOT):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if not PY.is_file():
        raise FileNotFoundError(PY)
    if not TORCHRUN.is_file():
        raise FileNotFoundError(TORCHRUN)
    if not DEIM_INIT.is_file():
        raise FileNotFoundError(DEIM_INIT)


def stage_irfs() -> tuple[Path, int, float]:
    cfg = ROOT / "configs/_auto_teacher_ir_x_deiminit_fold0_64_irfs.yml"
    out_dir = OUT / IR_TAG
    write_cfg(
        ROOT / "configs/dfine_teacher_ir_x_fold0.yml",
        cfg,
        save_dir=out_dir,
        train_root=TRAIN_ROOT,
        detector=DEIM_INIT,
        mono_teacher=True,
        class_balance_mode="irfs",
        keep_last_n=2,
        keep_best_n=2,
    )
    run_train(cfg, out_dir, AUTO / f"{IR_TAG}.log", mono="ir", port=29690)
    epoch, value = best_epoch(out_dir)
    return cfg, epoch, value


def stage_depth_irfs() -> tuple[Path, int, float]:
    cfg = ROOT / "configs/_auto_teacher_depth_x_deiminit_fold0_64_irfs.yml"
    out_dir = OUT / DEPTH_TAG
    write_cfg(
        ROOT / "configs/dfine_teacher_depth_x_fold0.yml",
        cfg,
        save_dir=out_dir,
        train_root=TRAIN_ROOT,
        detector=DEIM_INIT,
        mono_teacher=True,
        class_balance_mode="irfs",
        keep_last_n=2,
        keep_best_n=2,
    )
    run_train(cfg, out_dir, AUTO / f"{DEPTH_TAG}.log", mono="depth", port=29691)
    epoch, value = best_epoch(out_dir)
    return cfg, epoch, value


def stage_student(ir_epoch: int, depth_epoch: int) -> dict[str, object]:
    cfg = ROOT / "configs/_auto_run_x_m2d_deim_mal_irfs_deiminit_fold0_64.yml"
    out_dir = OUT / STUDENT_TAG
    write_cfg(
        ROOT / "configs/dfine_tristream_x_m2d_deim_mal_deiminit_fold0_64.yml",
        cfg,
        save_dir=out_dir,
        train_root=TRAIN_ROOT,
        detector=DEIM_INIT,
        ir_ckpt=OUT / IR_TAG / f"epoch{ir_epoch}.pt",
        depth_ckpt=OUT / DEPTH_TAG / f"epoch{depth_epoch}.pt",
        class_balance_mode="irfs",
        keep_last_n=8,
        keep_best_n=3,
    )
    run_train(cfg, out_dir, AUTO / f"{STUDENT_TAG}.log", mono=None, port=29692)
    student_epoch, student_map = best_epoch(out_dir)
    best_pred = OUT / f"{STUDENT_TAG}_pred_epoch{student_epoch}_tta_hflip"
    best_zip = SUBMISSIONS / f"{STUDENT_TAG}_epoch{student_epoch}_tta_hflip.zip"
    infer(
        cfg,
        out_dir / f"epoch{student_epoch}.pt",
        best_pred,
        best_zip,
        tta_hflip=True,
    )

    swa_ckpt = OUT / f"{STUDENT_TAG}_swa_last{SWA_LAST_K}.pt"
    build_swa_checkpoint(out_dir, swa_ckpt)
    swa_pred = OUT / f"{STUDENT_TAG}_pred_swa_last{SWA_LAST_K}"
    swa_zip = SUBMISSIONS / f"{STUDENT_TAG}_swa_last{SWA_LAST_K}.zip"
    infer(cfg, swa_ckpt, swa_pred, swa_zip, tta_hflip=False)

    result = {
        "seed": 42,
        "ir_teacher": {"best_epoch": ir_epoch, "val_map_50_95": None, "dir": str(OUT / IR_TAG)},
        "depth_teacher": {"best_epoch": depth_epoch, "val_map_50_95": None, "dir": str(OUT / DEPTH_TAG)},
        "student": {
            "best_epoch": student_epoch,
            "val_map_50_95": student_map,
            "dir": str(out_dir),
            "best_tta_zip": str(best_zip),
            "swa_ckpt": str(swa_ckpt),
            "swa_zip": str(swa_zip),
        },
    }
    (AUTO / f"{STUDENT_TAG}_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    ensure_common()
    if DONE.exists():
        print(DONE.read_text(encoding="utf-8"), flush=True)
        return 0
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()} {time.strftime('%F %T')}\n".encode())
        os.close(fd)
    except FileExistsError as exc:
        raise RuntimeError(f"sequence is already locked: {LOCK}") from exc

    try:
        wait_for_prerequisite()
        ir_cfg, ir_epoch, ir_map = stage_irfs()
        depth_cfg, depth_epoch, depth_map = stage_depth_irfs()
        result = stage_student(ir_epoch, depth_epoch)
        result["ir_teacher"]["val_map_50_95"] = ir_map
        result["depth_teacher"]["val_map_50_95"] = depth_map
        result["ir_teacher"]["config"] = str(ir_cfg)
        result["depth_teacher"]["config"] = str(depth_cfg)
        (AUTO / f"{STUDENT_TAG}_result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        DONE.write_text(time.strftime("%F %T"), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    finally:
        if LOCK.exists():
            LOCK.unlink()


if __name__ == "__main__":
    raise SystemExit(main())

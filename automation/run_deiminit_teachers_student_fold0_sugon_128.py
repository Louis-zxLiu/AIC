#!/usr/bin/env python3
"""Sugon Slurm stages for the independent 128-epoch DEIM-init experiment."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
import zipfile
from pathlib import Path

import yaml


ROOT = Path("/public/home/jqfan/AIC")
OUT = ROOT / "outputs"
AUTO = ROOT / "automation"
SUBMISSIONS = ROOT / "submissions"
PY = ROOT / "envs/aic128/bin/python"
TORCHRUN = ROOT / "envs/aic128/bin/torchrun"
DEIM_INIT = ROOT / "weights/deim_dfine_hgnetv2_x_coco_50e.pth"
EPOCHS = 128
NPROC = 2

IR_TAG = "teacher_ir_x_deiminit_fold0_128"
DEPTH_TAG = "teacher_depth_x_deiminit_fold0_128"
STUDENT_TAG = "run_x_m2d_deim_mal_deiminit_fold0_128"


def env_for() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(ROOT),
            "TORCH_HOME": str(ROOT / "cache/torch"),
            "XDG_CACHE_HOME": str(ROOT / "cache/xdg"),
            "PIP_CACHE_DIR": str(ROOT / "cache/pip"),
            "TMPDIR": str(ROOT / "tmp"),
            "PYTHONUNBUFFERED": "1",
            "PYTHONUTF8": "1",
            "OMP_NUM_THREADS": "6",
            "MKL_NUM_THREADS": "6",
        }
    )
    env.setdefault("CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(NPROC)))
    return env


def find_dataset_root(*, require_labels: bool) -> Path:
    candidates: list[Path] = []
    for path in (ROOT / "dataset").iterdir():
        if not path.is_dir():
            continue
        has_modalities = all((path / name).is_dir() for name in ("visible", "infrared", "depth"))
        has_labels = (path / "labels").is_dir()
        if has_modalities and has_labels == require_labels:
            candidates.append(path)
    if len(candidates) != 1:
        raise RuntimeError(f"ambiguous dataset root require_labels={require_labels}: {candidates}")
    return candidates[0]


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
    train["keep_last_n"] = 1
    train["keep_best_n"] = 1
    if mono_teacher:
        # Preserve the 4-GPU recipe's effective batch on a 2-GPU allocation
        # while keeping per-GPU memory reasonable for automatically assigned nodes.
        train["batch_size"] = 3
        train["grad_accum_steps"] = 8
        train["lr"] = 1.8666667e-4
    else:
        train["grad_accum_steps"] = 10
    if ir_ckpt is not None:
        cfg.setdefault("distill", {})["ir_teacher_ckpt"] = str(ir_ckpt)
    if depth_ckpt is not None:
        cfg.setdefault("distill", {})["depth_teacher_ckpt"] = str(depth_ckpt)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


def latest_checkpoint(out_dir: Path) -> Path | None:
    pairs = []
    for path in out_dir.glob("epoch*.pt"):
        match = re.fullmatch(r"epoch(\d+)", path.stem)
        if match:
            pairs.append((int(match.group(1)), path))
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
    if mono:
        cmd += ["--mono_modality", mono]
    latest = latest_checkpoint(out_dir)
    if latest is not None:
        cmd += ["--resume", str(latest)]
    run(cmd, log)


def validate_zip(zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
    txts = [name for name in names if name.endswith(".txt") and "/" not in name]
    if len(txts) != 1000 or len(names) != 1000:
        raise RuntimeError(f"zip root must contain exactly 1000 txt files: {zip_path}")
    pred_dir = OUT / f"{STUDENT_TAG}_pred_epoch{zip_path.stem.rsplit('epoch', 1)[-1]}"
    for txt in pred_dir.glob("*.txt"):
        with txt.open("r", encoding="utf-8") as handle:
            lines = handle.readlines()
        if len(lines) > 100:
            raise RuntimeError(f"{txt} has {len(lines)} lines")


def infer_student(student_cfg: Path, student_dir: Path, epoch: int) -> Path:
    test_root = find_dataset_root(require_labels=False)
    pred = OUT / f"{STUDENT_TAG}_pred_epoch{epoch}"
    zip_path = SUBMISSIONS / f"{STUDENT_TAG}_epoch{epoch}.zip"
    pred.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    run(
        [
            str(PY),
            "aic/tools/infer.py",
            "--config",
            str(student_cfg),
            "--checkpoint",
            str(student_dir / f"epoch{epoch}.pt"),
            "--test_root",
            str(test_root),
            "--out_dir",
            str(pred),
            "--zip_path",
            str(zip_path),
        ],
        AUTO / f"{STUDENT_TAG}_infer_epoch{epoch}.log",
    )
    txt_count = len(list(pred.glob("*.txt")))
    if txt_count != 1000:
        raise RuntimeError(f"expected 1000 txt files, got {txt_count}: {pred}")
    validate_zip(zip_path)
    return zip_path


def ensure_common() -> Path:
    for path in (OUT, AUTO, SUBMISSIONS, ROOT / "cache/torch", ROOT / "cache/xdg", ROOT / "cache/pip", ROOT / "tmp"):
        path.mkdir(parents=True, exist_ok=True)
    if not PY.is_file():
        raise FileNotFoundError(PY)
    if not DEIM_INIT.is_file():
        raise FileNotFoundError(DEIM_INIT)
    return find_dataset_root(require_labels=True)


def stage_ir() -> None:
    train_root = ensure_common()
    cfg = ROOT / "configs/_auto_teacher_ir_x_deiminit_fold0_128.yml"
    out_dir = OUT / IR_TAG
    write_cfg(
        ROOT / "configs/dfine_teacher_ir_x_fold0.yml",
        cfg,
        save_dir=out_dir,
        train_root=train_root,
        detector=DEIM_INIT,
        mono_teacher=True,
    )
    run_train(cfg, out_dir, AUTO / f"{IR_TAG}.log", mono="ir", port=29680)
    epoch, value = best_epoch(out_dir)
    (AUTO / f"{IR_TAG}_result.json").write_text(
        json.dumps({"best_epoch": epoch, "val_map_50_95": value, "dir": str(out_dir)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def stage_depth() -> None:
    train_root = ensure_common()
    cfg = ROOT / "configs/_auto_teacher_depth_x_deiminit_fold0_128.yml"
    out_dir = OUT / DEPTH_TAG
    write_cfg(
        ROOT / "configs/dfine_teacher_depth_x_fold0.yml",
        cfg,
        save_dir=out_dir,
        train_root=train_root,
        detector=DEIM_INIT,
        mono_teacher=True,
    )
    run_train(cfg, out_dir, AUTO / f"{DEPTH_TAG}.log", mono="depth", port=29681)
    epoch, value = best_epoch(out_dir)
    (AUTO / f"{DEPTH_TAG}_result.json").write_text(
        json.dumps({"best_epoch": epoch, "val_map_50_95": value, "dir": str(out_dir)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def stage_student() -> None:
    train_root = ensure_common()
    ir_epoch, ir_map = best_epoch(OUT / IR_TAG)
    depth_epoch, depth_map = best_epoch(OUT / DEPTH_TAG)
    cfg = ROOT / "configs/_auto_run_x_m2d_deim_mal_deiminit_fold0_128.yml"
    out_dir = OUT / STUDENT_TAG
    write_cfg(
        ROOT / "configs/dfine_tristream_x_m2d_deim_mal_deiminit_fold0_64.yml",
        cfg,
        save_dir=out_dir,
        train_root=train_root,
        detector=DEIM_INIT,
        ir_ckpt=OUT / IR_TAG / f"epoch{ir_epoch}.pt",
        depth_ckpt=OUT / DEPTH_TAG / f"epoch{depth_epoch}.pt",
    )
    run_train(cfg, out_dir, AUTO / f"{STUDENT_TAG}.log", mono=None, port=29682)
    student_epoch, student_map = best_epoch(out_dir)
    zip_path = infer_student(cfg, out_dir, student_epoch)
    result = {
        "seed": 42,
        "nproc_per_node": NPROC,
        "detector_init": str(DEIM_INIT),
        "ir_teacher": {"epoch": ir_epoch, "val_map_50_95": ir_map, "dir": str(OUT / IR_TAG)},
        "depth_teacher": {"epoch": depth_epoch, "val_map_50_95": depth_map, "dir": str(OUT / DEPTH_TAG)},
        "student": {"epoch": student_epoch, "val_map_50_95": student_map, "dir": str(out_dir), "zip": str(zip_path)},
    }
    (AUTO / f"{STUDENT_TAG}_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (AUTO / f"{STUDENT_TAG}.done").write_text(time.strftime("%F %T") + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["ir", "depth", "student"])
    args = parser.parse_args()
    {"ir": stage_ir, "depth": stage_depth, "student": stage_student}[args.stage]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""703: DEIM-D-FINE-X initialization ablation, teachers then student.

All outputs stay below the existing 703 AIC workspace.  The only model
initialization change relative to the standard X MAL/M2D recipe is the
DEIM-D-FINE-X detector checkpoint.
"""
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
DEIM_INIT = str(ROOT / "weights/deim_dfine_hgnetv2_x_coco_50e.pth")

ENV = os.environ.copy()
ENV.update(
    {
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "PYTHONPATH": str(ROOT),
        "NCCL_P2P_DISABLE": "1",
        "NCCL_IB_DISABLE": "1",
        "TORCH_HOME": str(ROOT / "cache/torch"),
        "XDG_CACHE_HOME": str(ROOT / "cache/xdg"),
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


def write_cfg(src: Path, dst: Path, *, save_dir: Path, detector: str,
              epochs: int = 96, full_data: bool = False,
              ir_ckpt: str | None = None, depth_ckpt: str | None = None,
              batch_size: int | None = None, grad_accum_steps: int | None = None,
              lr: float | None = None) -> None:
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    cfg["pretrained_detector"] = detector
    cfg["train"]["seed"] = 42
    cfg["train"]["data_root"] = TRAIN_ROOT
    cfg["train"]["full_data"] = full_data
    cfg["train"]["epochs"] = epochs
    cfg["train"]["save_dir"] = str(save_dir)
    if batch_size is not None:
        cfg["train"]["batch_size"] = batch_size
    if grad_accum_steps is not None:
        cfg["train"]["grad_accum_steps"] = grad_accum_steps
    if lr is not None:
        cfg["train"]["lr"] = lr
    if ir_ckpt is not None:
        cfg.setdefault("distill", {})["ir_teacher_ckpt"] = ir_ckpt
    if depth_ckpt is not None:
        cfg.setdefault("distill", {})["depth_teacher_ckpt"] = depth_ckpt
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


def best_epoch(out_dir: Path) -> tuple[int, float]:
    record = json.loads((out_dir / "curves.json").read_text(encoding="utf-8"))
    epochs = record.get("eval_epochs", [])
    values = record.get("eval_metrics", {}).get("mAP@50-95", [])
    pairs = [(int(epoch), float(value)) for epoch, value in zip(epochs, values) if value is not None]
    if not pairs:
        raise RuntimeError(f"no validation mAP in {out_dir / 'curves.json'}")
    epoch, value = max(pairs, key=lambda item: item[1])
    ckpt = out_dir / f"epoch{epoch}.pt"
    if not ckpt.is_file():
        raise RuntimeError(f"best checkpoint was pruned: {ckpt}")
    return epoch, value


def infer(cfg: Path, ckpt: Path, tag: str, epoch: int) -> str:
    pred = OUT / f"{tag}_pred_epoch{epoch}"
    zip_path = ROOT / "submissions" / f"{tag}_epoch{epoch}.zip"
    pred.mkdir(parents=True, exist_ok=True)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [PY, "aic/tools/infer.py", "--config", str(cfg), "--checkpoint", str(ckpt),
         "--test_root", TEST_ROOT, "--out_dir", str(pred), "--zip_path", str(zip_path)],
        AUTO / f"{tag}_infer_epoch{epoch}.log",
    )
    count = len(list(pred.glob("*.txt")))
    if count != 1000:
        raise RuntimeError(f"{tag} epoch{epoch}: expected 1000 txt, got {count}")
    return str(zip_path)


def main() -> int:
    for path in (ROOT / "cache/torch", ROOT / "cache/xdg", ROOT / "tmp"):
        path.mkdir(parents=True, exist_ok=True)
    if not Path(DEIM_INIT).is_file():
        raise FileNotFoundError(DEIM_INIT)

    # Keep the 96-epoch run completely separate from the abandoned 64-epoch run.
    ir_dir = OUT / "teacher_ir_x_deiminit_fold0_96"
    depth_dir = OUT / "teacher_depth_x_deiminit_fold0_96"
    ir_cfg = ROOT / "configs/_auto_teacher_ir_x_deiminit_fold0_96.yml"
    depth_cfg = ROOT / "configs/_auto_teacher_depth_x_deiminit_fold0_96.yml"
    write_cfg(ROOT / "configs/dfine_teacher_ir_x_fold0.yml", ir_cfg,
              save_dir=ir_dir, detector=DEIM_INIT,
              batch_size=6, grad_accum_steps=2, lr=1.8666667e-4)
    write_cfg(ROOT / "configs/dfine_teacher_depth_x_fold0.yml", depth_cfg,
              save_dir=depth_dir, detector=DEIM_INIT,
              batch_size=6, grad_accum_steps=2, lr=1.8666667e-4)

    run([TORCHRUN, "--standalone", "--nproc_per_node=4", "--master_port=29740",
         "aic/tools/train.py", "--config", str(ir_cfg), "--mono_modality", "ir"],
        AUTO / "teacher_ir_x_deiminit_fold0_96.log")
    ir_epoch, ir_map = best_epoch(ir_dir)
    run([TORCHRUN, "--standalone", "--nproc_per_node=4", "--master_port=29741",
         "aic/tools/train.py", "--config", str(depth_cfg), "--mono_modality", "depth"],
        AUTO / "teacher_depth_x_deiminit_fold0_96.log")
    depth_epoch, depth_map = best_epoch(depth_dir)

    student_dir = OUT / "run_x_m2d_deim_mal_deiminit_fold0_96"
    student_cfg = ROOT / "configs/_auto_run_x_m2d_mal_deiminit_fold0_96.yml"
    write_cfg(
        ROOT / "configs/dfine_tristream_x_m2d_deim_mal_deiminit_fold0_64.yml", student_cfg,
        save_dir=student_dir, detector=DEIM_INIT,
        ir_ckpt=str(ir_dir / f"epoch{ir_epoch}.pt"),
        depth_ckpt=str(depth_dir / f"epoch{depth_epoch}.pt"),
    )
    run([TORCHRUN, "--standalone", "--nproc_per_node=4", "--master_port=29742",
         "aic/tools/train.py", "--config", str(student_cfg)],
        AUTO / "run_x_m2d_deim_mal_deiminit_fold0_96.log")
    student_epoch, student_map = best_epoch(student_dir)
    zip_path = infer(student_cfg, student_dir / f"epoch{student_epoch}.pt",
                     "run_x_m2d_deim_mal_deiminit_fold0_96", student_epoch)
    result = {
        "seed": 42,
        "detector_init": DEIM_INIT,
        "ir_teacher": {"epoch": ir_epoch, "val_map_50_95": ir_map, "dir": str(ir_dir)},
        "depth_teacher": {"epoch": depth_epoch, "val_map_50_95": depth_map, "dir": str(depth_dir)},
        "student": {"epoch": student_epoch, "val_map_50_95": student_map,
                    "zip": zip_path, "dir": str(student_dir)},
    }
    (AUTO / "run_deiminit_teachers_student_fold0_703_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (AUTO / "run_deiminit_teachers_student_fold0_703.done").write_text(
        time.strftime("%F %T") + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

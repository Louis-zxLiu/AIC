#!/usr/bin/env python3
"""Serial DEIM recipe ablations for the AIC project.

All outputs stay below /home/u703/AIC_Project_sy/AIC on the remote host.
Historical M2D teachers were removed, so the chain rebuilds fold-0 teachers
first, selects their best EMA epochs, then runs schedule-only and MAL-only.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import yaml

ROOT = Path("/home/u703/AIC_Project_sy/AIC")
PY = "/home/u703/miniconda3/envs/AIC/bin/python"
TORCHRUN = "/home/u703/miniconda3/envs/AIC/bin/torchrun"
TEST_ROOT = "/home/u703/AIC_Project_sy/dataset/测试集"
AUTO = ROOT / "automation"
AUTO.mkdir(exist_ok=True)
ENV = os.environ.copy()
ENV.update({"PYTHONPATH": str(ROOT), "NCCL_P2P_DISABLE": "1", "NCCL_IB_DISABLE": "1"})


def latest_checkpoint(out_dir: Path):
    files = []
    for path in out_dir.glob("epoch*.pt"):
        try:
            files.append((int(path.stem[5:]), path))
        except ValueError:
            pass
    return max(files, default=None)


def completed_epoch(out_dir: Path, epochs: int) -> bool:
    curves = out_dir / "curves.json"
    if not curves.exists():
        return False
    try:
        rec = json.loads(curves.read_text(encoding="utf-8"))
        return max((int(x) for x in rec.get("eval_epochs", [])), default=-1) >= epochs - 1
    except Exception:
        return False


def run_train(config: str, mono: str | None = None) -> None:
    cfg = yaml.safe_load((ROOT / config).read_text(encoding="utf-8"))
    out_dir = Path(cfg["train"]["save_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [TORCHRUN, "--nproc_per_node=4", "aic/tools/train.py", "--config", config]
    if mono:
        cmd += ["--mono_modality", mono]
    latest = latest_checkpoint(out_dir)
    if latest and not completed_epoch(out_dir, int(cfg["train"]["epochs"])):
        cmd += ["--resume", str(latest[1])]
    log = AUTO / (Path(config).stem + ".log")
    print("[run]", " ".join(cmd), flush=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("\n===== START %s =====\n" % time.strftime("%F %T"))
        handle.flush()
        result = subprocess.run(cmd, cwd=ROOT, env=ENV, stdout=handle, stderr=subprocess.STDOUT)
        handle.write("\n===== EXIT %d %s =====\n" % (result.returncode, time.strftime("%F %T")))
    if result.returncode:
        raise SystemExit(f"{config} failed with exit code {result.returncode}")


def best_eval(out_dir: Path):
    rec = json.loads((out_dir / "curves.json").read_text(encoding="utf-8"))
    pairs = [
        (float(value), int(epoch))
        for epoch, value in zip(
            rec.get("eval_epochs", []), rec.get("eval_metrics", {}).get("mAP@50-95", [])
        )
        if value is not None
    ]
    if not pairs:
        raise RuntimeError(f"no validation metrics in {out_dir}")
    value, epoch = max(pairs)
    return epoch, value


def resolved_config(base_config: str, out_config: str, ir_ckpt: Path, depth_ckpt: Path) -> None:
    cfg = yaml.safe_load((ROOT / base_config).read_text(encoding="utf-8"))
    cfg["distill"]["ir_teacher_ckpt"] = str(ir_ckpt)
    cfg["distill"]["depth_teacher_ckpt"] = str(depth_ckpt)
    (ROOT / out_config).write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def resolved_baseline(out_config: str, ir_ckpt: Path, depth_ckpt: Path) -> None:
    """Create the same-epoch control used by the 60.53 ablation.

    It keeps the original cosine schedule and augmentation, while using the
    newly selected teachers and 64 epochs so the comparison has one variable:
    DEIM schedule or MAL versus this control.
    """
    cfg = yaml.safe_load(
        (ROOT / "configs/dfine_tristream_x_m2d_pkd_nocross_fold0_48_control.yml")
        .read_text(encoding="utf-8")
    )
    cfg["train"]["epochs"] = 64
    cfg["train"]["save_dir"] = str(ROOT / "outputs/run_x_m2d_pkd_nocross_baseline_fold0_64")
    cfg["distill"]["ir_teacher_ckpt"] = str(ir_ckpt)
    cfg["distill"]["depth_teacher_ckpt"] = str(depth_ckpt)
    (ROOT / out_config).write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def run_infer(config: str, out_dir: Path, epoch: int, tag: str) -> str:
    checkpoint = out_dir / f"epoch{epoch}.pt"
    pred_dir = ROOT / "outputs" / f"{tag}_pred_epoch{epoch}"
    zip_path = ROOT / "submissions" / f"{tag}_epoch{epoch}.zip"
    pred_dir.mkdir(parents=True, exist_ok=True)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        PY,
        "aic/tools/infer.py",
        "--config",
        config,
        "--checkpoint",
        str(checkpoint),
        "--test_root",
        TEST_ROOT,
        "--out_dir",
        str(pred_dir),
        "--zip_path",
        str(zip_path),
    ]
    log = AUTO / f"{tag}_infer.log"
    with log.open("a", encoding="utf-8") as handle:
        result = subprocess.run(cmd, cwd=ROOT, env=ENV, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise SystemExit(f"inference failed: {tag}")
    return str(zip_path)


run_train("configs/dfine_teacher_ir_x_fold0.yml", "ir")
run_train("configs/dfine_teacher_depth_x_fold0.yml", "depth")
ir_ep, ir_map = best_eval(ROOT / "outputs/teacher_ir_x_fold0")
depth_ep, depth_map = best_eval(ROOT / "outputs/teacher_depth_x_fold0")
ir_ckpt = ROOT / "outputs/teacher_ir_x_fold0" / f"epoch{ir_ep}.pt"
depth_ckpt = ROOT / "outputs/teacher_depth_x_fold0" / f"epoch{depth_ep}.pt"

for base_config, tag in [
    ("configs/dfine_tristream_x_m2d_deim_schedule_fold0_64.yml", "deim_schedule"),
    ("configs/dfine_tristream_x_m2d_deim_mal_fold0_64.yml", "deim_mal"),
]:
    generated = f"automation/{Path(base_config).stem}_resolved.yml"
    resolved_config(base_config, generated, ir_ckpt, depth_ckpt)
    run_train(generated)
    out_dir = Path(yaml.safe_load((ROOT / generated).read_text(encoding="utf-8"))["train"]["save_dir"])
    epoch, value = best_eval(out_dir)
    zip_path = run_infer(generated, out_dir, epoch, tag)
    result = {
        "best_epoch": epoch,
        "val_map_50_95": value,
        "zip": zip_path,
        "ir_teacher_epoch": ir_ep,
        "ir_teacher_map": ir_map,
        "depth_teacher_epoch": depth_ep,
        "depth_teacher_map": depth_map,
    }
    (AUTO / f"{tag}_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[result]", tag, result, flush=True)

# Run the direct 64-epoch control only after both recipe ablations have
# completed.  This makes the comparison fair even if a later run is resumed.
baseline_config = "automation/dfine_tristream_x_m2d_pkd_nocross_baseline_fold0_64_resolved.yml"
resolved_baseline(baseline_config, ir_ckpt, depth_ckpt)
run_train(baseline_config)
baseline_out = Path(yaml.safe_load((ROOT / baseline_config).read_text(encoding="utf-8"))["train"]["save_dir"])
baseline_ep, baseline_val = best_eval(baseline_out)
baseline_zip = run_infer(baseline_config, baseline_out, baseline_ep, "deim_baseline")
baseline_result = {
    "best_epoch": baseline_ep,
    "val_map_50_95": baseline_val,
    "zip": baseline_zip,
    "ir_teacher_epoch": ir_ep,
    "ir_teacher_map": ir_map,
    "depth_teacher_epoch": depth_ep,
    "depth_teacher_map": depth_map,
}
(AUTO / "deim_baseline_result.json").write_text(
    json.dumps(baseline_result, ensure_ascii=False, indent=2), encoding="utf-8"
)
print("[result] deim_baseline", baseline_result, flush=True)

(AUTO / "deim_sequence.done").write_text(time.strftime("%F %T"), encoding="utf-8")

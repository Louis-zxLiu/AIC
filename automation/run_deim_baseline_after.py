#!/usr/bin/env python3
"""Run the matched 64-epoch M2D control after the DEIM chain completes."""
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
AUTO = ROOT / "automation"
ENV = os.environ.copy()
ENV.update({"PYTHONPATH": str(ROOT), "NCCL_P2P_DISABLE": "1", "NCCL_IB_DISABLE": "1"})


def best(out_dir: Path):
    rec = json.loads((out_dir / "curves.json").read_text(encoding="utf-8"))
    vals = [
        (float(v), int(e))
        for e, v in zip(
            rec.get("eval_epochs", []), rec.get("eval_metrics", {}).get("mAP@50-95", [])
        )
        if v is not None
    ]
    if not vals:
        raise RuntimeError(f"no eval values in {out_dir}")
    value, epoch = max(vals)
    return epoch, value


while not (AUTO / "deim_sequence.done").exists():
    time.sleep(60)

ir_ep, ir_map = best(ROOT / "outputs/teacher_ir_x_fold0")
depth_ep, depth_map = best(ROOT / "outputs/teacher_depth_x_fold0")
ir_ckpt = ROOT / "outputs/teacher_ir_x_fold0" / f"epoch{ir_ep}.pt"
depth_ckpt = ROOT / "outputs/teacher_depth_x_fold0" / f"epoch{depth_ep}.pt"
cfg = yaml.safe_load(
    (ROOT / "configs/dfine_tristream_x_m2d_pkd_nocross_fold0_48_control.yml")
    .read_text(encoding="utf-8")
)
cfg["train"]["epochs"] = 64
cfg["train"]["save_dir"] = str(ROOT / "outputs/run_x_m2d_pkd_nocross_baseline_fold0_64")
cfg["distill"]["ir_teacher_ckpt"] = str(ir_ckpt)
cfg["distill"]["depth_teacher_ckpt"] = str(depth_ckpt)
config_rel = "automation/dfine_tristream_x_m2d_pkd_nocross_baseline_fold0_64_resolved.yml"
(ROOT / config_rel).write_text(
    yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
)
out_dir = Path(cfg["train"]["save_dir"])
out_dir.mkdir(parents=True, exist_ok=True)

cmd = [TORCHRUN, "--nproc_per_node=4", "aic/tools/train.py", "--config", config_rel]
latest = sorted(
    ((int(p.stem[5:]), p) for p in out_dir.glob("epoch*.pt") if p.stem[5:].isdigit()),
    reverse=True,
)
if latest:
    cmd += ["--resume", str(latest[0][1])]
log = AUTO / "deim_baseline_train.log"
with log.open("a", encoding="utf-8") as handle:
    handle.write(f"\n===== START {time.strftime('%F %T')} =====\n")
    result = subprocess.run(cmd, cwd=ROOT, env=ENV, stdout=handle, stderr=subprocess.STDOUT)
    handle.write(f"\n===== EXIT {result.returncode} {time.strftime('%F %T')} =====\n")
if result.returncode:
    raise SystemExit(result.returncode)

epoch, value = best(out_dir)
pred = ROOT / f"outputs/deim_baseline_pred_epoch{epoch}"
zip_path = ROOT / f"submissions/deim_baseline_epoch{epoch}.zip"
pred.mkdir(parents=True, exist_ok=True)
zip_path.parent.mkdir(parents=True, exist_ok=True)
infer = [
    PY,
    "aic/tools/infer.py",
    "--config",
    config_rel,
    "--checkpoint",
    str(out_dir / f"epoch{epoch}.pt"),
    "--test_root",
    "/home/u703/AIC_Project_sy/dataset/测试集",
    "--out_dir",
    str(pred),
    "--zip_path",
    str(zip_path),
]
with (AUTO / "deim_baseline_infer.log").open("a", encoding="utf-8") as handle:
    result = subprocess.run(infer, cwd=ROOT, env=ENV, stdout=handle, stderr=subprocess.STDOUT)
if result.returncode:
    raise SystemExit(result.returncode)
result_data = {
    "best_epoch": epoch,
    "val_map_50_95": value,
    "zip": str(zip_path),
    "ir_teacher_epoch": ir_ep,
    "ir_teacher_map": ir_map,
    "depth_teacher_epoch": depth_ep,
    "depth_teacher_map": depth_map,
}
(AUTO / "deim_baseline_result.json").write_text(
    json.dumps(result_data, ensure_ascii=False, indent=2), encoding="utf-8"
)
(AUTO / "deim_baseline.done").write_text(time.strftime("%F %T"), encoding="utf-8")
print(result_data, flush=True)

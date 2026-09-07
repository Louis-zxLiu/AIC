#!/usr/bin/env python3
"""Serial GPU-2/3 comparison: DEIM-X RGB-only, then native-X M2D."""
from __future__ import annotations
import json, os, subprocess, time
from pathlib import Path
import yaml

ROOT = Path("/home/u703/AIC_Project_sy/AIC")
OUT, AUTO = ROOT / "outputs", ROOT / "automation"
PY = "/home/u703/miniconda3/envs/AIC/bin/python"
TORCHRUN = "/home/u703/miniconda3/envs/AIC/bin/torchrun"
TRAIN_ROOT = "/home/u703/AIC_Project_sy/dataset/训练集"
TEST_ROOT = "/home/u703/AIC_Project_sy/dataset/测试集"
DEIM = str(ROOT / "weights/deim_dfine_hgnetv2_x_coco_50e.pth")
NATIVE = str(ROOT / "weights/dfine_x_coco.pth")
ENV = os.environ.copy()
ENV.update({"CUDA_VISIBLE_DEVICES":"2,3", "PYTHONPATH":str(ROOT), "NCCL_P2P_DISABLE":"1",
            "NCCL_IB_DISABLE":"1", "PYTHONUNBUFFERED":"1", "PYTHONUTF8":"1"})

def run(cmd, log):
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n===== START {time.strftime('%F %T')} =====\n"); f.flush()
        p = subprocess.run(cmd, cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT)
        f.write(f"\n===== EXIT {p.returncode} {time.strftime('%F %T')} =====\n")
    if p.returncode: raise RuntimeError(f"failed: {log}")

def cfg_write(src, dst, save_dir, detector, *, mono=False, ir=None, depth=None):
    c = yaml.safe_load(Path(src).read_text(encoding="utf-8"))
    c["pretrained_detector"] = detector
    c["load_pretrained_detector"] = True
    c["train"].update({"data_root":TRAIN_ROOT, "seed":42, "epochs":64, "save_dir":str(save_dir),
                        "batch_size":3, "grad_accum_steps":3, "lr":1.4e-4, "eval_interval_epochs":1,
                        "keep_last_n":24, "keep_best_n":3})
    if ir and depth:
        c.setdefault("distill", {}).update({"ir_teacher_ckpt":ir, "depth_teacher_ckpt":depth})
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True), encoding="utf-8")

def best(d):
    x = json.loads((d/"curves.json").read_text(encoding="utf-8"))
    pairs = list(zip(x["eval_epochs"], x["eval_metrics"]["mAP@50-95"]))
    e, v = max(((int(e), float(v)) for e,v in pairs if v is not None), key=lambda z:z[1])
    ck = d/f"epoch{e}.pt"
    if not ck.is_file(): raise RuntimeError(f"missing {ck}")
    return e,v,ck

def main():
    rgb_dir=OUT/"run_deim_x_rgbonly_fold0_64"; rgb_cfg=ROOT/"configs/_auto_run_deim_x_rgbonly_fold0_64.yml"
    cfg_write(ROOT/"configs/dfine_teacher_ir_x_fold0.yml", rgb_cfg, rgb_dir, DEIM)
    run([TORCHRUN,"--standalone","--nproc_per_node=2","--master_port=29820","aic/tools/train.py",
         "--config",str(rgb_cfg),"--mono_modality","rgb"], AUTO/"run_deim_x_rgbonly_fold0_64.log")
    re,rm,rck=best(rgb_dir)
    run([PY,"aic/tools/infer_mono.py","--config",str(rgb_cfg),"--checkpoint",str(rck),"--modality","rgb",
         "--test_root",TEST_ROOT,"--out_dir",str(OUT/f"run_deim_x_rgbonly_fold0_64_pred_epoch{re}"),
         "--zip_path",str(ROOT/f"submissions/run_deim_x_rgbonly_fold0_64_epoch{re}.zip")],
        AUTO/f"run_deim_x_rgbonly_fold0_64_infer_epoch{re}.log")

    student_dir=OUT/"run_native_x_m2d_newteachers_fold0_64"; student_cfg=ROOT/"configs/_auto_run_native_x_m2d_newteachers_fold0_64.yml"
    cfg_write(ROOT/"configs/dfine_tristream_x_m2d_deim_mal_deiminit_fold0_64.yml", student_cfg, student_dir, NATIVE,
              ir=str(OUT/"teacher_ir_x_deiminit_fold0_96"/"epoch91.pt"),
              depth=str(OUT/"teacher_depth_x_deiminit_fold0_96"/"epoch53.pt"))
    run([TORCHRUN,"--standalone","--nproc_per_node=2","--master_port=29821","aic/tools/train.py","--config",str(student_cfg)],
        AUTO/"run_native_x_m2d_newteachers_fold0_64.log")
    se,sm,sck=best(student_dir)
    run([PY,"aic/tools/infer.py","--config",str(student_cfg),"--checkpoint",str(sck),"--test_root",TEST_ROOT,
         "--out_dir",str(OUT/f"run_native_x_m2d_newteachers_fold0_64_pred_epoch{se}"),
         "--zip_path",str(ROOT/f"submissions/run_native_x_m2d_newteachers_fold0_64_epoch{se}.zip")],
        AUTO/f"run_native_x_m2d_newteachers_fold0_64_infer_epoch{se}.log")
    result={"rgb_deim":{"epoch":re,"val_map_50_95":rm,"zip":str(ROOT/f"submissions/run_deim_x_rgbonly_fold0_64_epoch{re}.zip")},
            "native_m2d":{"epoch":se,"val_map_50_95":sm,"zip":str(ROOT/f"submissions/run_native_x_m2d_newteachers_fold0_64_epoch{se}.zip")}}
    (AUTO/"run_rgb_deim_vs_native_m2d_703_result.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    (AUTO/"run_rgb_deim_vs_native_m2d_703.done").write_text(time.strftime('%F %T')+"\n",encoding="utf-8")
    print(json.dumps(result),flush=True)
if __name__ == "__main__": main()

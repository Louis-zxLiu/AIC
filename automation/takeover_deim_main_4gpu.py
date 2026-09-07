#!/usr/bin/env python3
"""After the already-running depth teacher exits, take over the student on 4 GPUs.

This avoids interrupting the in-progress teacher while preventing the old
two-GPU parent scheduler from launching its stale student command.
"""
from __future__ import annotations
import os, signal, subprocess, time
from pathlib import Path
import yaml

ROOT = Path('/home/u703/AIC_Project_sy/AIC')
OUT = ROOT / 'outputs'
AUTO = ROOT / 'automation'
PY = '/home/u703/miniconda3/envs/AIC/bin/python'
TORCHRUN = '/home/u703/miniconda3/envs/AIC/bin/torchrun'
PARENT = 230902
ENV = os.environ.copy()
ENV.update({'CUDA_VISIBLE_DEVICES':'0,1,2,3','PYTHONPATH':str(ROOT),
            'NCCL_P2P_DISABLE':'1','NCCL_IB_DISABLE':'1','PYTHONUNBUFFERED':'1'})

def pids(pattern):
    out = subprocess.run(['pgrep','-u','u703','-f',pattern], text=True, capture_output=True).stdout
    return [int(x) for x in out.split() if x.isdigit()]

def run(cmd, log):
    with open(log,'a',encoding='utf-8') as f:
        p=subprocess.run(cmd,cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT)
    if p.returncode: raise RuntimeError(f'failed {p.returncode}: {cmd}')

def main():
    depth_pat = '_auto_teacher_depth_x_full_seed42_ep64.yml'
    while pids(depth_pat):
        time.sleep(30)
    # Parent may already have spawned its stale 2-GPU student. Stop that
    # student and parent before launching the corrected four-GPU command.
    for pat in ['_auto_deim_mal_full2000_seed42_64.yml', 'run_deim_mal_full_teachers.py']:
        for pid in pids(pat):
            if pid != os.getpid():
                try: os.kill(pid, signal.SIGTERM)
                except ProcessLookupError: pass
    time.sleep(5)
    cfg = ROOT / 'configs/_auto_deim_mal_full2000_seed42_64.yml'
    if not cfg.exists():
        raise RuntimeError(f'missing generated student config {cfg}')
    data = yaml.safe_load(cfg.read_text(encoding='utf-8'))
    data['train']['batch_size'] = 2
    data['train']['grad_accum_steps'] = 2
    data['train']['lr'] = 1.8666667e-4
    data['train']['save_dir'] = str(OUT/'run_x_m2d_deim_mal_full2000_seed42_bs2_4gpu_64')
    cfg4 = ROOT / 'configs/_auto_deim_mal_full2000_seed42_bs2_4gpu_64.yml'
    cfg4.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding='utf-8')
    outdir = Path(data['train']['save_dir'])
    run([TORCHRUN,'--standalone','--nproc_per_node=4','--master_port=29632',
         'aic/tools/train.py','--config',str(cfg4)], AUTO/'run_deim_mal_main_4gpu.log')
    for ep in (47,63):
        pred=OUT/f'run_x_m2d_deim_mal_full2000_seed42_bs2_4gpu_64_pred_epoch{ep}'
        zip_path=ROOT/'submissions'/f'run_x_m2d_deim_mal_full2000_seed42_bs2_4gpu_64_epoch{ep}.zip'
        run([PY,'aic/tools/infer.py','--config',str(cfg4),'--checkpoint',str(outdir/f'epoch{ep}.pt'),
             '--test_root','/home/u703/AIC_Project_sy/dataset/测试集','--out_dir',str(pred),'--zip_path',str(zip_path)],
            AUTO/f'run_deim_mal_main_4gpu_infer_epoch{ep}.log')

if __name__=='__main__': main()

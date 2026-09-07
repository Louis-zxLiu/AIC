#!/usr/bin/env python3
"""Wait for the active Depth teacher, then run DEIM-MAL on GPU 2,3."""
from __future__ import annotations
import os, signal, subprocess, time
from pathlib import Path
import torch, yaml

ROOT=Path('/home/u703/AIC_Project_sy/AIC'); OUT=ROOT/'outputs'; AUTO=ROOT/'automation'
OLD_PARENT=1028646
IR=OUT/'auto_teacher_ir_x_full_seed42_bs6/epoch57.pt'
DEPTH=OUT/'auto_teacher_depth_x_full_seed42_d5_bs6/epoch63.pt'
STUDENT=OUT/'run_x_m2d_deim_mal_full2000_seed42_d5teachers_gpu23_bs2_64'
PY='/home/u703/miniconda3/envs/AIC/bin/python'; TORCHRUN='/home/u703/miniconda3/envs/AIC/bin/torchrun'
ENV=os.environ.copy(); ENV.update({'CUDA_VISIBLE_DEVICES':'2,3','PYTHONPATH':str(ROOT),
 'NCCL_P2P_DISABLE':'1','NCCL_IB_DISABLE':'1','PYTHONUNBUFFERED':'1'})

def run(cmd, log):
    with open(log,'a',encoding='utf-8') as f:
        r=subprocess.run(cmd,cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT)
    if r.returncode: raise RuntimeError(f'command failed {r.returncode}; see {log}')

# The teacher may have been launched by a different torchrun PID.  Waiting on
# a hard-coded PID is unsafe because completed torchrun children can remain as
# zombies.  The durable completion signal is the atomically-written epoch63
# checkpoint; wait for it, and briefly allow the writer to exit.
while not DEPTH.is_file(): time.sleep(30)
time.sleep(10)
try: os.kill(OLD_PARENT,signal.SIGKILL)
except ProcessLookupError: pass
if not DEPTH.is_file(): raise RuntimeError(f'Depth training ended without {DEPTH}')
c=torch.load(DEPTH,map_location='cpu',weights_only=False); s=c['ema'] if c.get('ema') is not None else c['model']
shape=tuple(s['backbone.stem_depth.stem1.conv.weight'].shape)
if shape != (32,5,3,3): raise RuntimeError(f'invalid Depth teacher shape {shape}')

cfg=yaml.safe_load((ROOT/'configs/dfine_tristream_x_m2d_deim_mal_fold0_64.yml').read_text(encoding='utf-8'))
cfg['train'].update({'data_root':'/home/u703/AIC_Project_sy/dataset/训练集','seed':42,
 'full_data':True,'batch_size':2,'grad_accum_steps':3,'epochs':64,'lr':1.4e-4,
 'save_dir':str(STUDENT),'keep_last_n':24})
cfg['distill']['ir_teacher_ckpt']=str(IR); cfg['distill']['depth_teacher_ckpt']=str(DEPTH)
config=ROOT/'configs/_auto_deim_mal_full2000_seed42_d5teachers_gpu23_bs2_64.yml'
config.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding='utf-8')
run([TORCHRUN,'--standalone','--nproc_per_node=2','--master_port=29641','aic/tools/train.py','--config',str(config)],
    AUTO/'deim_mal_full_seed42_d5teachers_gpu23.log')
for ep in (47,63):
    pred=OUT/f'run_x_m2d_deim_mal_full2000_seed42_d5teachers_gpu23_bs2_pred_epoch{ep}'
    zip_path=ROOT/'submissions'/f'run_x_m2d_deim_mal_full2000_seed42_d5teachers_gpu23_bs2_epoch{ep}.zip'
    run([PY,'aic/tools/infer.py','--config',str(config),'--checkpoint',str(STUDENT/f'epoch{ep}.pt'),
         '--test_root','/home/u703/AIC_Project_sy/dataset/测试集','--out_dir',str(pred),'--zip_path',str(zip_path)],
        AUTO/f'deim_mal_full_seed42_d5teachers_gpu23_infer_epoch{ep}.log')
(AUTO/'deim_mal_depth5_gpu23.done').write_text(time.strftime('%F %T'),encoding='utf-8')

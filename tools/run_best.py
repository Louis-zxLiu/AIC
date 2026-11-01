"""Reproduce the selected student or retrain its teachers and student."""
from pathlib import Path
import argparse,os,subprocess,sys,yaml
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['infer','train'])
 p.add_argument('--stage',choices=['student','teacher_ir','teacher_depth'],default='student')
 p.add_argument('--test-root',type=Path);p.add_argument('--data-root',type=Path)
 p.add_argument('--weights-dir',type=Path);p.add_argument('--checkpoint',type=Path)
 p.add_argument('--output-dir',type=Path);p.add_argument('--gpus',default='0');p.add_argument('--resume',type=Path)
 a=p.parse_args();env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=a.gpus,PYTHONPATH=str(ROOT),OMP_NUM_THREADS='4',NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1',PYTHONDONTWRITEBYTECODE='1')
 if a.mode=='infer':
  if not a.test_root:p.error('--test-root is required')
  if ',' in a.gpus:p.error('Inference uses one GPU; select one physical index')
  command=[sys.executable,'-B','-u','-m','aic.best56132.inference','--test-root',str(a.test_root.resolve())]
  checkpoint=a.checkpoint or (a.weights_dir or ROOT/'weights')/'student_epoch47_ema.pt'
  command += ['--checkpoint',str(checkpoint.resolve())]
  if a.output_dir:command += ['--output-dir',str(a.output_dir.resolve())]
 else:
  if not a.data_root:p.error('--data-root is required')
  weights=(a.weights_dir or ROOT/'training_weights').resolve();out=(a.output_dir or ROOT/'outputs'/a.stage).resolve();out.mkdir(parents=True,exist_ok=True)
  cfg=yaml.safe_load((ROOT/'configs'/f'{a.stage}.yml').read_text(encoding='utf-8'))
  cfg['train']['data_root']=str(a.data_root.resolve());cfg['train']['save_dir']=str(out);cfg['rf_pretrained']=str(weights/'rf-detr-xxlarge.pth')
  required=[Path(cfg['rf_pretrained'])]
  if a.stage=='student':
   for m,ep in [('ir',37),('depth',48)]:cfg['distill'][m+'_teacher_ckpt']=str(weights/f'teacher_{m}_epoch{ep}.pt');required.append(Path(cfg['distill'][m+'_teacher_ckpt']))
  for q in required:
   if not q.is_file():raise FileNotFoundError(q)
  resolved=out/'resolved_config.yml';resolved.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding='utf-8')
  env.update(DECAY_STOP_FILE=str(out/'stop_requested'),DECAY_NIGHT_GUARD='0')
  command=[sys.executable,'-B','-u'];count=len(a.gpus.split(','))
  if count>1:command += ['-m','torch.distributed.run','--standalone',f'--nproc_per_node={count}']
  command += ['-m','aic.best56132.train_entry','train','--config',str(resolved)]
  if a.stage.startswith('teacher_'):command += ['--mono_modality',a.stage.removeprefix('teacher_')]
  if a.resume:command += ['--resume',str(a.resume.resolve())]
 print(' '.join(command),flush=True);subprocess.run(command,cwd=ROOT,env=env,check=True)
if __name__=='__main__':main()

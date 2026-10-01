"""Portable entry point for the selected teachers, student and phase-2 inference."""
from pathlib import Path
import argparse
import os
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('mode',choices=['train','infer'])
    ap.add_argument('--stage',choices=['teacher_ir','teacher_depth','student_best'],default='student_best')
    ap.add_argument('--data-root',type=Path)
    ap.add_argument('--test-root',type=Path)
    ap.add_argument('--weights-dir',type=Path,default=ROOT/'weights')
    ap.add_argument('--output-dir',type=Path)
    ap.add_argument('--checkpoint',type=Path)
    ap.add_argument('--gpus',default='2,3')
    ap.add_argument('--resume',type=Path)
    args=ap.parse_args()
    if args.mode=='train' and args.data_root is None:ap.error('--data-root is required for training')
    if args.mode=='infer' and (args.test_root is None or args.stage!='student_best'):ap.error('inference requires --test-root and stage student_best')
    weights=args.weights_dir.resolve()
    out=(args.output_dir or ROOT/'outputs'/args.stage).resolve()
    out.mkdir(parents=True,exist_ok=True)
    cfg=yaml.safe_load((ROOT/'configs'/f'{args.stage}.yml').read_text(encoding='utf-8'))
    cfg['rf_pretrained']=str(weights/'rf-detr-xxlarge.pth')
    cfg['train']['save_dir']=str(out)
    if args.data_root:cfg['train']['data_root']=str(args.data_root.resolve())
    if 'distill' in cfg:
        cfg['distill']['ir_teacher_ckpt']=str(weights/'teacher_ir_epoch37.pt')
        cfg['distill']['depth_teacher_ckpt']=str(weights/'teacher_depth_epoch48.pt')
    resolved=out/'resolved_config.yml'
    resolved.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding='utf-8')
    required=[Path(cfg['rf_pretrained'])]
    if 'distill' in cfg:required += [Path(cfg['distill'][m+'_teacher_ckpt']) for m in ['ir','depth']]
    for path in required:
        if not path.is_file():raise FileNotFoundError(path)
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpus,PYTHONPATH=str(ROOT),NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1',OMP_NUM_THREADS='4')
    command=[sys.executable,'-u']
    if args.mode=='train':
        count=len(args.gpus.split(','))
        if count>1:command += ['-m','torch.distributed.run','--standalone',f'--nproc_per_node={count}']
    command += ['-m','aic.rf_nas.entry',args.mode,'--config',str(resolved)]
    if args.mode=='train':
        if args.stage.startswith('teacher_'):command += ['--mono_modality',args.stage.removeprefix('teacher_')]
        if args.resume:command += ['--resume',str(args.resume.resolve())]
    else:
        checkpoint=(args.checkpoint or weights/'student_epoch63_ema.pt').resolve()
        if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
        command += ['--checkpoint',str(checkpoint),'--test_root',str(args.test_root.resolve()),
                    '--out_dir',str(out/'predictions'),'--zip_path',str(out/'submission_epoch63_conf025.zip'),
                    '--score_thresh','0.25','--max_boxes','100']
    print(' '.join(command),flush=True)
    subprocess.run(command,cwd=ROOT,env=env,check=True)

if __name__=='__main__':main()

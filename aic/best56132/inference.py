"""Portable FP32 inference of the archived epoch47 EMA and original G6."""
from pathlib import Path
import argparse,hashlib,json,os,time,zipfile
import numpy as np
import torch,yaml
from torch.utils.data import DataLoader
from aic.best56132.views import RawViews,collate
from aic.best56132.g6 import prediction
from aic.rf_nas.model import RFNASDetector
from third_party_dfine import DFINEPostProcessor

CODE=Path(__file__).resolve().parents[2]
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def build_student(cfg,checkpoint):
 cfg=dict(cfg);cfg['inference_only']=True
 model=RFNASDetector(cfg)
 cp=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
 assert cp['epoch']==47,'This release reproduces the selected epoch47 EMA'
 assert cp.get('ema') is not None
 model.load_state_dict(cp['ema'],strict=True)
 model.eval();model.requires_grad_(False)
 return model
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--test-root',required=True,type=Path)
 ap.add_argument('--checkpoint',type=Path,default=CODE/'weights/student_epoch47_ema.pt')
 ap.add_argument('--output-dir',type=Path,default=CODE/'outputs/phase3')
 ap.add_argument('--num-workers',type=int,default=2)
 a=ap.parse_args();assert torch.cuda.is_available(),'CUDA GPU is required for full model inference'
 torch.set_num_threads(4)
 stems=sorted(p.stem for p in (a.test_root/'visible').iterdir() if p.suffix.lower() in ['.png','.jpg','.jpeg'])
 assert len(stems)==len(set(stems)) and len(stems)>0
 cfg=yaml.safe_load((CODE/'configs/student.yml').read_text(encoding='utf-8'))
 cfg['inference_only']=True;cfg['rf_pretrained']=None
 cfg['distill']['ir_teacher_ckpt']=None;cfg['distill']['depth_teacher_ckpt']=None
 cfg['train']['save_dir']=str(a.output_dir.resolve())
 a.output_dir.mkdir(parents=True,exist_ok=True)
 (a.output_dir/'resolved_config.yml').write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding='utf-8')
 model=build_student(cfg,a.checkpoint).cuda()
 assert model.backbone.teacher_init=={},'Inference must not load teachers'
 ds=RawViews(a.test_root,stems,2,.2,False)
 loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=a.num_workers,collate_fn=collate)
 post=DFINEPostProcessor(num_classes=12,use_focal_loss=True,num_top_queries=300)
 rows={};record=None;start=time.monotonic();pred_dir=a.output_dir/'predictions';pred_dir.mkdir(exist_ok=True)
 with torch.no_grad():
  for index,(rgb,ir,depth,targets) in enumerate(loader):
   # The scored run used FP32 inference, without AMP or model/flip ensembling.
   outputs=model(rgb.cuda(),ir.cuda(),depth.cuda())
   results=post(outputs,torch.stack([t['orig_size'] for t in targets]).cuda())
   for result,t in zip(results,targets):
    if t['view']==0:record={'stem':t['stem'],'size':t['full_size'],'views':[]}
    record['views'].append({'window':t['window'],'flipped':False,'boxes':result['boxes'].cpu().numpy(),'scores':result['scores'].cpu().numpy(),'labels':result['labels'].cpu().numpy()})
    if t['view']==4:
     rr=prediction(record);rows[t['stem']]=rr
     text=''.join(str(int(v[0]))+' '+' '.join(f'{float(x):.5f}' for x in v[1:])+'\n' for v in rr)
     (pred_dir/(t['stem']+'.txt')).write_text(text,encoding='utf-8',newline='\n')
   if (index+1)%50==0:print(json.dumps({'done_images':len(rows),'total_images':len(stems),'seconds':time.monotonic()-start}),flush=True)
 assert set(rows)==set(stems)
 dest=a.output_dir/'submission_epoch47_g6_conf025.zip'
 with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
  for stem in sorted(stems):z.write(pred_dir/(stem+'.txt'),stem+'.txt')
 with zipfile.ZipFile(dest) as z:assert z.testzip() is None and set(z.namelist())=={s+'.txt' for s in stems}
 report={'path':str(dest.resolve()),'sha256':sha(dest),'checkpoint_sha256':sha(a.checkpoint),'epoch':47,'weight_source':'EMA','images':len(rows),'boxes':sum(map(len,rows.values())),'score_threshold':.25,'max_boxes':100,'input_size':[880,1600],'views':5,'flip':False,'C1':False,'high_resolution_variant':False,'model_ensemble':False,'HBS_inference':'bypassed','teacher_required':False,'seconds':time.monotonic()-start,'device':torch.cuda.get_device_name(0)}
 (a.output_dir/'complete.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()

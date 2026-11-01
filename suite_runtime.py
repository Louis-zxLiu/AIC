"""Full checkpoint state and clean epoch-boundary resource handoff."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import os,json,torch
import torch.distributed as dist
STATE={}
def atomic(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);t=path.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2));t.replace(path)
def cpu(v):
 if isinstance(v,torch.Tensor):return v.detach().cpu().clone()
 if isinstance(v,dict):return {k:cpu(x) for k,x in v.items()}
 return v
def install(cfg,train):
 STATE['cfg']=cfg;old_distill=train.M2DDistillLoss
 def distill(*a,**kw):
  result=old_distill(*a,**kw);STATE['distill']=result;return result
 train.M2DDistillLoss=distill
 old_capture=train.capture_rng;old_restore=train.restore_rng
 def capture(generator):
  result=old_capture(generator);m=STATE.get('distill')
  if m is not None:result['rank_adapters']={mode:cpu(getattr(m,mode+'_adapters').state_dict()) for mode in ['ir','depth']}
  return result
 def restore(value,generator):
  old_restore(value,generator);m=STATE.get('distill')
  if value.get('handoff_cuda_seed') is not None:
   assert int(os.environ.get('RANK','0'))==1,'Only the newly added rank1 uses a new CUDA RNG stream'
   torch.cuda.manual_seed(int(value['handoff_cuda_seed']))
  if m is not None:
   for mode,values in value['rank_adapters'].items():getattr(m,mode+'_adapters').load_state_dict(values,strict=True)
 train.capture_rng=capture;train.restore_rng=restore
def audit_resume(ckpt,model,optimizer,scheduler,scaler,ema,distill,generator,rank,cfg):
 inner=model.module if hasattr(model,'module') else model
 assert all(torch.equal(v.detach().cpu(),ckpt['model'][k]) for k,v in inner.state_dict().items())
 assert ema.updates==ckpt['ema_updates']
 assert all(torch.equal(v.detach().cpu(),ckpt['ema'][k]) for k,v in ema.state_dict().items())
 assert optimizer.state_dict()['param_groups']==ckpt['optimizer']['param_groups']
 assert scheduler.last_epoch==ckpt['scheduler']['last_epoch']
 for mode,values in ckpt['rng_states'][rank]['rank_adapters'].items():
  assert all(torch.equal(v.detach().cpu(),values[k]) for k,v in getattr(distill,mode+'_adapters').state_dict().items())
 atomic(Path(cfg['train']['save_dir'])/f'resume_verified_epoch{ckpt["epoch"]}_rank{rank}.json',{'passed':True,'epoch':ckpt['epoch'],'world_size':ckpt['world_size'],'optimizer_updates':ckpt['optimizer_updates'],'scheduler_step':scheduler.last_epoch,'EMA_updates':ema.updates,'new_rank1_cuda_rng_seed':ckpt['rng_states'][rank].get('handoff_cuda_seed')})
def boundary_guard(completed_epoch):
 marker=Path(os.environ['DECAY_STOP_FILE']);stop=marker.exists()
 if os.environ.get('DECAY_NIGHT_GUARD')=='1':
  now=datetime.now(ZoneInfo('Asia/Shanghai'))
  stop|=not(now.hour>=22 or now.hour<6 or(now.hour==6 and now.minute<20))
 if dist.is_available() and dist.is_initialized():
  flag=torch.tensor([int(stop)],device=torch.device('cuda',torch.cuda.current_device()));dist.all_reduce(flag,op=dist.ReduceOp.MAX);stop=bool(flag.item())
 if stop:print(f'[early-decay] clean boundary stop after epoch{completed_epoch}',flush=True)
 return stop

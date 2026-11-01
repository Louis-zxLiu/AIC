"""RGB-only illumination augmentation with a separate deterministic RNG stream."""
import hashlib,json,os,pickle,random
from pathlib import Path
import numpy as np
import torch
from aic.data.transforms import SyncMultiModalTransform

POLICY={'enabled':False,'probability':.25,'gamma':[1.5,2.5],'gain':[.4,.8],'active_in_final_epochs':True}
SEED=42

class LowlightTransform(SyncMultiModalTransform):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.lowlight_epoch=0;self._illumination_rng=None;self._illumination_key=None
        self.lowlight_policy=dict(POLICY);self.lowlight_seed=SEED
    def set_epoch(self,epoch):
        super().set_epoch(epoch)
        self.lowlight_epoch=int(epoch);self._illumination_rng=None
    def _rgb(self,visible):
        if not self.train or not self.lowlight_policy['enabled']:return visible
        worker=torch.utils.data.get_worker_info();worker_id=worker.id if worker else 0
        key=(self.lowlight_seed,int(os.environ.get('RANK','0')),worker_id,self.lowlight_epoch)
        if self._illumination_rng is None or self._illumination_key!=key:
            digest=hashlib.sha256(repr(key).encode()).digest()
            self._illumination_rng=random.Random(int.from_bytes(digest[:8],'little'));self._illumination_key=key
        rng=self._illumination_rng
        if rng.random()>=self.lowlight_policy['probability']:return visible
        gamma=rng.uniform(*self.lowlight_policy['gamma']);gain=rng.uniform(*self.lowlight_policy['gain'])
        table=np.rint(np.clip(255*gain*(np.arange(256,dtype=np.float32)/255)**gamma,0,255)).astype(np.uint8)
        return table[visible]
    def __call__(self,visible,*args,**kwargs):
        return super().__call__(self._rgb(visible),*args,**kwargs)

def digest_model(model):
    h=hashlib.sha256()
    for key,value in model.state_dict().items():
        value=value.detach().cpu().contiguous()
        h.update(key.encode());h.update(str((tuple(value.shape),str(value.dtype))).encode());h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def install(cfg,train):
    global POLICY,SEED
    POLICY=dict(cfg['lowlight_augmentation']);SEED=int(cfg['train']['seed'])
    train.SyncMultiModalTransform=LowlightTransform
    original=train.build_tristream_dfine
    def audited_builder(config):
        model=original(config)
        if config.get('experiment',{}).get('audit_initialization',True):
            out=Path(config['train']['save_dir']);out.mkdir(parents=True,exist_ok=True)
            report={'model_sha256':digest_model(model),'torch_rng_sha256':hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                    'python_rng_sha256':hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
                    'numpy_rng_sha256':hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
                    'architecture':model.sampler.choose(False).as_dict(),'seed':SEED,'rank':int(os.environ.get('RANK','0')),
                    'teacher_initialization':model.backbone.teacher_init}
            (out/f'initialization_rank{report["rank"]}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        return model
    train.build_tristream_dfine=audited_builder

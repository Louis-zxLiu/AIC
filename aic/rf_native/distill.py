"""Retain the baseline PKD loss, replacing every HGNet teacher with RF-DETR."""
from torch import nn
import torch
from aic.models.distill import M2DDistillLoss,ResidualFeatureAdapter
from .model import build_native,RFBranch,load_branch

CONFIG=None


class RFDistill(M2DDistillLoss):
    def __init__(self,bb_cfg,ir_teacher_ckpt,depth_teacher_ckpt,**options):
        nn.Module.__init__(self)
        for name,default in [('weight',1.),('ir_weight',1.),('depth_weight',1.),('foreground_weight',2.),
                             ('cross_weight',0.),('reliability_floor',.25),('reliability_power',1.)]:
            setattr(self,name,float(options.get(name,default)))
        for name,default in [('use_depth_valid_mask',True),('use_adapters',True),('dynamic_reliability',False)]:
            setattr(self,name,bool(options.get(name,default)))
        self.global_sample_normalization = bool(options.get('global_sample_normalization', False))
        self.method=options.get('method','pkd_lcm'); self.level_weights=tuple(options.get('level_weights') or [1.])
        assert self.method=='pkd_lcm' and self.level_weights==(1.,) and self.cross_weight==0
        assert options.get('use_teacher_ema') is True
        for modality,path in [('ir',ir_teacher_ckpt),('depth',depth_teacher_ckpt)]:
            native,_=build_native(CONFIG)
            branch=RFBranch(native.backbone,modality)
            del native
            load_branch(branch,path,CONFIG['train'].get('split_sha256'))
            branch.requires_grad_(False); branch.eval()
            setattr(self,modality+'_teacher',branch)
            setattr(self,modality+'_teacher_source','ema')
            adapters=nn.ModuleList([ResidualFeatureAdapter(512,options.get('adapter_reduction',16))])
            setattr(self,modality+'_adapters',adapters)

    @torch.no_grad()
    def _teacher_feats(self,teacher,x,modality):
        teacher.eval()
        features,_,_=teacher(x)
        return [v.tensors for v in features]

    def forward(self,*args,**kwargs):
        # Standardization is sensitive to reduced-precision variance.
        with torch.autocast(device_type=args[0].device.type,enabled=False):
            return super().forward(args[0].float(),args[1].float(),
                                   [v.float() for v in args[2]],[v.float() for v in args[3]],**kwargs)

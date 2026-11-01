"""All three modalities use independent, pretrained RF-DETR 2XL backbones."""
import copy
import math
import torch
from torch import nn
from aic.models.opf_fusion import MultiScaleOPF

MODEL_CONFIG = None


def normalize(x):
    mean=x.new_tensor([.485,.456,.406])[None,:,None,None]
    std=x.new_tensor([.229,.224,.225])[None,:,None,None]
    return (x-mean)/std


class ModalityInput(nn.Module):
    def __init__(self, modality):
        super().__init__(); self.modality=modality
        if modality=='depth':
            self.project=nn.Conv2d(5,3,1)
            with torch.no_grad():
                self.project.weight.zero_(); self.project.bias.zero_()
                self.project.weight[:,0,0,0]=1.

    def forward(self,x):
        if self.modality=='depth': x=self.project(x[:,:5])
        elif x.shape[1]==1: x=x.repeat(1,3,1,1)
        assert x.shape[1]==3
        return normalize(x)


def build_native(cfg, *, initialize_weights=True):
    global MODEL_CONFIG
    if initialize_weights:
        from rfdetr_plus import RFDETR2XLarge
        wrapper=RFDETR2XLarge(pretrain_weights=cfg['rf_pretrained'],num_classes=12,
                             resolution=880,gradient_checkpointing=True)
        native=wrapper.model.model; MODEL_CONFIG=wrapper.model_config
        source=torch.load(cfg['rf_pretrained'],map_location='cpu',weights_only=False)['model']
        current=native.state_dict()
        body=[k for k in source if not k.startswith(('class_embed.','transformer.enc_out_class_embed.'))]
        errors=[k for k in body if k not in current or source[k].shape!=current[k].shape or not torch.equal(source[k],current[k])]
        if errors: raise RuntimeError(f'Pretrained RF body mismatch: {errors[:10]}')
    else:
        from rfdetr_plus.models.detection import RFDETR2XLargeConfig
        from rfdetr.models.lwdetr import build_model_from_config
        # Build the same graph directly, bypassing the RF wrapper's download/load
        # hooks. The default pretrain filename is only a builder flag disabling
        # DINO hub loading; this builder never opens that file. The caller must
        # strictly load the complete student checkpoint before any prediction.
        MODEL_CONFIG=RFDETR2XLargeConfig(num_classes=12,resolution=880,
                                        gradient_checkpointing=True)
        native=build_model_from_config(MODEL_CONFIG)
        body=[]
    hidden=native.class_embed.in_features
    def head():
        layer=nn.Linear(hidden,12); nn.init.constant_(layer.bias,-math.log(99.)); return layer
    native.class_embed=head()
    native.transformer.enc_out_class_embed=nn.ModuleList([head() for _ in range(native.group_detr)])
    return native,{'architecture':'RF-DETR 2XL','exact_pretrained_body_tensors':len(body),
                   'num_classes':12,'initialize_weights':initialize_weights}


class RFBranch(nn.Module):
    def __init__(self,native,modality):
        super().__init__(); self.input=ModalityInput(modality); self.native=native

    def forward(self,x):
        from rfdetr.utilities.tensors import nested_tensor_from_tensor_list
        return self.native(nested_tensor_from_tensor_list(self.input(x)))


def load_branch(branch,path,split_hash):
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    identity=ckpt.get('experiment_identity')
    if split_hash is None:
        # Full-data retraining deliberately has no validation split identity.
        # Reject a split-bound teacher so the two provenance modes cannot be
        # mixed accidentally, while accepting full-data checkpoints.
        assert not identity or not identity.get('split_sha256'), 'Expected a full-data teacher checkpoint'
    else:
        assert identity and identity.get('split_sha256')==split_hash,'Teacher split mismatch'
    assert ckpt.get('ema') is not None,'Teacher EMA is required'
    state=ckpt['ema']; mapped={}
    for key,value in state.items():
        if key.startswith('detector.backbone.'):
            mapped['native.'+key[len('detector.backbone.'):]]=value
        elif key.startswith('input.'):
            mapped[key]=value
    branch.load_state_dict(mapped,strict=True)
    return {'checkpoint':path,'source':'ema','loaded_tensors':len(mapped)}


class FusionBackbone(nn.Module):
    def __init__(self,native,cfg):
        super().__init__(); self.rgb=native
        self.ir=RFBranch(copy.deepcopy(native),'ir')
        self.depth=RFBranch(copy.deepcopy(native),'depth')
        self.teacher_init={}
        for mode in ['ir','depth']:
            self.teacher_init[mode]=load_branch(getattr(self,mode),cfg['distill'][mode+'_teacher_ckpt'],cfg['train'].get('split_sha256'))
        self.opf=MultiScaleOPF([512],num_heads=8,sr_ratios=[2],hard_depth_gate=True)
        self.active=None; self.saved=None

    def forward(self,samples):
        from rfdetr.utilities.tensors import NestedTensor
        rgb,pos,cross=self.rgb(samples)
        ir,depth=self.active
        ir_features,_,_=self.ir(ir); depth_features,_,_=self.depth(depth)
        assert len(rgb)==len(ir_features)==len(depth_features)==1
        rf=[v.tensors for v in rgb]; inf=[v.tensors for v in ir_features]; df=[v.tensors for v in depth_features]
        assert rf[0].shape==inf[0].shape==df[0].shape
        fused=self.opf(rf,inf,df,depth_valid_mask=depth[:,4:5])
        self.saved=(rf,inf,df)
        return [NestedTensor(fused[0],rgb[0].mask)],pos,cross


class RFDetector(nn.Module):
    def __init__(self,cfg,modality=None,frozen_backbone=None):
        super().__init__(); assert frozen_backbone is None
        self.modality=modality
        self.detector,self.pretrained_report=build_native(cfg)
        self.input=ModalityInput(modality or 'rgb')
        self.ir_alignment=None
        if modality is None:
            self.detector.backbone=FusionBackbone(self.detector.backbone,cfg)
            self.pretrained_report['teacher_initialization']=self.detector.backbone.teacher_init
            self.pretrained_report['independent_backbones']=True
        self.pretrained_report['modality']=modality or 'rgb_ir_depth'

    @property
    def backbone(self):return self.detector.backbone

    def forward(self,rgb,ir,depth,targets=None,return_backbone_feats=False):
        if self.modality:
            x={'ir':ir,'depth':depth,'rgb':rgb}[self.modality]
            return self.detector(self.input(x),targets)
        self.backbone.active=(ir,depth)
        try:
            out=self.detector(self.input(rgb),targets); features=self.backbone.saved
        finally:
            self.backbone.active=None; self.backbone.saved=None
        return (out,*features) if return_backbone_feats else out


class NativeCriterion(nn.Module):
    def __init__(self,**unused):
        super().__init__()
        from rfdetr.config import TrainConfig
        from rfdetr.models import build_criterion_from_config
        self.native,_=build_criterion_from_config(MODEL_CONFIG,TrainConfig(dataset_dir='.'))
        self.native.num_classes=12

    def forward(self,outputs,targets):
        def fp32(value):
            if isinstance(value,torch.Tensor): return value.float() if value.is_floating_point() else value
            if isinstance(value,dict): return {k:fp32(v) for k,v in value.items()}
            if isinstance(value,list): return [fp32(v) for v in value]
            return value
        with torch.autocast(device_type=outputs['pred_logits'].device.type,enabled=False):
            losses=self.native(fp32(outputs),targets)
        return {k:v*self.native.weight_dict[k] for k,v in losses.items() if k in self.native.weight_dict}


def param_groups(model,lr,backbone_lr_mult,weight_decay,**unused):
    groups={}
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        if '.opf.' in name or '.input.' in name or name.startswith('input.') or 'class_embed' in name:rate=lr
        elif '.backbone.' in name:rate=lr*backbone_lr_mult
        else:rate=lr*.1
        decay=0. if p.ndim<=1 or 'norm' in name.lower() or name.endswith('.bias') else weight_decay
        groups.setdefault((rate,decay),[]).append(p)
    return [dict(params=params,lr=rate,weight_decay=decay) for (rate,decay),params in groups.items()]

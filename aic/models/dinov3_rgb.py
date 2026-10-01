"""Large DINOv3 RGB encoders with the existing HGNet IR/Depth interface."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


def tensor_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class ViTPyramid(nn.Module):
    def __init__(self, channels=1024):
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv2d(channels, 384, 1) for _ in range(3)])
        self.scales = nn.ModuleList([nn.ConvTranspose2d(384, 384, 2, 2), nn.Identity(), nn.MaxPool2d(2, 2)])
        self.refine = nn.ModuleList([
            nn.Sequential(nn.Conv2d(384, 384, 3, padding=1, bias=False), LayerNorm2d(384), nn.Conv2d(384, out, 1))
            for out in (512, 1024, 2048)
        ])

    def forward(self, x):
        return [refine(scale(proj(x))) for proj, scale, refine in zip(self.projections, self.scales, self.refine)]


class DINOv3RGB(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        from transformers.utils import logging
        logging.disable_progress_bar()
        self.kind = cfg['type']
        self.use_checkpoint = bool(cfg.get('activation_checkpointing', True))
        path = cfg['weights']
        config = AutoConfig.from_pretrained(path, local_files_only=True)
        if self.kind == 'dinov3_convnext_large':
            if config.model_type != 'dinov3_convnext' or list(config.hidden_sizes) != [192,384,768,1536] or list(config.depths) != [3,3,27,3]:
                raise ValueError('expected DINOv3 ConvNeXt-L')
        elif self.kind == 'dinov3_vitl16':
            if config.model_type != 'dinov3_vit' or config.hidden_size != 1024 or config.num_hidden_layers != 24 or config.patch_size != 16:
                raise ValueError('expected standard DINOv3 ViT-L/16')
            config._attn_implementation = 'sdpa'
        else:
            raise ValueError(f'unsupported RGB backbone {self.kind}')
        if cfg.get('load_weights', True):
            self.model, info = AutoModel.from_pretrained(path, config=config, local_files_only=True, output_loading_info=True)
            if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
                raise RuntimeError(f'DINOv3 load is not complete: {info}')
        else:
            self.model = AutoModel.from_config(config)
        if self.kind == 'dinov3_convnext_large':
            self.bridge = nn.ModuleList([nn.Sequential(nn.Conv2d(c, o, 1, bias=False), LayerNorm2d(o)) for c,o in zip((384,768,1536),(512,1024,2048))])
            # The classification/global norm is not used by native stage outputs.
            self.model.layer_norm.requires_grad_(False)
        else:
            self.bridge = ViTPyramid()
            if self.use_checkpoint:
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            # No masked-image pretraining objective in detection.
            if getattr(self.model.embeddings, 'mask_token', None) is not None:
                self.model.embeddings.mask_token.requires_grad_(False)
        self.register_buffer('mean', torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([.229,.224,.225]).view(1,3,1,1))

    def forward(self, rgb):
        h,w = rgb.shape[-2:]
        if h % 32 or w % 32:
            raise ValueError('DINOv3 detection requires canvas sides divisible by 32')
        x = (rgb - self.mean) / self.std
        if self.kind == 'dinov3_convnext_large':
            outputs = []
            stages = self.model.stages if hasattr(self.model, 'stages') else self.model.model.stages
            for i, stage in enumerate(stages):
                x = checkpoint(stage, x, use_reentrant=False) if self.training and self.use_checkpoint else stage(x)
                if i:
                    outputs.append(self.bridge[i-1](x))
        else:
            tokens = self.model(pixel_values=x).last_hidden_state
            count = (h//16)*(w//16)
            special = 1 + self.model.config.num_register_tokens
            if tokens.shape[1] != count + special:
                raise ValueError('unexpected DINOv3 patch/register layout')
            grid = tokens[:, special:].transpose(1,2).reshape(rgb.shape[0],1024,h//16,w//16)
            outputs = self.bridge(grid)
        expected = [(h//s,w//s) for s in (8,16,32)]
        if [tuple(v.shape[-2:]) for v in outputs] != expected:
            raise RuntimeError('RGB pyramid has incorrect strides')
        return outputs


class HeterogeneousBackbone(nn.Module):
    def __init__(self, hgnet, rgb_cfg):
        super().__init__()
        if not hgnet.share_stages or hgnet.out_channels != [512,1024,2048]:
            raise ValueError('DINOv3 experiments require shared HGNetV2-B5 IR/Depth stages')
        self.stem_ir, self.stem_depth = hgnet.stem_ir, hgnet.stem_depth
        self.stages_ir, self.stages_depth = hgnet.stages_ir, hgnet.stages_depth
        self.depth_in_chs = hgnet.depth_in_chs
        self.out_channels = hgnet.out_channels
        self.return_idx = hgnet.return_idx
        self.rgb = DINOv3RGB(rgb_cfg)
        self.pretrained_param_ids = {id(p) for p in self.stages_ir.parameters()} | {id(p) for p in self.rgb.model.parameters()}
        self.use_checkpoint = bool(rgb_cfg.get('activation_checkpointing', True))

    def _modal(self, x, stem):
        x = stem(x)
        outputs=[]
        for i, stage in enumerate(self.stages_ir):
            x = checkpoint(stage, x, use_reentrant=False) if self.training and self.use_checkpoint else stage(x)
            if i in self.return_idx: outputs.append(x)
        return outputs

    def forward(self, rgb, ir, depth):
        return self.rgb(rgb), self._modal(ir, self.stem_ir), self._modal(depth[:,:self.depth_in_chs], self.stem_depth)


def attach_rgb_backbone(model, cfg):
    # Build the common detector FIRST with one fixed RNG stream. The parent
    # builder uses fork_rng so variant-specific construction cannot change it.
    common = {k:v for k,v in model.state_dict().items() if not k.startswith(('backbone.stem_rgb.','backbone.stages_rgb.'))}
    model.common_init_sha256 = tensor_hash(common)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(cfg.get('initialization_seed',42)))
        model.backbone = HeterogeneousBackbone(model.backbone, cfg['backbone']['rgb'])
    model.pretrained_report['common_init_sha256'] = model.common_init_sha256
    model.pretrained_report['rgb_parameters'] = sum(p.numel() for p in model.backbone.rgb.model.parameters())
    model.pretrained_report['rgb_type'] = cfg['backbone']['rgb']['type']
    return model

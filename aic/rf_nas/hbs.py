"""SET-HBS-inspired training-only smoothing of the tri-modal fused P4.

Reference: HuixinSun/SET, DNResBlock in mmdet/models/detectors/fcos_set.py.
Adaptations: one512-channel fused P4; cell-intersection masks with one-cell boundary
protection; zero-initialized output convolution; average clean/HBS detection
losses. No API, no inference-time feature modification, no extra distillation.
"""
import math
from pathlib import Path

import torch
from torch import nn

from aic.sugon.common import atomic_json


def foreground_mask(targets, height, width, device, padding_cells=1):
    result = torch.zeros((len(targets), 1, height, width), dtype=torch.bool, device=device)
    ys = torch.arange(height, device=device)[None, :, None]
    xs = torch.arange(width, device=device)[None, None, :]
    for i, target in enumerate(targets):
        boxes = target['boxes'].detach().float().to(device)
        if not boxes.numel():
            continue
        cx, cy, bw, bh = boxes.unbind(-1)
        x1 = ((cx - bw / 2) * width).floor().long().clamp(0, width - 1)
        y1 = ((cy - bh / 2) * height).floor().long().clamp(0, height - 1)
        x2 = ((cx + bw / 2) * width).ceil().long().clamp(1, width)
        y2 = ((cy + bh / 2) * height).ceil().long().clamp(1, height)
        x2, y2 = torch.maximum(x2, x1 + 1), torch.maximum(y2, y1 + 1)
        x1, y1 = (x1 - padding_cells).clamp_min(0), (y1 - padding_cells).clamp_min(0)
        x2, y2 = (x2 + padding_cells).clamp_max(width), (y2 + padding_cells).clamp_max(height)
        region = ((xs >= x1[:, None, None]) & (xs < x2[:, None, None]) &
                  (ys >= y1[:, None, None]) & (ys < y2[:, None, None]))
        result[i, 0] = region.any(0)
    return result


class BackgroundSmoother(nn.Module):
    def __init__(self, channels=512, reduction=4, kernel_size=5, padding_cells=1):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, kernel_size, padding=kernel_size // 2))
        nn.init.zeros_(self.conv_block[-1].weight)
        nn.init.zeros_(self.conv_block[-1].bias)
        self.padding_cells = padding_cells
        self.stats = {'calls': 0, 'foreground_fraction_sum': 0., 'background_change_sum': 0.}

    def forward(self, feature, targets, padding_mask=None):
        assert self.training, 'HBS must never be evaluated on GT-free inference'
        b, _, h, w = feature.shape
        assert len(targets) == b
        mask = foreground_mask(targets, h, w, feature.device, self.padding_cells)
        if padding_mask is not None:
            mask = mask | padding_mask[:, None]
        background = feature * (~mask).to(feature.dtype)
        transformed = background + self.conv_block(background)
        result = torch.where(mask, feature, transformed)
        self.stats['calls'] += 1
        self.stats['foreground_fraction_sum'] += float(mask.float().mean().detach())
        self.stats['background_change_sum'] += float((result.detach() - feature.detach()).abs().mean())
        return result


def install(model, cfg):
    block = cfg['hbs']
    assert block == {'enabled': True, 'channels': 512, 'reduction': 4, 'kernel_size': 5,
                     'padding_cells': 1, 'clean_loss_weight': .5, 'hbs_loss_weight': .5,
                     'location': 'opf_fused_p4', 'api_enabled': False}
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(cfg['train']['seed'] + 7927)
        smoother = BackgroundSmoother(block['channels'], block['reduction'], block['kernel_size'], block['padding_cells'])
    model.backbone.hbs_smoother = smoother
    model._hbs_grad_stats = {}
    from aic.models.dinov3_rgb import tensor_hash
    return {'module': 'HBS-inspired', 'parameters': sum(p.numel() for p in smoother.parameters()),
            'new_parameters': sum(p.numel() for p in smoother.parameters()),
            'baseline_subset_sha256': tensor_hash({k: v for k, v in model.state_dict().items() if '.hbs_' not in k}),
            'zero_function_at_init': True, 'inference_bypass': True,
            'reference': 'HuixinSun/SET DNResBlock; modified mask and normalized dual-view detection objective'}


def record_gradients(model):
    inner = model.module if hasattr(model, 'module') else model
    stats = inner._hbs_grad_stats
    for name, parameter in inner.named_parameters():
        if '.hbs_' not in name:
            continue
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        value = parameter.grad.detach().float().abs().max().item()
        row = stats.setdefault(name, {'updates': 0, 'nonzero_updates': 0, 'maximum': 0.})
        row['updates'] += 1
        row['nonzero_updates'] += int(value > 0)
        row['maximum'] = max(row['maximum'], value)


def epoch_record(model, cfg, epoch, rank, updates):
    inner = model.module if hasattr(model, 'module') else model
    smoother = inner.backbone.hbs_smoother
    atomic_json(Path(cfg['train']['save_dir']) / 'hbs' / f'epoch{epoch}_rank{rank}.json',
                {'epoch': epoch, 'rank': rank, 'optimizer_updates': updates,
                 'feature_stats': smoother.stats, 'gradients': inner._hbs_grad_stats})
    inner._hbs_grad_stats = {}
    smoother.stats = {'calls': 0, 'foreground_fraction_sum': 0., 'background_change_sum': 0.}

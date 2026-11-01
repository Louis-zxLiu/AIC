"""Balanced architecture sampling, synchronized by optimizer-update index."""
import hashlib
import json
import math
from pathlib import Path
import random


def architecture_key(arch):
    return 'w{windows_rgb}{windows_ir}{windows_depth}_d{decoder_layers}_q{queries}_o{opf_depth}'.format(**arch)


def scheduled_index(update, size, seed=42, probe=False):
    if probe:
        return update % size
    # One baseline anchor followed by four uniformly cycled alternatives.
    slot = update % 5
    if slot == 0:
        return None
    index = (update // 5) * 4 + slot - 1
    cycle, offset = divmod(index, size)
    order = list(range(size))
    random.Random(seed + cycle).shuffle(order)
    return order[offset]


def choose(sampler, training, architecture_cls):
    cfg = sampler.cfg
    if not training:
        selected = cfg['eval_arch']
    else:
        step = int(sampler.step.item())
        per_epoch = sampler.runtime_steps_per_epoch
        accum = sampler.runtime_accum
        warm_steps = int(cfg.get('warmup_arch_epochs', 2)) * per_epoch
        if step < warm_steps:
            selected = cfg['warmup_arch']
        else:
            update = (step - warm_steps) // accum
            index = scheduled_index(update, len(cfg['architecture_manifest']),
                                    cfg.get('sampling_seed', 42), cfg.get('probe', False))
            selected = cfg['eval_arch'] if index is None else cfg['architecture_manifest'][index]['arch']
        key = architecture_key(selected)
        if not hasattr(sampler, 'coverage'):
            sampler.coverage = {}
        sampler.coverage[key] = sampler.coverage.get(key, 0) + 1
        sampler.step.add_(1)
    sampler.last = architecture_cls(**selected)
    return sampler.last


def epoch_artifacts(model, ckpt, cfg, epoch):
    """Keep lightweight epoch55/63 and best weights, with no optimizer duplication."""
    import torch
    from aic.sugon.common import atomic_json
    folder = Path(cfg['train']['save_dir'])
    inner = model.module if hasattr(model, 'module') else model
    sampler = inner.sampler
    if cfg.get('nas', {}).get('expanded_search'):
        atomic_json(folder / 'coverage.json', {'epoch': epoch, 'steps': int(sampler.step.item()),
                    'counts': getattr(sampler, 'coverage', {}),
                    'steps_per_epoch': sampler.runtime_steps_per_epoch, 'accumulation': sampler.runtime_accum})
    curves = json.loads((folder / 'curves.json').read_text())
    vals = curves['eval_metrics'].get('mAP@50-95', [])
    is_best = bool(vals) and vals[-1] >= max(vals)
    compact = {k: ckpt[k] for k in ['epoch', 'ema', 'ema_updates', 'experiment_identity',
                                   'world_size', 'optimizer_updates'] if k in ckpt}
    compact['weight_source'] = 'ema'
    compact['purpose'] = 'Completed-stage inference/search weights; not an optimizer resume checkpoint'
    for name in ([f'ema_epoch{epoch}.pt'] if epoch in cfg['train'].get('nas_ema_epochs', [63]) else []) + (['best_ema.pt'] if is_best else []):
        target = folder / name
        temp = target.with_suffix('.pt.tmp')
        torch.save(compact, temp)
        temp.replace(target)
    # This snapshot only reads gradients; it does not alter optimization.
    gain = getattr(inner.backbone, 'opf2_gain', None)
    if gain is not None:
        atomic_json(folder / 'opf2_gate.json', {'epoch': epoch, 'gain': float(gain.detach().sigmoid().cpu())})

"""Pure schedule and read-only initialization audit for the serial KD experiment."""
import hashlib
import json
import math
from pathlib import Path


def decay_scale(epoch, step, steps_per_epoch, start_epoch, end_epoch, final_scale):
    if not 0 <= start_epoch < end_epoch or not 0 <= final_scale <= 1:
        raise ValueError('Invalid distillation decay interval')
    if epoch < start_epoch:
        return 1.0
    progress = min(max(((epoch - start_epoch) * steps_per_epoch + step) /
                       max((end_epoch - start_epoch) * steps_per_epoch - 1, 1), 0.0), 1.0)
    return final_scale + (1.0 - final_scale) * 0.5 * (1.0 + math.cos(math.pi * progress))


def record_initialization(model, distill, save_dir):
    import torch
    def digest(state):
        h = hashlib.sha256()
        for key, value in sorted(state.items()):
            value = value.detach().cpu().contiguous()
            h.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode())
            h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        return h.hexdigest()
    module = model.module if hasattr(model, 'module') else model
    record = {'student_sha256': digest(module.state_dict()),
              'adapters_sha256': digest({**{'ir.' + k: v for k, v in distill.ir_adapters.state_dict().items()},
                                        **{'depth.' + k: v for k, v in distill.depth_adapters.state_dict().items()}})}
    path = Path(save_dir) / 'initialization_digest.json'
    path.write_text(json.dumps(record, indent=2))
    return record

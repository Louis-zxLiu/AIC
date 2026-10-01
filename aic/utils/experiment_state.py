"""Identity and per-rank random state for reproducible experiment resumes."""
import random
import numpy as np
import torch
from aic.sugon.common import canonical_hash


def identity(cfg):
    return {'config_sha256':canonical_hash(cfg), 'split_sha256':cfg['train']['split_sha256']}


def capture_rng(generator):
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            'loader':generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu()); generator.set_state(state['loader'].cpu())
    if state['cuda'] is not None: torch.cuda.set_rng_state(state['cuda'].cpu())


def accumulation_window(step, steps, accumulation):
    start = (step // accumulation)*accumulation
    size = min(accumulation, steps-start)
    return size, (step+1 == start+size)

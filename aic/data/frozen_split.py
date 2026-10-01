"""Explicit frozen sample identities for the 703 DINOv3 experiments."""
import json
from pathlib import Path

from aic.sugon.common import canonical_hash
from aic.sugon.split import validate_split


def load_frozen_split(samples, path, data_root, expected_hash):
    split = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    validate_split(split, data_root)
    if split['sha256'] != expected_hash:
        raise ValueError('configured split hash differs from frozen manifest')
    by_stem = {sample.stem: sample for sample in samples}
    if len(by_stem) != len(samples) or set(by_stem) != set(split['train'] + split['val']):
        raise ValueError('scanned samples must exactly match the frozen manifest')
    return ([by_stem[s] for s in split['train']], [by_stem[s] for s in split['val']],
            {'split_sha256': split['sha256'], 'train_count': 1600, 'val_count': 400})

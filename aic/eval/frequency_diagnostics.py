"""Read-only validation artifacts; supports the frozen 703 evaluator API."""
from collections import Counter
import json
from pathlib import Path
from aic.sugon.common import atomic_json


def save_diagnostics(cfg, result, predictions, ground_truths):
    root = Path(cfg['train']['save_dir']) / 'validation'
    epoch = cfg['_eval_epoch']
    pred = {s: [[int(c), b.tolist(), float(score)] for c, b, score in rows]
            for s, rows in predictions.items()}
    gt = {s: [[int(c), b.tolist()] for c, b in rows] for s, rows in ground_truths.items()}
    assert set(pred) == set(gt) and len(pred) == 400
    gt_path = root / 'ground_truths.json'
    if gt_path.exists():
        assert json.loads(gt_path.read_text()) == gt, 'Validation targets changed'
    else:
        atomic_json(gt_path, gt)
    atomic_json(root / f'epoch{epoch}_predictions.json', pred)
    counts = Counter(c for rows in gt.values() for c, box in rows)
    metrics = {'epoch': epoch, 'images': 400, 'mAP': result.map_50_95,
               'AP50': result.map_50, 'AP75': result.map_75,
               'per_class_ap': result.per_class_ap50_95,
               'per_class_ap75': result.per_class_ap75,
               'per_class_support': {c: counts[c] for c in range(12)}}
    atomic_json(root / f'epoch{epoch}_metrics.json', metrics)
    return metrics

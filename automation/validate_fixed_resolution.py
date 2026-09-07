"""Evaluate one checkpoint at fixed rectangular resolutions on fold0 validation.

This is the low-risk Elastic-DETR screening experiment: no model or training
changes, only the validation resize size is changed. Each invocation handles
one or more sizes and writes one JSON result per size.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from aic.data.dataset import MultiModalDetDataset, collate_fn, kfold_split, scan_samples
from aic.data.transforms import SyncMultiModalTransform
from aic.eval.map_evaluator import evaluate, xywh_to_xyxy
from aic.models.tristream_dfine import build_tristream_dfine
from third_party_dfine import DFINEPostProcessor


def evaluate_size(model, cfg, checkpoint, size_hw, device):
    h, w = size_hw
    samples = scan_samples(cfg["train"]["data_root"], require_labels=True)
    _, val_samples = kfold_split(
        samples, n_folds=cfg["train"]["n_folds"], fold=cfg["train"]["fold"]
    )
    ds = MultiModalDetDataset(
        val_samples,
        SyncMultiModalTransform(size=(h, w), train=False),
        has_labels=True,
        disp8_mode=cfg["train"].get("disp8_mode", "use"),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)
    postproc = DFINEPostProcessor(
        num_classes=cfg["num_classes"],
        use_focal_loss=True,
        num_top_queries=cfg["decoder"]["num_queries"],
    )

    predictions: dict[str, list] = {}
    ground_truths: dict[str, list] = {}
    model.eval()
    with torch.no_grad():
        for rgb, ir, depth, targets in loader:
            outputs = model(rgb.to(device), ir.to(device), depth.to(device))
            orig_sizes = torch.stack([t["orig_size"] for t in targets]).to(device)
            results = postproc(outputs, orig_sizes)
            for i, target in enumerate(targets):
                stem = target["stem"]
                ow, oh = target["orig_size"].tolist()
                scale = np.array([ow, oh, ow, oh], dtype=np.float32)
                boxes = results[i]["boxes"].cpu().numpy() / scale
                scores = results[i]["scores"].cpu().numpy()
                labels = results[i]["labels"].cpu().numpy()
                predictions[stem] = [
                    (int(labels[j]), boxes[j], float(scores[j])) for j in range(len(labels))
                ]
                gt_boxes = target["boxes"].numpy()
                gt_labels = target["labels"].numpy()
                gt_xyxy = xywh_to_xyxy(gt_boxes) if gt_boxes.size else gt_boxes
                ground_truths[stem] = [
                    (int(gt_labels[j]), gt_xyxy[j]) for j in range(len(gt_labels))
                ]

    result = evaluate(predictions, ground_truths, class_ids=list(range(cfg["num_classes"])))
    return {
        "checkpoint": str(checkpoint),
        "size": [h, w],
        "num_val": len(val_samples),
        "mAP@50-95": result.map_50_95,
        "mAP@50": result.map_50,
        "mAP@75": result.map_75,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sizes", nargs="+", required=True, help="H,W pairs, e.g. 640,1120")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_tristream_dfine(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("ema", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for raw in args.sizes:
        h, w = (int(x) for x in raw.split(","))
        print(f"[start] size={[h, w]} device={device}", flush=True)
        result = evaluate_size(model, cfg, args.checkpoint, (h, w), device)
        out = out_dir / f"size_{h}x{w}.json"
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

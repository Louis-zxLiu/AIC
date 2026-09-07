"""Evaluate one single-modality teacher on a deterministic held-out fold."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from aic.data.dataset import MultiModalDetDataset, collate_fn, kfold_split, scan_samples
from aic.data.transforms import SyncMultiModalTransform
from aic.eval.map_evaluator import evaluate, xywh_to_xyxy
from aic.models.tristream_dfine import build_single_modality_dfine
from third_party_dfine import DFINEPostProcessor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--modality", required=True, choices=["ir", "depth"])
    parser.add_argument("--n_folds", type=int, default=20)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["train"]
    samples = scan_samples(train_cfg["data_root"], require_labels=True)
    _, val_samples = kfold_split(
        samples,
        n_folds=args.n_folds,
        fold=args.fold,
        seed=args.seed,
    )
    if len(val_samples) != 100:
        raise RuntimeError(f"expected 100 validation samples, found {len(val_samples)}")

    split_hash = hashlib.sha256("\n".join(s.stem for s in val_samples).encode()).hexdigest()
    transform = SyncMultiModalTransform(size=cfg["image_size"], train=False)
    dataset = MultiModalDetDataset(
        val_samples,
        transform,
        has_labels=True,
        disp8_mode=train_cfg.get("disp8_mode", "use"),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_single_modality_dfine(cfg, args.modality).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and checkpoint.get("ema") is not None:
        state = checkpoint["ema"]
        weight_source = "ema"
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
        weight_source = "model"
    else:
        state = checkpoint
        weight_source = "raw"
    model.load_state_dict(state, strict=True)
    model.eval()

    postprocessor = DFINEPostProcessor(
        num_classes=cfg["num_classes"],
        use_focal_loss=True,
        num_top_queries=cfg["decoder"]["num_queries"],
    )
    predictions: dict[str, list] = {}
    ground_truths: dict[str, list] = {}

    with torch.inference_mode():
        for rgb, ir, depth, targets in loader:
            outputs = model(rgb.to(device), ir.to(device), depth.to(device))
            orig_sizes = torch.stack([t["orig_size"] for t in targets]).to(device)
            results = postprocessor(outputs, orig_sizes)

            for index, target in enumerate(targets):
                stem = target["stem"]
                width, height = target["orig_size"].tolist()
                boxes = results[index]["boxes"].cpu().numpy()
                scores = results[index]["scores"].cpu().numpy()
                labels = results[index]["labels"].cpu().numpy()
                boxes /= np.array([width, height, width, height])
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
    record = {
        "checkpoint": str(Path(args.checkpoint)),
        "weight_source": weight_source,
        "modality": args.modality,
        "validation": {
            "n_folds": args.n_folds,
            "fold": args.fold,
            "seed": args.seed,
            "samples": len(val_samples),
            "sha256": split_hash,
        },
        "metrics": {
            "mAP@50-95": result.map_50_95,
            "mAP@50": result.map_50,
            "mAP@75": result.map_75,
        },
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

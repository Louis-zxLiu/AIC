"""Scan validation score thresholds and submit the best one for DEIM-MAL epoch39."""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from aic.data.dataset import MultiModalDetDataset, collate_fn, kfold_split, scan_samples
from aic.data.transforms import SyncMultiModalTransform
from aic.eval.map_evaluator import evaluate, xywh_to_xyxy
from aic.eval.submission_validator import validate_submission_dir
from aic.models.tristream_dfine import build_tristream_dfine
from third_party_dfine import DFINEPostProcessor


def make_gt(targets):
    gts = {}
    for target in targets:
        boxes = target["boxes"].numpy()
        labels = target["labels"].numpy()
        boxes = xywh_to_xyxy(boxes) if boxes.size else boxes
        gts[target["stem"]] = [(int(labels[i]), boxes[i]) for i in range(len(labels))]
    return gts


def collect_predictions(model, loader, postproc, device):
    predictions, ground_truths = {}, {}
    with torch.inference_mode():
        for rgb, ir, depth, targets in loader:
            outputs = model(rgb.to(device), ir.to(device), depth.to(device))
            sizes = torch.stack([target["orig_size"] for target in targets]).to(device)
            results = postproc(outputs, sizes)
            if loader.dataset.has_labels:
                ground_truths.update(make_gt(targets))
            for i, target in enumerate(targets):
                width, height = target["orig_size"].tolist()
                boxes = results[i]["boxes"].cpu().numpy()
                boxes = boxes / np.array([width, height, width, height], dtype=np.float32)
                scores = results[i]["scores"].cpu().numpy()
                labels = results[i]["labels"].cpu().numpy()
                predictions[target["stem"]] = [
                    (int(labels[j]), boxes[j], float(scores[j])) for j in range(len(labels))
                ]
    return predictions, ground_truths


def threshold_predictions(raw, threshold, max_boxes):
    filtered = {}
    for stem, detections in raw.items():
        detections = [d for d in detections if d[2] >= threshold]
        filtered[stem] = sorted(detections, key=lambda d: -d[2])[:max_boxes]
    return filtered


def write_submission(predictions, stems, out_dir, zip_path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        with (out_dir / f"{stem}.txt").open("w", encoding="utf-8") as handle:
            for label, box, score in predictions.get(stem, []):
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                bw = box[2] - box[0]
                bh = box[3] - box[1]
                values = [cx, cy, bw, bh]
                values = [float(np.clip(v, 0.0, 1.0)) for v in values]
                handle.write(
                    f"{label} {values[0]:.5f} {values[1]:.5f} {values[2]:.5f} "
                    f"{values[3]:.5f} {float(score):.5f}\n"
                )
    errors = validate_submission_dir(str(out_dir), list(stems))
    if errors:
        raise RuntimeError("submission validation failed: " + "; ".join(errors[:10]))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for stem in stems:
            archive.write(out_dir / f"{stem}.txt", arcname=f"{stem}.txt")
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        if len(names) != len(stems) or any("/" in name for name in names):
            raise RuntimeError(f"zip must contain exactly {len(stems)} root txt files, got {len(names)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train_root", required=True)
    parser.add_argument("--test_root", required=True)
    parser.add_argument("--result_json", required=True)
    parser.add_argument("--submission_dir", required=True)
    parser.add_argument("--zip_path", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    all_samples = scan_samples(args.train_root, require_labels=True)
    train_samples, val_samples = kfold_split(
        all_samples, n_folds=cfg["train"]["n_folds"], fold=cfg["train"]["fold"], seed=args.seed
    )
    test_samples = scan_samples(args.test_root, require_labels=False)
    if len(all_samples) != 2000 or len(train_samples) != 1600 or len(val_samples) != 400 or len(test_samples) != 1000:
        raise RuntimeError(
            f"unexpected split sizes: all={len(all_samples)} train={len(train_samples)} "
            f"val={len(val_samples)} test={len(test_samples)}"
        )

    transform = SyncMultiModalTransform(size=cfg["image_size"], train=False)
    val_ds = MultiModalDetDataset(val_samples, transform, has_labels=True, disp8_mode=cfg["train"].get("disp8_mode", "use"))
    test_ds = MultiModalDetDataset(test_samples, transform, has_labels=False, disp8_mode=cfg["train"].get("disp8_mode", "use"))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_tristream_dfine(cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = checkpoint.get("ema", checkpoint.get("model", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state, strict=True)
    model.eval()
    postproc = DFINEPostProcessor(
        num_classes=cfg["num_classes"], use_focal_loss=True, num_top_queries=cfg["decoder"]["num_queries"]
    )

    print(f"[scan] checkpoint={args.checkpoint} device={device} val={len(val_samples)} test={len(test_samples)}", flush=True)
    val_raw, ground_truths = collect_predictions(model, val_loader, postproc, device)
    thresholds = [0.01, 0.02, 0.03, 0.05, 0.08]
    max_boxes = int(cfg["inference"].get("max_boxes", 100))
    records = []
    for threshold in thresholds:
        pred = threshold_predictions(val_raw, threshold, max_boxes)
        result = evaluate(pred, ground_truths, class_ids=list(range(cfg["num_classes"])))
        record = {
            "threshold": threshold,
            "mAP@50-95": result.map_50_95,
            "mAP@50": result.map_50,
            "mAP@75": result.map_75,
            "boxes": sum(len(v) for v in pred.values()),
        }
        records.append(record)
        print("[scan] " + json.dumps(record, ensure_ascii=False), flush=True)
    best = max(records, key=lambda item: (item["mAP@50-95"], item["mAP@50"], -item["threshold"]))
    print("[scan] best=" + json.dumps(best, ensure_ascii=False), flush=True)

    test_raw, _ = collect_predictions(model, test_loader, postproc, device)
    test_pred = threshold_predictions(test_raw, best["threshold"], max_boxes)
    stems = [sample.stem for sample in test_samples]
    out_dir = Path(args.submission_dir)
    zip_path = Path(args.zip_path)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    write_submission(test_pred, stems, out_dir, zip_path)
    output = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "seed": args.seed,
        "split": {"all": len(all_samples), "train": len(train_samples), "val": len(val_samples), "test": len(test_samples)},
        "thresholds": records,
        "best": best,
        "submission": {"dir": str(out_dir), "zip": str(zip_path), "files": len(stems)},
    }
    Path(args.result_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.result_json).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[done] " + json.dumps(output, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

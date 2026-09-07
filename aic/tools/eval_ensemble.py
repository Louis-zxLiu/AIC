"""Evaluate checkpoint ensembles on the fixed local validation split."""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from aic.data.dataset import MultiModalDetDataset, collate_fn, kfold_split, scan_samples
from aic.data.transforms import SyncMultiModalTransform
from aic.eval.map_evaluator import evaluate, xywh_to_xyxy
from aic.models.tristream_dfine import build_tristream_dfine
from third_party_dfine import DFINEPostProcessor


def load_predictions(cfg, checkpoint, device):
    model = build_tristream_dfine(cfg).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("ema", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()

    samples = scan_samples(cfg["train"]["data_root"], require_labels=True)
    _, val_samples = kfold_split(
        samples,
        n_folds=cfg["train"]["n_folds"],
        fold=cfg["train"]["fold"],
    )
    ds = MultiModalDetDataset(
        val_samples,
        SyncMultiModalTransform(size=cfg["image_size"], train=False),
        has_labels=True,
        disp8_mode=cfg["train"].get("disp8_mode", "use"),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)
    postproc = DFINEPostProcessor(
        num_classes=cfg["num_classes"],
        use_focal_loss=True,
        num_top_queries=cfg["decoder"]["num_queries"],
    )

    predictions, ground_truths = {}, {}
    with torch.no_grad():
        for rgb, ir, depth, targets in loader:
            results = postproc(model(rgb.to(device), ir.to(device), depth.to(device)),
                               torch.stack([t["orig_size"] for t in targets]).to(device))
            for i, target in enumerate(targets):
                image_id = target["stem"]
                w, h = target["orig_size"].tolist()
                boxes = results[i]["boxes"].cpu().numpy() / np.array([w, h, w, h])
                labels = results[i]["labels"].cpu().numpy()
                scores = results[i]["scores"].cpu().numpy()
                predictions[image_id] = [
                    (int(labels[j]), boxes[j], float(scores[j])) for j in range(len(labels))
                ]
                gt_boxes = target["boxes"].numpy()
                gt_labels = target["labels"].numpy()
                gt_xyxy = xywh_to_xyxy(gt_boxes) if gt_boxes.size else gt_boxes
                ground_truths[image_id] = [
                    (int(gt_labels[j]), gt_xyxy[j]) for j in range(len(gt_labels))
                ]
    return predictions, ground_truths


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def fuse_predictions(all_predictions, method, threshold):
    image_ids = all_predictions[0].keys()
    fused = {}
    for image_id in image_ids:
        by_class = defaultdict(list)
        for model_predictions in all_predictions:
            for cls, box, score in model_predictions[image_id]:
                by_class[cls].append((box, score))
        output = []
        for cls, detections in by_class.items():
            detections.sort(key=lambda item: item[1], reverse=True)
            if method == "nms":
                kept = []
                for box, score in detections:
                    if all(iou(box, old_box) < threshold for old_box, _ in kept):
                        kept.append((box, score))
                output.extend((cls, box, score) for box, score in kept)
                continue

            clusters = []
            for box, score in detections:
                best = -1
                best_iou = threshold
                for k, cluster in enumerate(clusters):
                    overlap = iou(box, cluster["box"])
                    if overlap >= best_iou:
                        best, best_iou = k, overlap
                if best < 0:
                    clusters.append({"box": box.copy(), "score_sum": score, "max_score": score})
                else:
                    cluster = clusters[best]
                    total = cluster["score_sum"] + score
                    cluster["box"] = (cluster["box"] * cluster["score_sum"] + box * score) / total
                    cluster["score_sum"] = total
                    cluster["max_score"] = max(cluster["max_score"], score)
            output.extend((cls, c["box"], c["max_score"]) for c in clusters)
        fused[image_id] = output
    return fused


def report(name, predictions, ground_truths, num_classes):
    result = evaluate(predictions, ground_truths, class_ids=list(range(num_classes)))
    print(f"{name}: mAP@50-95={result.map_50_95:.4f} mAP@50={result.map_50:.4f} mAP@75={result.map_75:.4f}")
    return result.map_50_95


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_predictions = []
    ground_truths = None
    for checkpoint in args.checkpoint:
        print(f"[load] {checkpoint}", flush=True)
        predictions, ground_truths = load_predictions(cfg, checkpoint, device)
        all_predictions.append(predictions)
        report(checkpoint, predictions, ground_truths, cfg["num_classes"])
    for method in ("wbf", "nms"):
        for threshold in (0.50, 0.55, 0.60, 0.65, 0.70):
            fused = fuse_predictions(all_predictions, method, threshold)
            report(f"{method}@{threshold:.2f}", fused, ground_truths, cfg["num_classes"])


if __name__ == "__main__":
    main()

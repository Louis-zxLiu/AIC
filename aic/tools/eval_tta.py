"""Validate horizontal-flip TTA for one checkpoint on the fixed local split."""

from __future__ import annotations

import argparse

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from aic.data.dataset import MultiModalDetDataset, collate_fn, kfold_split, scan_samples
from aic.data.transforms import SyncMultiModalTransform
from aic.eval.map_evaluator import evaluate, xywh_to_xyxy
from aic.models.tristream_dfine import build_tristream_dfine
from aic.tools.infer import fuse_hflip_predictions
from third_party_dfine import DFINEPostProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--iou", type=float, nargs="+", default=[0.50, 0.55, 0.60, 0.65, 0.70])
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_tristream_dfine(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("ema", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()

    samples = scan_samples(cfg["train"]["data_root"], require_labels=True)
    _, val_samples = kfold_split(samples, n_folds=cfg["train"]["n_folds"], fold=cfg["train"]["fold"])
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
    score_thresh = cfg["inference"]["score_thresh"]
    max_boxes = cfg["inference"]["max_boxes"]
    baseline, flipped, ground_truths, sizes = {}, {}, {}, {}

    def collect(result, target, out):
        w, h = target["orig_size"].tolist()
        boxes = result["boxes"].cpu().numpy()
        scores = result["scores"].cpu().numpy()
        labels = result["labels"].cpu().numpy()
        keep = scores >= score_thresh
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        order = np.argsort(-scores)[:max_boxes]
        boxes, scores, labels = boxes[order], scores[order], labels[order]
        out[target["stem"]] = [
            (int(label), box / np.array([w, h, w, h], dtype=np.float32), float(score))
            for box, score, label in zip(boxes, scores, labels)
        ]

    with torch.no_grad():
        for rgb, ir, depth, targets in loader:
            rgb, ir, depth = rgb.to(device), ir.to(device), depth.to(device)
            orig_sizes = torch.stack([t["orig_size"] for t in targets]).to(device)
            base_results = postproc(model(rgb, ir, depth), orig_sizes)
            flip_results = postproc(
                model(torch.flip(rgb, [-1]), torch.flip(ir, [-1]), torch.flip(depth, [-1])),
                orig_sizes,
            )
            for i, target in enumerate(targets):
                collect(base_results[i], target, baseline)
                flip_boxes = flip_results[i]["boxes"].cpu().numpy()
                flip_scores = flip_results[i]["scores"].cpu().numpy()
                flip_labels = flip_results[i]["labels"].cpu().numpy()
                w, h = target["orig_size"].tolist()
                if len(flip_boxes):
                    old_x1 = flip_boxes[:, 0].copy()
                    flip_boxes[:, 0] = w - flip_boxes[:, 2]
                    flip_boxes[:, 2] = w - old_x1
                keep = flip_scores >= score_thresh
                flip_boxes, flip_scores, flip_labels = flip_boxes[keep], flip_scores[keep], flip_labels[keep]
                base_boxes = base_results[i]["boxes"].cpu().numpy()
                base_scores = base_results[i]["scores"].cpu().numpy()
                base_labels = base_results[i]["labels"].cpu().numpy()
                base_keep = base_scores >= score_thresh
                base_boxes, base_scores, base_labels = base_boxes[base_keep], base_scores[base_keep], base_labels[base_keep]
                flipped[target["stem"]] = (flip_boxes, flip_scores, flip_labels)
                sizes[target["stem"]] = (w, h)
                gt_boxes = target["boxes"].numpy()
                gt_labels = target["labels"].numpy()
                ground_truths[target["stem"]] = [
                    (int(label), box) for label, box in zip(gt_labels, xywh_to_xyxy(gt_boxes))
                ]

    # The fused test above is recomputed for each threshold below; report baseline first.
    base_result = evaluate(baseline, ground_truths, class_ids=list(range(cfg["num_classes"])))
    print(f"baseline: mAP@50-95={base_result.map_50_95:.4f} mAP@50={base_result.map_50:.4f}")
    for threshold in args.iou:
        fused_predictions = {}
        for stem in baseline:
            b = baseline[stem]
            f = flipped[stem]
            # Reconstruct pixel boxes from normalized baseline predictions using the sample size.
            w, h = sizes[stem]
            b_boxes = np.array([x[1] * np.array([w, h, w, h]) for x in b], dtype=np.float32)
            b_scores = np.array([x[2] for x in b], dtype=np.float32)
            b_labels = np.array([x[0] for x in b], dtype=np.int64)
            f_boxes, f_scores, f_labels = f
            boxes, scores, labels = fuse_hflip_predictions(
                (b_boxes, b_scores, b_labels), (f_boxes, f_scores, f_labels), threshold
            )
            order = np.argsort(-scores)[:max_boxes]
            fused_predictions[stem] = [
                (int(labels[j]), boxes[j] / np.array([w, h, w, h]), float(scores[j])) for j in order
            ]
        result = evaluate(fused_predictions, ground_truths, class_ids=list(range(cfg["num_classes"])))
        print(f"hflip_wbf@{threshold:.2f}: mAP@50-95={result.map_50_95:.4f} mAP@50={result.map_50:.4f}")


if __name__ == "__main__":
    main()

"""Compare conservative test-time augmentation variants on the fixed val split."""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from aic.data.dataset import MultiModalDetDataset, collate_fn, kfold_split, scan_samples
from aic.data.transforms import SyncMultiModalTransform
from aic.eval.map_evaluator import evaluate, xywh_to_xyxy
from aic.models.tristream_dfine import build_tristream_dfine
from third_party_dfine import DFINEPostProcessor


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + ab - inter, 1e-9)


def score_only_merge(base, aug, iou_threshold, alpha):
    """Keep base boxes and only boost scores when an aug view agrees."""
    b_boxes, b_scores, b_labels = base
    a_boxes, a_scores, a_labels = aug
    out_scores = b_scores.copy()
    for i, (box, label) in enumerate(zip(b_boxes, b_labels)):
        best = 0.0
        for a_box, a_score, a_label in zip(a_boxes, a_scores, a_labels):
            if int(a_label) == int(label):
                overlap = iou(box, a_box)
                if overlap >= iou_threshold:
                    best = max(best, float(a_score))
        if best > 0.0:
            out_scores[i] = min(1.0, out_scores[i] + alpha * best)
    return b_boxes.copy(), out_scores, b_labels.copy()


def resize_canvas(x, scale):
    """Scale a fixed-size tensor and center crop/pad it back to the same canvas."""
    _, _, h, w = x.shape
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    if x.shape[1] == 2:
        y0 = F.interpolate(x[:, :1], size=(nh, nw), mode="bilinear", align_corners=False)
        y1 = F.interpolate(x[:, 1:], size=(nh, nw), mode="nearest")
        y = torch.cat([y0, y1], dim=1)
    else:
        y = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
    canvas = torch.zeros_like(x)
    src_top = max(0, (nh - h) // 2)
    src_left = max(0, (nw - w) // 2)
    dst_top = max(0, (h - nh) // 2)
    dst_left = max(0, (w - nw) // 2)
    copy_h, copy_w = min(h, nh), min(w, nw)
    canvas[:, :, dst_top : dst_top + copy_h, dst_left : dst_left + copy_w] = y[
        :, :, src_top : src_top + copy_h, src_left : src_left + copy_w
    ]
    return canvas


def undo_scale(boxes, orig_w, orig_h, canvas_w, canvas_h, scale):
    if len(boxes) == 0:
        return boxes
    boxes = boxes.copy()
    factor = np.array([canvas_w / orig_w, canvas_h / orig_h, canvas_w / orig_w, canvas_h / orig_h])
    net = boxes * factor
    nh, nw = round(canvas_h * scale), round(canvas_w * scale)
    ox, oy = (canvas_w - nw) / 2.0, (canvas_h - nh) / 2.0
    net[:, [0, 2]] = (net[:, [0, 2]] - ox) / scale
    net[:, [1, 3]] = (net[:, [1, 3]] - oy) / scale
    net[:, [0, 2]] = np.clip(net[:, [0, 2]], 0, canvas_w)
    net[:, [1, 3]] = np.clip(net[:, [1, 3]], 0, canvas_h)
    return net / factor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--iou", type=float, nargs="+", default=[0.75, 0.85, 0.90])
    ap.add_argument("--alpha", type=float, nargs="+", default=[0.05, 0.10, 0.15])
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_tristream_dfine(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("ema", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()
    samples = scan_samples(cfg["train"]["data_root"], require_labels=True)
    _, val_samples = kfold_split(samples, cfg["train"]["n_folds"], cfg["train"]["fold"])
    ds = MultiModalDetDataset(
        val_samples,
        SyncMultiModalTransform(size=cfg["image_size"], train=False),
        has_labels=True,
        disp8_mode=cfg["train"].get("disp8_mode", "use"),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)
    post = DFINEPostProcessor(cfg["num_classes"], True, cfg["decoder"]["num_queries"])
    score_thresh, max_boxes = cfg["inference"]["score_thresh"], cfg["inference"]["max_boxes"]
    base, variants, gts, sizes = {}, defaultdict(dict), {}, {}

    def get_pred(result, w, h):
        boxes = result["boxes"].cpu().numpy()
        scores = result["scores"].cpu().numpy()
        labels = result["labels"].cpu().numpy()
        keep = scores >= score_thresh
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        order = np.argsort(-scores)[:max_boxes]
        return boxes[order], scores[order], labels[order]

    with torch.no_grad():
        for rgb, ir, depth, targets in loader:
            rgb, ir, depth = rgb.to(device), ir.to(device), depth.to(device)
            orig_sizes = torch.stack([t["orig_size"] for t in targets]).to(device)
            outputs = {
                "base": model(rgb, ir, depth),
                "hflip": model(torch.flip(rgb, [-1]), torch.flip(ir, [-1]), torch.flip(depth, [-1])),
                "scale095": model(resize_canvas(rgb, .95), resize_canvas(ir, .95), resize_canvas(depth, .95)),
                "scale105": model(resize_canvas(rgb, 1.05), resize_canvas(ir, 1.05), resize_canvas(depth, 1.05)),
                "rgb_dark": model((rgb * .85).clamp(0, 1), ir, depth),
                "rgb_bright": model((rgb * 1.15).clamp(0, 1), ir, depth),
            }
            results = {name: post(out, orig_sizes) for name, out in outputs.items()}
            for i, target in enumerate(targets):
                stem = target["stem"]
                w, h = target["orig_size"].tolist()
                sizes[stem] = (w, h)
                base[stem] = get_pred(results["base"][i], w, h)
                for name in ("hflip", "scale095", "scale105", "rgb_dark", "rgb_bright"):
                    boxes, scores, labels = get_pred(results[name][i], w, h)
                    if name == "hflip":
                        if len(boxes):
                            old_x1 = boxes[:, 0].copy()
                            boxes[:, 0], boxes[:, 2] = w - boxes[:, 2], w - old_x1
                    elif name.startswith("scale"):
                        scale = {"scale095": 0.95, "scale105": 1.05}[name]
                        boxes = undo_scale(boxes, w, h, cfg["image_size"][1], cfg["image_size"][0], scale)
                    variants[name][stem] = (boxes, scores, labels)
                gt_boxes, gt_labels = target["boxes"].numpy(), target["labels"].numpy()
                gts[stem] = [(int(label), box) for label, box in zip(gt_labels, xywh_to_xyxy(gt_boxes))]

    def normalized(pred):
        out = {}
        for stem, (boxes, scores, labels) in pred.items():
            w, h = sizes[stem]
            order = np.argsort(-scores)[:max_boxes]
            out[stem] = [(int(labels[j]), boxes[j] / np.array([w, h, w, h]), float(scores[j])) for j in order]
        return out

    def report(name, pred):
        r = evaluate(normalized(pred), gts, class_ids=list(range(cfg["num_classes"])))
        print(f"{name}: mAP@50-95={r.map_50_95:.4f} mAP@50={r.map_50:.4f}")
        return r.map_50_95

    report("baseline", base)
    for name in ("hflip", "scale095", "scale105", "rgb_dark", "rgb_bright"):
        best = 0.0
        for iou_threshold in args.iou:
            for alpha in args.alpha:
                merged = {stem: score_only_merge(base[stem], variants[name][stem], iou_threshold, alpha) for stem in base}
                best = max(best, report(f"{name} score@iou{iou_threshold:.2f} a{alpha:.2f}", merged))
        print(f"BEST {name}: {best:.4f}")


if __name__ == "__main__":
    main()

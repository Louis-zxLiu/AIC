"""Cache calibrated per-image reliability for one mono-modal teacher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from aic.data.dataset import MultiModalDetDataset, collate_fn, scan_samples
from aic.data.transforms import SyncMultiModalTransform
from aic.models.tristream_dfine import build_single_modality_dfine
from third_party_dfine import DFINEPostProcessor


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x0 = np.maximum(box[0], boxes[:, 0])
    y0 = np.maximum(box[1], boxes[:, 1])
    x1 = np.minimum(box[2], boxes[:, 2])
    y1 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x1 - x0, 0) * np.maximum(y1 - y0, 0)
    box_area = max((box[2] - box[0]) * (box[3] - box[1]), 0)
    boxes_area = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(
        boxes[:, 3] - boxes[:, 1], 0
    )
    return intersection / np.maximum(box_area + boxes_area - intersection, 1e-9)


def matched_gt_confidence(result: dict, target: dict, iou_threshold: float) -> float:
    scores = result["scores"].detach().cpu().numpy()
    labels = result["labels"].detach().cpu().numpy()
    boxes = result["boxes"].detach().cpu().numpy()
    gt_labels = target["labels"].numpy()
    gt_cxcywh = target["boxes"].numpy()
    if len(gt_labels) == 0:
        return 0.5
    width, height = target["orig_size"].tolist()
    gt_boxes = np.empty_like(gt_cxcywh)
    gt_boxes[:, 0] = (gt_cxcywh[:, 0] - gt_cxcywh[:, 2] / 2) * width
    gt_boxes[:, 1] = (gt_cxcywh[:, 1] - gt_cxcywh[:, 3] / 2) * height
    gt_boxes[:, 2] = (gt_cxcywh[:, 0] + gt_cxcywh[:, 2] / 2) * width
    gt_boxes[:, 3] = (gt_cxcywh[:, 1] + gt_cxcywh[:, 3] / 2) * height
    matched = []
    for gt_label, gt_box in zip(gt_labels, gt_boxes):
        candidates = labels == int(gt_label)
        if not candidates.any():
            matched.append(0.0)
            continue
        overlaps = box_iou(gt_box, boxes[candidates])
        valid_scores = scores[candidates][overlaps >= iou_threshold]
        matched.append(float(valid_scores.max()) if len(valid_scores) else 0.0)
    return float(np.mean(matched))


def calibrate(values: dict[str, float]) -> dict[str, float]:
    array = np.asarray(list(values.values()), dtype=np.float64)
    low, high = np.quantile(array, [0.05, 0.95])
    scale = max(float(high - low), 1e-6)
    return {
        stem: float(np.clip((value - low) / scale, 0.0, 1.0))
        for stem, value in values.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--modality", required=True, choices=("ir", "depth"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    samples = scan_samples(cfg["train"]["data_root"], require_labels=True)
    dataset = MultiModalDetDataset(
        samples,
        SyncMultiModalTransform(size=cfg["image_size"], train=False),
        has_labels=True,
        disp8_mode=cfg["train"].get("disp8_mode", "use"),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_single_modality_dfine(cfg, args.modality).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("ema") or checkpoint.get("model") or checkpoint
    model_state = model.state_dict()
    compatible = {
        key: value for key, value in state.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    result = model.load_state_dict(compatible, strict=False)
    active_prefixes = (
        f"backbone.stem_{args.modality}.",
        f"backbone.stages_{args.modality}.",
        "encoder.",
        "decoder.",
    )
    missing_active = [
        key for key in result.missing_keys if key.startswith(active_prefixes)
    ]
    if missing_active or result.unexpected_keys:
        raise RuntimeError(
            f"teacher load incomplete: missing_active={missing_active[:8]} "
            f"unexpected={result.unexpected_keys[:8]}"
        )
    print(
        f"[{args.modality}] checkpoint tensors loaded={len(compatible)}/{len(model_state)}",
        flush=True,
    )
    model.eval()
    postprocessor = DFINEPostProcessor(
        num_classes=cfg["num_classes"], use_focal_loss=True,
        num_top_queries=cfg["decoder"]["num_queries"],
    )
    raw = {}
    with torch.inference_mode():
        for index, (rgb, ir, depth, targets) in enumerate(loader):
            outputs = model(rgb.to(device), ir.to(device), depth.to(device))
            sizes = torch.stack([target["orig_size"] for target in targets]).to(device)
            raw[targets[0]["stem"]] = matched_gt_confidence(
                postprocessor(outputs, sizes)[0], targets[0], args.iou
            )
            if index % 100 == 0:
                print(f"[{args.modality}] {index}/{len(dataset)}", flush=True)
    calibrated = calibrate(raw)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(calibrated, sort_keys=True), encoding="utf-8")
    values = np.asarray(list(calibrated.values()))
    print(f"[{args.modality}] cached={len(calibrated)} mean={values.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()

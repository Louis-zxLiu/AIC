"""Inference for a single-modality D-FINE teacher/checkpoint."""
from __future__ import annotations

import argparse
import os
import zipfile

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from aic.data.dataset import MultiModalDetDataset, collate_fn, scan_samples
from aic.data.transforms import SyncMultiModalTransform
from aic.eval.submission_validator import validate_submission_dir
from aic.models.tristream_dfine import build_single_modality_dfine
from third_party_dfine import DFINEPostProcessor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--modality", required=True, choices=["rgb", "ir", "depth"])
    parser.add_argument("--test_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--zip_path", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_single_modality_dfine(cfg, args.modality).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("ema") if isinstance(ckpt, dict) and ckpt.get("ema") is not None else ckpt
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    model.eval()
    post = DFINEPostProcessor(cfg["num_classes"], True, cfg["decoder"]["num_queries"])
    samples = scan_samples(args.test_root, require_labels=False)
    ds = MultiModalDetDataset(
        samples,
        SyncMultiModalTransform(size=cfg["image_size"], train=False),
        has_labels=False,
        disp8_mode=cfg.get("train", {}).get("disp8_mode", "use"),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)
    with torch.inference_mode():
        for rgb, ir, depth, targets in loader:
            result = post(model(rgb.to(device), ir.to(device), depth.to(device)),
                          torch.stack([t["orig_size"] for t in targets]).to(device))
            for i, target in enumerate(targets):
                stem = target["stem"]
                w, h = target["orig_size"].tolist()
                boxes = result[i]["boxes"].cpu().numpy()
                scores = result[i]["scores"].cpu().numpy()
                labels = result[i]["labels"].cpu().numpy()
                keep = scores >= cfg["inference"]["score_thresh"]
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
                order = np.argsort(-scores)[: cfg["inference"]["max_boxes"]]
                boxes, scores, labels = boxes[order], scores[order], labels[order]
                cx = (boxes[:, 0] + boxes[:, 2]) / 2 / w
                cy = (boxes[:, 1] + boxes[:, 3]) / 2 / h
                bw = (boxes[:, 2] - boxes[:, 0]) / w
                bh = (boxes[:, 3] - boxes[:, 1]) / h
                with open(os.path.join(args.out_dir, stem + ".txt"), "w", encoding="utf-8") as f:
                    for label, x, y, ww, hh, score in zip(labels, cx, cy, bw, bh, scores):
                        f.write(f"{int(label)} {np.clip(x,0,1):.5f} {np.clip(y,0,1):.5f} "
                                f"{np.clip(ww,0,1):.5f} {np.clip(hh,0,1):.5f} {score:.5f}\n")
    errors = validate_submission_dir(args.out_dir, [s.stem for s in samples])
    if errors:
        raise RuntimeError("submission validation failed: " + "; ".join(errors[:5]))
    with zipfile.ZipFile(args.zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for sample in samples:
            zf.write(os.path.join(args.out_dir, sample.stem + ".txt"), arcname=sample.stem + ".txt")
    print(f"[OK] {len(samples)} files; {args.zip_path}", flush=True)


if __name__ == "__main__":
    main()

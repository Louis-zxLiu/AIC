"""推理脚本：读取测试集三模态图像，产出比赛要求格式的预测 txt 文件。

赛题红线（计划.md §五，务必遵守）：
  - 每张测试图必须产出对应文件，即使无检测目标也要写空文件——本脚本对
    scan_samples 返回的每个 stem 都显式 open(..., "w") 一次，即使该图
    检测框数为 0 也会创建空文件，不会因为"没有内容可写"而跳过。
  - 每文件最多 100 行，按 confidence 降序截断。
  - 每行格式: class_id norm_cx norm_cy norm_w norm_h confidence

用法：
    python aic/tools/infer.py --config configs/dfine_tristream.yml \
        --checkpoint outputs/run1/epoch47.pt \
        --test_root <测试集目录> --out_dir outputs/predictions
"""

from __future__ import annotations

import argparse
import os
import zipfile
from collections import defaultdict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from aic.data.dataset import MultiModalDetDataset, collate_fn, scan_samples
from aic.data.transforms import SyncMultiModalTransform
from aic.eval.submission_validator import validate_submission_dir
from aic.models.tristream_dfine import build_tristream_dfine
from third_party_dfine import DFINEPostProcessor


def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return inter / max(area_a + area_b - inter, 1e-9)


def fuse_hflip_predictions(base, flipped, iou_threshold: float, keep_unmatched_flipped: bool = False):
    """Two-view WBF that never merges two detections from the same view."""
    by_class = defaultdict(lambda: [[], []])
    for view_idx, result in enumerate((base, flipped)):
        boxes, scores, labels = result
        for box, score, label in zip(boxes, scores, labels):
            by_class[int(label)][view_idx].append((box, float(score)))

    fused = []
    for label, (base_dets, flip_dets) in by_class.items():
        clusters = [
            {"box": box.copy(), "weight": score, "score": score, "has_flip": False}
            for box, score in sorted(base_dets, key=lambda item: item[1], reverse=True)
        ]
        for box, score in sorted(flip_dets, key=lambda item: item[1], reverse=True):
            best_idx, best_iou = -1, iou_threshold
            for idx, cluster in enumerate(clusters):
                if cluster["has_flip"]:
                    continue
                overlap = box_iou_xyxy(box, cluster["box"])
                if overlap >= best_iou:
                    best_idx, best_iou = idx, overlap
            if best_idx < 0:
                if keep_unmatched_flipped:
                    clusters.append({"box": box.copy(), "weight": score, "score": score, "has_flip": True})
                continue
            cluster = clusters[best_idx]
            total_weight = cluster["weight"] + score
            cluster["box"] = (cluster["box"] * cluster["weight"] + box * score) / total_weight
            cluster["weight"] = total_weight
            cluster["score"] = max(cluster["score"], score)
            cluster["has_flip"] = True

        fused.extend((cluster["box"], cluster["score"], label) for cluster in clusters)

    fused.sort(key=lambda item: item[1], reverse=True)
    if not fused:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    boxes, scores, labels = zip(*fused)
    return np.stack(boxes), np.asarray(scores), np.asarray(labels)


def score_only_hflip_predictions(base, flipped, iou_threshold: float, alpha: float):
    """Keep original boxes and gently boost only flip-consistent scores."""
    boxes, scores, labels = (v.copy() for v in base)
    flip_boxes, flip_scores, flip_labels = flipped
    for i, (box, label) in enumerate(zip(boxes, labels)):
        best = 0.0
        for fbox, fscore, flabel in zip(flip_boxes, flip_scores, flip_labels):
            if int(label) == int(flabel) and box_iou_xyxy(box, fbox) >= iou_threshold:
                best = max(best, float(fscore))
        if best > 0.0:
            scores[i] = min(1.0, scores[i] + alpha * best)
    return boxes, scores, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--zip_path", default=None)
    # 推理分辨率可脱离 config 覆盖：实测 run2/epoch41 在 768 方形上 mAP@50-95 比 640
    # 高 2.4 分（0.5111 → 0.5355），因为 768 恰是多尺度训练档位的上界，仍在分布内；
    # 再往上（1120/1280）就掉出分布外开始下滑。
    parser.add_argument("--image_size", default=None, help="int 或 'h,w'；不传则用 config 值")
    # run2 及更早的权重是在 disp8 深度全零的输入上训出来的，推理必须传 zero 复刻该行为
    # （实测传 use 会让 disp8 子域从 0.5693 掉到 0.5448）。重训后的权重用 use。
    parser.add_argument("--disp8_mode", default=None, choices=["use", "zero"])
    parser.add_argument("--tta_hflip", action="store_true", help="average original and synchronized horizontal-flip views")
    parser.add_argument("--tta_iou", type=float, default=0.60, help="IoU threshold for two-view WBF")
    parser.add_argument("--tta_keep_unmatched", action="store_true", help="keep flip-only detections; disabled is safer")
    parser.add_argument("--tta_score_only_hflip", action="store_true", help="boost scores for flip-consistent original boxes")
    parser.add_argument("--tta_alpha", type=float, default=0.05, help="score boost coefficient for score-only flip TTA")
    parser.add_argument("--score_thresh", type=float, default=None, help="override inference score threshold")
    parser.add_argument("--max_boxes", type=int, default=None, help="override per-image output box limit")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    model = build_tristream_dfine(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "ema" in ckpt:
        # EMA 权重通常比原始权重更稳定（算法设计.md §3.1），优先使用。
        state = ckpt["ema"]
    else:
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    postproc = DFINEPostProcessor(
        num_classes=cfg["num_classes"], use_focal_loss=True, num_top_queries=cfg["decoder"]["num_queries"]
    )

    inf_cfg = cfg["inference"]
    score_thresh = args.score_thresh if args.score_thresh is not None else inf_cfg["score_thresh"]
    max_boxes = args.max_boxes if args.max_boxes is not None else inf_cfg["max_boxes"]

    samples = scan_samples(args.test_root, require_labels=False)
    if not samples:
        raise RuntimeError(f"no samples found under {args.test_root}")

    if args.image_size is None:
        infer_size = cfg["image_size"]
    elif "," in args.image_size:
        infer_size = [int(v) for v in args.image_size.split(",")]
    else:
        infer_size = int(args.image_size)
    disp8_mode = args.disp8_mode or cfg.get("train", {}).get("disp8_mode", "use")
    print(f"[infer] image_size={infer_size} disp8_mode={disp8_mode}")

    # 推理尺寸与训练尺寸不一致会大幅掉分，且每个权重的最优尺寸就是它自己的训练尺寸：
    # 实测 run3（训练 736x1280）在 768 上只有 0.4511，回到 736x1280 是 0.5428（差 9.2 分）；
    # run2（训练 640）反过来在 768 上最好。曾因此错误提交，线上从 54.78 掉到 51.507。
    # 不要跨模型套用"某个尺寸是甜点"的结论——要改推理尺寸必须先在 val 上实测。
    def _norm(s):
        return tuple(s) if isinstance(s, (list, tuple)) else (s, s)

    if _norm(infer_size) != _norm(cfg["image_size"]):
        print(
            f"[infer][警告] 推理尺寸 {infer_size} 与 config 训练尺寸 {cfg['image_size']} 不一致！\n"
            f"[infer][警告] 除非你已在 val 上实测该尺寸更优，否则这通常会掉分（历史上曾因此丢 3.3 线上分）。",
            flush=True,
        )

    tf = SyncMultiModalTransform(size=infer_size, train=False)
    ds = MultiModalDetDataset(samples, tf, has_labels=False, disp8_mode=disp8_mode)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

    written_stems = set()

    with torch.no_grad():
        for rgb, ir, depth, targets in loader:
            rgb, ir, depth = rgb.to(device), ir.to(device), depth.to(device)
            orig_sizes = torch.stack([t["orig_size"] for t in targets]).to(device)
            results = postproc(model(rgb, ir, depth), orig_sizes)
            flipped_results = None
            if args.tta_hflip or args.tta_score_only_hflip:
                flipped_results = postproc(
                    model(torch.flip(rgb, [-1]), torch.flip(ir, [-1]), torch.flip(depth, [-1])),
                    orig_sizes,
                )

            for i, t in enumerate(targets):
                stem = t["stem"]
                w, h = t["orig_size"].tolist()
                boxes_px = results[i]["boxes"].cpu().numpy()
                scores = results[i]["scores"].cpu().numpy()
                labels = results[i]["labels"].cpu().numpy()

                keep = scores >= score_thresh
                boxes_px, scores, labels = boxes_px[keep], scores[keep], labels[keep]

                if flipped_results is not None:
                    flip_boxes = flipped_results[i]["boxes"].cpu().numpy()
                    flip_scores = flipped_results[i]["scores"].cpu().numpy()
                    flip_labels = flipped_results[i]["labels"].cpu().numpy()
                    flip_keep = flip_scores >= score_thresh
                    flip_boxes = flip_boxes[flip_keep]
                    flip_scores = flip_scores[flip_keep]
                    flip_labels = flip_labels[flip_keep]
                    if len(flip_boxes):
                        old_x1 = flip_boxes[:, 0].copy()
                        flip_boxes[:, 0] = w - flip_boxes[:, 2]
                        flip_boxes[:, 2] = w - old_x1
                    if args.tta_score_only_hflip:
                        boxes_px, scores, labels = score_only_hflip_predictions(
                            (boxes_px, scores, labels),
                            (flip_boxes, flip_scores, flip_labels),
                            args.tta_iou,
                            args.tta_alpha,
                        )
                    else:
                        boxes_px, scores, labels = fuse_hflip_predictions(
                            (boxes_px, scores, labels),
                            (flip_boxes, flip_scores, flip_labels),
                            args.tta_iou,
                            args.tta_keep_unmatched,
                        )

                order = np.argsort(-scores)[:max_boxes]
                boxes_px, scores, labels = boxes_px[order], scores[order], labels[order]

                cx = (boxes_px[:, 0] + boxes_px[:, 2]) / 2 / w
                cy = (boxes_px[:, 1] + boxes_px[:, 3]) / 2 / h
                bw = (boxes_px[:, 2] - boxes_px[:, 0]) / w
                bh = (boxes_px[:, 3] - boxes_px[:, 1]) / h
                cx, cy, bw, bh = (np.clip(v, 0.0, 1.0) for v in (cx, cy, bw, bh))

                out_path = os.path.join(args.out_dir, stem + ".txt")
                with open(out_path, "w", encoding="utf-8") as f:
                    for j in range(len(labels)):
                        f.write(
                            f"{int(labels[j])} {cx[j]:.5f} {cy[j]:.5f} {bw[j]:.5f} {bh[j]:.5f} {scores[j]:.5f}\n"
                        )
                written_stems.add(stem)

    errors = validate_submission_dir(args.out_dir, [s.stem for s in samples])
    if errors:
        print(f"[FAIL] 提交格式校验发现 {len(errors)} 个问题：")
        for e in errors[:50]:
            print("  -", e)
        raise SystemExit(1)
    print(f"[OK] {len(written_stems)} 个预测文件写入 {args.out_dir}，格式校验通过。")

    if args.zip_path:
        with zipfile.ZipFile(args.zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for s in samples:
                fp = os.path.join(args.out_dir, s.stem + ".txt")
                zf.write(fp, arcname=s.stem + ".txt")
        print(f"[OK] 打包完成: {args.zip_path}")


if __name__ == "__main__":
    main()



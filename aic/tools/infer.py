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

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from aic.data.dataset import MultiModalDetDataset, collate_fn, scan_samples
from aic.data.transforms import SyncMultiModalTransform
from aic.eval.submission_validator import validate_submission_dir
from aic.models.tristream_dfine import build_tristream_dfine
from third_party_dfine import DFINEPostProcessor


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
    score_thresh = inf_cfg["score_thresh"]
    max_boxes = inf_cfg["max_boxes"]

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

    tf = SyncMultiModalTransform(size=infer_size, train=False)
    ds = MultiModalDetDataset(samples, tf, has_labels=False, disp8_mode=disp8_mode)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

    written_stems = set()

    with torch.no_grad():
        for rgb, ir, depth, targets in loader:
            rgb, ir, depth = rgb.to(device), ir.to(device), depth.to(device)
            outputs = model(rgb, ir, depth)
            orig_sizes = torch.stack([t["orig_size"] for t in targets]).to(device)
            results = postproc(outputs, orig_sizes)

            for i, t in enumerate(targets):
                stem = t["stem"]
                w, h = t["orig_size"].tolist()
                boxes_px = results[i]["boxes"].cpu().numpy()
                scores = results[i]["scores"].cpu().numpy()
                labels = results[i]["labels"].cpu().numpy()

                keep = scores >= score_thresh
                boxes_px, scores, labels = boxes_px[keep], scores[keep], labels[keep]

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



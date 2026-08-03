"""提交格式校验器，独立于训练/推理代码运行，抢在提交前拦截"零分陷阱"。

赛题红线（计划.md §五）：
  - 每张测试图必须有对应输出文件，即使该图无检测目标也要产出空文件；
    缺失文件 = 该图直接判 0 分，是最容易踩的坑。
  - 每个文件最多 100 个框，超出按 confidence 截断（低分先丢）。
  - 每行格式: [class_id, norm_cx, norm_cy, norm_w, norm_h, confidence]，
    坐标必须归一化到 [0,1]，class_id 必须在 [0,11] 合法范围内。
"""

from __future__ import annotations

import os

NUM_CLASSES = 12
MAX_BOXES_PER_IMAGE = 100


def validate_submission_dir(pred_dir: str, expected_stems: list[str]) -> list[str]:
    """返回错误信息列表，空列表代表校验通过。"""
    errors: list[str] = []

    existing = {os.path.splitext(f)[0] for f in os.listdir(pred_dir)} if os.path.isdir(pred_dir) else set()

    missing = [s for s in expected_stems if s not in existing]
    if missing:
        errors.append(f"缺失 {len(missing)} 个预测文件（将直接判 0 分），示例: {missing[:5]}")

    for stem in expected_stems:
        if stem not in existing:
            continue
        path = os.path.join(pred_dir, stem + ".txt")
        errors.extend(_validate_one_file(path, stem))

    return errors


def _validate_one_file(path: str, stem: str) -> list[str]:
    errs = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if len(lines) > MAX_BOXES_PER_IMAGE:
        errs.append(f"{stem}: {len(lines)} 行超过上限 {MAX_BOXES_PER_IMAGE}（应在写文件前截断）")

    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) != 6:
            errs.append(f"{stem} 第{i+1}行: 应为6个字段（class_id,cx,cy,w,h,conf），实际 {len(parts)} 个")
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h, conf = (float(x) for x in parts[1:])
        except ValueError:
            errs.append(f"{stem} 第{i+1}行: 数值解析失败: {line}")
            continue

        if not (0 <= cls < NUM_CLASSES):
            errs.append(f"{stem} 第{i+1}行: class_id={cls} 超出合法范围 [0,{NUM_CLASSES-1}]")
        for name, v in (("cx", cx), ("cy", cy), ("w", w), ("h", h)):
            if not (0.0 <= v <= 1.0):
                errs.append(f"{stem} 第{i+1}行: {name}={v} 未归一化到 [0,1]")
        if not (0.0 <= conf <= 1.0):
            errs.append(f"{stem} 第{i+1}行: confidence={conf} 未归一化到 [0,1]")

    return errs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--test_root", required=True, help="测试集根目录（含 visible/ 子目录，用于确定期望文件名）")
    args = parser.parse_args()

    vis_dir = os.path.join(args.test_root, "visible")
    stems = [os.path.splitext(f)[0] for f in os.listdir(vis_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    errors = validate_submission_dir(args.pred_dir, stems)
    if errors:
        print(f"[FAIL] 发现 {len(errors)} 个问题：")
        for e in errors[:50]:
            print("  -", e)
        raise SystemExit(1)
    print(f"[OK] {len(stems)} 个文件全部通过格式校验。")

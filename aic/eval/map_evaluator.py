"""本地 mAP@50-95 评测（COCO 定义，101 点插值），与训练代码完全解耦。

坐标系说明：所有框统一使用归一化 [0,1] 的 cxcywh 或 xyxy 均可，IoU 是
仿射不变量——对 x、y 各自独立缩放（即便宽高缩放比例不同，如 1920x1080 的
非正方形图像）不改变任意两个框的 IoU 值，因为交集/并集面积同时按相同的
Sx*Sy 缩放。因此直接在归一化坐标上计算 IoU 等价于在像素坐标上计算，
本评测器不需要还原到像素坐标。

mAP@50-95 定义：IoU 阈值 {0.50, 0.55, ..., 0.95} 共 10 个，每个阈值下按
101 点（recall = 0, 0.01, ..., 1.00）插值精度求 AP，10 个阈值的 AP 再对
类别取平均、对阈值取平均（等价于先对每个类别在 10 个阈值上平均，再对
类别平均）。没有任何 GT 的类别按 COCO 约定跳过（不计入平均，不是记为 0）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

IOU_THRESHOLDS = np.round(np.arange(0.50, 1.00, 0.05), 2)  # 0.50 ... 0.95
RECALL_POINTS = np.linspace(0.0, 1.0, 101)


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)


def box_iou_matrix(a_xyxy: np.ndarray, b_xyxy: np.ndarray) -> np.ndarray:
    """a: N,4  b: M,4  -> N,M IoU 矩阵。"""
    if a_xyxy.shape[0] == 0 or b_xyxy.shape[0] == 0:
        return np.zeros((a_xyxy.shape[0], b_xyxy.shape[0]), dtype=np.float64)

    ax1, ay1, ax2, ay2 = (a_xyxy[:, i][:, None] for i in range(4))
    bx1, by1, bx2, by2 = (b_xyxy[:, i][None, :] for i in range(4))

    ix1, iy1 = np.maximum(ax1, bx1), np.maximum(ay1, by1)
    ix2, iy2 = np.minimum(ax2, bx2), np.minimum(ay2, by2)
    iw, ih = np.clip(ix2 - ix1, 0, None), np.clip(iy2 - iy1, 0, None)
    inter = iw * ih

    area_a = np.clip(ax2 - ax1, 0, None) * np.clip(ay2 - ay1, 0, None)
    area_b = np.clip(bx2 - bx1, 0, None) * np.clip(by2 - by1, 0, None)
    union = area_a + area_b - inter

    return np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)


@dataclass
class Detection:
    image_id: str
    cls: int
    box_xyxy: np.ndarray  # (4,)
    score: float


@dataclass
class GroundTruth:
    image_id: str
    cls: int
    box_xyxy: np.ndarray  # (4,)


@dataclass
class EvalResult:
    per_class_ap50_95: dict[int, float] = field(default_factory=dict)
    per_class_ap50: dict[int, float] = field(default_factory=dict)
    per_class_ap75: dict[int, float] = field(default_factory=dict)
    map_50_95: float = 0.0
    map_50: float = 0.0
    map_75: float = 0.0


def _ap_101point(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """标准 COCO 101 点插值 AP：对每个 recall 采样点取 recall'>=r 处的最大精度。"""
    if recalls.size == 0:
        return 0.0
    # 使精度曲线单调不增（从右往左取最大值），是 COCO/VOC 标准做法。
    envelope = np.maximum.accumulate(precisions[::-1])[::-1]
    interp = np.zeros_like(RECALL_POINTS)
    idx = np.searchsorted(recalls, RECALL_POINTS, side="left")
    for i, ix in enumerate(idx):
        if ix < len(envelope):
            interp[i] = envelope[ix]
        else:
            interp[i] = 0.0
    return float(interp.mean())


def _ap_for_class_at_iou(
    dets: list[Detection], gts: list[GroundTruth], iou_thresh: float
) -> float | None:
    """返回 None 表示该类别没有任何 GT（COCO 约定：跳过，不计入平均）。"""
    if len(gts) == 0:
        return None

    gts_by_image: dict[str, list[int]] = {}
    gt_boxes_by_image: dict[str, np.ndarray] = {}
    for i, g in enumerate(gts):
        gts_by_image.setdefault(g.image_id, []).append(i)
    for img_id, idxs in gts_by_image.items():
        gt_boxes_by_image[img_id] = np.stack([gts[i].box_xyxy for i in idxs], axis=0)

    matched = {img_id: np.zeros(len(idxs), dtype=bool) for img_id, idxs in gts_by_image.items()}

    dets_sorted = sorted(dets, key=lambda d: -d.score)
    n_dets = len(dets_sorted)
    tp = np.zeros(n_dets, dtype=np.float64)
    fp = np.zeros(n_dets, dtype=np.float64)

    for i, d in enumerate(dets_sorted):
        img_id = d.image_id
        if img_id not in gt_boxes_by_image:
            fp[i] = 1.0
            continue
        ious = box_iou_matrix(d.box_xyxy[None, :], gt_boxes_by_image[img_id])[0]
        avail = ~matched[img_id]
        ious_avail = np.where(avail, ious, -1.0)
        best_j = int(np.argmax(ious_avail)) if ious_avail.size else -1
        if best_j >= 0 and ious_avail[best_j] >= iou_thresh:
            matched[img_id][best_j] = True
            tp[i] = 1.0
        else:
            fp[i] = 1.0

    n_gt = len(gts)
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / max(n_gt, 1)
    precisions = tp_cum / np.clip(tp_cum + fp_cum, 1e-9, None)

    return _ap_101point(recalls, precisions)


def evaluate(
    predictions: dict[str, list[tuple[int, np.ndarray, float]]],
    ground_truths: dict[str, list[tuple[int, np.ndarray]]],
    class_ids: list[int],
) -> EvalResult:
    """predictions/ground_truths: image_id -> list of (cls, box_xyxy, [score])。"""
    dets_by_cls: dict[int, list[Detection]] = {c: [] for c in class_ids}
    gts_by_cls: dict[int, list[GroundTruth]] = {c: [] for c in class_ids}

    for img_id, dets in predictions.items():
        for cls, box, score in dets:
            if cls in dets_by_cls:
                dets_by_cls[cls].append(Detection(img_id, cls, box, score))

    for img_id, gts in ground_truths.items():
        for cls, box in gts:
            if cls in gts_by_cls:
                gts_by_cls[cls].append(GroundTruth(img_id, cls, box))

    result = EvalResult()
    aps_per_class_all_iou: dict[int, list[float]] = {}

    for cls in class_ids:
        aps = []
        for thr in IOU_THRESHOLDS:
            ap = _ap_for_class_at_iou(dets_by_cls[cls], gts_by_cls[cls], float(thr))
            if ap is not None:
                aps.append(ap)
            if abs(thr - 0.50) < 1e-6 and ap is not None:
                result.per_class_ap50[cls] = ap
            if abs(thr - 0.75) < 1e-6 and ap is not None:
                result.per_class_ap75[cls] = ap
        if aps:
            aps_per_class_all_iou[cls] = aps
            result.per_class_ap50_95[cls] = float(np.mean(aps))

    if aps_per_class_all_iou:
        result.map_50_95 = float(np.mean([np.mean(v) for v in aps_per_class_all_iou.values()]))
    if result.per_class_ap50:
        result.map_50 = float(np.mean(list(result.per_class_ap50.values())))
    if result.per_class_ap75:
        result.map_75 = float(np.mean(list(result.per_class_ap75.values())))

    return result

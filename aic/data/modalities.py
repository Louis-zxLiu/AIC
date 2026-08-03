"""三模态图像读取与预处理。

三个模态在磁盘上的实际格式（已用官方示例数据核实，18/18 组核心格式一致）：
- visible : RGB uint8, 3ch, [0,255]
- infrared: RGB uint8, 3ch（三通道冗余，取任一通道即为热辐射灰度）
- depth   : PNG mode="I"（32bit 容器存 16bit 语义值），单位 mm，量程 [0, 19999]
            0 或极小值代表无效深度区域（无 depth 反射，非"距离为 0"）

样例数据中发现 1 个例外（00000008.jpg 系列）：三模态均退化为 8bit JPEG，
depth 被当成普通 RGB 图像存储而非真实深度值。这类样本深度通道不可信，
读取时检测到非 "I" mode 会自动置全图为 invalid mask，不会静默产出错误的深度值。
"""

from __future__ import annotations

import numpy as np
from PIL import Image

DEPTH_MIN_MM = 300      # 相机最近感知距离 0.3m
DEPTH_MAX_MM = 20000    # 相机最远感知距离 20m（含）取开区间上界之前的整数上限


def load_visible(path: str) -> np.ndarray:
    """返回 HxWx3 uint8 RGB。"""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def load_infrared(path: str) -> np.ndarray:
    """返回 HxW uint8 热辐射灰度图（从冗余三通道中取单通道）。"""
    img = Image.open(path)
    arr = np.array(img)
    if arr.ndim == 3:
        # 三通道视觉特征高度一致（赛题文档已说明），取任一通道等价，取通道0。
        arr = arr[..., 0]
    return arr.astype(np.uint8)


def load_depth_raw(path: str) -> tuple[np.ndarray, bool]:
    """读取深度图原始 mm 值。

    Returns:
        depth_mm: HxW int32/float，单位 mm。对非标准（退化为 RGB 的坏样本）格式，
                   返回全零数组并标记 is_valid_format=False。
        is_valid_format: 该文件是否为预期的单通道深度容器。
    """
    img = Image.open(path)
    # Pillow 对单通道整型 PNG 按存储字节序细分 mode（"I"=32bit 有符号，
    # "I;16"/"I;16B"/"I;16L"/"I;16N" 等=16bit），真实数据集中的正常深度图
    # 实测是 "I;16"，不是 "I"——只匹配 "I" 会把全部正常样本误判为退化格式。
    if img.mode.startswith("I"):
        depth_mm = np.array(img, dtype=np.int64)
        return depth_mm, True
    # 退化情况：depth 被存成了 RGB/L 等非深度容器格式，值不可信。
    arr = np.array(img)
    h, w = arr.shape[:2]
    return np.zeros((h, w), dtype=np.int64), False


def preprocess_depth(depth_mm: np.ndarray, is_valid_format: bool) -> tuple[np.ndarray, np.ndarray]:
    """深度图预处理：掩码 + 截断 + 逆深度归一化。

    直接 d/65535 是错的：真实量程只占理论值域([0,65535])的 30%，
    且 0 代表"无效"而非"距离0"，不做掩码会让模型把空洞学成近处物体。

    逆深度（1/d）而非线性归一化的理由：检测任务中近处物体通常更大更重要，
    逆深度在近处的动态范围更大，等价于对近处目标做了隐式加权。

    Returns:
        depth_norm: HxW float32 in [0,1]，无效区域填 0。
        valid_mask: HxW float32 {0,1}，作为额外通道随 depth_norm 一起送入网络，
                    使模型可学习区分"近距离"和"无有效深度"两种语义。
    """
    if not is_valid_format:
        h, w = depth_mm.shape
        return (
            np.zeros((h, w), dtype=np.float32),
            np.zeros((h, w), dtype=np.float32),
        )

    valid_mask = (depth_mm >= DEPTH_MIN_MM) & (depth_mm <= DEPTH_MAX_MM)
    clipped = np.clip(depth_mm, DEPTH_MIN_MM, DEPTH_MAX_MM).astype(np.float32)

    inv_depth = 1.0 / clipped
    inv_min = 1.0 / DEPTH_MAX_MM
    inv_max = 1.0 / DEPTH_MIN_MM
    depth_norm = (inv_depth - inv_min) / (inv_max - inv_min)
    depth_norm = np.clip(depth_norm, 0.0, 1.0).astype(np.float32)
    depth_norm = np.where(valid_mask, depth_norm, 0.0).astype(np.float32)

    return depth_norm, valid_mask.astype(np.float32)


def load_labels_yolo(path: str) -> np.ndarray:
    """读取 YOLO 格式标签：[class_id, cx, cy, w, h]，坐标已归一化到 [0,1]。

    空文件（无目标）返回 shape (0, 5) 的数组，这是合法情况，不是错误。
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:5])
            rows.append([cls, cx, cy, w, h])
    if not rows:
        return np.zeros((0, 5), dtype=np.float32)
    return np.array(rows, dtype=np.float32)

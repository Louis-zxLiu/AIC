"""三模态图像读取与预处理。

三个模态在磁盘上的实际格式（已用官方示例数据核实，18/18 组核心格式一致）：
- visible : RGB uint8, 3ch, [0,255]
- infrared: RGB uint8, 3ch（三通道冗余，取任一通道即为热辐射灰度）
- depth   : 两种格式并存，已在全量数据上统计+定标（见下）

depth 的两种格式（训练集 2000 张 / 测试集 1000 张全量扫描）：

  A) `.png` mode="I;16" —— 1851 张训练 / 845 张测试
     单位 mm，量程 [0, 19999]，**值大 = 远**。0 代表无效（无反射，非"距离 0"）。
     有效像素覆盖 69.3%，GT 框内有效像素 56.6%。

  B) `.jpg` mode="RGB"（三通道完全相同的 8bit 灰度）—— 149 张训练 / 155 张测试
     这批就是 640x360 的低分辨率子集。**不是坏文件**：234 个唯一值铺满 [0,255]，
     且按行斜率 +26.6/箱，**值大 = 近**，已经处在视差/逆深度空间。
     只在图像下 1/3 有效（上 2/3 精确为 0，传感器无覆盖），覆盖 26.3%，
     GT 框内有效像素仅 14.3%。JPEG 有损压缩，存在块效应噪声。

     历史坑：旧实现只认 mode 前缀 "I"，把 B 类整批判为 invalid 并全图置 0，
     等于**丢掉了 15.5% 测试图的整条 depth 分支**。B 类的值已是逆深度语义，
     与 preprocess_depth 对 A 类输出的 depth_norm（1=近）方向一致，因此只需
     除以 255 归一化，**不能再套 1/d 反演**（会把方向反过来）。
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


DEPTH_KIND_MM16 = "mm16"    # A 类：16bit 毫米距离，值大=远
DEPTH_KIND_DISP8 = "disp8"  # B 类：8bit 视差/逆深度，值大=近
DEPTH_KIND_BAD = "bad"      # 真正无法解析


def load_depth_raw(path: str) -> tuple[np.ndarray, str]:
    """读取深度图原始值，并识别它属于哪种格式。

    Returns:
        depth_raw: HxW int64。语义由 kind 决定，本函数不做任何数值变换。
        kind: DEPTH_KIND_MM16 / DEPTH_KIND_DISP8 / DEPTH_KIND_BAD
    """
    img = Image.open(path)
    # Pillow 对单通道整型 PNG 按存储字节序细分 mode（"I"=32bit 有符号，
    # "I;16"/"I;16B"/"I;16L"/"I;16N" 等=16bit），真实数据集中的正常深度图
    # 实测是 "I;16"，不是 "I"——只匹配 "I" 会把全部正常样本误判为退化格式。
    if img.mode.startswith("I"):
        return np.array(img, dtype=np.int64), DEPTH_KIND_MM16

    arr = np.array(img)
    if arr.ndim == 3:
        # 三通道冗余的 8bit 灰度视差图。要求三通道一致才认，避免把真彩色图误判成深度。
        if arr.shape[2] >= 3 and np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(
            arr[..., 1], arr[..., 2]
        ):
            return arr[..., 0].astype(np.int64), DEPTH_KIND_DISP8
        h, w = arr.shape[:2]
        return np.zeros((h, w), dtype=np.int64), DEPTH_KIND_BAD
    if arr.ndim == 2 and arr.dtype == np.uint8:
        return arr.astype(np.int64), DEPTH_KIND_DISP8

    h, w = arr.shape[:2]
    return np.zeros((h, w), dtype=np.int64), DEPTH_KIND_BAD


def preprocess_depth(depth_raw: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]:
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
    if kind == DEPTH_KIND_BAD:
        h, w = depth_raw.shape
        return (
            np.zeros((h, w), dtype=np.float32),
            np.zeros((h, w), dtype=np.float32),
        )

    if kind == DEPTH_KIND_DISP8:
        # 已是逆深度空间（值大=近），与下面 mm16 分支输出的 depth_norm 方向一致，
        # 直接线性归一化即可；再套 1/d 会把远近关系反过来。0 = 传感器无覆盖。
        valid_mask = depth_raw > 0
        depth_norm = (depth_raw.astype(np.float32) / 255.0).clip(0.0, 1.0)
        depth_norm = np.where(valid_mask, depth_norm, 0.0).astype(np.float32)
        return depth_norm, valid_mask.astype(np.float32)

    valid_mask = (depth_raw >= DEPTH_MIN_MM) & (depth_raw <= DEPTH_MAX_MM)
    clipped = np.clip(depth_raw, DEPTH_MIN_MM, DEPTH_MAX_MM).astype(np.float32)

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

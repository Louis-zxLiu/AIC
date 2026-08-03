"""三模态目标检测数据集。

目录结构（与官方样例一致）：
    root/
      visible/*.{png,jpg}
      infrared/*.{png,jpg}
      depth/*.{png,jpg}
      labels/*.txt   # 与图像同名（不含扩展名），YOLO 格式；测试集无此目录

collate_fn 与 D-FINE 官方保持同构：图像堆叠成 batch tensor，targets 是
list[dict]，每个 dict 至少含 "labels"、"boxes"（cxcywh, normalized）。
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset

from .modalities import (
    DEPTH_KIND_BAD,
    DEPTH_KIND_DISP8,
    load_depth_raw,
    load_infrared,
    load_labels_yolo,
    load_visible,
    preprocess_depth,
)
from .transforms import SyncMultiModalTransform

IMG_EXTS = (".png", ".jpg", ".jpeg")


@dataclass
class Sample:
    stem: str
    visible_path: str
    infrared_path: str
    depth_path: str
    label_path: str | None
    class_ids: list[int] = field(default_factory=list)


def _find_file(dir_path: str, stem: str) -> str | None:
    for ext in IMG_EXTS:
        p = os.path.join(dir_path, stem + ext)
        if os.path.exists(p):
            return p
    return None


def scan_samples(root: str, require_labels: bool = True) -> list[Sample]:
    vis_dir = os.path.join(root, "visible")
    ir_dir = os.path.join(root, "infrared")
    dep_dir = os.path.join(root, "depth")
    lbl_dir = os.path.join(root, "labels")

    samples = []
    for fname in sorted(os.listdir(vis_dir)):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in IMG_EXTS:
            continue
        ir_path = _find_file(ir_dir, stem)
        dep_path = _find_file(dep_dir, stem)
        if ir_path is None or dep_path is None:
            print(f"[warn] skip {stem}: missing infrared/depth counterpart")
            continue
        lbl_path = os.path.join(lbl_dir, stem + ".txt")
        if not os.path.exists(lbl_path):
            lbl_path = None
            if require_labels:
                print(f"[warn] skip {stem}: missing label file")
                continue

        class_ids = []
        if lbl_path is not None:
            arr = load_labels_yolo(lbl_path)
            class_ids = arr[:, 0].astype(int).tolist() if arr.size else []

        samples.append(
            Sample(
                stem=stem,
                visible_path=os.path.join(vis_dir, fname),
                infrared_path=ir_path,
                depth_path=dep_path,
                label_path=lbl_path,
                class_ids=class_ids,
            )
        )
    return samples


def kfold_split(samples: list[Sample], n_folds: int = 5, fold: int = 0, seed: int = 42):
    """按文件名排序后固定种子打乱再切分，保证多次运行 fold 划分一致。"""
    idx = list(range(len(samples)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    folds = [idx[i::n_folds] for i in range(n_folds)]
    val_idx = set(folds[fold])
    train = [samples[i] for i in idx if i not in val_idx]
    val = [samples[i] for i in idx if i in val_idx]
    return train, val


def compute_class_balanced_weights(samples: list[Sample], oversample_thr: float = 0.2) -> list[float]:
    """按 ClassBalancedDataset（mmdetection 风格，算法设计.md §3.4）计算样本采样权重。

    每个类别的重复采样系数 repeat_c = max(1, sqrt(oversample_thr / freq_c))，
    其中 freq_c 是含该类别的样本数占总样本数的比例。样本权重取其含有的
    所有类别里 repeat_c 的最大值——样本里只要含一个稀有类别，就应该被更
    频繁地采样，不能被同一样本里的多数类"平均掉"。

    不含任何标注框的样本（背景图/纯负样本）权重固定为 1.0，不参与类别
    频率计算，也不因为"没有稀有类别"被过采样或欠采样。
    """
    n = len(samples)
    if n == 0:
        return []

    class_sample_count: dict[int, int] = {}
    for s in samples:
        for c in set(s.class_ids):
            class_sample_count[c] = class_sample_count.get(c, 0) + 1

    repeat_factor = {
        c: max(1.0, (oversample_thr / (count / n)) ** 0.5) for c, count in class_sample_count.items()
    }

    weights = []
    for s in samples:
        if not s.class_ids:
            weights.append(1.0)
        else:
            weights.append(max(repeat_factor[c] for c in set(s.class_ids)))
    return weights


class MultiModalDetDataset(Dataset):
    """disp8_mode 控制 8bit 视差深度（149 训练 / 155 测试图，见 modalities.py 文档）：

      "use"  —— 正常解码使用。重训必须用这个，否则又把 15.5% 测试图的 depth 白扔。
      "zero" —— 整路置零，复刻旧行为。**只给 run2 及更早的 checkpoint 推理用**：
                那些权重是在 disp8 全零的输入上训出来的，喂真实深度反而掉分
                （实测 epoch41 @768：disp8 子域 0.5693 → 0.5448），属训练/推理不一致。
    """

    def __init__(
        self,
        samples: list[Sample],
        transform: SyncMultiModalTransform,
        has_labels: bool = True,
        disp8_mode: str = "use",
    ):
        if disp8_mode not in ("use", "zero"):
            raise ValueError(f"disp8_mode 必须是 'use' 或 'zero'，收到 {disp8_mode!r}")
        self.samples = samples
        self.transform = transform
        self.has_labels = has_labels
        self.disp8_mode = disp8_mode

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        s = self.samples[index]

        visible = load_visible(s.visible_path)
        infrared = load_infrared(s.infrared_path)
        depth_raw, depth_kind = load_depth_raw(s.depth_path)
        if depth_kind == DEPTH_KIND_BAD:
            print(f"[warn] {s.stem}: depth 无法解析为深度容器，该图 depth 分支置全零")
        if depth_kind == DEPTH_KIND_DISP8 and self.disp8_mode == "zero":
            # 刻意置零（复刻 run2 的训练输入），不是解析失败，所以不复用上面那条 warn。
            depth_kind = DEPTH_KIND_BAD
        depth_norm, valid_mask = preprocess_depth(depth_raw, depth_kind)

        if self.has_labels and s.label_path is not None:
            arr = load_labels_yolo(s.label_path)
            labels = arr[:, 0]
            boxes = arr[:, 1:5]
        else:
            labels = torch_empty_labels()
            boxes = torch_empty_boxes()

        vis, ir, depth_stack, target = self.transform(visible, infrared, depth_norm, valid_mask, boxes, labels)
        target["stem"] = s.stem
        return vis, ir, depth_stack, target


def torch_empty_labels():
    return np.zeros((0,), dtype=np.float32)


def torch_empty_boxes():
    return np.zeros((0, 4), dtype=np.float32)


class MultiScaleCollate:
    """算法设计.md §3.5：多尺度训练。batch 内图像已在 dataset 里统一 resize
    到 base_size（transform.size），这里按 D-FINE 官方 BatchImageCollateFunction
    的思路，在整个 batch 上再统一 resize 到围绕 base_size 浮动的目标尺寸
    （32 的倍数，与 backbone 的 stride 对齐）。

    对齐边界坐标不受影响：boxes 是归一化到 [0,1] 的 cxcywh，整批统一缩放
    图像分辨率不改变归一化坐标值，因此无需在这里改动 target。

    stop_epoch：训练末段（由外部 set_epoch 控制）关闭多尺度，固定回
    base_size，与强增强的 stop_epoch policy 同步收尾。
    """

    def __init__(
        self,
        base_size: int | tuple[int, int] | list[int] = 640,
        sizes: tuple = (512, 576, 640, 704, 768),
    ):
        # base_size / sizes 的元素都支持 int（方形）或 (h, w)（矩形）。矩形输入下
        # 各档必须保持与 base 相同的长宽比，否则多尺度会额外注入形状畸变。
        self.base_size = self._as_hw(base_size)
        self.sizes = tuple(self._as_hw(s) for s in sizes)
        self.active = True

    @staticmethod
    def _as_hw(s) -> tuple[int, int]:
        return (int(s), int(s)) if isinstance(s, int) else (int(s[0]), int(s[1]))

    def set_epoch_active(self, active: bool) -> None:
        self.active = active

    def __call__(self, batch):
        vis = torch.stack([b[0] for b in batch], dim=0)
        ir = torch.stack([b[1] for b in batch], dim=0)
        depth = torch.stack([b[2] for b in batch], dim=0)
        targets = [b[3] for b in batch]

        size = random.choice(self.sizes) if self.active else self.base_size
        if size != tuple(vis.shape[-2:]):
            vis = torch.nn.functional.interpolate(vis, size=size, mode="bilinear", align_corners=False)
            ir = torch.nn.functional.interpolate(ir, size=size, mode="bilinear", align_corners=False)
            # depth 的第2通道是 {0,1} valid_mask，用 nearest 保持硬边界语义，
            # 不与 depth_norm 通道一起做 bilinear（会产生 0~1 之间的过渡值）。
            depth_norm = torch.nn.functional.interpolate(depth[:, 0:1], size=size, mode="bilinear", align_corners=False)
            depth_mask = torch.nn.functional.interpolate(depth[:, 1:2], size=size, mode="nearest")
            depth = torch.cat([depth_norm, depth_mask], dim=1)

        return vis, ir, depth, targets


def collate_fn(batch):
    vis = torch.stack([b[0] for b in batch], dim=0)
    ir = torch.stack([b[1] for b in batch], dim=0)
    depth = torch.stack([b[2] for b in batch], dim=0)
    targets = [b[3] for b in batch]
    return vis, ir, depth, targets

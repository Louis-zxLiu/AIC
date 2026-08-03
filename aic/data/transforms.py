"""三模态同步数据增强。

核心约束（计划.md §2.2）：
- 几何变换（resize/crop/flip）三模态必须共享同一组随机参数，否则三路特征在
  空间上错位，OPF 跨模态注意力会学到错误的对应关系。
- 色彩/光度增强（亮度、对比度、色调等）只作用于 visible，因为 infrared/depth
  的像素值物理含义不是"颜色"，色彩扰动对它们没有意义甚至会破坏物理量。
- 模态 dropout（训练时以 p=0.1 整路置零一个非 RGB 模态）直接对应赛题对"部分
  模态缺失/失效时鲁棒性"的要求，推理时不启用。
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torchvision.transforms.functional as TF


class SyncMultiModalTransform:
    def __init__(
        self,
        size: int = 640,
        train: bool = True,
        hflip_p: float = 0.5,
        scale_range: tuple[float, float] = (0.5, 1.5),
        modality_dropout_p: float = 0.1,
        final_scale_range: tuple[float, float] = (0.9, 1.1),
        stop_decay_epoch_ratio: float = 0.8,
        total_epochs: int = 1,
    ):
        self.size = size
        self.train = train
        self.hflip_p = hflip_p
        self.base_scale_range = scale_range
        self.scale_range = scale_range
        self.base_modality_dropout_p = modality_dropout_p
        self.modality_dropout_p = modality_dropout_p
        self.final_scale_range = final_scale_range
        self.stop_decay_epoch_ratio = stop_decay_epoch_ratio
        self.total_epochs = max(total_epochs, 1)

    def set_epoch(self, epoch: int) -> None:
        """算法设计.md §3.5：训练末段（默认最后 20% epoch）渐进关闭强增强，
        让模型在接近真实数据分布的输入上收尾（借鉴 YOLOX close-mosaic 和
        D-FINE 官方 dataloader.yml 的 stop_epoch policy）。scale_range 线性
        收紧到 final_scale_range，modality_dropout_p 线性退火到 0，使模型
        在收尾阶段用完整三模态输入定型。
        """
        decay_start = self.stop_decay_epoch_ratio * self.total_epochs
        if epoch < decay_start or self.total_epochs <= decay_start:
            self.scale_range = self.base_scale_range
            self.modality_dropout_p = self.base_modality_dropout_p
            return

        progress = min(1.0, (epoch - decay_start) / max(self.total_epochs - decay_start, 1e-6))
        lo = self.base_scale_range[0] + progress * (self.final_scale_range[0] - self.base_scale_range[0])
        hi = self.base_scale_range[1] + progress * (self.final_scale_range[1] - self.base_scale_range[1])
        self.scale_range = (lo, hi)
        self.modality_dropout_p = self.base_modality_dropout_p * (1.0 - progress)

    def __call__(
        self,
        visible: np.ndarray,   # HxWx3 uint8
        infrared: np.ndarray,  # HxW uint8
        depth_norm: np.ndarray,   # HxW float32 [0,1]
        valid_mask: np.ndarray,   # HxW float32 {0,1}
        boxes_cxcywh: np.ndarray,  # Nx4 normalized [0,1], DETR convention
        labels: np.ndarray,       # N,
    ):
        h0, w0 = visible.shape[:2]

        vis = TF.to_tensor(visible)                      # 3,H,W float[0,1]
        ir = torch.from_numpy(infrared).float().unsqueeze(0) / 255.0   # 1,H,W
        dep = torch.from_numpy(depth_norm).unsqueeze(0)                # 1,H,W
        msk = torch.from_numpy(valid_mask).unsqueeze(0)                # 1,H,W

        boxes = torch.from_numpy(boxes_cxcywh.copy()) if boxes_cxcywh.size else torch.zeros((0, 4))
        labels_t = torch.from_numpy(labels.copy()).long() if labels.size else torch.zeros((0,), dtype=torch.long)

        if self.train:
            vis, ir, dep, msk, boxes = self._random_resized_crop(vis, ir, dep, msk, boxes)
            if random.random() < self.hflip_p:
                vis, ir, dep, msk = (TF.hflip(t) for t in (vis, ir, dep, msk))
                if boxes.numel():
                    boxes[:, 0] = 1.0 - boxes[:, 0]
            vis = self._photometric(vis)
        else:
            vis, ir, dep, msk = self._resize_only(vis, ir, dep, msk)

        if self.train and boxes.numel():
            keep = (boxes[:, 2] > 1e-4) & (boxes[:, 3] > 1e-4)
            boxes = boxes[keep]
            labels_t = labels_t[keep]

        if self.train and random.random() < self.modality_dropout_p:
            if random.random() < 0.5:
                ir = torch.zeros_like(ir)
            else:
                dep = torch.zeros_like(dep)
                msk = torch.zeros_like(msk)

        depth_stack = torch.cat([dep, msk], dim=0)  # 2,H,W

        target = {
            "labels": labels_t,
            "boxes": boxes.float(),
            "orig_size": torch.tensor([w0, h0], dtype=torch.float32),
        }
        return vis, ir, depth_stack, target

    def _resize_only(self, vis, ir, dep, msk):
        vis = TF.resize(vis, [self.size, self.size], antialias=True)
        ir = TF.resize(ir, [self.size, self.size], antialias=True)
        dep = TF.resize(dep, [self.size, self.size], antialias=True)
        msk = TF.resize(msk, [self.size, self.size], antialias=False)
        return vis, ir, dep, msk

    def _random_resized_crop(self, vis, ir, dep, msk, boxes):
        _, h, w = vis.shape
        scale = random.uniform(*self.scale_range)
        crop_h = min(h, max(1, int(h / scale)))
        crop_w = min(w, max(1, int(w / scale)))
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)

        vis = TF.resized_crop(vis, top, left, crop_h, crop_w, [self.size, self.size], antialias=True)
        ir = TF.resized_crop(ir, top, left, crop_h, crop_w, [self.size, self.size], antialias=True)
        dep = TF.resized_crop(dep, top, left, crop_h, crop_w, [self.size, self.size], antialias=True)
        msk = TF.resized_crop(msk, top, left, crop_h, crop_w, [self.size, self.size], antialias=False)

        if boxes.numel():
            cx, cy, bw, bh = boxes.unbind(-1)
            x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
            x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
            x1 = (x1 - left).clamp(0, crop_w)
            x2 = (x2 - left).clamp(0, crop_w)
            y1 = (y1 - top).clamp(0, crop_h)
            y2 = (y2 - top).clamp(0, crop_h)
            new_cx = (x1 + x2) / 2 / crop_w
            new_cy = (y1 + y2) / 2 / crop_h
            new_w = (x2 - x1) / crop_w
            new_h = (y2 - y1) / crop_h
            boxes = torch.stack([new_cx, new_cy, new_w, new_h], dim=-1)

        return vis, ir, dep, msk, boxes

    def _photometric(self, vis: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.8:
            vis = TF.adjust_brightness(vis, random.uniform(0.7, 1.3))
        if random.random() < 0.8:
            vis = TF.adjust_contrast(vis, random.uniform(0.7, 1.3))
        if random.random() < 0.5:
            vis = TF.adjust_saturation(vis, random.uniform(0.7, 1.3))
        if random.random() < 0.3:
            vis = TF.adjust_hue(vis, random.uniform(-0.05, 0.05))
        return vis.clamp(0, 1)

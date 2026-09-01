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

from .modalities import DEPTH_KIND_DISP8, DEPTH_KIND_MM16


class SyncMultiModalTransform:
    def __init__(
        self,
        size: int | tuple[int, int] | list[int] = 640,
        train: bool = True,
        hflip_p: float = 0.5,
        scale_range: tuple[float, float] = (0.5, 1.5),
        modality_dropout_p: float = 0.1,
        final_scale_range: tuple[float, float] = (0.9, 1.1),
        stop_decay_epoch_ratio: float = 0.8,
        total_epochs: int = 1,
        no_aug_epoch: int | None = None,
        min_box_px: float = 2.0,
    ):
        # size 支持 int（方形）或 (h, w)（矩形）。数据集全是 16:9，方形 resize 会带来
        # 1.78x 的形状畸变并把面积压到 0.44 倍，矩形输入同时解决畸变和小目标丢失。
        self.size_hw: tuple[int, int] = (
            (int(size), int(size)) if isinstance(size, int) else (int(size[0]), int(size[1]))
        )
        self.size = self.size_hw  # 兼容旧字段名
        self.min_box_px = min_box_px
        self.train = train
        self.hflip_p = hflip_p
        self.base_scale_range = scale_range
        self.scale_range = scale_range
        self.base_modality_dropout_p = modality_dropout_p
        self.modality_dropout_p = modality_dropout_p
        self.base_hflip_p = hflip_p
        self.final_scale_range = final_scale_range
        self.stop_decay_epoch_ratio = stop_decay_epoch_ratio
        self.total_epochs = max(total_epochs, 1)
        self.no_aug_epoch = no_aug_epoch
        self.photometric_enabled = True

    def set_epoch(self, epoch: int) -> None:
        """算法设计.md §3.5：训练末段（默认最后 20% epoch）渐进关闭强增强，
        让模型在接近真实数据分布的输入上收尾（借鉴 YOLOX close-mosaic 和
        D-FINE 官方 dataloader.yml 的 stop_epoch policy）。scale_range 线性
        收紧到 final_scale_range，modality_dropout_p 线性退火到 0，使模型
        在收尾阶段用完整三模态输入定型。
        """
        # DEIM-style close-out: the last epochs use deployment geometry only.
        # Keep this separate from the normal linear decay for clean ablations.
        self.hflip_p = self.base_hflip_p
        self.photometric_enabled = True
        if self.no_aug_epoch is not None and epoch >= self.no_aug_epoch:
            self.scale_range = (1.0, 1.0)
            self.modality_dropout_p = 0.0
            self.hflip_p = 0.0
            self.photometric_enabled = False
            return

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
        quality_mask: np.ndarray,  # HxW float32 [0,1]
        depth_kind: str,
        boxes_cxcywh: np.ndarray,  # Nx4 normalized [0,1], DETR convention
        labels: np.ndarray,       # N,
    ):
        h0, w0 = visible.shape[:2]

        vis = TF.to_tensor(visible)                      # 3,H,W float[0,1]
        ir = torch.from_numpy(infrared).float().unsqueeze(0) / 255.0   # 1,H,W
        dep = torch.from_numpy(depth_norm).unsqueeze(0)                # 1,H,W
        msk = torch.from_numpy(valid_mask).unsqueeze(0)                # 1,H,W
        quality = torch.from_numpy(quality_mask).unsqueeze(0)          # 1,H,W

        boxes = torch.from_numpy(boxes_cxcywh.copy()) if boxes_cxcywh.size else torch.zeros((0, 4))
        labels_t = torch.from_numpy(labels.copy()).long() if labels.size else torch.zeros((0,), dtype=torch.long)

        if self.train:
            vis, ir, dep, msk, quality, boxes = self._random_resized_crop(
                vis, ir, dep, msk, quality, boxes
            )
            if random.random() < self.hflip_p:
                vis, ir, dep, msk, quality = (
                    TF.hflip(t) for t in (vis, ir, dep, msk, quality)
                )
                if boxes.numel():
                    boxes[:, 0] = 1.0 - boxes[:, 0]
            vis = self._photometric(vis)
        else:
            vis, ir, dep, msk, quality = self._resize_only(vis, ir, dep, msk, quality)

        if self.train and boxes.numel():
            # 阈值按输出像素算，不用归一化常数：旧的 1e-4 归一化 ≈ 0.06px，形同没过滤，
            # 裁剪后只剩一条线的退化框会照样进 Hungarian 匹配，污染回归目标。对齐 D-FINE
            # 官方 SanitizeBoundingBoxes(min_size=1)，这里取 2px 略严一点。
            out_h, out_w = self.size_hw
            keep = (boxes[:, 2] * out_w >= self.min_box_px) & (boxes[:, 3] * out_h >= self.min_box_px)
            boxes = boxes[keep]
            labels_t = labels_t[keep]

        ir_distill_valid = 1.0
        depth_distill_valid = float(msk.any())
        if self.train and random.random() < self.modality_dropout_p:
            if random.random() < 0.5:
                ir = torch.zeros_like(ir)
                ir_distill_valid = 0.0
            else:
                dep = torch.zeros_like(dep)
                msk = torch.zeros_like(msk)
                quality = torch.zeros_like(quality)
                depth_distill_valid = 0.0

        mm16 = torch.full_like(dep, float(depth_kind == DEPTH_KIND_MM16))
        disp8 = torch.full_like(dep, float(depth_kind == DEPTH_KIND_DISP8))
        if depth_distill_valid == 0.0:
            mm16.zero_()
            disp8.zero_()
        quality = quality.clamp(0.0, 1.0) * msk
        depth_stack = torch.cat([dep, msk, mm16, disp8, quality], dim=0)

        target = {
            "labels": labels_t,
            "boxes": boxes.float(),
            "orig_size": torch.tensor([w0, h0], dtype=torch.float32),
            "ir_distill_valid": torch.tensor(ir_distill_valid, dtype=torch.float32),
            "depth_distill_valid": torch.tensor(depth_distill_valid, dtype=torch.float32),
        }
        return vis, ir, depth_stack, target

    def _resize_only(self, vis, ir, dep, msk, quality):
        out = list(self.size_hw)
        vis = TF.resize(vis, out, antialias=True)
        ir = TF.resize(ir, out, antialias=True)
        dep = TF.resize(dep, out, antialias=True)
        msk = TF.resize(msk, out, antialias=False)
        quality = TF.resize(quality, out, antialias=True)
        return vis, ir, dep, msk, quality

    def _random_resized_crop(self, vis, ir, dep, msk, quality, boxes):
        """scale>1 放大（裁小窗口），scale<1 缩小（先 zero-pad 出更大画布再取窗口）。

        旧实现用 `crop_h = min(h, int(h / scale))`，scale<1 时 int(h/scale) > h，被
        min 钳回 h —— 于是 scale_range 的整个下半段 (0.5, 1.0) 完全是 no-op，模型从
        未见过"目标变小"的样本。而 640 方形输入已经把目标面积压到 0.44 倍、47% 的框
        落进 <32px 的小目标桶，正是最需要 zoom-out 覆盖的尺度区间。这里补上 padding
        路径，语义对齐 D-FINE 官方的 RandomZoomOut(fill=0)。
        """
        _, h, w = vis.shape
        scale = random.uniform(*self.scale_range)
        win_h = max(1, int(round(h / scale)))
        win_w = max(1, int(round(w / scale)))

        # scale<1：窗口比原图大，补 0 到窗口尺寸（随机分配到四边，等价于随机平移）
        pad_h, pad_w = max(0, win_h - h), max(0, win_w - w)
        pad_top = random.randint(0, pad_h) if pad_h else 0
        pad_left = random.randint(0, pad_w) if pad_w else 0
        if pad_h or pad_w:
            padding = [pad_left, pad_top, pad_w - pad_left, pad_h - pad_top]  # l, t, r, b
            vis, ir, dep, msk, quality = (
                TF.pad(t, padding, fill=0) for t in (vis, ir, dep, msk, quality)
            )

        # padding 后画布是 (max(h,win_h), max(w,win_w))，窗口一定放得下
        top = random.randint(0, max(h, win_h) - win_h)
        left = random.randint(0, max(w, win_w) - win_w)

        out = list(self.size_hw)
        vis = TF.resized_crop(vis, top, left, win_h, win_w, out, antialias=True)
        ir = TF.resized_crop(ir, top, left, win_h, win_w, out, antialias=True)
        dep = TF.resized_crop(dep, top, left, win_h, win_w, out, antialias=True)
        msk = TF.resized_crop(msk, top, left, win_h, win_w, out, antialias=False)
        quality = TF.resized_crop(quality, top, left, win_h, win_w, out, antialias=True)

        if boxes.numel():
            # 原图像素 → padding 后画布 → 减去窗口原点，合并为单个偏移
            off_x, off_y = pad_left - left, pad_top - top
            cx, cy, bw, bh = boxes.unbind(-1)
            x1, y1 = (cx - bw / 2) * w + off_x, (cy - bh / 2) * h + off_y
            x2, y2 = (cx + bw / 2) * w + off_x, (cy + bh / 2) * h + off_y
            x1, x2 = x1.clamp(0, win_w), x2.clamp(0, win_w)
            y1, y2 = y1.clamp(0, win_h), y2.clamp(0, win_h)
            boxes = torch.stack(
                [(x1 + x2) / 2 / win_w, (y1 + y2) / 2 / win_h, (x2 - x1) / win_w, (y2 - y1) / win_h],
                dim=-1,
            )

        return vis, ir, dep, msk, quality, boxes

    def _photometric(self, vis: torch.Tensor) -> torch.Tensor:
        if not self.photometric_enabled:
            return vis
        if random.random() < 0.8:
            vis = TF.adjust_brightness(vis, random.uniform(0.7, 1.3))
        if random.random() < 0.8:
            vis = TF.adjust_contrast(vis, random.uniform(0.7, 1.3))
        if random.random() < 0.5:
            vis = TF.adjust_saturation(vis, random.uniform(0.7, 1.3))
        if random.random() < 0.3:
            vis = TF.adjust_hue(vis, random.uniform(-0.05, 0.05))
        return vis.clamp(0, 1)

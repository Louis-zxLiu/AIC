"""snl1（尺度归一化 L1）匈牙利匹配代价，作为官方 HungarianMatcher 的可选变体。

离线验证见 _match.py：把 cost_bbox 从"归一化坐标绝对 L1"换成"按 GT 自身
像素宽高归一化的相对 L1"，77% 的匹配分配发生变化，小目标（<32px）平均
ΔIoU +0.105。本文件把该公式适配进 D-FINE 官方 HungarianMatcher.forward()
的批量 flatten 张量结构（out_bbox: [bs*num_queries,4]，tgt_bbox: [total_targets,4]），
其余逻辑（cost_class、cost_giou、线性分配求解）完全复用父类，不改动
third_party/D-FINE 源码。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from third_party_dfine import HungarianMatcher
from src.zoo.dfine.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

from aic.models.gcd import gcd_cost_matrix


class SNL1HungarianMatcher(HungarianMatcher):
    """cost_type="snl1" 时，cost_bbox 按 GT 自身宽高做相对误差归一化；
    cost_type="base"（默认）时行为与官方 HungarianMatcher 完全一致。
    """

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0,
                 cost_type: str = "snl1", image_size=None):
        super().__init__(weight_dict, use_focal_loss=use_focal_loss, alpha=alpha, gamma=gamma)
        assert cost_type in ("base", "snl1"), f"未知 cost_type: {cost_type}"
        self.cost_type = cost_type
        # 归一化坐标 -> 像素坐标需要图像尺寸；训练/推理 image_size 固定，直接从
        # config 传入 [h, w]，与 cxcywh 的 w,h,w,h 分母顺序对齐时只需相对比例，
        # 用 [w, h] 广播即可，不要求绝对像素值精确，只影响 clamp(min=1.0) 的量纲。
        self.image_size_hw = image_size

    @torch.no_grad()
    def forward(self, outputs, targets, return_topk=False):
        if self.cost_type == "base":
            return super().forward(outputs, targets, return_topk=return_topk)

        bs, num_queries = outputs["pred_logits"].shape[:2]

        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [bs*num_queries, 4]

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])  # [total_targets, 4], cxcywh 归一化坐标

        if self.use_focal_loss:
            out_prob_t = out_prob[:, tgt_ids]
            neg_cost_class = (
                (1 - self.alpha) * (out_prob_t**self.gamma) * (-(1 - out_prob_t + 1e-8).log())
            )
            pos_cost_class = (
                self.alpha * ((1 - out_prob_t) ** self.gamma) * (-(out_prob_t + 1e-8).log())
            )
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -out_prob[:, tgt_ids]

        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # snl1：把归一化坐标转成"像素等效"坐标（用 [w,h] 广播即可，clamp 阈值
        # 1.0 像素防止极小 GT 除零），按 GT 自身 w,h,w,h 做相对误差归一化，
        # 对应 _match.py 的 cost_snl1 公式，扩展到 [bs*num_queries, total_targets] 的
        # 批量 cdist 形状（逐元素相对误差求和，而非 cdist 的 L1 距离）。
        h, w = (self.image_size_hw if self.image_size_hw is not None else (1.0, 1.0))
        scale = torch.tensor([w, h, w, h], device=out_bbox.device, dtype=out_bbox.dtype)
        out_bbox_px = out_bbox * scale
        tgt_bbox_px = tgt_bbox * scale

        denom = torch.cat([tgt_bbox_px[:, 2:], tgt_bbox_px[:, 2:]], dim=-1).clamp(min=1.0)  # [total_targets, 4]
        diff = (out_bbox_px[:, None, :] - tgt_bbox_px[None, :, :]).abs()  # [bs*num_queries, total_targets, 4]
        cost_bbox = (diff / denom[None, :, :]).sum(-1)  # [bs*num_queries, total_targets]

        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        C = torch.nan_to_num(C, nan=1.0)
        indices_pre = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices_pre
        ]

        if return_topk:
            return {
                "indices_o2m": self.get_top_k_matches(
                    C, sizes=sizes, k=return_topk, initial_indices=indices_pre
                )
            }

        return {"indices": indices}


class GCDHungarianMatcher(HungarianMatcher):
    """cost_type="gcd" 时，cost_bbox 用 Gaussian Combined Distance（计划.md §2.7，
    aic/models/gcd.py）替代原始 L1；cost_type="base" 时行为与官方
    HungarianMatcher 完全一致。GCD 具有仿射不变性，不需要像 snl1 那样传入
    image_size 做像素换算，归一化坐标直接算即可。
    """

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0,
                 cost_type: str = "gcd"):
        super().__init__(weight_dict, use_focal_loss=use_focal_loss, alpha=alpha, gamma=gamma)
        assert cost_type in ("base", "gcd"), f"未知 cost_type: {cost_type}"
        self.cost_type = cost_type

    @torch.no_grad()
    def forward(self, outputs, targets, return_topk=False):
        if self.cost_type == "base":
            return super().forward(outputs, targets, return_topk=return_topk)

        bs, num_queries = outputs["pred_logits"].shape[:2]

        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [bs*num_queries, 4]

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])  # [total_targets, 4], cxcywh 归一化坐标

        if self.use_focal_loss:
            out_prob_t = out_prob[:, tgt_ids]
            neg_cost_class = (
                (1 - self.alpha) * (out_prob_t**self.gamma) * (-(1 - out_prob_t + 1e-8).log())
            )
            pos_cost_class = (
                self.alpha * ((1 - out_prob_t) ** self.gamma) * (-(out_prob_t + 1e-8).log())
            )
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -out_prob[:, tgt_ids]

        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        cost_bbox = gcd_cost_matrix(out_bbox, tgt_bbox)  # [bs*num_queries, total_targets]

        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        C = torch.nan_to_num(C, nan=1.0)
        indices_pre = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices_pre
        ]

        if return_topk:
            return {
                "indices_o2m": self.get_top_k_matches(
                    C, sizes=sizes, k=return_topk, initial_indices=indices_pre
                )
            }

        return {"indices": indices}

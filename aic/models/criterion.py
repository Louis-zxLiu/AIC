"""GCDCriterion：在官方 DFINECriterion 基础上追加 loss_gcd 项（计划.md §2.7）。

只重写 loss_boxes，额外算一个 gcd_loss 并塞进返回的 losses dict，key 为
"loss_gcd"。是否真正生效完全由 config 的 criterion.weight_dict 是否包含
loss_gcd 决定——DFINECriterion.forward 里聚合 aux/dn/enc 各路 loss 时统一走
`{k: l_dict[k]*weight_dict[k] for k in l_dict if k in weight_dict}`，
没配置 loss_gcd 权重时该 key 会被自动过滤掉，等价于不加这个 loss，
不需要额外开关，也不会影响 vfl/local 等其它 loss 分支。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from third_party_dfine import DFINECriterion
from src.zoo.dfine.box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou

from aic.models.gcd import gcd_loss


class MALCriterion(DFINECriterion):
    """D-FINE criterion with DEIM-style Matchability-Aware Loss.

    This first ablation uses the matched box IoU as the matchability target q.
    Feature-similarity terms from the full DEIM implementation are deliberately
    omitted because our multimodal wrapper does not expose the same matching
    features; keeping q deterministic makes the loss a clean, reproducible
    single-variable comparison against VFL.
    """

    def __init__(self, *args, mal_gamma: float = 1.5, **kwargs):
        super().__init__(*args, **kwargs)
        if mal_gamma <= 0:
            raise ValueError("mal_gamma must be positive")
        self.mal_gamma = float(mal_gamma)

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        assert "pred_logits" in outputs and "pred_boxes" in outputs
        src_logits = outputs["pred_logits"]
        probabilities = src_logits.sigmoid().detach()
        gamma = float(getattr(self, "mal_gamma", 1.5))

        # Unmatched queries receive the MAL negative term for every class.
        log_p = F.logsigmoid(src_logits)
        log_not_p = F.logsigmoid(-src_logits)
        losses = -(probabilities.pow(gamma) * log_not_p)

        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel():
            target_labels = torch.cat(
                [t["labels"][j] for t, (_, j) in zip(targets, indices)], dim=0
            )
            if values is None:
                src_boxes = outputs["pred_boxes"][idx]
                target_boxes = torch.cat(
                    [t["boxes"][j] for t, (_, j) in zip(targets, indices)], dim=0
                )
                from src.zoo.dfine.box_ops import box_cxcywh_to_xyxy, box_iou

                q = torch.diag(
                    box_iou(
                        box_cxcywh_to_xyxy(src_boxes),
                        box_cxcywh_to_xyxy(target_boxes),
                    )[0]
                ).detach().clamp(0.0, 1.0)
            else:
                q = values.detach().clamp(0.0, 1.0)

            # Replace the matched class entry by quality-aware BCE.  The
            # q**gamma factor suppresses noisy low-quality positives while the
            # complementary term still discourages overconfident predictions.
            row, col = idx
            q_gamma = q.pow(gamma)
            losses[row, col, target_labels] = (
                -q_gamma * log_p[row, col, target_labels]
                -(1.0 - q_gamma) * log_not_p[row, col, target_labels]
            )

        loss = losses.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_vfl": loss}


class GCLCriterion(DFINECriterion):
    """D-FINE VFL with Rank-DETR's GIoU-aware positive target.

    The original D-FINE VFL uses matched IoU as the positive class target.
    GCL replaces it with ``(GIoU + 1) / 2`` while preserving the negative
    focal weighting and every auxiliary/denoising loss path in the parent
    criterion.  The target is detached, as in the original VFL implementation.
    """

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)

        src_logits = outputs["pred_logits"]
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        if idx[0].numel():
            src_boxes = outputs["pred_boxes"][idx]
            target_boxes = torch.cat(
                [t["boxes"][J] for t, (_, J) in zip(targets, indices)], dim=0
            )
            # GIoU is in [-1, 1]; map it to the valid VFL target range [0, 1].
            giou = torch.diag(
                generalized_box_iou(
                    box_cxcywh_to_xyxy(src_boxes),
                    box_cxcywh_to_xyxy(target_boxes),
                )
            ).detach()
            target_score_o[idx] = (
                ((giou + 1.0) * 0.5).clamp(0.0, 1.0).to(target_score_o.dtype)
            )

        target_score = target_score_o.unsqueeze(-1) * target
        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
        loss = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction="none"
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_vfl": loss}


class MALGCLCriterion(MALCriterion):
    """MAL negative weighting with GCL's GIoU-aware positive target.

    This is the explicit compatibility ablation: it keeps MAL's matchability-
    aware focal treatment for unmatched queries and replaces the matched IoU
    quality target by ``(GIoU + 1) / 2``.  All other D-FINE, M2D and EMA paths
    remain unchanged.
    """

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        assert "pred_logits" in outputs and "pred_boxes" in outputs
        src_logits = outputs["pred_logits"]
        probabilities = src_logits.sigmoid().detach()
        gamma = float(getattr(self, "mal_gamma", 1.5))

        log_p = F.logsigmoid(src_logits)
        log_not_p = F.logsigmoid(-src_logits)
        losses = -(probabilities.pow(gamma) * log_not_p)

        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel():
            target_labels = torch.cat(
                [t["labels"][j] for t, (_, j) in zip(targets, indices)], dim=0
            )
            # Compute GIoU from the matched boxes directly.  This intentionally
            # ignores any optional parent ``values`` metadata: MAL+GCL must use
            # the transformed GIoU target in every auxiliary/denoising path.
            src_boxes = outputs["pred_boxes"][idx]
            target_boxes = torch.cat(
                [t["boxes"][j] for t, (_, j) in zip(targets, indices)], dim=0
            )
            giou = torch.diag(
                generalized_box_iou(
                    box_cxcywh_to_xyxy(src_boxes),
                    box_cxcywh_to_xyxy(target_boxes),
                )
            )
            q = ((giou + 1.0) * 0.5).detach().clamp(0.0, 1.0)

            row, col = idx
            q_gamma = q.pow(gamma)
            losses[row, col, target_labels] = (
                -q_gamma * log_p[row, col, target_labels]
                -(1.0 - q_gamma) * log_not_p[row, col, target_labels]
            )

        loss = losses.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_vfl": loss}


class GCDCriterion(DFINECriterion):
    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}

        loss_giou = 1 - torch.diag(
            generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        )
        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
        losses["loss_giou"] = loss_giou.sum() / num_boxes

        # 与 GCDHungarianMatcher 用同一公式（aic/models/gcd.py gcd_loss），保证
        # 匹配目标与优化目标一致——这是 snl1（计划.md §2.2）失败教训的直接对策。
        losses["loss_gcd"] = gcd_loss(src_boxes, target_boxes).sum() / num_boxes

        return losses

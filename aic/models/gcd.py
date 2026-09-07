"""Gaussian Combined Distance (GCD)，arXiv 2510.27649 Eq.2-3/9。

把 bbox 建模为 2D 高斯 N(mu, Sigma)，Sigma = diag(w^2/4, h^2/4)，
D_gc^2 是两个高斯之间的对称距离（预测框协方差 + GT 框协方差各算一次
中心+形状项再相加），经证明具有仿射不变性（不依赖坐标是归一化还是
像素尺度），比 Wasserstein Distance 多了按框尺寸动态缩放的梯度权重
（原文 Eq.7-8），对小目标定位更敏感。

Eq.9 用 M_gcd = exp(-sqrt(D_gc^2)) 把无上界的距离压缩到 [0,1)，
1-M_gcd 可直接当 loss/cost 使用，量级与 GIoU loss（1-GIoU，[0,2]）
相近，不需要重新调 weight_dict。
"""

from __future__ import annotations

import torch

_EPS = 1e-6


def _gcd_sq(xp, yp, wp, hp, xt, yt, wt, ht):
    wp2, hp2 = wp.clamp(min=_EPS) ** 2, hp.clamp(min=_EPS) ** 2
    wt2, ht2 = wt.clamp(min=_EPS) ** 2, ht.clamp(min=_EPS) ** 2

    term_p = 0.5 * ((xp - xt) ** 2 / wp2 + (yp - yt) ** 2 / hp2) + 0.5 * (
        (wp - wt) ** 2 / (4 * wp2) + (hp - ht) ** 2 / (4 * hp2)
    )
    term_t = 0.5 * ((xt - xp) ** 2 / wt2 + (yt - yp) ** 2 / ht2) + 0.5 * (
        (wt - wp) ** 2 / (4 * wt2) + (ht - hp) ** 2 / (4 * ht2)
    )
    return term_p + term_t


def gcd_cost_matrix(boxes_p: torch.Tensor, boxes_t: torch.Tensor) -> torch.Tensor:
    """匈牙利匹配用：boxes_p [N,4]、boxes_t [M,4]（cxcywh，归一化或像素坐标均可，
    仿射不变性保证结果一致），返回 [N,M] 的 1-M_gcd 代价矩阵，越小越匹配。
    """
    xp, yp, wp, hp = (boxes_p[:, i : i + 1] for i in range(4))
    xt, yt, wt, ht = (boxes_t[None, :, i] for i in range(4))
    d2 = _gcd_sq(xp, yp, wp, hp, xt, yt, wt, ht)
    return 1.0 - torch.exp(-torch.sqrt(d2.clamp(min=0.0) + _EPS))


def gcd_loss(boxes_p: torch.Tensor, boxes_t: torch.Tensor) -> torch.Tensor:
    """训练 loss 用：boxes_p/boxes_t 均为 [N,4]（已按匹配结果对齐），
    返回 [N] 的逐框 1-M_gcd，与匹配代价用同一公式，保证匹配目标与
    优化目标一致（对策 snl1 的失败教训，见计划.md §2.2/§2.7）。
    """
    xp, yp, wp, hp = (boxes_p[:, i] for i in range(4))
    xt, yt, wt, ht = (boxes_t[:, i] for i in range(4))
    d2 = _gcd_sq(xp, yp, wp, hp, xt, yt, wt, ht)
    return 1.0 - torch.exp(-torch.sqrt(d2.clamp(min=0.0) + _EPS))

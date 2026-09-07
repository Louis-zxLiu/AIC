"""OPF (Orthogonal Projection Fusion) 三模态融合模块。

设计依据（计划.md §1.3）：脱胎于赛题主办方自己的 RDTTrack (NeurIPS 2025) 论文中的
正交投影思想，但原始公式存在三个问题，此处逐一修正：

1. 原文用单一全局标量系数控制投影去冗余的比例，这里改为按通道可学习的
   α（去冗余强度）和 β（恢复/重缩放强度）向量，让网络自己决定哪些通道在
   该模态更该保留、哪些该剔除，而不是所有通道一刀切。
2. 原文的 ‖·‖² 分母在特征接近零向量时数值不稳定，这里加 eps 并对分母做
   detach 处理防止梯度爆炸（分母中的二阶项会放大梯度噪声）。
3. 原文同时对 depth 和 ir 各自独立地对 RGB 做一次投影（并行），这里改成
   串行：先用 RGB 去相关 depth，再用"RGB + 已去相关的 depth"去相关 ir，
   避免 ir 中和 depth 冗余的部分被重复保留（并行版本会让两个模态都各自
   独立去掉和 RGB 的冗余，但不会去掉彼此之间的冗余）。

三个步骤：
  Step 1: OrthogonalDecorrelation  —— 上述改进的正交投影
  Step 2: ConfidenceGating         —— 每个模态一个空间置信度门（对深度尤其
                                        重要：无效深度区域的 valid_mask 均值
                                        直接作为该样本深度门控的先验）
  Step 3: CrossModalAttentionFusion —— RGB 作 query，去冗余+门控后的
                                        depth/ir 作 key/value，残差加回 RGB
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from aic.models.dwt import HaarDWT2D, HaarIDWT2D


class OrthogonalDecorrelation(nn.Module):
    """将 feat_b 中与 feat_a 冗余的成分投影去除，通道级可学习强度。"""

    def __init__(self, channels: int, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.full((channels,), 0.5))  # 去冗余强度
        self.beta = nn.Parameter(torch.ones(channels))           # 重缩放强度

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        # feat_a, feat_b: B,C,H,W。逐像素在通道维上做投影（把每个空间位置的
        # C维向量视为一个待去相关的向量），比逐通道在空间维上做更符合"同一
        # 位置不同模态语义冗余"的直觉。
        b, c, h, w = feat_a.shape
        a = feat_a.permute(0, 2, 3, 1).reshape(-1, c)  # N,C
        bb = feat_b.permute(0, 2, 3, 1).reshape(-1, c)

        dot = (a * bb).sum(dim=-1, keepdim=True)              # N,1
        norm_sq = (a * a).sum(dim=-1, keepdim=True) + self.eps  # 修正: 加 eps 防止除零
        coef = dot / norm_sq.detach()                          # 修正: 分母 detach 稳定梯度

        proj = coef * a                                        # N,C，b 在 a 方向上的投影
        alpha = self.alpha.clamp(0.0, 1.0)
        decorrelated = bb - alpha * proj                       # 通道级可控去冗余强度
        decorrelated = self.beta * decorrelated                # 通道级重缩放

        return decorrelated.reshape(b, h, w, c).permute(0, 3, 1, 2)


class ConfidenceGate(nn.Module):
    """基于特征自身统计（+可选显式先验，如深度 valid_mask）估计该模态在每个
    空间位置的可信度，退化/缺失模态自动被抑制。

    mask_prior：外部先验通道（如深度的 valid_mask 下采样到当前尺度），与
    特征拼接后一起决定门控值，而不是让网络仅从学到的特征里隐式反推"哪里
    的深度是无效的"。ir 分支没有对应先验，传 None 时退化为纯特征驱动。
    """

    def __init__(
        self,
        channels: int,
        use_mask_prior: bool = False,
        hard_mask_prior: bool = False,
    ):
        super().__init__()
        self.use_mask_prior = use_mask_prior
        self.hard_mask_prior = hard_mask_prior
        in_ch = channels + (1 if use_mask_prior else 0)
        self.gate = nn.Sequential(
            nn.Conv2d(in_ch, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor, mask_prior: torch.Tensor | None = None) -> torch.Tensor:
        x = feat
        if self.use_mask_prior and self.hard_mask_prior and mask_prior is not None:
            x = torch.cat([feat, mask_prior], dim=1)
        g = self.gate(x)  # B,1,H,W
        if self.use_mask_prior and mask_prior is not None:
            # The learned gate may refine confidence where depth exists, but
            # it must not resurrect a region whose measured quality is zero.
            g = g * mask_prior.clamp(0.0, 1.0)
        return feat * g


class CrossModalAttentionFusion(nn.Module):
    """RGB 作 query，非 RGB 模态作 key/value 的交叉注意力。

    K/V 空间降采样（PVT 的 Spatial-Reduction Attention）：原始实现里 Q 和 K/V
    都是全分辨率，注意力矩阵是 O((h·w)²) —— 640 输入的 stride-8 尺度上
    h·w=6400，单个 attn 矩阵就有 6400²×8heads，一个 block 还要算两次
    （depth、ir），直接把 batch_size 逼到 1，连带毁掉 BatchNorm 统计量并让每
    epoch 的 optimizer step 数掉到 50。

    这里 Q 保持全分辨率（输出仍是逐像素的，细粒度对齐不丢），K/V 先经一层
    stride=sr_ratio 的卷积做空间降采样，复杂度降到 O(h·w · h·w/sr²)。语义上
    这是承认"跨模态检索的 key 不需要逐像素"——三个模态本身是空间配准的，
    粗一级的 key 足以定位对应区域。

    sr_ratio 按尺度递减（浅层降得多、深层不降），使各尺度 K/V 长度大致相等。
    另外用 scaled_dot_product_attention 走 flash/mem-efficient kernel，注意力
    矩阵不会被物化出来。
    """

    def __init__(self, channels: int, num_heads: int = 8, sr_ratio: int = 1):
        super().__init__()
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1)
        if sr_ratio > 1:
            # 降采样用 stride 卷积而非 pooling：让网络自己学怎么把局部邻域压成
            # 一个 key，而不是无参数地平均掉。GroupNorm 而非 BatchNorm——这里
            # 的 batch 维统计对融合没有意义，且避免再引入一处 BN 依赖。
            self.kv_down = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=sr_ratio, stride=sr_ratio),
                nn.GroupNorm(num_groups=32 if channels % 32 == 0 else 1, num_channels=channels),
            )
        else:
            self.kv_down = nn.Identity()
        self.kv_proj = nn.Conv2d(channels, channels * 2, kernel_size=1)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=32 if channels % 32 == 0 else 1, num_channels=channels)

    def forward(self, rgb_feat: torch.Tensor, other_feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = rgb_feat.shape
        heads = self.num_heads
        hd = c // heads

        # SDPA 要求 (B, heads, seq, head_dim)
        q = self.q_proj(rgb_feat).reshape(b, heads, hd, h * w).transpose(-2, -1)

        kv_src = self.kv_down(other_feat)
        kv_len = kv_src.shape[-2] * kv_src.shape[-1]
        kv = self.kv_proj(kv_src).reshape(b, 2, heads, hd, kv_len)
        k = kv[:, 0].transpose(-2, -1)
        v = kv[:, 1].transpose(-2, -1)

        out = F.scaled_dot_product_attention(q, k, v)  # B,heads,h*w,hd
        out = out.transpose(-2, -1).reshape(b, c, h, w)
        out = self.out_proj(out)

        return self.norm(rgb_feat + out)


class OPFFusionBlock(nn.Module):
    """单一尺度上的完整 OPF 融合：去冗余 -> 置信度门控 -> 交叉注意力融合。

    depth 分支的 ConfidenceGate 接入 valid_mask 显式先验（见 forward 的
    depth_mask 参数）：无效深度区域的先验直接参与门控，而不是仅靠网络从
    depth 特征统计里隐式反推。ir 没有对应的显式有效性信号，门控保持纯
    特征驱动。

    use_dwt=True 时（计划.md §2.3 P1.5），三路输入先各自做一次 Haar DWT，
    拆成 1 个低频子带（LL）+ 3 个高频子带（LH/HL/HH）：
      - 低频 LL 承载全局几何/结构信息，走上述完整的 decorr+gate+attention
        流程（depth 本质低频主导，这条路径的正交解耦语义仍然成立）。
      - 高频 LH/HL/HH 承载边缘/纹理细节，参照 WaveMamba 的做法逐像素取三个
        模态里绝对值更大的那个，不学习融合权重——物理意义是"更强边缘信号
        更可信"，避免可学习融合在训练不足时把边缘细节和噪声一起抑制。
    最后 IDWT 把融合后的低频 + 选择后的高频合回单一分辨率特征。
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        sr_ratio: int = 1,
        use_dwt: bool = False,
        hard_depth_gate: bool = False,
    ):
        super().__init__()
        self.decorr_depth = OrthogonalDecorrelation(channels)
        self.decorr_ir = OrthogonalDecorrelation(channels)
        self.gate_depth = ConfidenceGate(
            channels,
            use_mask_prior=True,
            hard_mask_prior=hard_depth_gate,
        )
        self.gate_ir = ConfidenceGate(channels, use_mask_prior=False)
        self.fuse_depth = CrossModalAttentionFusion(channels, num_heads, sr_ratio)
        self.fuse_ir = CrossModalAttentionFusion(channels, num_heads, sr_ratio)

        self.use_dwt = use_dwt
        if use_dwt:
            self.dwt = HaarDWT2D(channels)
            self.idwt = HaarIDWT2D(channels)

    def _fuse_low_freq(
        self, rgb: torch.Tensor, ir: torch.Tensor, depth: torch.Tensor, depth_mask: torch.Tensor | None
    ) -> torch.Tensor:
        depth_decorr = self.decorr_depth(rgb, depth)
        depth_gated = self.gate_depth(depth_decorr, mask_prior=depth_mask)
        fused = self.fuse_depth(rgb, depth_gated)

        ir_decorr = self.decorr_ir(fused, ir)
        ir_gated = self.gate_ir(ir_decorr)
        fused = self.fuse_ir(fused, ir_gated)

        return fused

    @staticmethod
    def _select_high_freq(rgb_hf: torch.Tensor, ir_hf: torch.Tensor, depth_hf: torch.Tensor) -> torch.Tensor:
        # 逐像素取三者中绝对值最大的（WaveMamba 式高频融合），保留符号。
        stacked = torch.stack([rgb_hf, ir_hf, depth_hf], dim=0)  # 3,B,C,H,W
        idx = stacked.abs().argmax(dim=0, keepdim=True)
        return stacked.gather(0, idx).squeeze(0)

    def forward(
        self, rgb: torch.Tensor, ir: torch.Tensor, depth: torch.Tensor, depth_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if not self.use_dwt:
            return self._fuse_low_freq(rgb, ir, depth, depth_mask)

        h, w = rgb.shape[-2:]
        rgb_ll, rgb_lh, rgb_hl, rgb_hh = self.dwt(rgb)
        ir_ll, ir_lh, ir_hl, ir_hh = self.dwt(ir)
        depth_ll, depth_lh, depth_hl, depth_hh = self.dwt(depth)

        depth_mask_ll = None
        if depth_mask is not None:
            depth_mask_ll = F.interpolate(depth_mask, size=rgb_ll.shape[-2:], mode="nearest")

        fused_ll = self._fuse_low_freq(rgb_ll, ir_ll, depth_ll, depth_mask_ll)
        fused_lh = self._select_high_freq(rgb_lh, ir_lh, depth_lh)
        fused_hl = self._select_high_freq(rgb_hl, ir_hl, depth_hl)
        fused_hh = self._select_high_freq(rgb_hh, ir_hh, depth_hh)

        return self.idwt(fused_ll, fused_lh, fused_hl, fused_hh, out_hw=(h, w))


class MultiScaleOPF(nn.Module):
    """对 backbone 输出的每个尺度分别应用独立的 OPF 融合模块。

    per-scale 独立参数（而非跨尺度共享）：因为不同尺度语义层级不同
    （浅层更多纹理/边缘冗余，深层更多语义级冗余），共享会强迫网络在
    所有尺度上用同一套去冗余策略。

    sr_ratios：各尺度交叉注意力的 K/V 空间降采样倍率，默认 (4, 2, 1) 对应
    stride 8/16/32 三个尺度——降采样后三者的 K/V 序列长度相同（640 输入下
    各为 20×20=400），跨模态检索的代价在各尺度上均衡，不会被最浅层主导。
    """

    def __init__(
        self,
        channels_per_scale: list[int],
        num_heads: int = 8,
        sr_ratios: tuple[int, ...] | list[int] | None = None,
        use_dwt: bool = False,
        hard_depth_gate: bool = False,
    ):
        super().__init__()
        if sr_ratios is None:
            # 由深到浅递增：最后一个尺度（最深、分辨率最低）不降采样。
            n = len(channels_per_scale)
            sr_ratios = [2 ** (n - 1 - i) for i in range(n)]
        if len(sr_ratios) != len(channels_per_scale):
            raise ValueError(
                f"sr_ratios 长度 {len(sr_ratios)} 与尺度数 {len(channels_per_scale)} 不一致"
            )
        self.hard_depth_gate = bool(hard_depth_gate)
        self.blocks = nn.ModuleList(
            [
                OPFFusionBlock(
                    c,
                    num_heads,
                    sr,
                    use_dwt=use_dwt,
                    hard_depth_gate=hard_depth_gate,
                )
                for c, sr in zip(channels_per_scale, sr_ratios)
            ]
        )

    def forward(
        self,
        rgb_feats: list[torch.Tensor],
        ir_feats: list[torch.Tensor],
        depth_feats: list[torch.Tensor],
        depth_valid_mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        # depth_valid_mask is now a continuous quality map. Area resampling
        # preserves valid coverage instead of dropping sparse pixels through
        # nearest-neighbor sampling at stride 8/16/32.
        masks_per_scale = [None] * len(self.blocks)
        if depth_valid_mask is not None:
            resize_mode = "area" if self.hard_depth_gate else "nearest"
            masks_per_scale = [
                F.interpolate(depth_valid_mask, size=feat.shape[-2:], mode=resize_mode)
                for feat in rgb_feats
            ]
        return [
            block(r, i, d, depth_mask=m)
            for block, r, i, d, m in zip(self.blocks, rgb_feats, ir_feats, depth_feats, masks_per_scale)
        ]

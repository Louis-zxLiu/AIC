"""三流共享骨干（Tri-Stream Shared-Backbone HGNetv2）。

设计依据（计划.md §1.2）：赛题数据仅 2000 组，若三个模态各用一套独立
HGNetv2（如 double-co-detr 的双流独立骨干设计），参数量 x3 会在如此小的
数据集上过拟合。此处三个模态共享同一套 stage 权重（即共享除 stem 外的
全部骨干参数），只有输入层 stem 按模态通道数各自独立：
    RGB   stem: in_chs=3 （标准彩色）
    IR    stem: in_chs=1 （单通道热辐射灰度）
    Depth stem: in_chs=2 （逆深度归一化值 + 有效性 mask）

共享 stage 权重的物理直觉：三个模态本质上都是在描述同一物理场景的空间
结构（边缘、纹理、目标轮廓），只是成像原理不同，这部分先验没有理由三份
独立学习；差异主要体现在低层输入统计特性上，交给各自 stem 吸收。

新 stem 的 1x1/3x3 首层卷积从预训练 RGB stem 的对应层按输入通道求和平均
初始化（而非随机初始化），让 IR/Depth 分支从"RGB 预训练特征的合理近似"
起步，而不是从零学习，这在小数据集上能显著加快收敛、降低过拟合风险。
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from third_party_dfine import HGNetv2, StemBlock


def _init_stem_from_rgb(new_stem: StemBlock, rgb_stem: StemBlock, new_in_chs: int) -> None:
    """用预训练 RGB stem 首层卷积权重，按输入通道求和平均后广播初始化新 stem。"""
    with torch.no_grad():
        rgb_conv = rgb_stem.stem1.conv
        new_conv = new_stem.stem1.conv
        # rgb_conv 可能是 Sequential（padding="same" 时）或直接 Conv2d，统一取出真正的卷积层。
        rgb_conv2d = rgb_conv[0] if isinstance(rgb_conv, nn.Sequential) else rgb_conv
        new_conv2d = new_conv[0] if isinstance(new_conv, nn.Sequential) else new_conv

        # rgb weight: out_c, 3, k, k -> 按输入通道均值得到 out_c, 1, k, k，再广播到 new_in_chs。
        avg_w = rgb_conv2d.weight.data.mean(dim=1, keepdim=True)  # out_c,1,k,k
        new_conv2d.weight.data.copy_(avg_w.repeat(1, new_in_chs, 1, 1))
        if new_conv2d.bias is not None and rgb_conv2d.bias is not None:
            new_conv2d.bias.data.copy_(rgb_conv2d.bias.data)


class TriStreamHGNetv2(nn.Module):
    """三模态共享 stage、独立 stem 的骨干。forward 一次处理一个模态。

    pretrained_param_ids：`stem_rgb` 和 `stages` 的参数 id 集合，供
    训练脚本的 `build_param_groups` 区分"预训练来的权重"和"全新初始化
    的权重"（`stem_ir`/`stem_depth`）。两者都在 `backbone.` 前缀下，但
    优化器上应该用不同的学习率倍率：前者要小心保护预训练特征，后者是
    要从零学起的新模块，不该被同样地压低学习率。
    """

    def __init__(
        self,
        name: str = "B2",
        use_lab: bool = False,
        return_idx=(1, 2, 3),
        pretrained: bool = True,
        freeze_stem_only: bool = True,
        freeze_at: int = 0,
        freeze_norm: bool = True,
    ):
        super().__init__()
        # freeze_* 三个参数原先没有显式传递，静默继承了 HGNetv2 的默认值
        # （见 third_party/D-FINE/src/nn/backbone/hgnetv2.py 的签名）。这里
        # 写明确，值与官方 dfine_hgnetv2_l_coco.yml 一致：
        #   freeze_stem_only=True + freeze_at=0 —— 只冻结 stem，stages 全部可训练
        #   freeze_norm=True     —— 把 backbone 里的 BatchNorm 换成 FrozenBatchNorm2d
        # 注意 freeze_norm 只作用于此刻已存在的模块，即 RGB 分支；下面新建的
        # stem_ir/stem_depth 保持活的 BN（会被 SyncBN 转换），这是有意的——它们
        # 是从零学起的新模块，需要自己的归一化统计量。
        rgb_backbone = HGNetv2(
            name=name,
            use_lab=use_lab,
            return_idx=list(return_idx),
            pretrained=pretrained,
            freeze_stem_only=freeze_stem_only,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
        )

        self.stages = rgb_backbone.stages
        self.return_idx = list(return_idx)
        self.out_channels = [rgb_backbone._out_channels[i] for i in return_idx]

        stem_channels = rgb_backbone.arch_configs[name]["stem_channels"]

        self.stem_rgb = rgb_backbone.stem
        self.stem_ir = StemBlock(in_chs=1, mid_chs=stem_channels[1], out_chs=stem_channels[2], use_lab=use_lab)
        self.stem_depth = StemBlock(in_chs=2, mid_chs=stem_channels[1], out_chs=stem_channels[2], use_lab=use_lab)

        if pretrained:
            _init_stem_from_rgb(self.stem_ir, self.stem_rgb, new_in_chs=1)
            _init_stem_from_rgb(self.stem_depth, self.stem_rgb, new_in_chs=2)

        self.pretrained_param_ids = {id(p) for p in self.stem_rgb.parameters()} | {
            id(p) for p in self.stages.parameters()
        }

    def _forward_stem(self, x: torch.Tensor, stem: StemBlock) -> list[torch.Tensor]:
        x = stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor, depth: torch.Tensor):
        rgb_feats = self._forward_stem(rgb, self.stem_rgb)
        ir_feats = self._forward_stem(ir, self.stem_ir)
        depth_feats = self._forward_stem(depth, self.stem_depth)
        return rgb_feats, ir_feats, depth_feats

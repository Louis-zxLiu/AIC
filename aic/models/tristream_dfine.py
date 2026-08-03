"""三流 D-FINE：TriStreamHGNetv2 backbone + OPF 融合 + 官方 HybridEncoder/DFINETransformer。

插入点严格对应官方 DFINE.forward（third_party/D-FINE/src/zoo/dfine/dfine.py）：
    x = backbone(x); x = encoder(x); x = decoder(x, targets)
OPF 融合发生在 backbone 输出之后、encoder 输入之前——backbone 换成三流版本，
三路各尺度特征经 MultiScaleOPF 融合成单路后，直接送入未经任何修改的官方
HybridEncoder / DFINETransformer / DFINECriterion / DFINEPostProcessor。
这样融合模块之后的所有组件都是原版 D-FINE 代码，最大程度复用其已验证正确
的训练/推理逻辑，避免引入新 bug。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import TriStreamHGNetv2
from .opf_fusion import MultiScaleOPF


class TriStreamDFINE(nn.Module):
    def __init__(self, backbone: TriStreamHGNetv2, opf: MultiScaleOPF, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.opf = opf
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor, depth: torch.Tensor, targets=None):
        rgb_feats, ir_feats, depth_feats = self.backbone(rgb, ir, depth)
        # depth 输入是 [depth_norm, valid_mask] 堆叠的 2 通道张量（见
        # aic/data/transforms.py depth_stack），第二通道是显式有效性先验，
        # 传给 OPF 用于 ConfidenceGate（见 opf_fusion.py）。
        depth_valid_mask = depth[:, 1:2, :, :]
        fused = self.opf(rgb_feats, ir_feats, depth_feats, depth_valid_mask=depth_valid_mask)
        enc_out = self.encoder(fused)
        out = self.decoder(enc_out, targets)
        return out

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self


def build_tristream_dfine(cfg: dict) -> TriStreamDFINE:
    """从字典配置（见 configs/dfine_tristream.yml 的 python 端加载结果）构建模型。"""
    from third_party_dfine import DFINETransformer, HybridEncoder

    bb_cfg = cfg["backbone"]
    backbone = TriStreamHGNetv2(
        name=bb_cfg["name"],
        use_lab=bb_cfg.get("use_lab", False),
        return_idx=tuple(bb_cfg.get("return_idx", [1, 2, 3])),
        pretrained=bb_cfg.get("pretrained", True),
    )

    opf = MultiScaleOPF(
        channels_per_scale=backbone.out_channels,
        num_heads=cfg["opf"].get("num_heads", 8),
        sr_ratios=cfg["opf"].get("sr_ratios"),
    )

    enc_cfg = dict(cfg["encoder"])
    enc_cfg["in_channels"] = backbone.out_channels
    encoder = HybridEncoder(**enc_cfg)

    dec_cfg = dict(cfg["decoder"])
    dec_cfg["feat_channels"] = encoder.out_channels
    dec_cfg["feat_strides"] = encoder.out_strides
    dec_cfg["num_classes"] = cfg["num_classes"]
    decoder = DFINETransformer(**dec_cfg)

    model = TriStreamDFINE(backbone, opf, encoder, decoder)

    # backbone.pretrained 只给了 HGNetv2 的分类权重，encoder/decoder 仍是随机
    # 初始化。若配置了检测预训练权重，在这里覆盖进去（12 类头因 shape 不符会
    # 自动跳过，保持随机初始化）。详见 aic/models/pretrained.py。
    ckpt = cfg.get("pretrained_detector")
    if ckpt:
        from .pretrained import load_dfine_detector_weights

        load_dfine_detector_weights(model, ckpt)

    return model

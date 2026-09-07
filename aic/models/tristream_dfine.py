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
from .ir_alignment import MultiScaleIRAlignment
from .opf_fusion import MultiScaleOPF


class ResidualModalityFusion(nn.Module):
    """RGB-anchored residual fusion with lightweight spatial quality gates.

    The RGB branch keeps the pretrained feature as the anchor. IR and Depth
    contribute projected residuals, so unreliable auxiliary inputs cannot
    replace the pretrained RGB representation wholesale.
    """

    def __init__(self, channels_per_scale: list[int] | tuple[int, ...], gate_reduction: int = 16, gate_bias: float = -1.5):
        super().__init__()
        self.ir_proj = nn.ModuleList()
        self.depth_proj = nn.ModuleList()
        self.ir_gate = nn.ModuleList()
        self.depth_gate = nn.ModuleList()
        for channels in channels_per_scale:
            hidden = max(int(channels) // int(gate_reduction), 16)
            self.ir_proj.append(nn.Conv2d(channels, channels, 1))
            self.depth_proj.append(nn.Conv2d(channels, channels, 1))
            self.ir_gate.append(nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.GELU(), nn.Conv2d(hidden, 1, 1)))
            self.depth_gate.append(nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.GELU(), nn.Conv2d(hidden, 1, 1)))
        self.gate_bias = float(gate_bias)
        self.reset_parameters()

    @staticmethod
    def _identity_init(layer: nn.Conv2d) -> None:
        nn.init.zeros_(layer.weight)
        nn.init.zeros_(layer.bias)
        with torch.no_grad():
            diagonal = min(layer.weight.shape[0], layer.weight.shape[1])
            layer.weight[:diagonal, :diagonal, 0, 0].copy_(torch.eye(diagonal, device=layer.weight.device, dtype=layer.weight.dtype))

    def reset_parameters(self) -> None:
        for layer in list(self.ir_proj) + list(self.depth_proj):
            self._identity_init(layer)
        for gates in list(self.ir_gate) + list(self.depth_gate):
            nn.init.zeros_(gates[-1].weight)
            nn.init.constant_(gates[-1].bias, self.gate_bias)

    def forward(
        self,
        rgb_feats: list[torch.Tensor],
        ir_feats: list[torch.Tensor],
        depth_feats: list[torch.Tensor],
        depth_quality: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        fused = []
        for idx, (rgb, ir, depth) in enumerate(zip(rgb_feats, ir_feats, depth_feats)):
            ir_gate = torch.sigmoid(self.ir_gate[idx](ir))
            depth_gate = torch.sigmoid(self.depth_gate[idx](depth))
            if depth_quality is not None:
                quality = torch.nn.functional.interpolate(
                    depth_quality, size=depth.shape[-2:], mode="bilinear", align_corners=False
                ).clamp(0.0, 1.0)
                depth_gate = depth_gate * quality
            fused.append(rgb + ir_gate * self.ir_proj[idx](ir) + depth_gate * self.depth_proj[idx](depth))
        return fused


class SingleModalityDFINE(nn.Module):
    """单模态 D-FINE，用于计划.md §2.8 M²D-LIF linear probing 诊断。

    backbone 仍是 TriStreamHGNetv2（三个 stem 都建），但只跑其中一个模态的
    forward 路径，直接把该模态的多尺度特征喂给 encoder，不经过 OPF——诊断
    对比的是"该模态 backbone 特征的质量"，OPF 的去相关/门控/交叉注意力
    在只有一路真实输入时是对零张量的无意义运算，且其可学习参数会在纯单
    模态数据上学到退化解，引入不必要的混杂变量，因此here去掉这一层。

    frozen_backbone=True 时对应"探针"模式：backbone 来自联合训练好的
    checkpoint，冻结不参与训练，只训练 encoder/decoder，衡量该 backbone
    产出的特征本身有多好；frozen_backbone=False 对应"单模态基线"模式：
    backbone 从预训练权重正常初始化，跟随 encoder/decoder 一起从零训练，
    作为对比的参照系。
    """

    def __init__(self, backbone: TriStreamHGNetv2, modality: str, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        if modality not in ("rgb", "ir", "depth"):
            raise ValueError(f"modality 必须是 rgb/ir/depth，收到 {modality!r}")
        self.backbone = backbone
        self.modality = modality
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor, depth: torch.Tensor, targets=None):
        x = {"rgb": rgb, "ir": ir, "depth": depth[:, : self.backbone.depth_in_chs]}[
            self.modality
        ]
        stem = {"rgb": self.backbone.stem_rgb, "ir": self.backbone.stem_ir, "depth": self.backbone.stem_depth}[
            self.modality
        ]
        stages = {
            "rgb": self.backbone.stages_rgb,
            "ir": self.backbone.stages_ir,
            "depth": self.backbone.stages_depth,
        }[self.modality]
        feats = self.backbone._forward_stem(x, stem, stages, use_checkpoint=False)
        enc_out = self.encoder(feats)
        out = self.decoder(enc_out, targets)
        return out


def build_single_modality_dfine(cfg: dict, modality: str, frozen_backbone: TriStreamHGNetv2 | None = None):
    """构建单模态 D-FINE（诊断专用）。

    frozen_backbone 为 None 时新建一个 backbone（单模态基线，从预训练权重
    正常初始化，随 encoder/decoder 一起训练）；传入一个已加载 checkpoint
    的 TriStreamHGNetv2 实例时，冻结其全部参数并复用（探针模式）。两种
    模式的 encoder/decoder 都全新初始化——诊断比较的是 backbone 特征质量，
    头部能力必须两侧对齐，不能有一侧头部是热启动的。
    """
    from third_party_dfine import DFINETransformer, HybridEncoder

    if frozen_backbone is not None:
        backbone = frozen_backbone
        backbone.requires_grad_(False)
        backbone.eval()
    else:
        bb_cfg = cfg["backbone"]
        backbone = TriStreamHGNetv2(
            name=bb_cfg["name"],
            use_lab=bb_cfg.get("use_lab", False),
            return_idx=tuple(bb_cfg.get("return_idx", [1, 2, 3])),
            pretrained=bb_cfg.get("pretrained", True),
            share_stages=bb_cfg.get("share_stages", True),
            depth_in_chs=bb_cfg.get("depth_in_chs", 2),
        )

    enc_cfg = dict(cfg["encoder"])
    enc_cfg["in_channels"] = backbone.out_channels
    encoder = HybridEncoder(**enc_cfg)

    dec_cfg = dict(cfg["decoder"])
    dec_cfg["feat_channels"] = encoder.out_channels
    dec_cfg["feat_strides"] = encoder.out_strides
    dec_cfg["num_classes"] = cfg["num_classes"]
    decoder = DFINETransformer(**dec_cfg)

    model = SingleModalityDFINE(backbone, modality, encoder, decoder)

    # A mono teacher must inherit the detector body initialization as well as
    # the HGNetV2 backbone.  Without this, encoder/decoder start randomly and
    # the teacher is not comparable to the D-FINE-X baseline.
    ckpt = cfg.get("pretrained_detector")
    if cfg.get("load_pretrained_detector", False) and ckpt:
        from .pretrained import load_dfine_detector_weights

        load_dfine_detector_weights(model, ckpt)

    return model


class TriStreamDFINE(nn.Module):
    def __init__(
        self,
        backbone: TriStreamHGNetv2,
        opf: MultiScaleOPF,
        encoder: nn.Module,
        decoder: nn.Module,
        ir_alignment: MultiScaleIRAlignment | None = None,
        residual_fusion: ResidualModalityFusion | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.opf = opf
        self.encoder = encoder
        self.decoder = decoder
        self.ir_alignment = ir_alignment
        self.residual_fusion = residual_fusion

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor, depth: torch.Tensor, targets=None, return_backbone_feats: bool = False):
        rgb_feats, ir_feats, depth_feats = self.backbone(rgb, ir, depth)
        # The fifth channel is a continuous reliability map that already
        # combines hard validity, local coverage, and JPEG channel agreement.
        depth_quality_mask = (
            depth[:, 4:5, :, :] if self.backbone.depth_in_chs >= 5 else depth[:, 1:2, :, :]
        )
        # Distillation consumes the untouched IR backbone features returned below.
        # Alignment is an identity-initialized copy used only by the OPF branch.
        ir_fusion_feats = (
            self.ir_alignment(rgb_feats, ir_feats)
            if self.ir_alignment is not None
            else ir_feats
        )
        if self.residual_fusion is not None:
            fused = self.residual_fusion(rgb_feats, ir_fusion_feats, depth_feats, depth_quality_mask)
        else:
            fused = self.opf(rgb_feats, ir_fusion_feats, depth_feats, depth_valid_mask=depth_quality_mask)
        enc_out = self.encoder(fused)
        out = self.decoder(enc_out, targets)
        # return_backbone_feats：M2D 蒸馏（计划.md §2.8 后续）需要 OPF 融合*之前*
        # 的 IR/Depth 多尺度特征去对齐各自单模态 teacher 的 backbone 输出——融合
        # 后的 fused/enc_out 已经是三模态混合的单路表示，无法把 loss 单独归因到
        # 某个模态，因此必须在这里（OPF 之前）暴露，而不是 encoder 输出之后。
        if return_backbone_feats:
            return out, rgb_feats, ir_feats, depth_feats
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
        share_stages=bb_cfg.get("share_stages", True),
        depth_in_chs=bb_cfg.get("depth_in_chs", 2),
    )

    residual_cfg = dict(cfg.get("residual_fusion", {}))
    residual_fusion = None
    if residual_cfg.get("enabled", False):
        residual_fusion = ResidualModalityFusion(
            channels_per_scale=backbone.out_channels,
            gate_reduction=int(residual_cfg.get("gate_reduction", 16)),
            gate_bias=float(residual_cfg.get("gate_bias", -1.5)),
        )
        opf = None
    else:
        opf = MultiScaleOPF(
            channels_per_scale=backbone.out_channels,
            num_heads=cfg["opf"].get("num_heads", 8),
            sr_ratios=cfg["opf"].get("sr_ratios"),
            use_dwt=cfg["opf"].get("use_dwt", False),
            hard_depth_gate=cfg["opf"].get("hard_depth_gate", False),
        )

    enc_cfg = dict(cfg["encoder"])
    enc_cfg["in_channels"] = backbone.out_channels
    freqfusion_cfg = dict(cfg.get("freqfusion", {}))
    if freqfusion_cfg.get("enabled", False):
        from .freq_fusion import FreqFusionHybridEncoder

        encoder = FreqFusionHybridEncoder(freqfusion=freqfusion_cfg, **enc_cfg)
    else:
        encoder = HybridEncoder(**enc_cfg)

    dec_cfg = dict(cfg["decoder"])
    dec_cfg["feat_channels"] = encoder.out_channels
    dec_cfg["feat_strides"] = encoder.out_strides
    dec_cfg["num_classes"] = cfg["num_classes"]
    decoder = DFINETransformer(**dec_cfg)

    # Construct the optional module after the detector body. With a fixed seed,
    # this keeps every baseline detector parameter initialization identical.
    alignment_cfg = dict(cfg.get("ir_alignment", {}))
    ir_alignment = None
    if alignment_cfg.get("enabled", False):
        ir_alignment = MultiScaleIRAlignment(
            channels_per_scale=backbone.out_channels,
            strides=cfg["encoder"].get("feat_strides", [8, 16, 32]),
            hidden_channels=int(alignment_cfg.get("hidden_channels", 16)),
            max_offset_pixels=float(alignment_cfg.get("max_offset_pixels", 32.0)),
            gate_bias=float(alignment_cfg.get("gate_bias", -2.0)),
        )

    model = TriStreamDFINE(
        backbone, opf, encoder, decoder, ir_alignment=ir_alignment, residual_fusion=residual_fusion
    )

    # backbone.pretrained 只给了 HGNetv2 的分类权重，encoder/decoder 仍是随机
    # 初始化。若配置了检测预训练权重，在这里覆盖进去（12 类头因 shape 不符会
    # 自动跳过，保持随机初始化）。详见 aic/models/pretrained.py。
    ckpt = cfg.get("pretrained_detector")
    if ckpt:
        from .pretrained import load_dfine_detector_weights

        load_dfine_detector_weights(model, ckpt)

    return model

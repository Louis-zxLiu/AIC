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


class P3IRLocalInteraction(nn.Module):
    """Stride-8 IR local-window residual for small-object detail recovery.

    The final projection is zero initialized, so enabling this module preserves
    the baseline function at initialization while allowing local RGB/IR
    interactions to be learned at the highest detector feature resolution.
    """

    def __init__(self, channels: int, hidden_channels: int | None = None):
        super().__init__()
        hidden = int(hidden_channels or max(channels // 2, 32))
        self.block = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32 if hidden % 32 == 0 else 1, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        nn.init.zeros_(self.block[-1].weight)
        nn.init.zeros_(self.block[-1].bias)

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        return rgb + self.block(torch.cat([rgb, ir], dim=1))


class ModalStageAdapter(nn.Module):
    """Low-rank residual adapters for the non-RGB high-level streams."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 32)
        self.down = nn.Conv2d(channels, hidden, 1)
        self.depthwise = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.act = nn.GELU()
        self.up = nn.Conv2d(hidden, channels, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.depthwise(self.down(x))))


class P3IRDeformableInteraction(nn.Module):
    """Small zero-init local IR sampler at stride 8.

    It predicts bounded offsets from RGB/IR P3 and samples a compact 3x3
    neighborhood from IR. The zero-initialized output projection preserves the
    native detector at initialization.
    """

    def __init__(self, channels: int, points: int = 9, max_offset: float = 2.0):
        super().__init__()
        self.points = int(points)
        self.max_offset = float(max_offset)
        hidden = max(channels // 4, 32)
        self.offset = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, self.points * 2, 1),
        )
        self.value = nn.Conv2d(channels, channels, 1)
        self.out = nn.Conv2d(channels * 2, channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        b, _, h, w = rgb.shape
        offsets = torch.tanh(self.offset(torch.cat([rgb, ir], dim=1)))
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=rgb.device, dtype=rgb.dtype),
            torch.linspace(-1.0, 1.0, w, device=rgb.device, dtype=rgb.dtype),
            indexing="ij",
        )
        base = torch.stack([xx, yy], dim=-1).view(1, h, w, 1, 2)
        grid_offsets = offsets.permute(0, 2, 3, 1).view(b, h, w, self.points, 2)
        scale = rgb.new_tensor([2.0 / max(w - 1, 1), 2.0 / max(h - 1, 1)])
        grid = (base + grid_offsets * self.max_offset * scale).clamp(-1.0, 1.0)
        sampled = []
        value = self.value(ir)
        for point in range(self.points):
            sampled.append(torch.nn.functional.grid_sample(
                value, grid[..., point, :], mode="bilinear", padding_mode="border", align_corners=True
            ))
        local = torch.stack(sampled, dim=0).mean(dim=0)
        return rgb + self.out(torch.cat([rgb, local], dim=1))


class P3DepthGeometryInteraction(nn.Module):
    """Use Depth P3 validity/geometry to gate a local IR residual."""

    def __init__(self, channels: int):
        super().__init__()
        hidden = max(channels // 4, 32)
        self.gate = nn.Sequential(
            nn.Conv2d(channels + 1, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
            nn.Sigmoid(),
        )
        self.offset = nn.Conv2d(channels + 1, 2, 3, padding=1)
        self.out = nn.Conv2d(channels * 2, channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor, depth: torch.Tensor, quality: torch.Tensor | None) -> torch.Tensor:
        q = torch.ones_like(depth[:, :1]) if quality is None else quality
        q = torch.nn.functional.interpolate(q, size=depth.shape[-2:], mode="bilinear", align_corners=False)
        q = torch.nn.functional.interpolate(q, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        depth = torch.nn.functional.interpolate(depth, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        gate = self.gate(torch.cat([depth, q], dim=1)) * q.clamp(0.0, 1.0)
        delta = torch.tanh(self.offset(torch.cat([depth, q], dim=1)))
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, rgb.shape[-2], device=rgb.device, dtype=rgb.dtype),
            torch.linspace(-1.0, 1.0, rgb.shape[-1], device=rgb.device, dtype=rgb.dtype),
            indexing="ij",
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(rgb.shape[0], -1, -1, -1).clone()
        scale = rgb.new_tensor([2.0 / max(rgb.shape[-1] - 1, 1), 2.0 / max(rgb.shape[-2] - 1, 1)])
        grid = (grid + delta.permute(0, 2, 3, 1) * scale).clamp(-1.0, 1.0)
        local = torch.nn.functional.grid_sample(ir, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return rgb + self.out(torch.cat([rgb, gate * local], dim=1))


class SharedPrivateEvidence(nn.Module):
    """Factor modal features into an OPF-shared path and private residuals.

    The private branch starts at zero, so the experiment is function-preserving
    at initialization.  Only the IR private P3 residual is exposed to B1;
    Depth private features never become a direct classification value path.
    """

    def __init__(self, channels_per_scale, reduction: int = 16):
        super().__init__()
        self.private = nn.ModuleDict()
        for modality in ("rgb", "ir", "depth"):
            self.private[modality] = nn.ModuleList()
            for channels in channels_per_scale:
                hidden = max(int(channels) // int(reduction), 32)
                block = nn.Sequential(
                    nn.Conv2d(channels, hidden, 1),
                    nn.GELU(),
                    nn.Conv2d(hidden, channels, 1),
                )
                nn.init.zeros_(block[-1].weight)
                nn.init.zeros_(block[-1].bias)
                self.private[modality].append(block)

    def forward(self, rgb_feats, ir_feats, depth_feats):
        private = {}
        shared = {}
        for name, feats in (("rgb", rgb_feats), ("ir", ir_feats), ("depth", depth_feats)):
            private[name] = [block(feat) for block, feat in zip(self.private[name], feats)]
            shared[name] = [feat - delta for feat, delta in zip(feats, private[name])]
        return shared["rgb"], shared["ir"], shared["depth"], private["ir"]


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
    if cfg.get("query_modal_router", {}).get("enabled", False):
        dec_cfg["query_modal_router"] = True
        dec_cfg["modal_feat_channels"] = [backbone.out_channels[0]] * 3
        dec_cfg["query_modal_router_ir_only"] = cfg.get("query_modal_router", {}).get("ir_only", True)
    decoder = DFINETransformer(**dec_cfg)

    model = SingleModalityDFINE(backbone, modality, encoder, decoder)

    # A mono teacher must inherit the detector body initialization as well as
    # the HGNetV2 backbone.  Without this, encoder/decoder start randomly and
    # the teacher is not comparable to the D-FINE-X baseline.
    ckpt = cfg.get("pretrained_detector")
    if cfg.get("load_pretrained_detector", False) and ckpt:
        from .pretrained import load_dfine_detector_weights

        load_dfine_detector_weights(
            model,
            ckpt,
            class_mapping=cfg.get("pretrained_class_mapping"),
        )

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
        p3_ir_local: P3IRLocalInteraction | None = None,
        modal_adapters: nn.ModuleDict | None = None,
        p3_ir_deform: P3IRDeformableInteraction | None = None,
        p3_depth_geometry: P3DepthGeometryInteraction | None = None,
        shared_private: SharedPrivateEvidence | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.opf = opf
        self.encoder = encoder
        self.decoder = decoder
        self.ir_alignment = ir_alignment
        self.residual_fusion = residual_fusion
        self.p3_ir_local = p3_ir_local
        self.modal_adapters = modal_adapters
        self.p3_ir_deform = p3_ir_deform
        self.p3_depth_geometry = p3_depth_geometry
        self.shared_private = shared_private

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
        if self.modal_adapters is not None:
            ir_fusion_feats = [self.modal_adapters["ir"][i](feat) for i, feat in enumerate(ir_fusion_feats)]
            depth_feats = [self.modal_adapters["depth"][i](feat) for i, feat in enumerate(depth_feats)]
        ir_private_feats = None
        if self.shared_private is not None:
            rgb_feats, ir_fusion_feats, depth_feats, ir_private_feats = self.shared_private(
                rgb_feats, ir_fusion_feats, depth_feats
            )
        if self.residual_fusion is not None:
            fused = self.residual_fusion(rgb_feats, ir_fusion_feats, depth_feats, depth_quality_mask)
        else:
            fused = self.opf(rgb_feats, ir_fusion_feats, depth_feats, depth_valid_mask=depth_quality_mask)
        if self.p3_ir_local is not None:
            fused[0] = self.p3_ir_local(fused[0], ir_fusion_feats[0])
        if self.p3_ir_deform is not None:
            detail_ir = ir_fusion_feats[0]
            if ir_private_feats is not None:
                detail_ir = detail_ir + ir_private_feats[0]
            fused[0] = self.p3_ir_deform(fused[0], detail_ir)
        if self.p3_depth_geometry is not None:
            fused[0] = self.p3_depth_geometry(
                fused[0], ir_fusion_feats[0], depth_feats[0], depth_quality_mask
            )
        enc_out = self.encoder(fused)
        out = self.decoder(
            enc_out,
            targets,
            # Query-level fusion needs the full pyramid.  Keep the quality map
            # as the fourth item so depth evidence remains validity-gated.
            modal_feats=(rgb_feats, ir_fusion_feats, depth_feats, depth_quality_mask),
            raw_modalities=(rgb, ir, depth[:, : self.backbone.depth_in_chs]),
        )
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
    if cfg.get('backbone', {}).get('rgb'):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(cfg.get('initialization_seed', 42)))
            return _build_tristream_dfine(cfg)
    return _build_tristream_dfine(cfg)


def _build_tristream_dfine(cfg: dict) -> TriStreamDFINE:
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
    tri_query_cfg = dict(cfg.get("tri_modal_query_decoder", {}))
    if tri_query_cfg.get("enabled", False):
        dec_cfg["tri_modal_query_decoder"] = True
        dec_cfg["tri_modal_query_heads"] = int(tri_query_cfg.get("heads", 8))
    if cfg.get("query_modal_router", {}).get("enabled", False):
        dec_cfg["query_modal_router"] = True
        dec_cfg["modal_feat_channels"] = [backbone.out_channels[0]] * 3
        dec_cfg["query_modal_router_ir_only"] = cfg.get("query_modal_router", {}).get("ir_only", True)
    region_cfg = dict(cfg.get("query_region_fusion", {}))
    if region_cfg.get("enabled", False):
        dec_cfg["query_region_fusion"] = True
        dec_cfg["query_region_points"] = int(region_cfg.get("points", 9))
        dec_cfg["query_region_max_offset"] = float(region_cfg.get("max_offset", 0.25))
        dec_cfg["query_region_hidden_dim"] = int(region_cfg.get("hidden_dim", 128))
    region_refine_cfg = dict(cfg.get("query_region_refine", {}))
    if region_refine_cfg.get("enabled", False):
        dec_cfg["query_region_refine"] = True
        dec_cfg["query_region_points"] = int(region_refine_cfg.get("points", 9))
        dec_cfg["query_region_hidden_dim"] = int(region_refine_cfg.get("hidden_dim", 128))
    raw_zoom_cfg = dict(cfg.get("raw_zoom_refine", {}))
    if raw_zoom_cfg.get("enabled", False):
        dec_cfg.update(
            {
                "raw_zoom_refine": True,
                "raw_zoom_modal_channels": [3, 1, int(bb_cfg.get("depth_in_chs", 2))],
                "raw_zoom_crop_size": int(raw_zoom_cfg.get("crop_size", 16)),
                "raw_zoom_context_scale": float(raw_zoom_cfg.get("context_scale", 2.0)),
                "raw_zoom_min_crop_pixels": float(raw_zoom_cfg.get("min_crop_pixels", 32.0)),
                "raw_zoom_patch_dim": int(raw_zoom_cfg.get("patch_dim", 96)),
                "raw_zoom_depth_quality_channel": int(raw_zoom_cfg.get("depth_quality_channel", 4)),
            }
        )
    dec_cfg["encoder_oto_supervision"] = bool(
        cfg.get("encoder_oto_supervision", {}).get("enabled", False)
    )
    dec_cfg["modal_feat_channels"] = (
        list(backbone.out_channels)
        if tri_query_cfg.get("enabled", False)
        else [backbone.out_channels[0]] * 3
    )
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
        backbone,
        opf,
        encoder,
        decoder,
        ir_alignment=ir_alignment,
        residual_fusion=residual_fusion,
        p3_ir_local=(
            P3IRLocalInteraction(
                backbone.out_channels[0],
                hidden_channels=int(cfg.get("p3_ir_local", {}).get("hidden_channels", 192)),
            )
            if cfg.get("p3_ir_local", {}).get("enabled", False)
            else None
        ),
        modal_adapters=(
            nn.ModuleDict({
                "ir": nn.ModuleList([
                    ModalStageAdapter(c, reduction=int(cfg.get("modal_adapter", {}).get("reduction", 16)))
                    for c in backbone.out_channels
                ]),
                "depth": nn.ModuleList([
                    ModalStageAdapter(c, reduction=int(cfg.get("modal_adapter", {}).get("reduction", 16)))
                    for c in backbone.out_channels
                ]),
            })
            if cfg.get("modal_adapter", {}).get("enabled", False)
            else None
        ),
        p3_ir_deform=(
            P3IRDeformableInteraction(
                backbone.out_channels[0],
                points=int(cfg.get("p3_ir_deform", {}).get("points", 9)),
                max_offset=float(cfg.get("p3_ir_deform", {}).get("max_offset", 2.0)),
            )
            if cfg.get("p3_ir_deform", {}).get("enabled", False)
            else None
        ),
        p3_depth_geometry=(
            P3DepthGeometryInteraction(backbone.out_channels[0])
            if cfg.get("p3_depth_geometry", {}).get("enabled", False)
            else None
        ),
        shared_private=(
            SharedPrivateEvidence(
                backbone.out_channels,
                reduction=int(cfg.get("shared_private_evidence", {}).get("reduction", 16)),
            )
            if cfg.get("shared_private_evidence", {}).get("enabled", False)
            else None
        ),
    )

    # backbone.pretrained 只给了 HGNetv2 的分类权重，检测预训练在这里覆盖。
    # 分类头默认仍按 shape 严格匹配；本实验通过配置显式打开 COCO->AIC
    # 语义行迁移，未映射的类别继续保留随机初始化。
    ckpt = cfg.get("pretrained_detector")
    if ckpt:
        from .pretrained import load_dfine_detector_weights

        model.pretrained_report = load_dfine_detector_weights(
            model,
            ckpt,
            class_mapping=cfg.get("pretrained_class_mapping"),
            strict_body=bool(bb_cfg.get('rgb')),
        )

    if bb_cfg.get('rgb'):
        from .dinov3_rgb import attach_rgb_backbone
        if not ckpt:
            model.pretrained_report = {}
        model = attach_rgb_backbone(model, cfg)
    return model

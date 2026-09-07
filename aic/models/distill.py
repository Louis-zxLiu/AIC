"""M2D-LIF 蒸馏（计划.md §2.8 后续）：修复 Fusion Degradation。

诊断结论（memory `fusion-degradation-confirmed`）：联合训练拖累了 IR/Depth
backbone 的表征质量（frozen probe 明显弱于独立单模态训练），RGB 不受影响。
修复思路：用两个独立训练到收敛的单模态模型（`--mono_modality ir/depth`，
无 `--frozen_backbone_ckpt`，即从预训练权重正常训练的完整模型）当 teacher，
在三流联合训练时让 student 的 IR/Depth backbone 输出去逼近对应 teacher 的
backbone 输出。

对齐点选在 backbone 输出（OPF 融合之前），不是 encoder 输出之后：融合后的
fused/encoder 输出已经是三模态混合的单路表示，无法把蒸馏 loss 单独归因到
"IR 该长什么样"或"Depth 该长什么样"；只有 OPF 之前的多尺度特征还能一一
对应到某个模态自己的 teacher。

度量用 cosine 而不是 MSE：两个 teacher 是完全独立训练的模型，其 backbone
输出的特征幅度/尺度和 student 之间没有任何理由对齐，MSE 会被幅度差异主导、
淹没真正关心的方向对齐信号；cosine 只约束方向，是特征蒸馏的标准做法。
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import TriStreamHGNetv2


def _load_teacher_backbone(
    ckpt_path: str,
    backbone: TriStreamHGNetv2,
    modality: str,
    use_ema: bool = False,
) -> str:
    """Load only the active mono-modal stem and stages from a teacher."""
    if modality not in ("ir", "depth"):
        raise ValueError(f"unsupported teacher modality: {modality}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    source = "ema" if use_ema and checkpoint.get("ema") is not None else "model"
    state = {k.removeprefix("module."): v for k, v in checkpoint[source].items()}

    stem_prefix = f"backbone.stem_{modality}."
    stem_state = {k[len(stem_prefix) :]: v for k, v in state.items() if k.startswith(stem_prefix)}
    if not stem_state:
        raise RuntimeError(f"teacher checkpoint has no {stem_prefix} weights: {ckpt_path}")
    getattr(backbone, f"stem_{modality}").load_state_dict(stem_state, strict=True)

    stage_state = None
    for prefix in (f"backbone.stages_{modality}.", "backbone.stages_rgb.", "backbone.stages."):
        candidate = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
        if candidate:
            stage_state = candidate
            break
    if stage_state is None:
        raise RuntimeError(f"teacher checkpoint has no stages weights: {ckpt_path}")
    getattr(backbone, f"stages_{modality}").load_state_dict(stage_state, strict=True)
    return source


class ResidualFeatureAdapter(nn.Module):
    """Small training-only projection that separates two shared-stage targets."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 32)
        self.down = nn.Conv2d(channels, hidden, 1)
        self.act = nn.GELU()
        self.up = nn.Conv2d(hidden, channels, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))


class M2DDistillLoss(nn.Module):
    """Selectable legacy cosine or PKD+LCM distillation for IR and Depth."""

    def __init__(
        self,
        bb_cfg: dict,
        ir_teacher_ckpt: str,
        depth_teacher_ckpt: str,
        weight: float = 1.0,
        ir_weight: float = 1.0,
        depth_weight: float = 1.0,
        use_teacher_ema: bool = False,
        foreground_weight: float = 1.0,
        level_weights: list[float] | tuple[float, ...] | None = None,
        use_depth_valid_mask: bool = False,
        cross_weight: float = 1.0,
        use_adapters: bool = True,
        adapter_reduction: int = 16,
        dynamic_reliability: bool = False,
        reliability_floor: float = 0.25,
        reliability_power: float = 1.0,
        method: str = "pkd_lcm",
    ):
        super().__init__()
        if method not in ("cosine", "pkd_lcm"):
            raise ValueError(f"unsupported M2D distillation method: {method!r}")
        self.method = method
        self.weight = float(weight)
        self.ir_weight = float(ir_weight)
        self.depth_weight = float(depth_weight)
        self.foreground_weight = float(foreground_weight)
        self.level_weights = tuple(float(v) for v in (level_weights or (1.0, 1.0, 1.0)))
        self.use_depth_valid_mask = bool(use_depth_valid_mask)
        self.cross_weight = float(cross_weight) if method == "pkd_lcm" else 0.0
        self.use_adapters = bool(use_adapters) and method == "pkd_lcm"
        self.dynamic_reliability = bool(dynamic_reliability)
        self.reliability_floor = float(reliability_floor)
        self.reliability_power = float(reliability_power)
        if self.foreground_weight < 1.0:
            raise ValueError("distill.foreground_weight must be >= 1.0")
        if not self.level_weights or any(v <= 0 for v in self.level_weights):
            raise ValueError("distill.level_weights must contain positive values")
        if self.cross_weight < 0:
            raise ValueError("distill.cross_weight must be >= 0")
        if self.ir_weight < 0 or self.depth_weight < 0:
            raise ValueError("distill.ir_weight and distill.depth_weight must be >= 0")
        if self.ir_weight == 0 and self.depth_weight == 0:
            raise ValueError("at least one modality distillation weight must be positive")
        if self.cross_weight > 0 and (self.ir_weight == 0 or self.depth_weight == 0):
            raise ValueError("cross-modal distillation requires both modality weights to be positive")
        if not 0.0 < self.reliability_floor <= 1.0 or self.reliability_power <= 0.0:
            raise ValueError("invalid dynamic reliability floor/power")

        self.ir_teacher = TriStreamHGNetv2(
            name=bb_cfg["name"],
            use_lab=bb_cfg.get("use_lab", False),
            return_idx=tuple(bb_cfg.get("return_idx", [1, 2, 3])),
            pretrained=False,
            share_stages=bb_cfg.get("share_stages", True),
            depth_in_chs=bb_cfg.get("depth_in_chs", 2),
        )
        self.ir_teacher_source = _load_teacher_backbone(
            ir_teacher_ckpt,
            self.ir_teacher,
            "ir",
            use_ema=use_teacher_ema,
        )
        self.ir_teacher.requires_grad_(False)
        self.ir_teacher.eval()

        self.depth_teacher = TriStreamHGNetv2(
            name=bb_cfg["name"],
            use_lab=bb_cfg.get("use_lab", False),
            return_idx=tuple(bb_cfg.get("return_idx", [1, 2, 3])),
            pretrained=False,
            share_stages=bb_cfg.get("share_stages", True),
            depth_in_chs=bb_cfg.get("depth_in_chs", 2),
        )
        self.depth_teacher_source = _load_teacher_backbone(
            depth_teacher_ckpt,
            self.depth_teacher,
            "depth",
            use_ema=use_teacher_ema,
        )
        self.depth_teacher.requires_grad_(False)
        self.depth_teacher.eval()

        channels = tuple(self.ir_teacher.out_channels)
        if len(channels) != len(self.level_weights):
            raise ValueError(
                f"teacher feature levels ({len(channels)}) do not match level_weights ({len(self.level_weights)})"
            )
        if self.use_adapters:
            self.ir_adapters = nn.ModuleList(
                ResidualFeatureAdapter(c, reduction=adapter_reduction) for c in channels
            )
            self.depth_adapters = nn.ModuleList(
                ResidualFeatureAdapter(c, reduction=adapter_reduction) for c in channels
            )
        else:
            # Keep stable checkpoint attributes without adding trainable parameters.
            self.ir_adapters = nn.ModuleList(nn.Identity() for _ in channels)
            self.depth_adapters = nn.ModuleList(nn.Identity() for _ in channels)

    @staticmethod
    def _foreground_mask(targets: list[dict], height: int, width: int, reference: torch.Tensor) -> torch.Tensor:
        """Rasterize normalized cxcywh boxes onto feature cells using box/cell overlap."""
        masks = reference.new_zeros((len(targets), height, width))
        cell_x0 = torch.arange(width, device=reference.device, dtype=reference.dtype) / width
        cell_x1 = cell_x0 + 1.0 / width
        cell_y0 = torch.arange(height, device=reference.device, dtype=reference.dtype) / height
        cell_y1 = cell_y0 + 1.0 / height
        for batch_idx, target in enumerate(targets):
            boxes = target["boxes"].to(device=reference.device, dtype=reference.dtype)
            if boxes.numel() == 0:
                continue
            cx, cy, box_w, box_h = boxes.unbind(-1)
            x0 = (cx - box_w / 2).clamp(0, 1)
            x1 = (cx + box_w / 2).clamp(0, 1)
            y0 = (cy - box_h / 2).clamp(0, 1)
            y1 = (cy + box_h / 2).clamp(0, 1)
            overlap_x = (cell_x1[None] > x0[:, None]) & (cell_x0[None] < x1[:, None])
            overlap_y = (cell_y1[None] > y0[:, None]) & (cell_y0[None] < y1[:, None])
            masks[batch_idx] = (overlap_y[:, :, None] & overlap_x[:, None, :]).any(dim=0)
        return masks

    @staticmethod
    def _availability(
        targets: list[dict] | None,
        key: str | tuple[str, ...],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if targets is None:
            return reference.new_ones((reference.shape[0], 1, 1))
        keys = (key,) if isinstance(key, str) else key
        values = []
        for target in targets:
            value = reference.new_tensor(1.0)
            for item in keys:
                value = value * target.get(item, reference.new_tensor(1.0)).to(
                    device=reference.device, dtype=reference.dtype
                )
            values.append(value)
        return torch.stack(
            values
        ).reshape(-1, 1, 1)

    @staticmethod
    def _simam_attention(feature: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        mean = feature.mean(dim=(-2, -1), keepdim=True)
        variance = (feature - mean).square().mean(dim=(-2, -1), keepdim=True)
        energy = (feature - mean).square() / (4.0 * (variance + eps)) + 0.5
        return torch.sigmoid(energy)

    def _dynamic_modality_weights(
        self,
        targets: list[dict] | None,
        depth_quality: torch.Tensor | None,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = reference.shape[0]
        shape = (batch_size, 1, 1)
        if not self.dynamic_reliability or targets is None:
            return (
                reference.new_full(shape, self.ir_weight),
                reference.new_full(shape, self.depth_weight),
            )

        def target_values(key: str) -> torch.Tensor:
            return torch.stack([
                target[key].to(device=reference.device, dtype=reference.dtype)
                for target in targets
            ]).reshape(shape)

        ir_available = self._availability(targets, "ir_distill_valid", reference)
        depth_available = self._availability(targets, "depth_distill_valid", reference)
        ir_confidence = target_values("ir_teacher_confidence").clamp(0.0, 1.0)
        depth_confidence = target_values("depth_teacher_confidence").clamp(0.0, 1.0)
        if depth_quality is None:
            depth_coverage = reference.new_ones(shape)
        else:
            depth_coverage = depth_quality.mean(dim=(-2, -1), keepdim=False).reshape(shape)

        ir_raw = self.ir_weight * ir_available * ir_confidence.clamp_min(
            self.reliability_floor
        ).pow(self.reliability_power)
        depth_raw = self.depth_weight * depth_available * depth_confidence.clamp_min(
            self.reliability_floor
        ).pow(self.reliability_power) * depth_coverage.clamp(0.0, 1.0)
        raw_sum = ir_raw + depth_raw
        total_weight = self.ir_weight + self.depth_weight
        active = raw_sum > 0
        ir_dynamic = torch.where(active, total_weight * ir_raw / raw_sum.clamp_min(1e-6), ir_raw)
        depth_dynamic = torch.where(
            active, total_weight * depth_raw / raw_sum.clamp_min(1e-6), depth_raw
        )
        return ir_dynamic, depth_dynamic

    @staticmethod
    def _pkd_point_loss(
        student: torch.Tensor,
        teacher: torch.Tensor,
        spatial_weight: torch.Tensor,
    ) -> torch.Tensor:
        def normalize(x: torch.Tensor) -> torch.Tensor:
            weight = spatial_weight[:, None]
            denominator = weight.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
            mean = (x * weight).sum(dim=(-2, -1), keepdim=True) / denominator
            variance = ((x - mean).square() * weight).sum(
                dim=(-2, -1), keepdim=True
            ) / denominator
            return (x - mean) * torch.rsqrt(variance + 1e-6)

        # Half-MSE between standardized responses equals 1-Pearson correlation
        # when the complete spatial map is used.
        return 0.5 * (normalize(student) - normalize(teacher)).square().mean(dim=1)

    def _point_weight(
        self,
        point_loss: torch.Tensor,
        targets: list[dict] | None,
        availability: torch.Tensor,
        quality_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        height, width = point_loss.shape[-2:]
        point_weight = torch.ones_like(point_loss)
        if targets is not None and self.foreground_weight > 1.0:
            foreground = self._foreground_mask(targets, height, width, point_loss)
            point_weight = point_weight + (self.foreground_weight - 1.0) * foreground
        if quality_mask is not None:
            level_quality = F.interpolate(quality_mask, size=(height, width), mode="area")[:, 0]
            point_weight = point_weight * level_quality.clamp(0.0, 1.0)
        return point_weight * availability

    def _pkd_loss(
        self,
        student_feats: list[torch.Tensor],
        teacher_feats: list[torch.Tensor],
        targets: list[dict] | None = None,
        quality_mask: torch.Tensor | None = None,
        availability_key: str | tuple[str, ...] = "ir_distill_valid",
        attention: list[torch.Tensor] | None = None,
        sample_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if len(student_feats) != len(teacher_feats) or len(student_feats) != len(self.level_weights):
            raise ValueError(
                "student/teacher feature levels must match distill.level_weights: "
                f"student={len(student_feats)} teacher={len(teacher_feats)} weights={len(self.level_weights)}"
            )
        availability = self._availability(targets, availability_key, student_feats[0])
        losses = []
        for level, (student, teacher) in enumerate(zip(student_feats, teacher_feats)):
            if attention is not None:
                student = student * attention[level]
                teacher = teacher * attention[level]
            point_weight = self._point_weight(
                student[:, 0],
                targets=targets,
                availability=availability,
                quality_mask=quality_mask,
            )
            point_loss = self._pkd_point_loss(student, teacher, point_weight)
            numerator = (point_loss * point_weight).sum(dim=(-2, -1))
            denominator = point_weight.sum(dim=(-2, -1))
            per_sample = numerator / denominator.clamp_min(1.0)
            active = denominator > 0
            if sample_weight is None:
                weight = active.to(per_sample.dtype)
            else:
                weight = sample_weight.reshape(-1) * active.to(per_sample.dtype)
            losses.append((per_sample * weight).sum() / weight.sum().clamp_min(1e-6))
        level_weight = losses[0].new_tensor(self.level_weights)
        aggregate = (torch.stack(losses) * level_weight).sum() / level_weight.sum()
        return aggregate, losses

    def _cosine_loss(
        self,
        student_feats: list[torch.Tensor],
        teacher_feats: list[torch.Tensor],
        targets: list[dict] | None = None,
        quality_mask: torch.Tensor | None = None,
        availability_key: str | tuple[str, ...] = "ir_distill_valid",
        sample_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Legacy LIM: foreground-aware pointwise cosine distance."""
        if len(student_feats) != len(teacher_feats) or len(student_feats) != len(self.level_weights):
            raise ValueError(
                "student/teacher feature levels must match distill.level_weights: "
                f"student={len(student_feats)} teacher={len(teacher_feats)} weights={len(self.level_weights)}"
            )
        availability = self._availability(targets, availability_key, student_feats[0])
        losses = []
        for student, teacher in zip(student_feats, teacher_feats):
            point_loss = 1.0 - F.cosine_similarity(student, teacher, dim=1, eps=1e-6)
            point_weight = self._point_weight(
                point_loss,
                targets=targets,
                availability=availability,
                quality_mask=quality_mask,
            )
            numerator = (point_loss * point_weight).sum(dim=(-2, -1))
            denominator = point_weight.sum(dim=(-2, -1))
            per_sample = numerator / denominator.clamp_min(1.0)
            active = denominator > 0
            if sample_weight is None:
                weight = active.to(per_sample.dtype)
            else:
                weight = sample_weight.reshape(-1) * active.to(per_sample.dtype)
            losses.append((per_sample * weight).sum() / weight.sum().clamp_min(1e-6))
        level_weight = losses[0].new_tensor(self.level_weights)
        aggregate = (torch.stack(losses) * level_weight).sum() / level_weight.sum()
        return aggregate, losses

    @torch.no_grad()
    def _teacher_feats(self, teacher: TriStreamHGNetv2, x: torch.Tensor, modality: str) -> list[torch.Tensor]:
        stem = {"ir": teacher.stem_ir, "depth": teacher.stem_depth}[modality]
        stages = {"ir": teacher.stages_ir, "depth": teacher.stages_depth}[modality]
        if modality == "depth":
            x = x[:, : teacher.depth_in_chs]
        return teacher._forward_stem(x, stem, stages, use_checkpoint=False)

    def forward(
        self,
        ir_input: torch.Tensor,
        depth_input: torch.Tensor,
        student_ir_feats: list[torch.Tensor],
        student_depth_feats: list[torch.Tensor],
        targets: list[dict] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        teacher_ir_feats = None
        teacher_depth_feats = None
        projected_ir = None
        projected_depth = None
        if self.ir_weight > 0 or self.cross_weight > 0:
            teacher_ir_feats = self._teacher_feats(self.ir_teacher, ir_input, "ir")
            projected_ir = [adapter(feat) for adapter, feat in zip(self.ir_adapters, student_ir_feats)]
        if self.depth_weight > 0 or self.cross_weight > 0:
            teacher_depth_feats = self._teacher_feats(self.depth_teacher, depth_input, "depth")
            projected_depth = [
                adapter(feat) for adapter, feat in zip(self.depth_adapters, student_depth_feats)
            ]
        depth_quality = None
        if self.use_depth_valid_mask:
            depth_quality = depth_input[:, 4:5] if depth_input.shape[1] >= 5 else depth_input[:, 1:2]
        dynamic_ir_weight, dynamic_depth_weight = self._dynamic_modality_weights(
            targets, depth_quality, student_ir_feats[0]
        )

        same_modality_loss = self._pkd_loss if self.method == "pkd_lcm" else self._cosine_loss
        zero = student_ir_feats[0].new_zeros(())
        ir_levels = [zero for _ in self.level_weights]
        depth_levels = [zero for _ in self.level_weights]
        loss_ir = zero
        loss_depth = zero
        if self.ir_weight > 0:
            loss_ir, ir_levels = same_modality_loss(
                projected_ir,
                teacher_ir_feats,
                targets=targets,
                availability_key="ir_distill_valid",
                sample_weight=dynamic_ir_weight,
            )
        if self.depth_weight > 0:
            loss_depth, depth_levels = same_modality_loss(
                projected_depth,
                teacher_depth_feats,
                targets=targets,
                quality_mask=depth_quality,
                availability_key="depth_distill_valid",
                sample_weight=dynamic_depth_weight,
            )

        cross_levels = [level.new_zeros(()) for level in ir_levels]
        loss_cross = loss_ir.new_zeros(())
        if self.cross_weight > 0:
            ir_attention = [self._simam_attention(feature) for feature in teacher_ir_feats]
            depth_attention = [self._simam_attention(feature) for feature in teacher_depth_feats]
            depth_to_ir, depth_to_ir_levels = self._pkd_loss(
                projected_depth,
                teacher_ir_feats,
                targets=targets,
                quality_mask=depth_quality,
                availability_key=("ir_distill_valid", "depth_distill_valid"),
                attention=ir_attention,
            )
            ir_to_depth, ir_to_depth_levels = self._pkd_loss(
                projected_ir,
                teacher_depth_feats,
                targets=targets,
                quality_mask=depth_quality,
                availability_key=("ir_distill_valid", "depth_distill_valid"),
                attention=depth_attention,
            )
            loss_cross = depth_to_ir + ir_to_depth
            cross_levels = [
                left + right
                for left, right in zip(depth_to_ir_levels, ir_to_depth_levels)
            ]

        weighted_ir = self.weight * dynamic_ir_weight.mean() * loss_ir
        weighted_depth = self.weight * dynamic_depth_weight.mean() * loss_depth
        weighted_cross = self.weight * self.cross_weight * loss_cross
        total = weighted_ir + weighted_depth + weighted_cross
        metrics = {
            "loss_ir": weighted_ir.detach(),
            "loss_depth": weighted_depth.detach(),
            "loss_cross": weighted_cross.detach(),
            "weight_ir_dynamic": dynamic_ir_weight.mean().detach(),
            "weight_depth_dynamic": dynamic_depth_weight.mean().detach(),
        }
        for level, value in enumerate(ir_levels):
            metrics[f"loss_ir_l{level}"] = value.detach()
        for level, value in enumerate(depth_levels):
            metrics[f"loss_depth_l{level}"] = value.detach()
        for level, value in enumerate(cross_levels):
            metrics[f"loss_cross_l{level}"] = value.detach()
        return total, metrics


class FDRCrossModelDistillLoss(nn.Module):
    """Cross-model distillation of D-FINE's fine-grained box distributions.

    The frozen mono-modal teachers provide ``pred_corners`` (four categorical
    distance distributions).  Student and teacher queries are paired through
    their independent GT Hungarian matches, avoiding the invalid assumption
    that DETR query indices are aligned across models.
    """

    def __init__(self, cfg: dict, ir_teacher_ckpt: str, depth_teacher_ckpt: str,
                 weight: float = 0.25, temperature: float = 5.0,
                 ir_weight: float = 1.0, depth_weight: float = 1.0):
        super().__init__()
        if weight < 0 or temperature <= 0:
            raise ValueError("invalid FDR distillation weight/temperature")
        self.weight = float(weight)
        self.temperature = float(temperature)
        self.ir_weight = float(ir_weight)
        self.depth_weight = float(depth_weight)
        from .tristream_dfine import build_single_modality_dfine
        self.teachers = nn.ModuleDict()
        for modality, path in (("ir", ir_teacher_ckpt), ("depth", depth_teacher_ckpt)):
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            state = ckpt.get("ema") if ckpt.get("ema") is not None else ckpt.get("model", ckpt)
            state = {k.removeprefix("module."): v for k, v in state.items()}
            teacher_cfg = copy.deepcopy(cfg)
            teacher_cfg["backbone"]["pretrained"] = False
            teacher_cfg["load_pretrained_detector"] = False
            teacher = build_single_modality_dfine(teacher_cfg, modality)

            model_state = teacher.state_dict()
            incompatible_shapes = {
                key: (tuple(value.shape), tuple(model_state[key].shape))
                for key, value in state.items()
                if key in model_state and value.shape != model_state[key].shape
            }
            # A mono checkpoint contains all three stems even though only one
            # is used.  It is valid to omit an incompatible inactive stem, but
            # never adapt or silently skip the active modality, encoder or
            # decoder tensors.
            active_prefixes = (
                f"backbone.stem_{modality}.",
                f"backbone.stages_{modality}.",
                "encoder.",
                "decoder.",
            )
            active_mismatches = {
                key: shapes
                for key, shapes in incompatible_shapes.items()
                if key.startswith(active_prefixes)
            }
            if active_mismatches:
                raise RuntimeError(
                    f"FDR teacher {modality} has active shape mismatches: {active_mismatches}"
                )
            compatible_state = {
                key: value
                for key, value in state.items()
                if key in model_state and key not in incompatible_shapes
            }
            result = teacher.load_state_dict(compatible_state, strict=False)
            skipped = set(incompatible_shapes)
            unexpected_missing = [key for key in result.missing_keys if key not in skipped]
            if unexpected_missing or result.unexpected_keys:
                raise RuntimeError(
                    f"FDR teacher {modality} checkpoint is incomplete: "
                    f"missing={unexpected_missing[:20]} unexpected={result.unexpected_keys[:20]}"
                )
            if incompatible_shapes:
                print(
                    f"[distill] FDR {modality} ignored inactive shape mismatches: "
                    f"{incompatible_shapes}",
                    flush=True,
                )
            teacher.requires_grad_(False)
            teacher.eval()
            self.teachers[modality] = teacher

    @staticmethod
    def _aligned_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        def xyxy(x):
            cx, cy, w, h = x.unbind(-1)
            return torch.stack((cx-w/2, cy-h/2, cx+w/2, cy+h/2), -1)
        a, b = xyxy(boxes1), xyxy(boxes2)
        lt, rb = torch.maximum(a[:, :2], b[:, :2]), torch.minimum(a[:, 2:], b[:, 2:])
        inter = (rb-lt).clamp_min(0).prod(-1)
        area_a = (a[:, 2:]-a[:, :2]).clamp_min(0).prod(-1)
        area_b = (b[:, 2:]-b[:, :2]).clamp_min(0).prod(-1)
        return inter / (area_a + area_b - inter).clamp_min(1e-8)

    @torch.no_grad()
    def _teacher_forward(self, modality, rgb, ir, depth, targets):
        teacher = self.teachers[modality]
        # D-FINE exposes ``pred_corners`` only on its training decoder path.
        # The teacher remains frozen and gradient-free; temporarily selecting
        # that path is solely to read the fine-grained distributions, then the
        # module is restored to eval mode before returning.
        teacher.train()
        outputs = teacher(rgb, ir, depth, targets)
        teacher.eval()
        return outputs

    def _one(self, student_outputs, teacher_outputs, targets, matcher):
        pred = student_outputs.get("pred_corners")
        target = teacher_outputs.get("pred_corners")
        if pred is None or target is None:
            return student_outputs["pred_logits"].sum() * 0.0, 0
        sidx = matcher(student_outputs, targets)["indices"]
        tidx = matcher(teacher_outputs, targets)["indices"]
        total = pred.sum() * 0.0
        count = 0
        T = self.temperature
        reg_max = pred.shape[-1] // 4 - 1
        for b, ((sq, sg), (tq, tg)) in enumerate(zip(sidx, tidx)):
            smap = {int(g): int(q) for q, g in zip(sq, sg)}
            tmap = {int(g): int(q) for q, g in zip(tq, tg)}
            common = sorted(set(smap).intersection(tmap))
            if not common:
                continue
            g = torch.as_tensor(common, device=pred.device, dtype=torch.long)
            si = torch.as_tensor([smap[int(x)] for x in common], device=pred.device)
            ti = torch.as_tensor([tmap[int(x)] for x in common], device=pred.device)
            sp = pred[b, si].reshape(-1, 4, reg_max + 1)
            tp = target[b, ti].detach().reshape(-1, 4, reg_max + 1)
            sb = student_outputs["pred_boxes"][b, si]
            tb = teacher_outputs["pred_boxes"][b, ti].detach()
            labels = targets[b]["labels"][g]
            score = teacher_outputs["pred_logits"][b, ti].detach().sigmoid().gather(1, labels[:, None]).squeeze(1)
            quality = self._aligned_iou(tb, targets[b]["boxes"][g]).detach()
            w = (score * quality).clamp_min(1e-4)
            kl = F.kl_div(F.log_softmax(sp / T, -1), F.softmax(tp / T, -1), reduction="none").sum(-1).mean(-1)
            total = total + (kl * w).sum() / w.sum().clamp_min(1e-6)
            count += len(common)
        return total / max(len(targets), 1), count

    def forward(self, rgb, ir, depth, student_outputs, targets, matcher):
        zero = student_outputs["pred_logits"].sum() * 0.0
        total, pairs = zero, 0
        metrics = {}
        for modality, inp_weight in (("ir", self.ir_weight), ("depth", self.depth_weight)):
            if inp_weight <= 0:
                continue
            with torch.no_grad():
                teacher_outputs = self._teacher_forward(modality, rgb, ir, depth, targets)
            loss, n = self._one(student_outputs, teacher_outputs, targets, matcher)
            total = total + inp_weight * loss
            pairs += n
            metrics[f"loss_fdr_{modality}"] = (inp_weight * loss).detach()
        metrics["fdr_pairs"] = student_outputs["pred_logits"].new_tensor(float(pairs))
        return self.weight * total, metrics


class TriStreamSelfDistillLoss(nn.Module):
    """Static same-architecture teacher with GT-aligned DETR query distillation.

    Student and teacher queries are matched to ground truth independently, then
    paired through the GT index. This avoids assuming that equal query indices
    represent the same object, which is not valid for DETR-style detectors.
    """

    def __init__(
        self,
        model_cfg: dict,
        teacher_ckpt: str,
        use_teacher_ema: bool = True,
        cls_weight: float = 2.0,
        bbox_weight: float = 5.0,
        giou_weight: float = 2.0,
        temperature: float = 2.0,
        teacher_score_thresh: float = 0.25,
        teacher_gt_iou_thresh: float = 0.5,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("self_distill.temperature must be positive")
        if not 0.0 <= teacher_score_thresh <= 1.0:
            raise ValueError("self_distill.teacher_score_thresh must be in [0, 1]")
        if not 0.0 <= teacher_gt_iou_thresh <= 1.0:
            raise ValueError("self_distill.teacher_gt_iou_thresh must be in [0, 1]")

        # Avoid loading classification/detector pretraining before immediately
        # replacing every tensor with the frozen teacher checkpoint.
        teacher_cfg = copy.deepcopy(model_cfg)
        teacher_cfg["backbone"]["pretrained"] = False
        teacher_cfg.pop("pretrained_detector", None)
        teacher_cfg.pop("distill", None)
        teacher_cfg.pop("dino_distill", None)
        teacher_cfg.pop("self_distill", None)

        from .tristream_dfine import build_tristream_dfine

        self.teacher = build_tristream_dfine(teacher_cfg)
        checkpoint = torch.load(teacher_ckpt, map_location="cpu", weights_only=False)
        source = "ema" if use_teacher_ema and checkpoint.get("ema") is not None else "model"
        state = checkpoint[source]
        if isinstance(state, dict) and "module" in state and isinstance(state["module"], dict):
            state = state["module"]
        state = {key.removeprefix("module."): value for key, value in state.items()}
        self.teacher.load_state_dict(state, strict=True)
        self.teacher.requires_grad_(False)
        self.teacher.eval()

        self.teacher_ckpt = teacher_ckpt
        self.teacher_source = source
        self.cls_weight = float(cls_weight)
        self.bbox_weight = float(bbox_weight)
        self.giou_weight = float(giou_weight)
        self.temperature = float(temperature)
        self.teacher_score_thresh = float(teacher_score_thresh)
        self.teacher_gt_iou_thresh = float(teacher_gt_iou_thresh)

    def train(self, mode: bool = True):
        # The wrapper follows the caller's mode, but the static teacher must
        # always retain inference-time normalization and deterministic outputs.
        super().train(mode)
        self.teacher.eval()
        return self

    @staticmethod
    def _aligned_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        cx1, cy1, w1, h1 = boxes1.unbind(-1)
        cx2, cy2, w2, h2 = boxes2.unbind(-1)
        a1 = torch.stack((cx1 - w1 / 2, cy1 - h1 / 2, cx1 + w1 / 2, cy1 + h1 / 2), dim=-1)
        a2 = torch.stack((cx2 - w2 / 2, cy2 - h2 / 2, cx2 + w2 / 2, cy2 + h2 / 2), dim=-1)
        lt = torch.maximum(a1[:, :2], a2[:, :2])
        rb = torch.minimum(a1[:, 2:], a2[:, 2:])
        inter = (rb - lt).clamp(min=0).prod(-1)
        area1 = (a1[:, 2:] - a1[:, :2]).clamp(min=0).prod(-1)
        area2 = (a2[:, 2:] - a2[:, :2]).clamp(min=0).prod(-1)
        return inter / (area1 + area2 - inter).clamp(min=1e-8)

    @staticmethod
    def _aligned_giou_loss(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        cx1, cy1, w1, h1 = boxes1.unbind(-1)
        cx2, cy2, w2, h2 = boxes2.unbind(-1)
        a1 = torch.stack((cx1 - w1 / 2, cy1 - h1 / 2, cx1 + w1 / 2, cy1 + h1 / 2), dim=-1)
        a2 = torch.stack((cx2 - w2 / 2, cy2 - h2 / 2, cx2 + w2 / 2, cy2 + h2 / 2), dim=-1)
        lt = torch.maximum(a1[:, :2], a2[:, :2])
        rb = torch.minimum(a1[:, 2:], a2[:, 2:])
        inter = (rb - lt).clamp(min=0).prod(-1)
        area1 = (a1[:, 2:] - a1[:, :2]).clamp(min=0).prod(-1)
        area2 = (a2[:, 2:] - a2[:, :2]).clamp(min=0).prod(-1)
        union = (area1 + area2 - inter).clamp(min=1e-8)
        iou = inter / union
        enclosing_lt = torch.minimum(a1[:, :2], a2[:, :2])
        enclosing_rb = torch.maximum(a1[:, 2:], a2[:, 2:])
        enclosing = (enclosing_rb - enclosing_lt).clamp(min=0).prod(-1).clamp(min=1e-8)
        giou = iou - (enclosing - union) / enclosing
        return 1.0 - giou

    @torch.no_grad()
    def teacher_forward(self, rgb: torch.Tensor, ir: torch.Tensor, depth: torch.Tensor) -> dict:
        self.teacher.eval()
        return self.teacher(rgb, ir, depth)

    def forward(self, student_outputs: dict, teacher_outputs: dict, targets: list[dict], matcher):
        student_indices = matcher(student_outputs, targets)["indices"]
        teacher_indices = matcher(teacher_outputs, targets)["indices"]
        reference = student_outputs["pred_logits"]
        zero = reference.sum() * 0.0

        cls_sum = zero
        bbox_sum = zero
        giou_sum = zero
        weight_sum = zero
        matched_pairs = 0
        accepted_pairs = 0

        for batch_idx, (student_pair, teacher_pair) in enumerate(zip(student_indices, teacher_indices)):
            student_query, student_gt = student_pair
            teacher_query, teacher_gt = teacher_pair
            student_by_gt = {int(gt): int(query) for query, gt in zip(student_query, student_gt)}
            teacher_by_gt = {int(gt): int(query) for query, gt in zip(teacher_query, teacher_gt)}
            common_gt = sorted(set(student_by_gt).intersection(teacher_by_gt))
            matched_pairs += len(common_gt)
            if not common_gt:
                continue

            device = reference.device
            gt_idx = torch.tensor(common_gt, device=device, dtype=torch.long)
            student_idx = torch.tensor([student_by_gt[idx] for idx in common_gt], device=device)
            teacher_idx = torch.tensor([teacher_by_gt[idx] for idx in common_gt], device=device)
            labels = targets[batch_idx]["labels"][gt_idx]
            gt_boxes = targets[batch_idx]["boxes"][gt_idx]

            student_logits = student_outputs["pred_logits"][batch_idx, student_idx]
            student_boxes = student_outputs["pred_boxes"][batch_idx, student_idx]
            teacher_logits = teacher_outputs["pred_logits"][batch_idx, teacher_idx].detach()
            teacher_boxes = teacher_outputs["pred_boxes"][batch_idx, teacher_idx].detach()
            teacher_probs = teacher_logits.sigmoid()
            teacher_scores = teacher_probs.gather(1, labels[:, None]).squeeze(1)
            teacher_iou = self._aligned_iou(teacher_boxes, gt_boxes)
            valid = (teacher_scores >= self.teacher_score_thresh) & (
                teacher_iou >= self.teacher_gt_iou_thresh
            )
            if not valid.any():
                continue

            accepted_pairs += int(valid.sum().item())
            student_logits = student_logits[valid]
            student_boxes = student_boxes[valid]
            teacher_logits = teacher_logits[valid]
            teacher_boxes = teacher_boxes[valid]
            weights = (teacher_scores[valid] * teacher_iou[valid]).detach().clamp(min=1e-4)

            soft_targets = torch.sigmoid(teacher_logits / self.temperature)
            cls_each = F.binary_cross_entropy_with_logits(
                student_logits / self.temperature,
                soft_targets,
                reduction="none",
            ).mean(-1) * (self.temperature**2)
            bbox_each = F.l1_loss(student_boxes, teacher_boxes, reduction="none").mean(-1)
            giou_each = self._aligned_giou_loss(student_boxes, teacher_boxes)

            cls_sum = cls_sum + (cls_each * weights).sum()
            bbox_sum = bbox_sum + (bbox_each * weights).sum()
            giou_sum = giou_sum + (giou_each * weights).sum()
            weight_sum = weight_sum + weights.sum()

        if accepted_pairs:
            cls_loss = cls_sum / weight_sum.clamp(min=1e-8)
            bbox_loss = bbox_sum / weight_sum.clamp(min=1e-8)
            giou_loss = giou_sum / weight_sum.clamp(min=1e-8)
        else:
            cls_loss = bbox_loss = giou_loss = zero

        total = (
            self.cls_weight * cls_loss
            + self.bbox_weight * bbox_loss
            + self.giou_weight * giou_loss
        )
        metrics = {
            "loss_self_cls": cls_loss.detach(),
            "loss_self_bbox": bbox_loss.detach(),
            "loss_self_giou": giou_loss.detach(),
            "self_pairs": reference.new_tensor(float(accepted_pairs)),
            "self_coverage": reference.new_tensor(
                float(accepted_pairs) / max(float(matched_pairs), 1.0)
            ),
        }
        return total, metrics


class DINOv2RGBFeatureDistillLoss(nn.Module):
    """Distill frozen DINOv2 patch features into the RGB backbone only."""

    def __init__(
        self,
        checkpoint_dir: str,
        student_channels: tuple[int, ...] = (512, 1024, 2048),
        weight: float = 0.5,
        foreground_weight: float = 2.0,
        level_weights: tuple[float, ...] = (0.0, 1.0, 1.0),
        patch_size: int = 14,
    ):
        super().__init__()
        from transformers import AutoModel

        self.weight = float(weight)
        self.foreground_weight = float(foreground_weight)
        self.level_weights = tuple(float(v) for v in level_weights)
        self.patch_size = int(patch_size)
        if len(student_channels) != len(self.level_weights):
            raise ValueError("dino_distill level_weights must match student feature levels")
        if self.foreground_weight < 1.0 or max(self.level_weights, default=0.0) <= 0:
            raise ValueError("DINOv2 distill weights must be valid")

        self.teacher = AutoModel.from_pretrained(checkpoint_dir, local_files_only=True)
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        hidden_size = int(self.teacher.config.hidden_size)
        self.adapters = nn.ModuleList(nn.Conv2d(ch, hidden_size, 1) for ch in student_channels)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @staticmethod
    def _foreground_mask(targets: list[dict], height: int, width: int, reference: torch.Tensor) -> torch.Tensor:
        return M2DDistillLoss._foreground_mask(targets, height, width, reference)

    def forward(
        self,
        rgb_input: torch.Tensor,
        student_feats: list[torch.Tensor],
        targets: list[dict] | None = None,
    ) -> torch.Tensor:
        if len(student_feats) != len(self.adapters):
            raise ValueError("DINOv2 student feature levels do not match adapters")
        normalized = (rgb_input - self.mean) / self.std
        with torch.no_grad():
            output = self.teacher(pixel_values=normalized, interpolate_pos_encoding=True)
            tokens = output.last_hidden_state[:, 1:]
            patch_h = rgb_input.shape[-2] // self.patch_size
            patch_w = rgb_input.shape[-1] // self.patch_size
            if patch_h * patch_w != tokens.shape[1]:
                raise RuntimeError(
                    f"DINOv2 patch grid mismatch: expected {patch_h}x{patch_w}, tokens={tokens.shape[1]}"
                )
            teacher_map = tokens.transpose(1, 2).reshape(tokens.shape[0], -1, patch_h, patch_w).detach()

        losses = []
        for idx, (student, adapter, level_weight) in enumerate(
            zip(student_feats, self.adapters, self.level_weights)
        ):
            if level_weight <= 0:
                continue
            target = F.interpolate(teacher_map, size=student.shape[-2:], mode="bilinear", align_corners=False)
            projected = adapter(student)
            point_loss = 1.0 - F.cosine_similarity(projected, target, dim=1, eps=1e-6)
            point_weight = torch.ones_like(point_loss)
            if targets is not None and self.foreground_weight > 1.0:
                foreground = self._foreground_mask(targets, point_loss.shape[-2], point_loss.shape[-1], point_loss)
                point_weight = point_weight + (self.foreground_weight - 1.0) * foreground
            losses.append((point_loss * point_weight).sum() / point_weight.sum().clamp_min(1.0) * level_weight)
        return self.weight * torch.stack(losses).sum() / sum(v for v in self.level_weights if v > 0)

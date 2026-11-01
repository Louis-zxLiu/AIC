"""Elastic RF-DETR teachers and tri-modal student.

The installed RF-DETR 2XL has five decoder layers and exposes all intermediate
predictions when auxiliary loss is enabled.  We keep every parameter registered
and select an intermediate layer as the active subnet output.  This is safe for
DDP and gives the same predictions as an exported early-exit decoder; export can
physically remove the unused suffix after search.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from torch import nn

from aic.models.opf_fusion import MultiScaleOPF
from aic.rf_native.model import ModalityInput, RFBranch, build_native, load_branch


@dataclass(frozen=True)
class Architecture:
    windows_rgb: int = 2
    windows_ir: int = 2
    windows_depth: int = 2
    decoder_layers: int = 5
    queries: int = 300
    opf_depth: int = 1

    def as_dict(self):
        return dict(windows_rgb=self.windows_rgb, windows_ir=self.windows_ir,
                    windows_depth=self.windows_depth, decoder_layers=self.decoder_layers,
                    queries=self.queries, opf_depth=self.opf_depth)


def _set_windows(module: nn.Module, count: int) -> None:
    """Update every cached window-count field used by the installed backbone."""
    if count not in (2, 4):
        raise ValueError(f"unsupported num_windows={count}")
    touched = 0
    for child in module.modules():
        if hasattr(child, "num_windows"):
            child.num_windows = count
            touched += 1
        config = getattr(child, "config", None)
        if config is not None and hasattr(config, "num_windows"):
            config.num_windows = count
    if not touched:
        raise RuntimeError("RF backbone exposes no window controls")


def _select_decoder_output(outputs: dict, depth: int) -> dict:
    if depth not in (3, 4, 5):
        raise ValueError(f"unsupported decoder depth={depth}")
    auxiliaries = list(outputs.get("aux_outputs", []))
    layers = auxiliaries + [{k: outputs[k] for k in ("pred_logits", "pred_boxes") if k in outputs}]
    if len(layers) != 5:
        raise RuntimeError(f"expected five RF-DETR decoder outputs, got {len(layers)}")
    chosen = dict(outputs)
    chosen.update(layers[depth - 1])
    chosen["aux_outputs"] = layers[: depth - 1]
    chosen["nas_decoder_layers"] = depth
    return chosen


class ArchitectureSampler(nn.Module):
    def __init__(self, cfg: dict, student: bool):
        super().__init__()
        self.cfg = cfg.get("nas", {})
        self.student = student
        self.register_buffer("step", torch.zeros((), dtype=torch.long), persistent=True)
        self.last: Architecture | None = None

    def _arch(self, index: int, *, maximal: bool = False) -> Architecture:
        windows = tuple(self.cfg.get("windows", [2, 4]))
        depths = tuple(self.cfg.get("decoder_layers", [3, 4, 5]))
        queries = tuple(self.cfg.get("queries", [150, 225, 300]))
        opfs = tuple(self.cfg.get("opf_depth", [1, 2])) if self.student else (1,)
        if maximal:
            return Architecture(max(windows), max(windows), max(windows), max(depths), max(queries), max(opfs))
        wr = windows[index % len(windows)]; index //= len(windows)
        wi = windows[index % len(windows)] if self.student else wr; index //= len(windows) if self.student else 1
        wd = windows[index % len(windows)] if self.student else wr; index //= len(windows) if self.student else 1
        dec = depths[index % len(depths)]; index //= len(depths)
        query = queries[index % len(queries)]; index //= len(queries)
        opf = opfs[index % len(opfs)]
        return Architecture(wr, wi, wd, dec, query, opf)

    def choose(self, training: bool) -> Architecture:
        if self.cfg.get("expanded_search"):
            from .search_schedule import choose
            return choose(self, training, Architecture)
        force = self.cfg.get("force_arch")
        if force:
            arch = Architecture(**force)
        elif not training:
            arch = Architecture(**self.cfg.get("eval_arch", {}))
        else:
            current = int(self.step.item())
            epoch = current // max(int(self.cfg.get("steps_per_epoch", 1)), 1)
            if epoch < 4:
                arch = Architecture()
            elif epoch < 8:
                arch = self._arch(current, maximal=False)
                arch = Architecture(arch.windows_rgb, arch.windows_ir, arch.windows_depth, 5, 300, 1)
            else:
                slot = current % 10
                arch = Architecture() if slot < 2 else self._arch(current, maximal=True) if slot < 4 else self._arch(current)
            self.step.add_(1)
        self.last = arch
        return arch


class ElasticFusionBackbone(nn.Module):
    def __init__(self, native, cfg):
        super().__init__()
        self.rgb = native
        self.ir = RFBranch(copy.deepcopy(native), "ir")
        self.depth = RFBranch(copy.deepcopy(native), "depth")
        self.teacher_init = {}
        if not cfg.get("inference_only", False):
            for mode in ("ir", "depth"):
                self.teacher_init[mode] = load_branch(
                    getattr(self, mode), cfg["distill"][mode + "_teacher_ckpt"], cfg["train"].get("split_sha256")
                )
        self.opf1 = MultiScaleOPF([512], num_heads=8, sr_ratios=[2], hard_depth_gate=True)
        self.opf2 = MultiScaleOPF([512], num_heads=8, sr_ratios=[2], hard_depth_gate=True)
        self.opf2_gain = nn.Parameter(torch.tensor(math.log(0.01 / 0.99)))
        self.active_inputs = None
        self.active_arch: Architecture | None = None
        self.saved = None
        self.hbs_cache = None
        self.hbs_cached_mode = False
        self.hbs_targets = None

    def forward(self, samples):
        from rfdetr.utilities.tensors import NestedTensor
        if self.hbs_cached_mode:
            assert self.training and self.hbs_cache is not None and self.hbs_targets is not None
            features, pos, cross = self.hbs_cache
            tensor, mask = features[0].decompose()
            smoothed = self.hbs_smoother(tensor, self.hbs_targets, mask)
            return [NestedTensor(smoothed, mask)], pos, cross
        if self.active_inputs is None or self.active_arch is None:
            raise RuntimeError("elastic fusion context was not installed")
        ir_input, depth_input = self.active_inputs
        arch = self.active_arch
        _set_windows(self.rgb, arch.windows_rgb)
        _set_windows(self.ir, arch.windows_ir)
        _set_windows(self.depth, arch.windows_depth)
        rgb, pos, cross = self.rgb(samples)
        ir, _, _ = self.ir(ir_input)
        depth, _, _ = self.depth(depth_input)
        if not (len(rgb) == len(ir) == len(depth) == 1):
            raise RuntimeError("RF-DETR 2XL NAS expects one projected feature level")
        rf, inf, df = [v.tensors for v in rgb], [v.tensors for v in ir], [v.tensors for v in depth]
        fused = self.opf1(rf, inf, df, depth_valid_mask=depth_input[:, 4:5])
        if arch.opf_depth == 2:
            refined = self.opf2(fused, inf, df, depth_valid_mask=depth_input[:, 4:5])
            gain = self.opf2_gain.sigmoid()
            fused = [a + gain * (b - a) for a, b in zip(fused, refined)]
        self.saved = (rf, inf, df)
        result = [NestedTensor(fused[0], rgb[0].mask)]
        if self.training and hasattr(self, 'hbs_smoother'):
            self.hbs_cache = (result, pos, cross)
        return result, pos, cross


class RFNASDetector(nn.Module):
    def __init__(self, cfg, modality=None, frozen_backbone=None):
        super().__init__()
        if frozen_backbone is not None:
            raise ValueError("frozen external backbone is unsupported")
        self.cfg = cfg
        self.modality = modality
        self.detector, self.pretrained_report = build_native(cfg, initialize_weights=not cfg.get("inference_only", False))
        self.input = ModalityInput(modality or "rgb")
        self.sampler = ArchitectureSampler(cfg, student=modality is None)
        self.detector.transformer.register_forward_pre_hook(self._slice_group_queries, with_kwargs=True)
        self.ir_alignment = None
        if modality is None:
            self.detector.backbone = ElasticFusionBackbone(self.detector.backbone, cfg)
            self.pretrained_report["teacher_initialization"] = self.detector.backbone.teacher_init
            self.pretrained_report["independent_backbones"] = True
        self.pretrained_report.update(modality=modality or "rgb_ir_depth", nas=True,
                                      native_decoder_layers=len(self.detector.transformer.decoder.layers))
        if modality is None and cfg.get('hbs', {}).get('enabled', False):
            from .hbs import install
            self.pretrained_report['hbs'] = install(self, cfg)

    @property
    def backbone(self):
        return self.detector.backbone

    def _activate_queries(self, count: int):
        self.detector.num_queries = count
        self.detector.transformer.num_queries = count

    def _slice_group_queries(self, module, args, kwargs):
        # LWDETR only slices embeddings in eval mode.  In training, preserve all
        # Group-DETR groups but take Q entries from every group (never the first
        # Q entries of the flattened 13-group table).
        if not self.training:
            return args, kwargs
        count = int(self.detector.num_queries)
        groups = int(self.detector.group_detr)
        mutable = list(args)
        for position in (3, 4):
            table = mutable[position]
            per_group = table.shape[0] // groups
            if count > per_group or table.shape[0] % groups:
                raise RuntimeError(f"invalid grouped query slice: total={table.shape[0]} groups={groups} q={count}")
            mutable[position] = table.reshape(groups, per_group, *table.shape[1:])[:, :count].reshape(
                groups * count, *table.shape[1:]
            )
        return tuple(mutable), kwargs

    def forward(self, rgb, ir, depth, targets=None, return_backbone_feats=False):
        arch = self.sampler.choose(self.training)
        self._activate_queries(arch.queries)
        if self.modality:
            _set_windows(self.detector.backbone, arch.windows_rgb)
            x = {"ir": ir, "depth": depth, "rgb": rgb}[self.modality]
            out = self.detector(self.input(x), targets)
            return _select_decoder_output(out, arch.decoder_layers)
        self.backbone.active_inputs = (ir, depth)
        self.backbone.active_arch = arch
        augmented = None
        try:
            out = self.detector(self.input(rgb), targets)
            features = self.backbone.saved
            if self.training and self.cfg.get('hbs', {}).get('enabled', False):
                assert targets is not None
                self.backbone.hbs_cached_mode = True
                self.backbone.hbs_targets = targets
                augmented = self.detector(self.input(rgb), targets)
        finally:
            self.backbone.hbs_cache = None
            self.backbone.hbs_cached_mode = False
            self.backbone.hbs_targets = None
            self.backbone.active_inputs = None
            self.backbone.active_arch = None
            self.backbone.saved = None
        out = _select_decoder_output(out, arch.decoder_layers)
        if augmented is not None:
            out['hbs_outputs'] = _select_decoder_output(augmented, arch.decoder_layers)
        return (out, *features) if return_backbone_feats else out


class NASCriterion(nn.Module):
    def __init__(self, **unused):
        super().__init__()
        from rfdetr.config import TrainConfig
        from rfdetr.models import build_criterion_from_config
        from aic.rf_native import model as native_model
        self.native, _ = build_criterion_from_config(native_model.MODEL_CONFIG, TrainConfig(dataset_dir="."))
        self.native.num_classes = 12

    def forward(self, outputs, targets):
        hbs_outputs = outputs.pop('hbs_outputs', None)
        depth = int(outputs.pop("nas_decoder_layers"))
        def fp32(value):
            if isinstance(value, torch.Tensor): return value.float() if value.is_floating_point() else value
            if isinstance(value, dict): return {k: fp32(v) for k, v in value.items()}
            if isinstance(value, list): return [fp32(v) for v in value]
            return value
        with torch.autocast(device_type=outputs["pred_logits"].device.type, enabled=False):
            raw = self.native(fp32(outputs), targets)
        result = {}
        decoder_scale = 5.0 / depth
        for key, value in raw.items():
            if key in self.native.weight_dict:
                result[key] = value * self.native.weight_dict[key] * (1.0 if key.endswith("_enc") else decoder_scale)
        if hbs_outputs is None:
            return result
        auxiliary = self.forward(hbs_outputs, targets)
        return {**{k: .5 * v for k, v in result.items()},
                **{'hbs_' + k: .5 * v for k, v in auxiliary.items()}}


def param_groups(model, lr, backbone_lr_mult, weight_decay, **unused):
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        if any(token in name for token in (".opf1.", ".opf2.", "opf2_gain", ".input.", ".hbs_")) or name.startswith("input.") or "class_embed" in name:
            rate = lr
        elif ".backbone." in name:
            rate = lr * backbone_lr_mult
        else:
            rate = lr * 0.1
        decay = 0.0 if parameter.ndim <= 1 or "norm" in name.lower() or name.endswith(".bias") else weight_decay
        groups.setdefault((rate, decay), []).append(parameter)
    return [dict(params=params, lr=rate, weight_decay=decay) for (rate, decay), params in groups.items()]

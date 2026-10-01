"""单模型 DDP 训练脚本（计划.md §3.2，用户已明确要求单模型 DDP 而非多实验并行）。

启动方式（远端 Linux 服务器，双卡）：
    NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 torchrun --nproc_per_node=2 aic/tools/train.py --config configs/dfine_tristream.yml
单卡调试：
    python aic/tools/train.py --config configs/dfine_tristream.yml

NCCL_P2P_DISABLE=1 是消费级 Ada 架构（4080 系列）已知问题的规避手段：
这类卡的 P2P（GPU 间直连）能力检测在部分驱动/主板组合下会误报为可用，
实际通信会挂死，训练卡在第一次 all_reduce 不报错也不进展。魔改4080更
不具备数据中心卡的 NVLink，PCIe-only 拓扑下这个问题概率更高，因此默认
禁用 P2P（退化为走 host memory 中转，慢但正确）。
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import random
import re
import time
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from aic.data.dataset import (
    MultiModalDetDataset,
    MultiScaleCollate,
    collate_fn,
    compute_class_balanced_weights,
    kfold_split,
    scan_samples,
)
from aic.data.transforms import SyncMultiModalTransform
from aic.models.criterion import GCDCriterion, GCLCriterion, MALCriterion, MALGCLCriterion
from aic.models.distill import DINOv2RGBFeatureDistillLoss, M2DDistillLoss, TriStreamSelfDistillLoss
try:
    from aic.models.distill import FDRCrossModelDistillLoss
except ImportError:
    # Older deployed workspaces may not include the optional FDR ablation.
    FDRCrossModelDistillLoss = None
from aic.models.matcher import GCDHungarianMatcher, SNL1HungarianMatcher
from aic.models.backbone import TriStreamHGNetv2
from aic.models.tristream_dfine import build_single_modality_dfine, build_tristream_dfine
from aic.utils.curves import CurveLogger
from aic.utils.experiment_state import identity, capture_rng, restore_rng, accumulation_window
from third_party_dfine import DFINECriterion, HungarianMatcher


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return (not is_dist()) or dist.get_rank() == 0


def setup_distributed(timeout_minutes=30) -> tuple[int, int, int]:
    """从 torchrun 注入的环境变量初始化进程组。单卡直接跑时不进入分布式模式。"""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size,
                            timeout=timedelta(minutes=timeout_minutes))
    return rank, world_size, local_rank


def build_param_groups(
    model,
    lr: float,
    backbone_lr_mult: float,
    weight_decay: float,
    freqfusion_lr_mult: float = 1.0,
):
    """四路参数分组（算法设计.md §3.2/§3.3）：
    1. 预训练骨干（stem_rgb + stages）非 norm 参数 —— lr*backbone_lr_mult, weight_decay
    2. 预训练骨干 norm/bias 参数                    —— lr*backbone_lr_mult, weight_decay=0
    3. 新模块（stem_ir/stem_depth/OPF/encoder/decoder）非 norm 参数 —— lr, weight_decay
    4. 新模块 norm/bias 参数                                        —— lr, weight_decay=0

    norm/bias 零 weight_decay 是 D-FINE 官方配置（optimizer.yml）的做法，对
    LayerNorm/BatchNorm/GroupNorm 的 scale/bias 加 L2 正则会拖慢收敛甚至损害
    精度，是 transformer 类架构的常见实践。

    stem_ir/stem_depth 虽然参数名以 "backbone." 开头，但它们是全新初始化、
    要从零学习模态特异统计特性的模块，不应该像预训练来的 stem_rgb/stages
    一样被压低学习率（否则新分支收敛速度被不必要地拖慢）——通过
    `TriStreamHGNetv2.pretrained_param_ids` 精确区分，而不是用参数名前缀。
    """
    norm_types = (torch.nn.LayerNorm, torch.nn.BatchNorm2d, torch.nn.GroupNorm, torch.nn.BatchNorm1d)
    norm_param_ids = set()
    for m in model.modules():
        if isinstance(m, norm_types) or type(m).__name__ in ('DINOv3ConvNextLayerNorm', 'DINOv3ViTLayerNorm'):
            norm_param_ids.update(id(p) for p in m.parameters())

    inner = model.module if hasattr(model, "module") else model
    pretrained_ids = inner.backbone.pretrained_param_ids

    groups = {
        "pretrained": [],
        "pretrained_norm": [],
        "new": [],
        "new_norm": [],
        "freqfusion": [],
        "freqfusion_norm": [],
    }
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_pretrained = id(p) in pretrained_ids
        is_norm_or_bias = id(p) in norm_param_ids or name.endswith(".bias")
        is_freqfusion = "freq_fusions" in name or "freqfusion_gates" in name
        if is_freqfusion:
            key = "freqfusion_norm" if is_norm_or_bias else "freqfusion"
        else:
            key = ("pretrained" if is_pretrained else "new") + ("_norm" if is_norm_or_bias else "")
        groups[key].append(p)

    param_groups = [
        {"params": groups["pretrained"], "lr": lr * backbone_lr_mult, "weight_decay": weight_decay},
        {"params": groups["pretrained_norm"], "lr": lr * backbone_lr_mult, "weight_decay": 0.0},
        {"params": groups["new"], "lr": lr, "weight_decay": weight_decay},
        {"params": groups["new_norm"], "lr": lr, "weight_decay": 0.0},
        {"params": groups["freqfusion"], "lr": lr * freqfusion_lr_mult, "weight_decay": weight_decay},
        {"params": groups["freqfusion_norm"], "lr": lr * freqfusion_lr_mult, "weight_decay": 0.0},
    ]
    return [g for g in param_groups if g["params"]]


class ModelEMA:
    """D-FINE 官方默认开启的权重指数滑动平均（optimizer.yml: use_ema=True,
    decay=0.9999, warmups=1000）。对小数据集、梯度噪声较大的场景收益更明显
    （算法设计.md §3.1）。验证和最终 checkpoint 都用 EMA 权重。
    """

    def __init__(self, model, decay: float = 0.9999, warmups: int = 1000):
        inner = model.module if hasattr(model, "module") else model
        self.decay = decay
        self.warmups = warmups
        self.updates = 0
        self.shadow = {k: v.detach().clone().float() for k, v in inner.state_dict().items()}

    def update(self, model):
        inner = model.module if hasattr(model, "module") else model
        self.updates += 1
        d = self.decay * (1 - torch.exp(torch.tensor(-self.updates / self.warmups))).item() if self.warmups else self.decay
        with torch.no_grad():
            for k, v in inner.state_dict().items():
                if v.dtype.is_floating_point:
                    self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1 - d)
                else:
                    self.shadow[k] = v.detach().clone()

    def state_dict(self):
        return self.shadow

    def apply_to(self, model):
        """把 EMA 权重临时应用到 model 上，返回原始 state_dict 以便恢复。"""
        inner = model.module if hasattr(model, "module") else model
        original = {k: v.detach().clone() for k, v in inner.state_dict().items()}
        ema_state = {k: v.to(dtype=original[k].dtype) for k, v in self.shadow.items()}
        inner.load_state_dict(ema_state)
        return original

    def restore(self, model, original):
        inner = model.module if hasattr(model, "module") else model
        inner.load_state_dict(original)


def move_targets_to_device(targets, device):
    out = []
    for t in targets:
        out.append({k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()})
    return out


def apply_student_modality_dropout(
    ir: torch.Tensor,
    depth: torch.Tensor,
    probability: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop IR or Depth per sample while preserving clean tensors for a teacher."""
    if probability <= 0.0:
        return ir, depth, ir.new_zeros(())
    batch_size = ir.shape[0]
    drop = torch.rand(batch_size, device=ir.device) < probability
    drop_ir = drop & (torch.rand(batch_size, device=ir.device) < 0.5)
    drop_depth = drop & ~drop_ir
    student_ir = ir.clone()
    student_depth = depth.clone()
    student_ir[drop_ir] = 0
    student_depth[drop_depth] = 0
    return student_ir, student_depth, drop.float().mean()


class RepeatFactorDistributedSampler(torch.utils.data.Sampler[int]):
    """Distributed repeat-factor sampler for image-level long-tail balancing."""

    def __init__(
        self,
        weights: list[float],
        *,
        num_replicas: int,
        rank: int,
        seed: int,
        shuffle: bool = True,
    ) -> None:
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        if not weights:
            raise ValueError("weights must be non-empty")
        self.weights = [max(1.0, float(w)) for w in weights]
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        self.base_indices = self._build_base_indices()
        self.num_samples = int(math.ceil(len(self.base_indices) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def _build_base_indices(self) -> list[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        indices: list[int] = []
        for idx, weight in enumerate(self.weights):
            whole = int(math.floor(weight))
            indices.extend([idx] * whole)
            frac = weight - whole
            if frac > 0 and torch.rand((), generator=generator).item() < frac:
                indices.append(idx)
        return indices or list(range(len(self.weights)))

    def __iter__(self):
        indices = list(self.base_indices)
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            order = torch.randperm(len(indices), generator=generator).tolist()
            indices = [indices[i] for i in order]
        if len(indices) < self.total_size:
            repeats = math.ceil((self.total_size - len(indices)) / len(indices))
            indices += (indices * repeats)[: self.total_size - len(indices)]
        else:
            indices = indices[: self.total_size]
        return iter(indices[self.rank : self.total_size : self.num_replicas])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @property
    def repeat_size(self) -> int:
        return len(self.base_indices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--nas_search_manifest", default=None)
    parser.add_argument("--nas_search_output", default=None)
    parser.add_argument(
        "--warm_start_model",
        default=None,
        help="Load model/EMA weights only, then create a fresh optimizer and scheduler.",
    )
    parser.add_argument(
        "--mono_modality",
        default=None,
        choices=["rgb", "ir", "depth"],
        help="计划.md §2.8 M²D-LIF linear probing 诊断专用：只用单一模态训练，"
        "backbone 走 SingleModalityDFINE（无 OPF），不设置时走正常三流训练",
    )
    parser.add_argument(
        "--frozen_backbone_ckpt",
        default=None,
        help="--mono_modality 探针模式：从该 checkpoint 加载 backbone 权重并冻结，"
        "只训练 encoder/decoder。不设置时 backbone 从预训练权重正常初始化并参与训练"
        "（单模态基线模式）。仅在设置了 --mono_modality 时生效",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    context_run = 'context_views' in cfg.get('train', {})
    curriculum_run = cfg.get('train', {}).get('data_curriculum', {}).get('enabled', False)
    if cfg.get('fusion', {}).get('type') == 'freqband' or context_run or curriculum_run:
        from aic.utils.frequency_experiment import verify_execution
        verify_execution(cfg)
    rank, world_size, local_rank = setup_distributed(cfg['train'].get('ddp_timeout_minutes', 30))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    train_cfg = cfg["train"]
    if curriculum_run:
        if world_size != 2 or train_cfg['batch_size'] != 2 or not train_cfg.get('split_manifest'):
            raise ValueError('data curriculum requires two ranks, paired batch2 and frozen split')
        if context_run or train_cfg.get('class_balance', {}).get('enabled') or args.warm_start_model:
            raise ValueError('data curriculum forbids crop replacement, resampling or warm-start')
    if context_run:
        args.warm_start_model = args.warm_start_model or train_cfg['warm_start_checkpoint']
        if cfg.get('load_pretrained_detector', True) or cfg['backbone'].get('pretrained', True):
            raise ValueError('context pair must load only its frozen warm-start checkpoint')
        if not train_cfg.get('split_manifest') or train_cfg.get('class_balance', {}).get('enabled'):
            raise ValueError('context pair requires frozen split and one visit per source')
        if world_size != 2 or not cfg['backbone'].get('share_stages'):
            raise ValueError('context pair requires two ranks and shared backbone stages')
    self_distill_cfg = cfg.get("self_distill") if not args.mono_modality else None
    seed = int(train_cfg.get("seed", 42))
    process_seed = seed + rank
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)
    save_dir = train_cfg["save_dir"]
    if is_main_process():
        os.makedirs(save_dir, exist_ok=True)

    curve_logger = CurveLogger(save_dir) if is_main_process() else None

    samples = scan_samples(train_cfg["data_root"], require_labels=True)
    explicit_split = bool(train_cfg.get('split_manifest'))
    frequency_run = cfg.get('fusion', {}).get('type') == 'freqband'
    validation_provenance = cfg.get('validation_provenance', {'validation_independent': False, 'teacher_overlap': 329})
    if explicit_split:
        from aic.data.frozen_split import load_frozen_split
        train_samples, val_samples, split_info = load_frozen_split(
            samples, train_cfg['split_manifest'], train_cfg['data_root'], train_cfg['split_sha256'])
        if train_cfg.get('full_data'):
            raise ValueError('frozen split cannot be combined with full_data')
    elif train_cfg.get("full_data", False):
        train_samples, val_samples = samples, []
    else:
        train_samples, val_samples = kfold_split(samples, n_folds=train_cfg["n_folds"], fold=train_cfg["fold"])
    has_val = len(val_samples) > 0
    if is_main_process():
        print(f"[data] train={len(train_samples)} val={len(val_samples)} total={len(samples)}")
    if train_cfg.get('preflight_overfit_samples'):
        if not cfg.get('experiment', {}).get('preflight'):
            raise ValueError('tiny subset is restricted to preflight')
        train_samples = train_samples[:int(train_cfg['preflight_overfit_samples'])]

    train_tf = SyncMultiModalTransform(
        size=cfg["image_size"],
        train=True,
        # Self-distillation needs a clean copy for the static teacher. Its
        # student-only modality dropout is applied on GPU in the train loop.
        modality_dropout_p=0.0 if self_distill_cfg else train_cfg["modality_dropout_p"],
        scale_range=tuple(train_cfg.get("scale_range", [0.5, 1.5])),
        final_scale_range=tuple(train_cfg.get("final_scale_range", [0.9, 1.1])),
        stop_decay_epoch_ratio=float(train_cfg.get("stop_decay_epoch_ratio", 0.8)),
        total_epochs=train_cfg["epochs"],
        no_aug_epoch=(
            int(train_cfg["no_aug_epoch"])
            if train_cfg.get("no_aug_epoch") is not None
            else None
        ),
    )
    val_tf = SyncMultiModalTransform(size=cfg["image_size"], train=False)
    disp8_mode = train_cfg.get("disp8_mode", "use")
    dataset_distill_cfg = cfg.get("distill", {}) if not args.mono_modality else {}
    confidence_args = {
        "ir_confidence_cache": dataset_distill_cfg.get("ir_confidence_cache"),
        "depth_confidence_cache": dataset_distill_cfg.get("depth_confidence_cache"),
    }
    train_ds = MultiModalDetDataset(
        train_samples,
        train_tf,
        has_labels=True,
        disp8_mode=disp8_mode,
        mosaic_prob=float(train_cfg.get("mosaic_prob", 0.0)),
        mosaic_start_epoch=int(train_cfg.get("mosaic_start_epoch", 4)),
        mosaic_stop_epoch=(
            int(train_cfg["mosaic_stop_epoch"])
            if train_cfg.get("mosaic_stop_epoch") is not None
            else None
        ),
        **confidence_args,
        context_views=train_cfg.get('context_views'),
    )
    val_ds = MultiModalDetDataset(val_samples, val_tf, has_labels=True, disp8_mode=disp8_mode)

    # 类别不均衡重采样（算法设计.md §3.4，计划.md §2.4 tier 1）：ClassBalancedDataset
    # 风格的按类别频率过采样。DDP 下使用同步的 repeat-factor 索引表，再按 rank 切片。
    class_balance_cfg = train_cfg.get("class_balance", {})
    use_class_balance = class_balance_cfg.get("enabled", False)

    if use_class_balance:
        oversample_thr = class_balance_cfg.get("oversample_thr", 0.2)
        weights = compute_class_balanced_weights(
            train_samples,
            oversample_thr=oversample_thr,
            mode=class_balance_cfg.get("mode", "image"),
        )
        if world_size > 1:
            train_sampler = RepeatFactorDistributedSampler(
                weights,
                num_replicas=world_size,
                rank=rank,
                seed=seed,
                shuffle=True,
            )
        else:
            train_sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        if is_main_process():
            detail = (
                f"repeat_size={train_sampler.repeat_size} per_rank={len(train_sampler)}"
                if world_size > 1
                else f"samples={len(weights)}"
            )
            print(f"[data] class-balanced sampling enabled, oversample_thr={oversample_thr} {detail}")
    elif world_size > 1:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True,
                                           seed=seed if frequency_run or context_run or curriculum_run else 0)
    else:
        train_sampler = None

    multi_scale_cfg = train_cfg.get("multi_scale", {})
    if multi_scale_cfg.get("enabled", False):
        train_collate = MultiScaleCollate(
            base_size=cfg["image_size"], sizes=tuple(multi_scale_cfg.get("sizes", [512, 576, 640, 704, 768]))
        )
    else:
        train_collate = collate_fn

    train_generator = torch.Generator()
    train_generator.manual_seed(process_seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=train_cfg["num_workers"],
        collate_fn=train_collate,
        drop_last=not explicit_split,
        pin_memory=True,
        generator=train_generator,
    )
    if curriculum_run:
        from aic.data.data_curriculum import FullFramePairDataset, FullFramePairBatchSampler
        train_ds = FullFramePairDataset(train_samples, cfg)
        train_sampler = FullFramePairBatchSampler(train_ds, rank, world_size)
        train_collate = collate_fn
        train_loader = DataLoader(train_ds, batch_sampler=train_sampler,
                                  num_workers=train_cfg['num_workers'], collate_fn=train_collate,
                                  pin_memory=True, generator=train_generator)
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(seed),
    )

    if args.mono_modality:
        frozen_bb = None
        if args.frozen_backbone_ckpt:
            bb_cfg = cfg["backbone"]
            frozen_bb = TriStreamHGNetv2(
                name=bb_cfg["name"],
                use_lab=bb_cfg.get("use_lab", False),
                return_idx=tuple(bb_cfg.get("return_idx", [1, 2, 3])),
                pretrained=False,
                share_stages=bb_cfg.get("share_stages", True),
                depth_in_chs=bb_cfg.get("depth_in_chs", 2),
            )
            state = torch.load(args.frozen_backbone_ckpt, map_location="cpu")["model"]
            bb_state = {k[len("backbone.") :]: v for k, v in state.items() if k.startswith("backbone.")}
            if "stages.0.blocks.0.layers.0.conv.weight" in bb_state and frozen_bb.share_stages:
                # 兼容旧 checkpoint（backbone.py 重构前保存的），当时共享 stages
                # 只用一个扁平的 `stages` 属性名注册，新代码里 share_stages=True
                # 仍是同一份权重，只是分别注册在 stages_rgb/stages_ir/stages_depth
                # 三个属性名下（state_dict 按注册名展开，即使三者指向同一对象）。
                for k in [k for k in bb_state if k.startswith("stages.")]:
                    suffix = k[len("stages.") :]
                    v = bb_state.pop(k)
                    bb_state[f"stages_rgb.{suffix}"] = v
                    bb_state[f"stages_ir.{suffix}"] = v
                    bb_state[f"stages_depth.{suffix}"] = v
            frozen_bb.load_state_dict(bb_state)
            if is_main_process():
                print(f"[probe] loaded frozen backbone from {args.frozen_backbone_ckpt}, {len(bb_state)} tensors")
        model = build_single_modality_dfine(cfg, args.mono_modality, frozen_bb).to(device)
    else:
        if curriculum_run and args.resume:
            import copy
            build_cfg = copy.deepcopy(cfg)
            build_cfg['load_pretrained_detector'] = False
            build_cfg['backbone']['pretrained'] = False
        else:
            build_cfg = cfg
        model = build_tristream_dfine(build_cfg).to(device)
        if curriculum_run and not args.resume:
            from aic.models.pretrained import audit_frequency_initialization
            audit_frequency_initialization(model)

    if explicit_split and is_main_process():
        from aic.sugon.common import atomic_json
        atomic_json(os.path.join(save_dir, 'resume_initialization.json' if args.resume else 'initialization.json'), model.pretrained_report)

    warm_checkpoint = None
    if args.warm_start_model:
        warm_checkpoint = torch.load(args.warm_start_model, map_location="cpu", weights_only=False)
        warm_state = warm_checkpoint
        if context_run and not warm_checkpoint.get('ema'):
            raise ValueError('context warm start requires historical EMA')
        if isinstance(warm_state, dict) and warm_state.get("ema") is not None:
            warm_state = warm_state["ema"]
        elif isinstance(warm_state, dict) and "model" in warm_state:
            warm_state = warm_state["model"]
        warm_result = model.load_state_dict(warm_state, strict=context_run)
        if context_run:
            from aic.models.dinov3_rgb import tensor_hash
            model.pretrained_report.update(source=args.warm_start_model, weight_source='ema',
                                          strict=True, warm_model_sha256=tensor_hash(warm_state))
        del warm_state
        if is_main_process():
            print(
                f"[warm-start] loaded {args.warm_start_model}; "
                f"missing={len(warm_result.missing_keys)} unexpected={len(warm_result.unexpected_keys)}"
            )

    if args.nas_search_manifest:
        if not args.warm_start_model or not args.nas_search_output:
            raise ValueError("NAS search requires --warm_start_model and --nas_search_output")
        import json
        manifest = json.loads(open(args.nas_search_manifest, encoding="utf-8").read())
        if not isinstance(manifest, list) or not manifest:
            raise ValueError("NAS search manifest must be a non-empty list")
        inner = model.module if hasattr(model, "module") else model
        completed = []
        if os.path.exists(args.nas_search_output):
            completed = json.loads(open(args.nas_search_output, encoding="utf-8").read()).get("results", [])
        done_ids = {row["id"] for row in completed}
        for item in manifest:
            if item["id"] in done_ids:
                continue
            resolution = tuple(int(v) for v in item["resolution"])
            arch = {k: int(v) for k, v in item["arch"].items()}
            cfg["image_size"] = list(resolution)
            cfg.setdefault("nas", {})["eval_arch"] = arch
            inner.sampler.cfg["eval_arch"] = arch
            val_ds.transform.size_hw = resolution
            val_ds.transform.size = resolution
            cfg["decoder"]["num_queries"] = arch["queries"]
            result = _run_eval(inner, val_loader, device, 1, cfg)
            row = {
                **item,
                "checkpoint": str(args.warm_start_model),
                "map_50_95": float(result.map_50_95),
                "map_50": float(result.map_50),
                "map_75": float(result.map_75),
                "per_class_ap50_95": {str(k): float(v) for k, v in result.per_class_ap50_95.items()},
            }
            completed.append(row)
            from aic.sugon.common import atomic_json
            atomic_json(args.nas_search_output, {"results": completed})
            print(json.dumps(row, sort_keys=True), flush=True)
        return

    # M2D 蒸馏（计划.md §2.8 后续，memory `fusion-degradation-confirmed`）：只在
    # config 里配置了 distill 段时启用，不配置就是原来的行为，与旧 config 完全
    # 兼容。只对三流联合训练生效——mono_modality 探针本身就是诊断/teacher 训练
    # 用的，不需要也不应该被蒸馏。
    distill_cfg = cfg.get("distill") if not args.mono_modality else None
    distill_loss_fn = None
    if distill_cfg:
        distill_loss_fn = M2DDistillLoss(
            bb_cfg=cfg["backbone"],
            ir_teacher_ckpt=distill_cfg["ir_teacher_ckpt"],
            depth_teacher_ckpt=distill_cfg["depth_teacher_ckpt"],
            weight=distill_cfg.get("weight", 1.0),
            ir_weight=distill_cfg.get("ir_weight", 1.0),
            depth_weight=distill_cfg.get("depth_weight", 1.0),
            use_teacher_ema=distill_cfg.get("use_teacher_ema", False),
            foreground_weight=distill_cfg.get("foreground_weight", 1.0),
            level_weights=distill_cfg.get("level_weights"),
            use_depth_valid_mask=distill_cfg.get("use_depth_valid_mask", False),
            cross_weight=distill_cfg.get("cross_weight", 1.0),
            use_adapters=distill_cfg.get("use_adapters", True),
            adapter_reduction=distill_cfg.get("adapter_reduction", 16),
            dynamic_reliability=distill_cfg.get("dynamic_reliability", False),
            reliability_floor=distill_cfg.get("reliability_floor", 0.25),
            reliability_power=distill_cfg.get("reliability_power", 1.0),
            method=distill_cfg.get("method", "pkd_lcm"),
            global_sample_normalization=curriculum_run,
        ).to(device)
        if is_main_process():
            print(
                f"[distill] M2D 蒸馏已启用，weight={distill_cfg.get('weight', 1.0)} "
                f"ir_weight={distill_loss_fn.ir_weight} "
                f"depth_weight={distill_loss_fn.depth_weight} "
                f"teacher={distill_loss_fn.ir_teacher_source}/{distill_loss_fn.depth_teacher_source} "
                f"foreground_weight={distill_loss_fn.foreground_weight} "
                f"level_weights={distill_loss_fn.level_weights} "
                f"depth_valid_mask={distill_loss_fn.use_depth_valid_mask} "
                f"method={distill_loss_fn.method} "
                f"cross_weight={distill_loss_fn.cross_weight} "
                f"use_adapters={distill_loss_fn.use_adapters} "
                f"dynamic_reliability={distill_loss_fn.dynamic_reliability} "
                f"decay_start_epoch={distill_cfg.get('decay_start_epoch', 'none')} "
                f"final_scale={distill_cfg.get('final_scale', 1.0)}"
            )

    fdr_cfg = cfg.get("fdr_distill") if not args.mono_modality else None
    fdr_loss_fn = None
    if fdr_cfg:
        fdr_loss_fn = FDRCrossModelDistillLoss(
            cfg=cfg,
            ir_teacher_ckpt=fdr_cfg["ir_teacher_ckpt"],
            depth_teacher_ckpt=fdr_cfg["depth_teacher_ckpt"],
            weight=fdr_cfg.get("weight", 0.25),
            temperature=fdr_cfg.get("temperature", 5.0),
            ir_weight=fdr_cfg.get("ir_weight", 1.0),
            depth_weight=fdr_cfg.get("depth_weight", 1.0),
        ).to(device)
        if is_main_process():
            print(f"[distill] FDR cross-model enabled: weight={fdr_loss_fn.weight}, T={fdr_loss_fn.temperature}")

    dino_cfg = cfg.get("dino_distill") if not args.mono_modality else None
    dino_loss_fn = None
    if dino_cfg:
        dino_loss_fn = DINOv2RGBFeatureDistillLoss(
            checkpoint_dir=dino_cfg["checkpoint_dir"],
            student_channels=tuple(model.backbone.out_channels),
            weight=dino_cfg.get("weight", 0.5),
            foreground_weight=dino_cfg.get("foreground_weight", 2.0),
            level_weights=tuple(dino_cfg.get("level_weights", [0.0, 1.0, 1.0])),
            patch_size=dino_cfg.get("patch_size", 14),
        ).to(device)
        if is_main_process():
            print(
                f"[distill] DINOv2 RGB teacher enabled, weight={dino_loss_fn.weight} "
                f"foreground_weight={dino_loss_fn.foreground_weight} "
                f"level_weights={dino_loss_fn.level_weights} "
                f"warmup_epochs={dino_cfg.get('warmup_epochs', 0)}"
            )

    self_distill_loss_fn = None
    if self_distill_cfg:
        if distill_loss_fn is not None or dino_loss_fn is not None:
            raise ValueError("self_distill must be isolated from M2D and DINO distillation")
        self_distill_loss_fn = TriStreamSelfDistillLoss(
            model_cfg=cfg,
            teacher_ckpt=self_distill_cfg["teacher_ckpt"],
            use_teacher_ema=self_distill_cfg.get("use_teacher_ema", True),
            cls_weight=self_distill_cfg.get("cls_weight", 2.0),
            bbox_weight=self_distill_cfg.get("bbox_weight", 5.0),
            giou_weight=self_distill_cfg.get("giou_weight", 2.0),
            temperature=self_distill_cfg.get("temperature", 2.0),
            teacher_score_thresh=self_distill_cfg.get("teacher_score_thresh", 0.25),
            teacher_gt_iou_thresh=self_distill_cfg.get("teacher_gt_iou_thresh", 0.5),
        ).to(device)
        if is_main_process():
            print(
                f"[self-distill] static tri-stream teacher enabled: "
                f"{self_distill_loss_fn.teacher_ckpt} ({self_distill_loss_fn.teacher_source}); "
                f"weights=cls:{self_distill_loss_fn.cls_weight},"
                f"bbox:{self_distill_loss_fn.bbox_weight},giou:{self_distill_loss_fn.giou_weight}; "
                f"temperature={self_distill_loss_fn.temperature} "
                f"score_thresh={self_distill_loss_fn.teacher_score_thresh} "
                f"gt_iou_thresh={self_distill_loss_fn.teacher_gt_iou_thresh}"
            )

    if world_size > 1 and torch.cuda.is_available():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    matcher_cfg = dict(cfg["matcher"])
    cost_type = matcher_cfg.pop("cost_type", "base")
    if cost_type == "base":
        matcher = HungarianMatcher(**matcher_cfg)
    elif cost_type == "gcd":
        matcher = GCDHungarianMatcher(**matcher_cfg, cost_type=cost_type)
    else:
        matcher = SNL1HungarianMatcher(**matcher_cfg, cost_type=cost_type, image_size=cfg["image_size"])
    criterion_type = str(cfg.get("criterion_type", "dfine")).lower()
    if criterion_type == "mal":
        criterion_cls = MALCriterion
    elif criterion_type in ("mal_gcl", "mal+gcl"):
        criterion_cls = MALGCLCriterion
    elif criterion_type == "gcl":
        criterion_cls = GCLCriterion
    elif criterion_type == "gcd" or (criterion_type == "dfine" and cost_type == "gcd"):
        criterion_cls = GCDCriterion
    elif criterion_type in ("dfine", "rf_native", "rf_nas"):
        criterion_cls = DFINECriterion
    else:
        raise ValueError(f"unsupported criterion_type: {criterion_type!r}")
    criterion_cfg = dict(cfg["criterion"])
    criterion = criterion_cls(
        matcher=matcher,
        num_classes=cfg["num_classes"],
        **criterion_cfg,
    ).to(device)
    if cfg.get("encoder_oto_supervision", {}).get("enabled", False):
        criterion.encoder_oto_topk = int(cfg["encoder_oto_supervision"].get("topk", 4))
        criterion.encoder_oto_weight = float(cfg["encoder_oto_supervision"].get("weight", 0.5))

    if world_size > 1:
        # find_unused_parameters=True: 模态 dropout 会让某些 iteration 里 IR 或 Depth 的
        # stem 分支完全没有梯度贡献（输入被置零后仍前向传播，但极端情况下某分支参数可能
        # 未参与 loss），DDP 默认要求所有参数每步都参与反向，否则报错。等确认训练稳定、
        # 各分支每步都必然被用到后，可切换为 static_graph=True 换取更快的通信效率。
        # broadcast_buffers=False：默认每次 forward 前都把 rank0 的 buffer 广播给
        # 其它 rank。这里没必要——backbone 的 BN 已被换成 FrozenBatchNorm2d（buffer
        # 恒定不变），其余可训练 BN 都走了 SyncBatchNorm（本身就跨进程同步统计量）。
        # 关掉可以省下每步一次全 buffer 广播，同时消除一类 collective 序列错位的风险。
        model = DDP(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            find_unused_parameters=True,
            broadcast_buffers=False,
        )

    param_groups = build_param_groups(
        model,
        train_cfg["lr"],
        train_cfg["backbone_lr_mult"],
        train_cfg["weight_decay"],
        freqfusion_lr_mult=float(train_cfg.get("freqfusion_lr_mult", 1.0)),
    )
    distill_params = []
    if distill_loss_fn is not None:
        distill_params = [p for p in distill_loss_fn.parameters() if p.requires_grad]
        if distill_params:
            param_groups.append(
                {"params": distill_params, "lr": train_cfg["lr"], "weight_decay": train_cfg["weight_decay"]}
            )
    dino_params = []
    if dino_loss_fn is not None:
        dino_params = [p for p in dino_loss_fn.parameters() if p.requires_grad]
        if dino_params:
            param_groups.append(
                {"params": dino_params, "lr": train_cfg["lr"], "weight_decay": train_cfg["weight_decay"]}
            )
    optimizer = torch.optim.AdamW(param_groups)
    auxiliary_params = distill_params + dino_params
    if explicit_split and world_size > 1:
        # These adapters live outside DDP; synchronize their initialization as
        # well as their gradients so both ranks optimize the same parameters.
        for parameter in auxiliary_params:
            dist.broadcast(parameter.data, src=0)
    if context_run and distill_loss_fn is not None:
        if 'm2d_adapters' not in warm_checkpoint:
            raise ValueError('historical checkpoint must include M2D adapters')
        distill_loss_fn.ir_adapters.load_state_dict(warm_checkpoint['m2d_adapters']['ir'], strict=True)
        distill_loss_fn.depth_adapters.load_state_dict(warm_checkpoint['m2d_adapters']['depth'], strict=True)
    del warm_checkpoint
    if explicit_split and distill_loss_fn is not None:
        from aic.models.dinov3_rgb import tensor_hash
        adapter_state = {**{'ir.'+k:v for k,v in distill_loss_fn.ir_adapters.state_dict().items()},
                         **{'depth.'+k:v for k,v in distill_loss_fn.depth_adapters.state_dict().items()}}
        model_inner = model.module if world_size > 1 else model
        model_inner.pretrained_report['m2d_adapter_init_sha256'] = tensor_hash(adapter_state)
        if is_main_process():
            atomic_json(os.path.join(save_dir, 'resume_initialization.json' if args.resume else 'initialization.json'), model_inner.pretrained_report)
    clip_params = list(model.parameters()) + auxiliary_params

    grad_accum = train_cfg["grad_accum_steps"]

    # Read the resume metadata once.  Besides avoiding a second 1.6GB checkpoint
    # load, this lets the scheduler keep the original optimizer-step density when
    # resuming with a different DDP world size (for example, 2 -> 4 GPUs).
    resume_probe = None
    resume_scheduler_step = None
    resume_epoch = None
    if args.resume and os.path.exists(args.resume):
        resume_probe = torch.load(args.resume, map_location="cpu", weights_only=False)
        if explicit_split and resume_probe.get('experiment_identity') != identity(cfg):
            raise ValueError('resume configuration/split identity differs from checkpoint')
        if (frequency_run or context_run or curriculum_run) and resume_probe.get('world_size') != world_size:
            raise ValueError('experiment resume must retain the original DDP world size')
        resume_scheduler_step = int(resume_probe.get("scheduler", {}).get("last_epoch", -1))
        resume_epoch = int(resume_probe.get("epoch", -1))

    # 关键：scheduler.step() 只在梯度累积边界被调用（见训练循环），所以调度器
    # 的 step 计数单位是 **optimizer step**，不是 dataloader step。这里必须除以
    # grad_accum，否则 warmup/cosine 的时间轴会被放大 grad_accum 倍——首轮实验就
    # 栽在这里：warmup 名义 2 epoch，实际要 2×16=32 epoch 才走完，cosine 衰减
    # 全程没启动，实测 lr 比预期低了一个数量级。
    # steps_per_epoch 是 optimizer step 数（调度器单位）；dataloader_steps 是
    # 实际迭代次数，仅用于日志进度显示。两者相差 grad_accum 倍。
    dataloader_steps = len(train_loader)
    dino_warmup_steps = (
        int(float(dino_cfg.get("warmup_epochs", 0)) * dataloader_steps)
        if dino_cfg is not None
        else 0
    )
    distill_decay_start = int(distill_cfg.get("decay_start_epoch", train_cfg["epochs"])) if distill_cfg else train_cfg["epochs"]
    distill_warmup_steps = (
        int(float(distill_cfg.get("warmup_epochs", 0)) * dataloader_steps)
        if distill_cfg else 0
    )
    distill_final_scale = float(distill_cfg.get("final_scale", 1.0)) if distill_cfg else 1.0
    if not 0.0 <= distill_final_scale <= 1.0:
        raise ValueError("distill.final_scale must be in [0, 1]")
    if not 0 <= distill_decay_start < train_cfg["epochs"] and distill_final_scale != 1.0:
        raise ValueError("distill.decay_start_epoch must be within the training schedule")

    self_distill_start = int(self_distill_cfg.get("start_epoch", 4)) if self_distill_cfg else 0
    self_distill_ramp = float(self_distill_cfg.get("ramp_epochs", 4.0)) if self_distill_cfg else 0.0
    self_distill_decay = float(self_distill_cfg.get("decay_start_epoch", 36.0)) if self_distill_cfg else 0.0
    self_distill_stop = float(self_distill_cfg.get("stop_epoch", 40.0)) if self_distill_cfg else 0.0
    self_distill_peak = float(self_distill_cfg.get("peak_scale", 1.0)) if self_distill_cfg else 0.0
    student_dropout_p = float(self_distill_cfg.get("student_modality_dropout_p", 0.1)) if self_distill_cfg else 0.0
    student_dropout_stop = float(self_distill_cfg.get("student_dropout_stop_epoch", self_distill_stop)) if self_distill_cfg else 0.0
    if self_distill_cfg:
        if not 0 <= self_distill_start < train_cfg["epochs"]:
            raise ValueError("self_distill.start_epoch must be within the training schedule")
        if not self_distill_start + self_distill_ramp <= self_distill_decay < self_distill_stop <= train_cfg["epochs"]:
            raise ValueError("self_distill schedule must satisfy start+ramp <= decay < stop <= epochs")
        if self_distill_peak < 0 or not 0.0 <= student_dropout_p <= 1.0:
            raise ValueError("invalid self_distill peak scale or student modality dropout")

    def self_distill_scale(epoch_value: int, step_value: int) -> float:
        if self_distill_loss_fn is None:
            return 0.0
        position = epoch_value + step_value / max(dataloader_steps, 1)
        if position < self_distill_start:
            return 0.0
        if self_distill_ramp > 0 and position < self_distill_start + self_distill_ramp:
            return self_distill_peak * (position - self_distill_start) / self_distill_ramp
        if position < self_distill_decay:
            return self_distill_peak
        if position < self_distill_stop:
            progress = (position - self_distill_decay) / max(self_distill_stop - self_distill_decay, 1e-8)
            return self_distill_peak * 0.5 * (1.0 + math.cos(math.pi * progress))
        return 0.0
    effective_steps = min(dataloader_steps, int(train_cfg.get('max_steps_per_epoch', 0)) or dataloader_steps)
    steps_per_epoch = max(math.ceil(effective_steps / grad_accum) if explicit_split else dataloader_steps // grad_accum, 1)
    schedule_steps_per_epoch = steps_per_epoch
    if resume_scheduler_step is not None and resume_scheduler_step >= 0 and resume_epoch is not None and resume_epoch >= 0:
        old_steps_per_epoch = round(resume_scheduler_step / max(resume_epoch + 1, 1))
        if old_steps_per_epoch > 0:
            schedule_steps_per_epoch = max(schedule_steps_per_epoch, old_steps_per_epoch)
            if is_main_process() and schedule_steps_per_epoch != steps_per_epoch:
                print(
                    f"[scheduler] preserving resume step density: current={steps_per_epoch}, "
                    f"checkpoint={old_steps_per_epoch}, using={schedule_steps_per_epoch}"
                )
    total_steps = schedule_steps_per_epoch * train_cfg["epochs"]
    warmup_steps = schedule_steps_per_epoch * train_cfg["warmup_epochs"]

    lr_schedule = str(train_cfg.get("lr_schedule", "cosine")).lower()
    if lr_schedule not in ("cosine", "flat_cosine"):
        raise ValueError(f"unsupported train.lr_schedule: {lr_schedule!r}")
    flat_ratio = float(train_cfg.get("flat_ratio", 0.33))
    if not 0.0 <= flat_ratio < 1.0:
        raise ValueError("train.flat_ratio must be in [0, 1)")
    flat_start_step = min(
        max(int(total_steps * flat_ratio), warmup_steps),
        max(total_steps - 1, warmup_steps),
    )

    def lr_lambda(step):
        if step < warmup_steps:
            factor = (step + 1) / max(warmup_steps, 1)
        else:
            if lr_schedule == "flat_cosine" and step < flat_start_step:
                progress = 0.0
            else:
                decay_start = flat_start_step if lr_schedule == "flat_cosine" else warmup_steps
                progress = (step - decay_start) / max(total_steps - decay_start, 1)
            progress = min(max(progress, 0.0), 1.0)
            factor = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265)).item())
        stage2_start = int(train_cfg.get("stage2_start_epoch", train_cfg["epochs"]))
        stage2_lr_mult = float(train_cfg.get("stage2_lr_mult", 1.0))
        if step // max(schedule_steps_per_epoch, 1) >= stage2_start:
            factor *= stage2_lr_mult
        return factor

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    amp_dtype = torch.bfloat16 if train_cfg["amp_dtype"] == "bfloat16" else torch.float16
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    ema_cfg = train_cfg.get("ema", {})
    ema = (
        ModelEMA(model, decay=ema_cfg.get("decay", 0.9999), warmups=ema_cfg.get("warmups", 1000))
        if ema_cfg.get("enabled", False)
        else None
    )

    start_epoch = 0
    optimizer_updates = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = resume_probe if resume_probe is not None else torch.load(args.resume, map_location=device)
        (model.module if world_size > 1 else model).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        if dino_loss_fn is not None:
            if "dino_adapters" in ckpt:
                dino_loss_fn.adapters.load_state_dict(ckpt["dino_adapters"])
            elif is_main_process():
                print("[resume] warning: checkpoint has no DINO adapter state; adapters remain reinitialized")
        if distill_loss_fn is not None:
            if "m2d_adapters" in ckpt:
                distill_loss_fn.ir_adapters.load_state_dict(ckpt["m2d_adapters"]["ir"])
                distill_loss_fn.depth_adapters.load_state_dict(ckpt["m2d_adapters"]["depth"])
            elif is_main_process():
                if explicit_split:
                    raise ValueError('frozen-split resume requires M2D adapter state')
                print("[resume] warning: checkpoint has no M2D adapter state; adapters remain reinitialized")
        start_epoch = ckpt["epoch"] + 1
        optimizer_updates = ckpt.get('optimizer_updates', start_epoch * steps_per_epoch)
        if ema is not None and "ema" in ckpt:
            ema.shadow = {k: v.to(device) for k, v in ckpt["ema"].items()}
            ema.updates = ckpt.get("ema_updates", 0)
        if explicit_split:
            restore_rng(ckpt['rng_states'][rank], train_generator)
            if curve_logger is not None:
                # Discard records written after the last atomically saved epoch.
                curve_logger.truncate_after(ckpt['epoch'], dataloader_steps)
                curve_logger.save_json()
        if is_main_process():
            print(f"[resume] from {args.resume}, starting epoch {start_epoch}")
        resume_probe = None

    # checkpoint 轮转用：断点恢复时从 curves.json 取回历史最佳，避免把之前的
    # 最优权重当成旧文件删掉。
    best_map, best_epoch = -1.0, -1
    if is_main_process() and curve_logger is not None:
        hist = os.path.join(save_dir, "curves.json")
        if os.path.exists(hist):
            try:
                import json

                with open(hist, "r", encoding="utf-8") as f:
                    rec = json.load(f)
                eps = rec.get("eval_epochs", [])
                vals = rec.get("eval_metrics", {}).get("mAP@50-95", [])
                for ep, v in zip(eps, vals):
                    if v is not None and v > best_map:
                        best_map, best_epoch = float(v), int(ep)
                print(f"[ckpt] 历史最佳 epoch{best_epoch} mAP@50-95={best_map:.4f}")
            except Exception as exc:
                print(f"[ckpt] 读取 curves.json 失败，轮转按无历史处理: {exc}")

    end_epoch = min(train_cfg['epochs'], int(os.environ.get('AIC_STOP_AFTER_EPOCH', train_cfg['epochs'])))
    for epoch in range(start_epoch, end_epoch):
        model.train()
        memory_probe = bool(train_cfg.get("memory_report_path"))
        if frequency_run or context_run or curriculum_run or memory_probe:
            torch.cuda.reset_peak_memory_stats(device)
            if train_cfg.get('preflight_trace'):
                from aic.utils.experiment_state import state_digest
                inner = model.module if world_size > 1 else model
                epoch_entry_state = {
                    'model': state_digest(inner.state_dict()), 'optimizer': state_digest(optimizer.state_dict()),
                    'scheduler': state_digest(scheduler.state_dict()), 'rng': state_digest(capture_rng(train_generator)),
                    'ema': state_digest(ema.state_dict()) if ema else None,
                    'adapters': state_digest({'ir':distill_loss_fn.ir_adapters.state_dict(),
                                             'depth':distill_loss_fn.depth_adapters.state_dict()}) if distill_loss_fn else None,
                }
        if cfg.get("encoder_oto_supervision", {}).get("enabled", False):
            warmup_epochs = int(cfg["encoder_oto_supervision"].get("warmup_epochs", 2))
            target_weight = float(cfg["encoder_oto_supervision"].get("weight", 0.5))
            progress = (epoch + 1 - warmup_epochs) / max(warmup_epochs, 1)
            criterion.encoder_oto_weight = target_weight * min(1.0, max(0.0, progress))
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        if curriculum_run:
            from aic.sugon.common import atomic_json
            hashes = [None] * world_size
            dist.all_gather_object(hashes, train_sampler.summary['plan_sha256'])
            if len(set(hashes)) != 1:
                raise ValueError('DDP curriculum plans differ')
            if is_main_process():
                atomic_json(os.path.join(save_dir, 'run_info', f'plan_epoch{epoch}.json'),
                            {'summary': train_sampler.summary, 'batches': train_sampler.plans})

        train_tf.set_epoch(epoch)
        train_ds.set_epoch(epoch)
        stage2_start = int(train_cfg.get("stage2_start_epoch", train_cfg["epochs"]))
        if epoch >= stage2_start:
            # Explicit 48/16 close-out: fixed training size, no modality dropout,
            # and no strong geometric scale jitter in the refinement stage.
            train_tf.modality_dropout_p = float(train_cfg.get("stage2_modality_dropout_p", 0.0))
            train_tf.scale_range = tuple(train_cfg.get("stage2_scale_range", [1.0, 1.0]))
        if isinstance(train_collate, MultiScaleCollate):
            decay_start = train_tf.stop_decay_epoch_ratio * train_tf.total_epochs
            train_collate.set_epoch_active(
                epoch < decay_start and epoch < stage2_start
            )

        epoch_start = time.time()
        if context_run or curriculum_run:
            exposure_rows = []
        optimizer.zero_grad()
        max_steps_per_epoch = int(train_cfg.get("max_steps_per_epoch", 0))
        for step, (rgb, ir, depth, targets) in enumerate(train_loader):
            if max_steps_per_epoch > 0 and step >= max_steps_per_epoch:
                break
            if context_run:
                exposure_rows.extend({'stem':t['stem'], **t['context_view'],
                                      'labels':t['labels'].tolist(), 'boxes':t['boxes'].tolist(),
                                      'input_hw':list(rgb.shape[-2:])} for t in targets)
            if curriculum_run:
                exposure_rows.extend({'stem':t['stem'], **t['curriculum_view'],
                                      'labels':t['labels'].tolist(), 'boxes':t['boxes'].tolist(),
                                      'input_hw':list(rgb.shape[-2:])} for t in targets)
            if (frequency_run or context_run or curriculum_run) and step == 0:
                first_samples = [t['stem'] for t in targets]
            rgb, ir, depth = rgb.to(device, non_blocking=True), ir.to(device, non_blocking=True), depth.to(
                device, non_blocking=True
            )
            targets = move_targets_to_device(targets, device)
            current_self_scale = self_distill_scale(epoch, step)
            current_dropout_p = student_dropout_p if epoch + step / max(dataloader_steps, 1) < student_dropout_stop else 0.0
            student_ir, student_depth, student_dropout_rate = apply_student_modality_dropout(
                ir, depth, current_dropout_p
            )
            teacher_outputs = None
            if current_self_scale > 0.0:
                with torch.autocast(
                    device_type="cuda" if use_amp else "cpu", dtype=amp_dtype, enabled=use_amp
                ):
                    teacher_outputs = self_distill_loss_fn.teacher_forward(rgb, ir, depth)

            # 梯度累积期间跳过 DDP 的梯度 allreduce：只有到累积边界那一步才需要
            # 同步。原先每个 micro-step 都同步，等于把通信量放大了 grad_accum 倍，
            # 而且这些中间结果的 allreduce 完全是白做的（梯度还要继续累加）。
            # find_unused_parameters=True 会让每次同步额外扫一遍计算图，更贵。
            window_size, is_boundary = accumulation_window(step, effective_steps, grad_accum) if explicit_split else (grad_accum, (step+1)%grad_accum == 0)
            sync_ctx = (
                contextlib.nullcontext()
                if is_boundary or world_size == 1
                else model.no_sync()
            )

            with sync_ctx:
                with torch.autocast(device_type="cuda" if use_amp else "cpu", dtype=amp_dtype, enabled=use_amp):
                    if distill_loss_fn is not None or dino_loss_fn is not None or fdr_loss_fn is not None:
                        outputs, rgb_feats, ir_feats, depth_feats = model(
                            rgb, student_ir, student_depth, targets, return_backbone_feats=True
                        )
                    else:
                        outputs = model(rgb, student_ir, student_depth, targets)
                    fusion_fg_logits = outputs.pop('fusion_fg_logits', None)
                    loss_dict = criterion(outputs, targets)
                    # criterion 内部已经对每个 key（包括 _aux_i/_dn_i/_enc_i/_pre 等派生 key）
                    # 乘过 weight_dict 系数了（见 dfine_criterion.py forward），所以这里要把
                    # loss_dict 的全部 value 加总，而不是只挑 key 本身在 weight_dict 里的那几个
                    # —— 之前的写法只加了 4 个基础 key，丢掉了全部辅助解码层损失、去噪损失、
                    # encoder 辅助损失（占总损失量级的大头），等于关掉了 D-FINE 的深监督和
                    # denoising 训练机制。跟官方 det_engine.py 的 `sum(loss_dict.values())` 保持一致。
                    total = sum(loss_dict.values())
                    if fusion_fg_logits is not None:
                        from aic.models.frequency_fusion import fusion_foreground_loss
                        fg_scale = cfg['fusion']['foreground_weight'] * min(
                            (optimizer_updates + 1) / cfg['fusion']['foreground_warmup_updates'], 1.)
                        loss_dict['loss_fusion_fg'] = fusion_foreground_loss(fusion_fg_logits, targets) * fg_scale
                        total = total + loss_dict['loss_fusion_fg']
                    inner_model = model.module if world_size > 1 else model
                    if frequency_run:
                        loss_dict.update(inner_model.opf.last_metrics())
                    if getattr(inner_model, "ir_alignment", None) is not None:
                        # Diagnostic tensors are detached and deliberately added only
                        # after the criterion sum, so they never alter optimization.
                        loss_dict.update(inner_model.ir_alignment.last_metrics())
                    if distill_loss_fn is not None:
                        natural_indices = [i for i, t in enumerate(targets) if t.get('natural_view', False)] if curriculum_run else list(range(len(targets)))
                        if curriculum_run and len(natural_indices) != 1:
                            raise ValueError('each rank must distill exactly one natural view')
                        distill_total, distill_metrics = distill_loss_fn(
                            ir[natural_indices],
                            depth[natural_indices],
                            [f[natural_indices] for f in ir_feats],
                            [f[natural_indices] for f in depth_feats],
                            targets=[targets[i] for i in natural_indices],
                        )
                        distill_scale = min(
                            1.0,
                            (epoch * dataloader_steps + step + 1) / max(distill_warmup_steps, 1),
                        ) if distill_warmup_steps > 0 else 1.0
                        if epoch >= distill_decay_start and distill_final_scale < 1.0:
                            decay_steps = max(
                                (train_cfg["epochs"] - distill_decay_start) * dataloader_steps - 1,
                                1,
                            )
                            decay_step = (epoch - distill_decay_start) * dataloader_steps + step
                            progress = min(max(decay_step / decay_steps, 0.0), 1.0)
                            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                            distill_scale = distill_final_scale + (1.0 - distill_final_scale) * cosine
                        loss_dict["distill_scale"] = distill_total.new_tensor(distill_scale)
                        loss_dict["loss_distill"] = distill_total * distill_scale
                        loss_dict.update(
                            {key: value * distill_scale for key, value in distill_metrics.items()}
                        )
                        total = total + loss_dict["loss_distill"]
                    if fdr_loss_fn is not None:
                        fdr_total, fdr_metrics = fdr_loss_fn(
                            rgb, ir, depth, outputs, targets, matcher
                        )
                        loss_dict["loss_fdr"] = fdr_total
                        loss_dict.update(fdr_metrics)
                        total = total + fdr_total
                    if dino_loss_fn is not None:
                        dino_weighted = dino_loss_fn(rgb, rgb_feats, targets=targets)
                        dino_scale = min(
                            1.0,
                            (epoch * dataloader_steps + step + 1) / max(dino_warmup_steps, 1),
                        ) if dino_warmup_steps > 0 else 1.0
                        loss_dict["loss_dino_raw"] = (
                            dino_weighted.detach() / max(dino_loss_fn.weight, 1e-12)
                        )
                        loss_dict["dino_weight"] = dino_weighted.new_tensor(
                            dino_loss_fn.weight * dino_scale
                        )
                        loss_dict["loss_dino"] = dino_weighted * dino_scale
                        total = total + loss_dict["loss_dino"]
                    if self_distill_loss_fn is not None:
                        loss_dict["self_distill_scale"] = total.new_tensor(current_self_scale)
                        loss_dict["student_dropout_rate"] = student_dropout_rate.detach()
                        if teacher_outputs is not None:
                            self_total, self_metrics = self_distill_loss_fn(
                                outputs, teacher_outputs, targets, matcher
                            )
                            loss_dict["loss_self_distill"] = self_total * current_self_scale
                            loss_dict.update(self_metrics)
                            total = total + loss_dict["loss_self_distill"]
                        else:
                            loss_dict["loss_self_distill"] = total.new_zeros(())
                    loss = total / window_size
                    if explicit_split and not torch.isfinite(loss).all():
                        raise FloatingPointError(f'non-finite training loss at epoch={epoch}, step={step}, rank={rank}')

                scaler.scale(loss).backward()

            if is_boundary:
                # Training-only distillation adapters live outside the student DDP
                # wrapper, so synchronize their accumulated gradients explicitly.
                if world_size > 1 and auxiliary_params:
                    for param in auxiliary_params:
                        if param.grad is not None:
                            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                            param.grad.div_(world_size)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(clip_params, train_cfg["grad_clip_norm"], error_if_nonfinite=frequency_run or context_run or curriculum_run)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_stepped = scaler.get_scale() >= previous_scale
                if optimizer_stepped:
                    scheduler.step()
                    optimizer_updates += 1
                optimizer.zero_grad()
                if ema is not None and optimizer_stepped:
                    ema.update(model)

            if is_main_process() and step % train_cfg["log_interval"] == 0:
                base_keys = (
                    "loss_vfl", "loss_bbox", "loss_giou", "loss_gcd", "loss_fgl", "loss_ddf", "loss_fusion_fg",
                    "loss_distill", "loss_ir", "loss_depth", "loss_cross",
                    "distill_scale",
                    "weight_ir_dynamic", "weight_depth_dynamic",
                    "loss_ir_l0", "loss_ir_l1", "loss_ir_l2",
                    "loss_depth_l0", "loss_depth_l1", "loss_depth_l2",
                    "loss_cross_l0", "loss_cross_l1", "loss_cross_l2",
                    "loss_fdr", "loss_fdr_ir", "loss_fdr_depth", "fdr_pairs",
                    "align_gate_l0", "align_gate_l1", "align_gate_l2",
                    "align_offset_l0", "align_offset_l1", "align_offset_l2",
                    "loss_dino_raw", "loss_dino", "dino_weight",
                    "loss_self_distill", "loss_self_cls", "loss_self_bbox", "loss_self_giou",
                    "self_pairs", "self_coverage", "self_distill_scale", "student_dropout_rate",
                    "loss_oto_vfl", "loss_oto_bbox", "loss_oto_giou",
                )
                loss_str = " ".join(f"{k}={loss_dict[k].item():.4f}" for k in base_keys if k in loss_dict)
                lr_now = optimizer.param_groups[-1]["lr"]
                total_loss = loss.item() * window_size
                print(f"[epoch {epoch} step {step}/{dataloader_steps}] lr={lr_now:.2e} total={total_loss:.4f} {loss_str}")

                global_step = epoch * dataloader_steps + step
                step_losses = {k: loss_dict[k].item() for k in base_keys if k in loss_dict}
                if frequency_run:
                    step_losses.update({k: float(v) for k,v in inner_model.opf.last_metrics().items()})
                step_losses["total"] = total_loss
                # 辅助层损失分组求和：D-FINE 深监督机制产生的派生 key 按后缀分四组
                # （_aux_i=5 层中间 decoder 输出，_pre=首层传统检测头，_enc_i=encoder
                # 辅助头，_dn_i/_dn_pre=去噪分支），主头 5 项之外全部计入这四组之一，
                # 分组数值之和 + 主头总和 == total（sum(loss_dict.values())）。
                group_sums = {"grp_aux": 0.0, "grp_enc": 0.0, "grp_dn": 0.0, "grp_pre": 0.0}
                for k, v in loss_dict.items():
                    # D-FINE uses ``_aux_i`` while native RF-DETR names
                    # intermediate decoder losses ``loss_*_i``.  Keep both
                    # conventions visible in the same diagnostic bucket.
                    if "_aux_" in k or re.fullmatch(r"loss_(?:ce|bbox|giou|keypoints_[a-z0-9_]+)_\d+", k):
                        group_sums["grp_aux"] += v.item()
                    elif "_enc_" in k or k.endswith("_enc"):
                        group_sums["grp_enc"] += v.item()
                    elif "_dn_" in k:
                        group_sums["grp_dn"] += v.item()
                    elif k.endswith("_pre"):
                        group_sums["grp_pre"] += v.item()
                step_losses.update(group_sums)
                curve_logger.log_train_step(global_step, step_losses, lr_now)
                curve_logger.plot()
                curve_logger.save_json()

        rng_states = None
        if explicit_split:
            if train_cfg.get('max_steps_per_epoch') and distill_loss_fn is not None and world_size > 1:
                adapter_digest = tensor_hash({**{'ir.'+k:v for k,v in distill_loss_fn.ir_adapters.state_dict().items()},
                                              **{'depth.'+k:v for k,v in distill_loss_fn.depth_adapters.state_dict().items()}})
                adapter_digests = [None] * world_size
                dist.all_gather_object(adapter_digests, adapter_digest)
                if len(set(adapter_digests)) != 1:
                    raise RuntimeError('M2D adapter parameters diverged across DDP ranks')
            local_rng = capture_rng(train_generator)
            rng_states = [None] * world_size
            if is_dist():
                dist.all_gather_object(rng_states, local_rng)
            else:
                rng_states[0] = local_rng
        if context_run or curriculum_run:
            from aic.sugon.common import atomic_json
            atomic_json(os.path.join(save_dir, 'run_info', f'exposure_epoch{epoch}_rank{rank}.json'), exposure_rows)
        if frequency_run or context_run or curriculum_run:
            if optimizer_updates != (epoch + 1) * steps_per_epoch:
                raise RuntimeError('optimizer update count does not match complete accumulation windows')
            if train_cfg.get('preflight_trace'):
                from aic.sugon.common import atomic_json
                atomic_json(os.path.join(save_dir, f'trace_epoch{epoch}_rank{rank}.json'),
                            {'first_samples': first_samples, 'last_loss': float(loss.detach()),
                             'entry_state': epoch_entry_state,
                             'updates': optimizer_updates, 'lr': scheduler.get_last_lr(),
                             'ema_updates': ema.updates if ema else None})
        if memory_probe:
            from aic.sugon.common import atomic_json
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            report_dir = os.path.join(save_dir, str(train_cfg["memory_report_path"]))
            os.makedirs(report_dir, exist_ok=True)
            atomic_json(os.path.join(report_dir, f"epoch{epoch}_rank{rank}.json"), {
                "epoch": epoch,
                "rank": rank,
                "device": str(device),
                "total_bytes": int(total_bytes),
                "free_bytes_after_train": int(free_bytes),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                "allocated_fraction": float(torch.cuda.max_memory_allocated(device) / total_bytes),
                "reserved_fraction": float(torch.cuda.max_memory_reserved(device) / total_bytes),
                "optimizer_updates": optimizer_updates,
            })
        if is_main_process():
            print(f"[epoch {epoch}] done in {time.time() - epoch_start:.1f}s")
            if curriculum_run:
                from aic.sugon.common import atomic_json
                atomic_json(os.path.join(save_dir, 'run_info', f'runtime_epoch{epoch}.json'),
                            {'training_seconds':time.time()-epoch_start,'optimizer_updates':optimizer_updates,
                             'peak_allocated_mib':torch.cuda.max_memory_allocated(device)/2**20,
                             'view_plan':train_sampler.summary})

        if is_main_process() and (epoch + 1) % train_cfg["eval_interval_epochs"] == 0:
            # EMA 权重更稳定，验证/落盘都用它；仅在 rank0 临时应用+恢复，不影响
            # 其它 rank 的训练状态（DDP 模式下 eval 只在主进程跑，见上面 is_main_process 判断）。
            #
            # eval 前向传播必须过 model.module（原始模块），不能过 DDP 包装器本身：
            # SyncBatchNorm 意味着 backbone 的 BatchNorm2d 被替换为跨进程同步实现，而
            # DDP 默认 broadcast_buffers=True，每次调用被包装模块的 forward() 都会做一次
            # buffer broadcast collective。这段代码只在 rank0 执行，若 eval 循环里对
            # `model`（DDP 包装器）调用 400 次 forward，就是给 rank0 侧插入 400 次其它
            # rank 不知情、也不会参与的 broadcast——collective 是按调用顺序匹配的，不是
            # 按语义匹配，rank1 完全没有对应调用。这不会在 eval 循环本身报错（rank0 自己
            # 发出的 broadcast 无人接收，NCCL 不会立刻检测到），但会让两个 rank 的 collective
            # 调用序列错位；等下一个 epoch 开始所有 rank 一起再次调用 model(...) 时，
            # rank1 等的 broadcast 对应的是错位后的另一次调用，双方永远配不上，最终 30
            # 分钟 watchdog 超时（NCCL BROADCAST timeout）。改用 model.module 走原始
            # nn.Module.__call__，不经过 DDP 的 forward 钩子，就不会有任何 collective。
            eval_module = model.module if world_size > 1 else model
            eval_state_backup = ema.apply_to(model) if ema is not None else None
            eval_result = None
            if has_val:
                cfg['_eval_epoch'] = epoch
                eval_result = _run_eval(eval_module, val_loader, device, world_size, cfg)
                cfg.pop('_eval_epoch', None)
                curve_logger.log_eval(
                    epoch,
                    {
                        "mAP@50-95": eval_result.map_50_95,
                        "mAP@50": eval_result.map_50,
                        "mAP@75": eval_result.map_75,
                        "raw_mAP@50-95": getattr(eval_result, "raw_map_50_95", eval_result.map_50_95),
                        "raw_mAP@50": getattr(eval_result, "raw_map_50", eval_result.map_50),
                        "raw_mAP@75": getattr(eval_result, "raw_map_75", eval_result.map_75),
                        **({f'mAP@{v}': eval_result.per_iou_map[v/100] for v in (80, 85, 90)}
                           if hasattr(eval_result, 'per_iou_map') else {}),
                    },
                )
                curve_logger.plot()
                curve_logger.save_json()

            if explicit_split and eval_state_backup is not None:
                ema.restore(model, eval_state_backup)
                eval_state_backup = None
            ckpt_path = os.path.join(save_dir, f"epoch{epoch}.pt")
            state = (model.module if world_size > 1 else model).state_dict()
            ckpt = {
                "model": state,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
            }
            if explicit_split:
                ckpt.update(experiment_identity=identity(cfg), rng_states=rng_states,
                            optimizer_updates=optimizer_updates, world_size=world_size,
                            **validation_provenance)
            if frequency_run:
                from aic.sugon.common import atomic_json
                atomic_json(os.path.join(save_dir, f'runtime_epoch{epoch}.json'),
                            {'train_seconds': time.time()-epoch_start, 'optimizer_updates': optimizer_updates,
                             'peak_allocated_mib': torch.cuda.max_memory_allocated(device)/2**20,
                             'fusion_metrics': {k: float(v) for k,v in eval_module.opf.last_metrics().items()}})
            if ema is not None:
                ckpt["ema"] = ema.state_dict()
                ckpt["ema_updates"] = ema.updates
            if dino_loss_fn is not None:
                ckpt["dino_adapters"] = dino_loss_fn.adapters.state_dict()
            if distill_loss_fn is not None:
                ckpt["m2d_adapters"] = {
                    "ir": distill_loss_fn.ir_adapters.state_dict(),
                    "depth": distill_loss_fn.depth_adapters.state_dict(),
                }
            if self_distill_loss_fn is not None:
                ckpt["self_distill_teacher"] = {
                    "checkpoint": self_distill_loss_fn.teacher_ckpt,
                    "source": self_distill_loss_fn.teacher_source,
                }
            # 先写临时文件再原子改名：写盘失败（如磁盘满）不会留下截断的 .pt，
            # 断点恢复时也不会读到坏文件。
            tmp_path = ckpt_path + ".tmp"
            try:
                torch.save(ckpt, tmp_path)
                os.replace(tmp_path, ckpt_path)
            except Exception:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise
            print(f"[ckpt] saved {ckpt_path}")
            if explicit_split:
                from aic.sugon.common import atomic_json
                atomic_json(os.path.join(save_dir, 'latest.json'), {'epoch':epoch, 'checkpoint':os.path.basename(ckpt_path), **identity(cfg)})
                if eval_result is not None and eval_result.map_50_95 > best_map:
                    best_map, best_epoch = float(eval_result.map_50_95), epoch
                    atomic_json(os.path.join(save_dir, 'best.json'), {'epoch':epoch,'round':epoch+1,
                        'checkpoint':os.path.basename(ckpt_path),'map_50_95':best_map,
                        'map_50':float(eval_result.map_50),'map_75':float(eval_result.map_75),
                        'weight_source':'ema', **validation_provenance, **identity(cfg)})

            # checkpoint 轮转：每个约 1.5GB，完整保留会迅速占满磁盘。
            # 保留最近 keep_last_n 个，并额外保留验证集 top keep_best_n 个。
            keep_last_n = int(train_cfg.get("keep_last_n", 3))
            if keep_last_n > 0:
                saved = []
                for fn in os.listdir(save_dir):
                    m = re.fullmatch(r"epoch(\d+)\.pt", fn)
                    if m:
                        saved.append((int(m.group(1)), fn))
                saved.sort()
                best_epochs = set()
                if has_val:
                    try:
                        import json

                        with open(os.path.join(save_dir, "curves.json"), "r", encoding="utf-8") as f:
                            rec = json.load(f)
                        saved_epochs = {ep for ep, _ in saved}
                        pairs = [
                            (int(ep), float(v))
                            for ep, v in zip(
                                rec.get("eval_epochs", []),
                                rec.get("eval_metrics", {}).get("mAP@50-95", []),
                            )
                            if v is not None and int(ep) in saved_epochs
                        ]
                        keep_best_n = int(train_cfg.get("keep_best_n", keep_last_n))
                        best_epochs = {
                            ep for ep, _ in sorted(pairs, key=lambda item: item[1], reverse=True)[:keep_best_n]
                        }
                    except Exception as exc:
                        print(f"[ckpt] unable to read validation history for top-k retention: {exc}")
                protected = {ep for ep, _ in saved[-keep_last_n:]} | best_epochs
                protected |= {int(ep) for ep in train_cfg.get("retain_epochs", [])}
                for ep, fn in saved:
                    if ep not in protected:
                        os.remove(os.path.join(save_dir, fn))
                        print(f"[ckpt] pruned {fn}")


            if eval_state_backup is not None:
                ema.restore(model, eval_state_backup)

        if is_dist():
            dist.barrier()

    if is_dist():
        dist.destroy_process_group()


@torch.no_grad()
def _run_eval(model, val_loader, device, world_size, cfg):
    from third_party_dfine import DFINEPostProcessor

    import numpy as np

    from aic.eval.map_evaluator import evaluate, xywh_to_xyxy

    model.eval()
    postproc = DFINEPostProcessor(
        num_classes=cfg["num_classes"], use_focal_loss=True, num_top_queries=cfg["decoder"]["num_queries"]
    )

    predictions: dict[str, list] = {}
    raw_predictions: dict[str, list] = {}
    ground_truths: dict[str, list] = {}

    for batch_index, (rgb, ir, depth, targets) in enumerate(val_loader):
        if cfg.get('train', {}).get('eval_max_images', 0) and batch_index >= cfg['train']['eval_max_images']:
            break
        rgb, ir, depth = rgb.to(device), ir.to(device), depth.to(device)
        outputs = model(rgb, ir, depth)
        orig_sizes = torch.stack([t["orig_size"] for t in targets]).to(device)
        results = postproc(outputs, orig_sizes)

        for i, t in enumerate(targets):
            stem = t["stem"]
            w, h = t["orig_size"].tolist()
            boxes_px = results[i]["boxes"].cpu().numpy()
            scores = results[i]["scores"].cpu().numpy()
            labels = results[i]["labels"].cpu().numpy()
            boxes_norm = boxes_px / np.array([w, h, w, h])
            raw_detections = [
                (int(labels[j]), boxes_norm[j], float(scores[j])) for j in range(len(labels))
            ]
            raw_predictions[stem] = raw_detections
            cap = int(cfg.get("train", {}).get("eval_max_boxes", 0))
            predictions[stem] = (
                sorted(raw_detections, key=lambda item: -item[2])[:cap]
                if cap > 0 else raw_detections
            )
            if cfg.get('train', {}).get('submission_eval', False):
                from aic.eval.submission_postprocess import submission_rows, evaluator_rows
                predictions[stem] = evaluator_rows(submission_rows(boxes_px, scores, labels, w, h,
                    cfg['inference']['score_thresh'], cfg['inference']['max_boxes']))

            gt_boxes = t["boxes"].numpy()
            gt_labels = t["labels"].numpy()
            gt_xyxy = xywh_to_xyxy(gt_boxes) if gt_boxes.size else gt_boxes
            ground_truths[stem] = [(int(gt_labels[j]), gt_xyxy[j]) for j in range(len(gt_labels))]

    result = evaluate(predictions, ground_truths, class_ids=list(range(cfg["num_classes"])))
    if cfg.get('train', {}).get('save_eval_predictions'):
        from aic.eval.frequency_diagnostics import save_diagnostics
        save_diagnostics(cfg, result, predictions, ground_truths)
    if int(cfg.get("train", {}).get("eval_max_boxes", 0)) > 0:
        raw_result = evaluate(raw_predictions, ground_truths, class_ids=list(range(cfg["num_classes"])))
        result.raw_map_50_95 = raw_result.map_50_95
        result.raw_map_50 = raw_result.map_50
        result.raw_map_75 = raw_result.map_75
    print(
        f"[eval] top{int(cfg.get('train', {}).get('eval_max_boxes', 0)) or 'all'} "
        f"mAP@50-95={result.map_50_95:.4f} mAP@50={result.map_50:.4f} mAP@75={result.map_75:.4f}"
        + (f" raw={result.raw_map_50_95:.4f}" if hasattr(result, "raw_map_50_95") else "")
    )
    model.train()
    return result


if __name__ == "__main__":
    main()

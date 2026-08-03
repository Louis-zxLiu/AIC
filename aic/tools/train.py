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
import os
import re
import time

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
from aic.models.tristream_dfine import build_tristream_dfine
from aic.utils.curves import CurveLogger
from third_party_dfine import DFINECriterion, HungarianMatcher


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return (not is_dist()) or dist.get_rank() == 0


def setup_distributed() -> tuple[int, int, int]:
    """从 torchrun 注入的环境变量初始化进程组。单卡直接跑时不进入分布式模式。"""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def build_param_groups(model, lr: float, backbone_lr_mult: float, weight_decay: float):
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
        if isinstance(m, norm_types):
            norm_param_ids.update(id(p) for p in m.parameters())

    inner = model.module if hasattr(model, "module") else model
    pretrained_ids = inner.backbone.pretrained_param_ids

    groups = {"pretrained": [], "pretrained_norm": [], "new": [], "new_norm": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_pretrained = id(p) in pretrained_ids
        is_norm_or_bias = id(p) in norm_param_ids or name.endswith(".bias")
        key = ("pretrained" if is_pretrained else "new") + ("_norm" if is_norm_or_bias else "")
        groups[key].append(p)

    param_groups = [
        {"params": groups["pretrained"], "lr": lr * backbone_lr_mult, "weight_decay": weight_decay},
        {"params": groups["pretrained_norm"], "lr": lr * backbone_lr_mult, "weight_decay": 0.0},
        {"params": groups["new"], "lr": lr, "weight_decay": weight_decay},
        {"params": groups["new_norm"], "lr": lr, "weight_decay": 0.0},
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    train_cfg = cfg["train"]
    save_dir = train_cfg["save_dir"]
    if is_main_process():
        os.makedirs(save_dir, exist_ok=True)

    curve_logger = CurveLogger(save_dir) if is_main_process() else None

    samples = scan_samples(train_cfg["data_root"], require_labels=True)
    train_samples, val_samples = kfold_split(samples, n_folds=train_cfg["n_folds"], fold=train_cfg["fold"])
    if is_main_process():
        print(f"[data] train={len(train_samples)} val={len(val_samples)} total={len(samples)}")

    train_tf = SyncMultiModalTransform(
        size=cfg["image_size"],
        train=True,
        modality_dropout_p=train_cfg["modality_dropout_p"],
        total_epochs=train_cfg["epochs"],
    )
    val_tf = SyncMultiModalTransform(size=cfg["image_size"], train=False)
    disp8_mode = train_cfg.get("disp8_mode", "use")
    train_ds = MultiModalDetDataset(train_samples, train_tf, has_labels=True, disp8_mode=disp8_mode)
    val_ds = MultiModalDetDataset(val_samples, val_tf, has_labels=True, disp8_mode=disp8_mode)

    # 类别不均衡重采样（算法设计.md §3.4，计划.md §2.4 tier 1）：ClassBalancedDataset
    # 风格的按类别频率过采样。DistributedSampler 不支持自定义样本权重，DDP 模式下暂时
    # 退化为均匀 shuffle（仍需先跑通单卡验证过采样收益后再移植到分布式，见文档说明）。
    class_balance_cfg = train_cfg.get("class_balance", {})
    use_class_balance = class_balance_cfg.get("enabled", False)

    if world_size > 1:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    elif use_class_balance:
        weights = compute_class_balanced_weights(
            train_samples, oversample_thr=class_balance_cfg.get("oversample_thr", 0.2)
        )
        train_sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        if is_main_process():
            print(f"[data] class-balanced sampling enabled, oversample_thr={class_balance_cfg.get('oversample_thr', 0.2)}")
    else:
        train_sampler = None

    multi_scale_cfg = train_cfg.get("multi_scale", {})
    if multi_scale_cfg.get("enabled", False):
        train_collate = MultiScaleCollate(
            base_size=cfg["image_size"], sizes=tuple(multi_scale_cfg.get("sizes", [512, 576, 640, 704, 768]))
        )
    else:
        train_collate = collate_fn

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=train_cfg["num_workers"],
        collate_fn=train_collate,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn
    )

    model = build_tristream_dfine(cfg).to(device)

    if world_size > 1 and torch.cuda.is_available():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    matcher = HungarianMatcher(**cfg["matcher"])
    criterion = DFINECriterion(matcher=matcher, num_classes=cfg["num_classes"], **cfg["criterion"]).to(device)

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
        model, train_cfg["lr"], train_cfg["backbone_lr_mult"], train_cfg["weight_decay"]
    )
    optimizer = torch.optim.AdamW(param_groups)

    grad_accum = train_cfg["grad_accum_steps"]

    # 关键：scheduler.step() 只在梯度累积边界被调用（见训练循环），所以调度器
    # 的 step 计数单位是 **optimizer step**，不是 dataloader step。这里必须除以
    # grad_accum，否则 warmup/cosine 的时间轴会被放大 grad_accum 倍——首轮实验就
    # 栽在这里：warmup 名义 2 epoch，实际要 2×16=32 epoch 才走完，cosine 衰减
    # 全程没启动，实测 lr 比预期低了一个数量级。
    # steps_per_epoch 是 optimizer step 数（调度器单位）；dataloader_steps 是
    # 实际迭代次数，仅用于日志进度显示。两者相差 grad_accum 倍。
    dataloader_steps = len(train_loader)
    steps_per_epoch = max(dataloader_steps // grad_accum, 1)
    total_steps = steps_per_epoch * train_cfg["epochs"]
    warmup_steps = steps_per_epoch * train_cfg["warmup_epochs"]

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    amp_dtype = torch.bfloat16 if train_cfg["amp_dtype"] == "bfloat16" else torch.float16
    use_amp = torch.cuda.is_available()

    ema_cfg = train_cfg.get("ema", {})
    ema = (
        ModelEMA(model, decay=ema_cfg.get("decay", 0.9999), warmups=ema_cfg.get("warmups", 1000))
        if ema_cfg.get("enabled", False)
        else None
    )

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        (model.module if world_size > 1 else model).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        if ema is not None and "ema" in ckpt:
            ema.shadow = {k: v.to(device) for k, v in ckpt["ema"].items()}
            ema.updates = ckpt.get("ema_updates", 0)
        if is_main_process():
            print(f"[resume] from {args.resume}, starting epoch {start_epoch}")

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

    for epoch in range(start_epoch, train_cfg["epochs"]):
        model.train()
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)

        train_tf.set_epoch(epoch)
        if isinstance(train_collate, MultiScaleCollate):
            decay_start = train_tf.stop_decay_epoch_ratio * train_tf.total_epochs
            train_collate.set_epoch_active(epoch < decay_start)

        epoch_start = time.time()
        optimizer.zero_grad()
        for step, (rgb, ir, depth, targets) in enumerate(train_loader):
            rgb, ir, depth = rgb.to(device, non_blocking=True), ir.to(device, non_blocking=True), depth.to(
                device, non_blocking=True
            )
            targets = move_targets_to_device(targets, device)

            # 梯度累积期间跳过 DDP 的梯度 allreduce：只有到累积边界那一步才需要
            # 同步。原先每个 micro-step 都同步，等于把通信量放大了 grad_accum 倍，
            # 而且这些中间结果的 allreduce 完全是白做的（梯度还要继续累加）。
            # find_unused_parameters=True 会让每次同步额外扫一遍计算图，更贵。
            is_boundary = (step + 1) % grad_accum == 0
            sync_ctx = (
                contextlib.nullcontext()
                if is_boundary or world_size == 1
                else model.no_sync()
            )

            with sync_ctx:
                with torch.autocast(device_type="cuda" if use_amp else "cpu", dtype=amp_dtype, enabled=use_amp):
                    outputs = model(rgb, ir, depth, targets)
                    loss_dict = criterion(outputs, targets)
                    # criterion 内部已经对每个 key（包括 _aux_i/_dn_i/_enc_i/_pre 等派生 key）
                    # 乘过 weight_dict 系数了（见 dfine_criterion.py forward），所以这里要把
                    # loss_dict 的全部 value 加总，而不是只挑 key 本身在 weight_dict 里的那几个
                    # —— 之前的写法只加了 4 个基础 key，丢掉了全部辅助解码层损失、去噪损失、
                    # encoder 辅助损失（占总损失量级的大头），等于关掉了 D-FINE 的深监督和
                    # denoising 训练机制。跟官方 det_engine.py 的 `sum(loss_dict.values())` 保持一致。
                    loss = sum(loss_dict.values()) / grad_accum

                loss.backward()

            if is_boundary:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update(model)

            if is_main_process() and step % train_cfg["log_interval"] == 0:
                base_keys = ("loss_vfl", "loss_bbox", "loss_giou", "loss_fgl", "loss_ddf")
                loss_str = " ".join(f"{k}={loss_dict[k].item():.4f}" for k in base_keys if k in loss_dict)
                lr_now = optimizer.param_groups[-1]["lr"]
                total_loss = loss.item() * grad_accum
                print(f"[epoch {epoch} step {step}/{dataloader_steps}] lr={lr_now:.2e} total={total_loss:.4f} {loss_str}")

                global_step = epoch * dataloader_steps + step
                step_losses = {k: loss_dict[k].item() for k in base_keys if k in loss_dict}
                step_losses["total"] = total_loss
                # 辅助层损失分组求和：D-FINE 深监督机制产生的派生 key 按后缀分四组
                # （_aux_i=5 层中间 decoder 输出，_pre=首层传统检测头，_enc_i=encoder
                # 辅助头，_dn_i/_dn_pre=去噪分支），主头 5 项之外全部计入这四组之一，
                # 分组数值之和 + 主头总和 == total（sum(loss_dict.values())）。
                group_sums = {"grp_aux": 0.0, "grp_enc": 0.0, "grp_dn": 0.0, "grp_pre": 0.0}
                for k, v in loss_dict.items():
                    if "_aux_" in k:
                        group_sums["grp_aux"] += v.item()
                    elif "_enc_" in k:
                        group_sums["grp_enc"] += v.item()
                    elif "_dn_" in k:
                        group_sums["grp_dn"] += v.item()
                    elif k.endswith("_pre"):
                        group_sums["grp_pre"] += v.item()
                step_losses.update(group_sums)
                curve_logger.log_train_step(global_step, step_losses, lr_now)
                curve_logger.plot()
                curve_logger.save_json()

        if is_main_process():
            print(f"[epoch {epoch}] done in {time.time() - epoch_start:.1f}s")

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
            eval_result = _run_eval(eval_module, val_loader, device, world_size, cfg)
            curve_logger.log_eval(
                epoch,
                {
                    "mAP@50-95": eval_result.map_50_95,
                    "mAP@50": eval_result.map_50,
                    "mAP@75": eval_result.map_75,
                },
            )
            curve_logger.plot()
            curve_logger.save_json()

            ckpt_path = os.path.join(save_dir, f"epoch{epoch}.pt")
            state = (model.module if world_size > 1 else model).state_dict()
            ckpt = {
                "model": state,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
            }
            if ema is not None:
                ckpt["ema"] = ema.state_dict()
                ckpt["ema_updates"] = ema.updates
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

            # checkpoint 轮转：每个 1.5GB，48 epoch 全留会写满磁盘。
            # 只保留最近 keep_last_n 个 + 历史最佳。
            keep_last_n = int(train_cfg.get("keep_last_n", 3))
            if keep_last_n > 0:
                if eval_result.map_50_95 > best_map:
                    best_map = eval_result.map_50_95
                    best_epoch = epoch
                saved = []
                for fn in os.listdir(save_dir):
                    m = re.fullmatch(r"epoch(\d+)\.pt", fn)
                    if m:
                        saved.append((int(m.group(1)), fn))
                saved.sort()
                for ep, fn in saved[:-keep_last_n]:
                    if ep != best_epoch:
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
    ground_truths: dict[str, list] = {}

    for rgb, ir, depth, targets in val_loader:
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
            predictions[stem] = [
                (int(labels[j]), boxes_norm[j], float(scores[j])) for j in range(len(labels))
            ]

            gt_boxes = t["boxes"].numpy()
            gt_labels = t["labels"].numpy()
            gt_xyxy = xywh_to_xyxy(gt_boxes) if gt_boxes.size else gt_boxes
            ground_truths[stem] = [(int(gt_labels[j]), gt_xyxy[j]) for j in range(len(gt_labels))]

    result = evaluate(predictions, ground_truths, class_ids=list(range(cfg["num_classes"])))
    print(f"[eval] mAP@50-95={result.map_50_95:.4f} mAP@50={result.map_50:.4f} mAP@75={result.map_75:.4f}")
    model.train()
    return result


if __name__ == "__main__":
    main()

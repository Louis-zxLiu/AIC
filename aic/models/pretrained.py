"""加载官方 D-FINE COCO 检测预训练权重到三流模型。

为什么需要这个：`backbone.pretrained=True` 只加载 HGNetv2 的 **分类** 权重
（PPHGNetV2_B4_stage1.pth），HybridEncoder 和 DFINETransformer 是完全随机
初始化的。DETR 系的解码器（可学习 query + 迭代式框精修）从零训练需要极大
的数据量和 optimizer step 数，在 1600 张图上基本不可能收敛——这正是首轮实验
mAP 停在 1e-4 量级的主因之一。

`dfine_l_coco.pth`（COCO 54.0 mAP）的 encoder/decoder 配置与我们
configs/dfine_tristream.yml 里的 encoder/decoder 段逐字段一致，因此权重可以
直接对上。匹配策略沿用官方 `_matched_state`（third_party/D-FINE/src/solver/
_solver.py:277）的 key + shape 双重校验：形状不符的（12 类 vs 80 类的分类头、
以及 stem 因输入通道数不同而不同的部分）自动跳过，保持随机初始化。
"""

from __future__ import annotations

import torch


def _remap_backbone_keys(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """把官方 backbone 的 key 映射到 TriStreamHGNetv2 的属性名。

    官方 HGNetv2 里 stem 就叫 `stem`，我们把 RGB 分支的 stem 改名成 `stem_rgb`
    以便和 `stem_ir`/`stem_depth` 并列。`stages` 名字相同，无需处理。IR/Depth
    的 stem 不在这里赋值——它们由 `_init_stem_from_rgb` 从 RGB stem 的通道均值
    初始化（见 backbone.py），比拿 COCO 的 3 通道权重硬塞更合理。
    """
    out = {}
    for k, v in sd.items():
        if k.startswith("backbone.stem."):
            out[k.replace("backbone.stem.", "backbone.stem_rgb.", 1)] = v
        elif k.startswith("backbone.stages."):
            # Official D-FINE/DEIM checkpoints use one `stages` branch.  The
            # tri-stream wrapper names the RGB branch `stages_rgb`; map it so
            # the shared RGB/IR/Depth stages can inherit the detector weights.
            out[k.replace("backbone.stages.", "backbone.stages_rgb.", 1)] = v
        else:
            out[k] = v
    return out


def load_dfine_detector_weights(model: torch.nn.Module, path: str, verbose: bool = True) -> dict:
    """就地把 D-FINE COCO 检测权重加载进 model，返回匹配统计。

    只加载形状一致的参数，其余保持模型当前（随机 / 分类预训练）的初始化。
    """
    if path.startswith("http"):
        state = torch.hub.load_state_dict_from_url(path, map_location="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=False)

    if "ema" in state and state["ema"] is not None:
        src = state["ema"]["module"]
    elif "model" in state:
        src = state["model"]
    else:
        src = state
    src = {k.removeprefix("module."): v for k, v in src.items()}
    src = _remap_backbone_keys(src)

    cur = model.state_dict()
    matched, missed, unmatched = {}, [], []
    for k, v in cur.items():
        if k not in src:
            missed.append(k)
        elif v.shape == src[k].shape:
            matched[k] = src[k]
        else:
            unmatched.append(k)

    model.load_state_dict(matched, strict=False)

    stats = {
        "matched": len(matched),
        "missed": len(missed),
        "unmatched": len(unmatched),
        "missed_keys": missed,
        "unmatched_keys": unmatched,
    }
    if verbose:
        print(
            f"[pretrained] {path}: matched={stats['matched']} "
            f"missed={stats['missed']} unmatched(shape)={stats['unmatched']}"
        )
        # 逐模块汇报覆盖率——encoder/decoder 覆盖率低说明 key 映射出了问题，
        # 这类静默失败比报错更危险，所以显式打出来。
        for prefix in ("backbone", "opf", "encoder", "decoder"):
            total = sum(1 for k in cur if k.startswith(prefix + "."))
            got = sum(1 for k in matched if k.startswith(prefix + "."))
            if total:
                print(f"[pretrained]   {prefix}: {got}/{total} ({100.0 * got / total:.1f}%)")
        if unmatched:
            print(f"[pretrained]   shape-mismatch (保持随机初始化): {unmatched[:8]}")
    return stats

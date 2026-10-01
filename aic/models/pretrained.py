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

import os

import torch


# COCO category ids follow the official 80-class ordering.  The AIC labels
# that do not have a defensible COCO counterpart are intentionally omitted:
# their target rows stay at the model's fresh initialization.
COCO_AIC_CLASS_MAPPING = {
    0: (0,),                          # person <- person
    1: (8,),                          # boat <- boat
    2: (14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 77),  # animal
    3: (13, 56, 57),                 # seat <- bench/chair/couch
    5: (1,),                          # bicycle <- bicycle
    6: (2, 5, 7),                     # car <- car/bus/truck
    7: (32,),                         # ball <- sports ball
    8: (9,),                          # light <- traffic light
}

_CLASS_HEAD_KEYS = {
    "decoder.enc_score_head.weight",
    "decoder.enc_score_head.bias",
    "decoder.denoising_class_embed.weight",
}


def _is_class_head_key(key: str) -> bool:
    return key in _CLASS_HEAD_KEYS or key.startswith("decoder.dec_score_head.") and key.endswith(
        (".weight", ".bias")
    )


def _map_class_rows(
    target: torch.Tensor,
    source: torch.Tensor,
    mapping: dict[int, tuple[int, ...]],
    *,
    source_classes: int = 80,
    target_classes: int = 12,
) -> tuple[torch.Tensor | None, list[int]]:
    """Copy or average COCO rows into a target classifier tensor.

    The target clone is important: omitted AIC classes must retain the fresh
    initialization created by the target model.  The helper supports both a
    row-major classifier and a transposed classifier so a future D-FINE head
    layout cannot silently receive a wrong-axis mapping.
    """
    if target.ndim == 1 and source.ndim == 1:
        source_axis = target_axis = 0
    elif target.ndim == 2 and source.ndim == 2:
        source_rows = source_classes + 1 if source.shape[0] == source_classes + 1 else source_classes
        target_rows = target_classes + 1 if target.shape[0] == target_classes + 1 else target_classes
        if source.shape[0] == source_rows and target.shape[0] == target_rows:
            source_axis = target_axis = 0
        elif source.shape[1] in (source_classes, source_classes + 1) and target.shape[1] in (
            target_classes,
            target_classes + 1,
        ) and source.shape[1] - target.shape[1] == source_classes - target_classes:
            source_axis = target_axis = 1
        else:
            return None, []
    else:
        return None, []

    if source.shape[:source_axis] + source.shape[source_axis + 1 :] != target.shape[:target_axis] + target.shape[target_axis + 1 :]:
        return None, []

    # Denoising embeddings have one additional COCO/AIC padding row.  It is
    # deliberately left from `target`; the source padding row is not a class.
    result = target.detach().clone()
    mapped: list[int] = []
    for target_id, source_ids in mapping.items():
        if target_id >= target_classes or not source_ids:
            continue
        if any(source_id < 0 or source_id >= source_classes for source_id in source_ids):
            raise ValueError(f"invalid COCO source id in mapping for AIC class {target_id}: {source_ids}")
        values = source.index_select(source_axis, torch.as_tensor(source_ids, dtype=torch.long))
        value = values.mean(dim=source_axis)
        index = [slice(None)] * result.ndim
        index[target_axis] = target_id
        result[tuple(index)] = value
        mapped.append(target_id)
    return result, mapped


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


def load_dfine_detector_weights(
    model: torch.nn.Module,
    path: str,
    verbose: bool = True,
    class_mapping: dict | None = None,
    strict_body: bool = False,
) -> dict:
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
    if strict_body:
        # Shared stages are aliases in state_dict. Fill every alias explicitly
        # and initialize both auxiliary stems from the SAME detector stem.
        expanded = dict(src)
        cur_shapes = model.state_dict()
        for key, value in src.items():
            if key.startswith('backbone.stages_rgb.'):
                for name in ('stages_ir', 'stages_depth'):
                    expanded[key.replace('stages_rgb', name, 1)] = value
            if key.startswith('backbone.stem_rgb.'):
                for name in ('stem_ir', 'stem_depth'):
                    dest = key.replace('stem_rgb', name, 1)
                    if dest not in cur_shapes:
                        continue
                    target = cur_shapes[dest]
                    if value.shape == target.shape:
                        expanded[dest] = value
                    elif value.ndim == 4 and value.shape[1] == 3 and target.shape[0] == value.shape[0]:
                        expanded[dest] = value.mean(1, keepdim=True).repeat(1, target.shape[1], 1, 1)
        src = expanded

    cur = model.state_dict()
    mapping_enabled = bool(class_mapping and class_mapping.get("enabled", False))
    # The remote 703 deployment may retain an older model wrapper.  The
    # experiment launcher uses this isolated environment flag so the loader
    # can enable the same behavior without replacing that wrapper wholesale.
    mapping_enabled = mapping_enabled or os.environ.get("AIC_PRETRAINED_CLASS_MAPPING") == "1"
    if strict_body and mapping_enabled:
        raise ValueError('DINOv3 queue forbids semantic class mapping')
    semantic_mapping = COCO_AIC_CLASS_MAPPING if mapping_enabled else None
    matched, missed, unmatched = {}, [], []
    mapped_keys: list[str] = []
    for k, v in cur.items():
        if k not in src:
            missed.append(k)
        elif v.shape == src[k].shape:
            matched[k] = src[k]
        elif semantic_mapping is not None and _is_class_head_key(k):
            mapped, target_ids = _map_class_rows(v, src[k], semantic_mapping)
            if mapped is not None and target_ids:
                matched[k] = mapped
                mapped_keys.append(k)
            else:
                unmatched.append(k)
        else:
            unmatched.append(k)

    if strict_body:
        allowed = lambda k: (_is_class_head_key(k) or k.startswith('opf.') or '.query_region_' in k
                              or k.endswith('.num_batches_tracked'))
        unexpected = [k for k in missed + unmatched if not allowed(k)]
        if unexpected:
            raise RuntimeError(f'unexpected pretrained detector omissions: {unexpected}')
    model.load_state_dict(matched, strict=False)

    stats = {
        "matched": len(matched),
        "missed": len(missed),
        "unmatched": len(unmatched),
        "missed_keys": missed,
        "unmatched_keys": unmatched,
        "semantic_mapped_keys": mapped_keys,
    }
    if strict_body:
        by_module = {}
        for name, parameter in model.named_parameters():
            prefix = name.split('.')[0]
            record = by_module.setdefault(prefix, {'parameters': 0, 'loaded_parameters': 0})
            record['parameters'] += parameter.numel()
            if name in matched:
                record['loaded_parameters'] += parameter.numel()
        stats['unique_parameter_coverage'] = by_module
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
        if semantic_mapping is not None:
            print(
                f"[pretrained]   semantic class transfer: {len(mapped_keys)} tensors; "
                "unmapped AIC classes remain fresh"
            )
    return stats

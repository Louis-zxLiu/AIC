"""
Elastic Resolution 快速验证脚本

验证"不同图用不同分辨率"是否能提升小目标 AP 和 AP75/AP90
不实现完整 Elastic-DETR，用启发式规则模拟

用法：
python aic/tools/validate_elastic_resolution.py \
    --config configs/dfine_tristream_hires.yml \
    --checkpoint runs/run3_deim_mal/epoch_best.pt \
    --data-root data/ \
    --output elastic_validation.json
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# TODO: 根据实际项目结构调整 import
# from aic.data.dataset import MultiModalDataset
# from aic.models.tristream_dfine import TriStreamDFINE
# from aic.tools.infer import load_model, preprocess_image, postprocess_predictions


def estimate_small_object_density(img_path, coco_gt=None, img_id=None):
    """
    估算图像中小目标的密度（用于分辨率选择）

    方案 1：如果有 GT 标注，直接统计小目标数量
    方案 2：如果无 GT（测试集），用边缘密度 + 纹理复杂度启发式估计
    """
    if coco_gt and img_id:
        # 方案 1：基于 GT 标注（仅用于 val 集验证）
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)

        small_count = 0
        for ann in anns:
            area = ann['area']
            if area < 32 * 32:  # COCO small object 定义
                small_count += 1

        return small_count

    else:
        # 方案 2：启发式估计（用于测试集）
        # TODO: 实现基于图像内容的小目标密度估计
        # 可用边缘检测、纹理分析、或轻量预测网络
        raise NotImplementedError("测试集的小目标密度估计需要实现")


def select_resolution_rule_based(small_object_density, strategy='adaptive'):
    """
    基于规则的分辨率选择

    strategy:
        'adaptive': 根据小目标密度自适应选择
        'fixed_high': 全部用高分辨率（对照组）
        'fixed_low': 全部用低分辨率（对照组）
    """
    if strategy == 'fixed_high':
        return 960
    elif strategy == 'fixed_low':
        return 640
    elif strategy == 'adaptive':
        # 自适应规则（可调整阈值）
        if small_object_density >= 5:
            return 960  # 小目标很多 → 最高分辨率
        elif small_object_density >= 2:
            return 896  # 中等小目标 → 高分辨率
        elif small_object_density >= 1:
            return 768  # 少量小目标 → 中分辨率
        else:
            return 640  # 无小目标或都是大目标 → 低分辨率
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def run_inference_multi_resolution(
    model,
    dataloader,
    coco_gt,
    strategy='adaptive',
    device='cuda'
):
    """
    多分辨率推理并收集结果
    """
    model.eval()
    all_predictions = []
    resolution_stats = {640: 0, 768: 0, 896: 0, 960: 0}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Inference ({strategy})"):
            img_id = batch['img_id']
            img_path = batch['img_path']

            # 估算小目标密度
            small_density = estimate_small_object_density(
                img_path, coco_gt, img_id
            )

            # 选择分辨率
            resolution = select_resolution_rule_based(small_density, strategy)
            resolution_stats[resolution] += 1

            # TODO: 根据实际模型接口调整
            # 动态调整输入分辨率
            # img_resized = resize_image(batch['image'], target_size=resolution)
            # outputs = model(img_resized, batch['ir'], batch['depth'])
            # predictions = postprocess_predictions(outputs, original_size=batch['original_size'])

            # all_predictions.extend(predictions)

            pass  # 替换为实际推理代码

    return all_predictions, resolution_stats


def evaluate_coco_metrics(predictions, coco_gt, annotation_file):
    """
    COCO 评估，返回详细指标
    """
    # 保存预测结果为 COCO 格式
    pred_file = 'temp_predictions.json'
    with open(pred_file, 'w') as f:
        json.dump(predictions, f)

    # COCO 评估
    coco_dt = coco_gt.loadRes(pred_file)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # 提取详细指标
    metrics = {
        'AP': coco_eval.stats[0],
        'AP50': coco_eval.stats[1],
        'AP75': coco_eval.stats[2],
        'AP_small': coco_eval.stats[3],
        'AP_medium': coco_eval.stats[4],
        'AP_large': coco_eval.stats[5],
        'AR_1': coco_eval.stats[6],
        'AR_10': coco_eval.stats[7],
        'AR_100': coco_eval.stats[8],
        'AR_small': coco_eval.stats[9],
        'AR_medium': coco_eval.stats[10],
        'AR_large': coco_eval.stats[11],
    }

    Path(pred_file).unlink()  # 清理临时文件
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data-root', type=str, default='data/')
    parser.add_argument('--annotation', type=str, default='data/annotations/instances_val.json')
    parser.add_argument('--output', type=str, default='elastic_validation.json')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    print("=" * 80)
    print("Elastic Resolution 快速验证")
    print("=" * 80)

    # 加载模型
    print(f"\n[1/4] 加载模型: {args.checkpoint}")
    # TODO: 实现模型加载
    # model = load_model(args.config, args.checkpoint, args.device)
    model = None

    # 加载数据集
    print(f"\n[2/4] 加载数据集: {args.annotation}")
    coco_gt = COCO(args.annotation)
    # TODO: 实现 dataloader
    # dataloader = create_dataloader(args.data_root, args.annotation)
    dataloader = []

    # 运行三种策略的推理
    strategies = ['fixed_low', 'adaptive', 'fixed_high']
    results = {}

    for strategy in strategies:
        print(f"\n[3/4] 运行推理: {strategy}")
        predictions, res_stats = run_inference_multi_resolution(
            model, dataloader, coco_gt, strategy, args.device
        )

        print(f"    分辨率分布: {res_stats}")
        print(f"    平均分辨率: {sum(k*v for k,v in res_stats.items()) / sum(res_stats.values()):.1f}")

        print(f"\n[4/4] 评估指标: {strategy}")
        metrics = evaluate_coco_metrics(predictions, coco_gt, args.annotation)

        results[strategy] = {
            'metrics': metrics,
            'resolution_stats': res_stats,
            'avg_resolution': sum(k*v for k,v in res_stats.items()) / sum(res_stats.values())
        }

        print(f"    AP: {metrics['AP']:.4f}")
        print(f"    AP50: {metrics['AP50']:.4f}")
        print(f"    AP75: {metrics['AP75']:.4f}")
        print(f"    AP_small: {metrics['AP_small']:.4f}")

    # 对比分析
    print("\n" + "=" * 80)
    print("对比分析")
    print("=" * 80)

    baseline = results['fixed_low']
    adaptive = results['adaptive']
    high = results['fixed_high']

    print(f"\n{'Metric':<15} {'640固定':<12} {'自适应':<12} {'960固定':<12} {'增益(自适应-640)':<20}")
    print("-" * 80)

    for metric_name in ['AP', 'AP50', 'AP75', 'AP_small']:
        base_val = baseline['metrics'][metric_name]
        adap_val = adaptive['metrics'][metric_name]
        high_val = high['metrics'][metric_name]
        gain = adap_val - base_val

        print(f"{metric_name:<15} {base_val:<12.4f} {adap_val:<12.4f} {high_val:<12.4f} {gain:+.4f} ({gain/base_val*100:+.2f}%)")

    print(f"\n平均分辨率:")
    print(f"  640固定: 640")
    print(f"  自适应: {adaptive['avg_resolution']:.1f}")
    print(f"  960固定: 960")

    # 保存结果
    print(f"\n保存结果到: {args.output}")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    # 结论
    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)

    if adaptive['metrics']['AP_small'] > baseline['metrics']['AP_small'] + 0.01:
        print("✅ 自适应分辨率对小目标有显著提升！")
        print("   建议继续实施完整 Elastic-DETR 方案")
    elif adaptive['metrics']['AP'] > baseline['metrics']['AP'] + 0.005:
        print("✅ 自适应分辨率整体有提升，但小目标增益不显著")
        print("   建议优化分辨率选择规则，或增加分辨率档位")
    else:
        print("❌ 自适应分辨率无明显收益")
        print("   可能原因：(1) 分辨率选择规则不合理")
        print("           (2) 当前分辨率已经够用")
        print("           (3) 瓶颈在其他地方（损失、架构、数据）")
        print("   建议回到诊断实验，定位真正的瓶颈")


if __name__ == '__main__':
    main()

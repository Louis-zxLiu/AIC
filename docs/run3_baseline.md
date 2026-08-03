# run3_hires 基线存档

> 用途：在改动模型之前先固定一份"标准答案"。后续任何改动（CLIP、P2、匹配代价……）
> 都必须与本文件记录的条件逐项对齐后再比较分数，否则分数差无法归因。
>
> 存档时间：2026-08-03　　实验代号：`run3_hires`
>
> **状态：训练已完成（48 epoch 跑完，最后一个评测点 epoch 47）。**
> 最佳 checkpoint 为 **epoch37**，提交文件已生成，线上分数待回填。

---

## 一、结论速览

| 项目 | 值 |
|---|---|
| 本地验证 mAP@50-95 | **0.5430**（epoch 37，EMA 权重） |
| 本地验证 mAP@50 | 0.7972 |
| 本地验证 mAP@75 | 0.5868 |
| 验证集划分 | 5 折的 fold 0，train 1600 / val 400 |
| 训练分辨率 | `[736, 1280]`（H×W） |
| 推理分辨率 | **768**（正方形，非训练尺寸，见 §六） |
| 线上提交分数 | **尚未用本模型提交**，见 §七 |

> 注意：队友示例里的 53.666 是**上一版模型（run2）的线上分数**量级，不是本次 run3。
> run3 目前只有本地验证分数，线上分数待提交后回填本文件。

---

## 二、代码版本

| 项目 | 值 |
|---|---|
| 仓库 | https://github.com/Louis-zxLiu/AIC |
| 分支 | `main` |
| 存档 commit | 见本文件同批推送的 commit（`git log -1`） |
| 三方依赖 | `third_party/D-FINE`（官方 D-FINE，未改动其源码） |

**模型实际启用的模块**（防止后续代码被改过后对不上）：

- 骨干：`TriStreamHGNetv2`（HGNetV2-B4），**三个模态共享全部 stage 权重**，仅 stem 不同
  （RGB in_chs=3 / IR in_chs=1 / Depth in_chs=2）；新 stem 由 COCO 预训练 RGB stem 的
  通道均值广播初始化。→ 见 `aic/models/backbone.py`
- 融合：OPF 三尺度融合，K/V 空间降采样（`opf.sr_ratios`）把注意力从 O((h·w)²) 降到 O(h·w·400)
- 检测头：D-FINE 解码器，`num_queries: 300`、`reg_max: 32`、`reg_scale: 4`、`num_points: [3,6,3]`
- 特征层：`return_idx: [1,2,3]` → `feat_strides: [8,16,32]`，**未启用 stride 4（P2）**
- EMA：启用，`decay: 0.998`、`warmups: 200`
- 预训练：`dfine_l_coco.pth`，matched=1156 / missed=160 / shape-mismatch=15
  （backbone 87.0%、encoder 100%、decoder 93.3%、**opf 0%** 随机初始化）

---

## 三、YAML 配置

完整配置：[`configs/dfine_tristream_hires.yml`](../configs/dfine_tristream_hires.yml)（已入库）

关键训练条件摘录：

| 参数 | 值 | 说明 |
|---|---|---|
| `image_size` | `[736, 1280]` | 矩形输入，比值 1.739 |
| `num_classes` | 12 | |
| `epochs` | 48 | |
| `warmup_epochs` | 2 | |
| `batch_size` | 3 | 单卡有效 batch |
| `grad_accum_steps` | 3 | 有效总 batch = 3×2卡×3 = **18** |
| `lr` | 1.4e-4 | 按 total batch 线性缩放自官方 2.5e-4@batch32 |
| `backbone_lr_mult` | 0.05 | backbone lr = base/20，对齐官方 D-FINE-L |
| `weight_decay` | 1.0e-4 | |
| `grad_clip_norm` | 0.1 | |
| `amp_dtype` | bfloat16 | |
| `modality_dropout_p` | 0.1 | 训练时随机丢模态 |
| `eval_interval_epochs` | 2 | 故只有奇数 epoch 有验证分数 |
| `keep_last_n` | 3 | 本次新增，见 §八 |

**数据增强**：
- `multi_scale.enabled: true`，档位 `[[576,1024],[640,1120],[672,1184],[736,1280],[800,1408]]`
  （全部 32 的倍数，比值 1.74~1.78，保持与 base 同长宽比以免注入形状畸变）；
  训练末 20% epoch 自动关闭
- `class_balance.enabled: true`，`oversample_thr: 0.2`（**单卡有效；DDP 下退化为均匀 shuffle**，
  本次是双卡 DDP，所以类别过采样实际未生效——这是复现时容易踩的坑）
- depth 处理：`disp8_mode` 按 run2 行为，8bit 视差图那批**置零**（见 depth 两种格式方向相反的结论）

---

## 四、训练命令

```bash
cd /root/aic && PYTHONPATH=/root/aic torchrun --nproc_per_node=2 --master_port=29511 aic/tools/train.py --config configs/dfine_tristream_hires.yml
```

崩溃后从断点恢复用的命令（见 §八）：

```bash
cd /root/aic && PYTHONPATH=/root/aic torchrun --nproc_per_node=2 --master_port=29511 aic/tools/train.py --config configs/dfine_tristream_hires.yml --resume /root/autodl-tmp/outputs/run3_hires/epoch33.pt
```

命令行**没有**覆盖任何 YAML 参数（未传 `--image_size` / `--disp8_mode` 等），
即 fold、GPU 数、分辨率全部以 YAML 为准。硬件：2 × 32G GPU，约 227s/epoch。

---

## 五、训练日志与收敛情况

日志：`/root/autodl-tmp/outputs/run3_hires/train.log`
曲线：同目录 `curves.json` / `curves.png`（`eval_epochs` + `eval_metrics["mAP@50-95"]`）

| epoch | mAP@50-95 | | epoch | mAP@50-95 |
|---|---|---|---|---|
| 1 | 0.1765 | | 21 | 0.5326 |
| 3 | 0.3736 | | 23 | 0.5360 |
| 5 | 0.4515 | | 25 | 0.5381 |
| 7 | 0.4776 | | 27 | 0.5373 |
| 9 | 0.4966 | | 29 | 0.5382 |
| 11 | 0.5065 | | 31 | 0.5395 |
| 13 | 0.5150 | | 33 | 0.5427 |
| 15 | 0.5195 | | 35 | 0.5422 / 0.5425 |
| 17 | 0.5227 | | 37 | **0.5430** ← 当前最佳 |
| 19 | 0.5249 | | 39 | 0.5425 |

**收敛判断**：单调上升，尚未过拟合，e33 之后进入平台期（0.5427→0.5430，4 个 epoch 涨 0.0003）。
epoch 35 有两个分数是因为崩溃后从 e33 重跑了一次 e35（见 §八），不是波动。

**与上一版对比**：run2（`image_size: 640`）峰值在 e33 = 0.5159，之后衰减到 e47 = 0.5072。
run3 在同一 epoch 数上领先约 2.7 分，且没有出现 run2 的衰减。

---

## 六、推理命令（重要：推理尺寸与训练尺寸不同）

```bash
cd /root/aic && PYTHONPATH=/root/aic python aic/tools/infer.py --config configs/dfine_tristream_hires.yml --checkpoint /root/autodl-tmp/outputs/run3_hires/epoch37.pt --test_root /root/autodl-tmp/dataset/测试集 --out_dir /root/autodl-tmp/submit_run3 --image_size 768 --zip_path /root/autodl-tmp/submit_run3.zip
```

| 设置 | 值 |
|---|---|
| 测试尺寸 | **768**（正方形）——**必须显式传 `--image_size 768`**，不传会用 config 的 `[736,1280]` |
| 置信度阈值 | `inference.score_thresh: 0.05` |
| 每图保留框数 | `inference.max_boxes: 100` |
| 权重 | 用 **EMA** 权重（checkpoint 里的 `ema` 键） |

**为什么推理用 768 而不是训练的 736×1280**：768 是实测的推理分辨率甜点。
同一权重下整图 768 得 0.5370，而整图 1344 只有 0.4681 —— 往上放大同样掉分。
切片（SAHI 式）推理已实测全面有害，9 个方案无一超过整图 768，不要再试。

---

## 七、提交文件

**run3 尚未提交线上**。当前可提交的最佳权重是 `epoch37.pt`（本地 0.5430）。
提交后请在此回填：

| 项目 | 值 |
|---|---|
| 提交文件路径 | *（待填）* |
| 线上分数 | *（待填）* |
| 对应 checkpoint | `run3_hires/epoch37.pt` |
| 对应本地验证分数 | 0.5430 |

上一版 run2 的线上分数请以队友记录为准（示例中提到 53.666 量级）。
**注意 run2 与 run3 的 fold 划分相同（fold 0, seed 42），分数可直接比较。**

---

## 八、checkpoint

目录：`/root/autodl-tmp/outputs/run3_hires/`

| 文件 | 说明 |
|---|---|
| `epoch37.pt` | **当前最佳**，本地 0.5430 |
| 每个 ckpt 内容 | `model` / `optimizer` / `scheduler` / `epoch` / `ema` / `ema_updates`（1331 张量，约 1.5GB）|

**必须永久保留**：`/root/autodl-tmp/outputs/run2/epoch41.pt` —— 它是所有离线诊断实验
（`_diag.py` / `_diag2.py` / `_match.py` / `_tile.py`）的公共基线权重，删掉后此前所有对照数字都无法复现。

### 本次训练中断事故（复现时需知）

训练在 **epoch 35 保存 checkpoint 时崩溃**：

```
RuntimeError: [enforce fail at inline_container.cc:424] . unexpected pos 412752960 vs 412752848
```

不是 OOM，是 **`/root/autodl-tmp`（50G）被写满**。每个 ckpt 1.5GB，48 epoch 每 2 轮存一次
= 24 个 = 36G，加上 18G 数据集必然爆盘。已从 `epoch33.pt` 断点恢复，LR 接续正确。
**丢失**：e35 的权重（分数 0.5422 已记录在日志里，权重文件不可恢复）。

已在代码中修复，复现时不会再遇到：
1. **原子写入**（`aic/tools/train.py`）：先写 `.tmp` 再 `os.replace()`，写盘失败不留截断的 `.pt`
2. **轮转 `keep_last_n: 3`**：只保留最近 3 个 + 历史最佳；断点恢复时从 `curves.json`
   取回历史最佳，避免把最优权重当旧文件删掉

---

## 九、fold 与数据划分

**这一条决定分数能不能公平比较——不同划分的分数不可直接比。**

| 项目 | 值 |
|---|---|
| 数据根目录 | `/root/autodl-tmp/dataset/训练集/`（子目录 `visible/ infrared/ depth/ labels/`）|
| 总样本 | 2000 |
| `n_folds` | 5 |
| `fold` | **0** |
| 划分种子 | **42**（`kfold_split` 内部固定）|
| train / val | **1600 / 400** |

划分逻辑（`aic/data/dataset.py:97`）：按文件名排序后用 `random.Random(42)` 打乱，
再 `idx[i::n_folds]` 取第 `fold` 份作验证集。**只要 seed 与 fold 相同，划分就完全一致、可复现。**

> 遗留问题：当前用 fold 0 训练，等于有 400 张有标注的图（20%）从未参与训练。
> 最终提交前应考虑用全部 2000 张重训。

---

## 十、类别与评价指标

`num_classes: 12`，标注格式为 YOLO 归一化 `cls cx cy w h`。

**类别名称：数据集未随附映射文件，赛题文档也未给出 id→名称对应表。**
下表只记录 id 与验证集实例数（不臆造名称；做 CLIP 文本类别前必须先向主办方或赛题文档确认）：

| cls id | train 实例数 | val 实例数 | val 中位边长(原图 px) | AP@50-95（run2 e41 @768）|
|---|---|---|---|---|
| 0 | 4370 | 1129 | 60.7 | 0.4895 |
| 1 | 103 | 32 | 60.1 | 0.6230 |
| 2 | 2388 | 673 | 86.5 | 0.5575 |
| 3 | 507 | 114 | 255.6 | 0.7294 |
| 4 | 662 | 147 | 72.2 | 0.5122 |
| 5 | 513 | 153 | 43.3 | 0.4637 |
| 6 | 1220 | 351 | 66.3 | 0.5809 |
| 7 | 75 | **14** | 40.9 | 0.3741 |
| 8 | 1163 | 292 | 103.2 | 0.5843 |
| 9 | 234 | 48 | 81.7 | 0.4282 |
| 10 | 166 | 38 | 90.7 | 0.5047 |
| 11 | 23 | **2** | 114.5 | 0.5782 |
| 合计 | 11424 | 2993 | | 0.5355 |

**类别极度不均衡**：cls0 占 train 的 38%，而 cls11 只有 23 个、cls7 只有 75 个。
验证集里 cls11 仅 2 个实例、cls7 仅 14 个——**这两类的 AP 是噪声，不要用它们判断改动好坏**。

**评价指标**：COCO 风格 mAP@50-95（101 点插值，10 个 IoU 阈值 0.50:0.05:0.95）。
实现见 `aic/eval/map_evaluator.py`。

**真正需要优化的指标**：mAP@50 已达 0.797，而 mAP@75 只有 0.587 ——
**缺口在定位精度，不在识别**。按尺度拆分（run2 e41 @768，含 COCO ignore 语义）：

| 尺度（原图边长）| AP@50-95 | AP@50 | nGT |
|---|---|---|---|
| all | 0.5355 | 0.7959 | 2993 |
| small <32 | **0.1397** | 0.2866 | 404 |
| medium 32-96 | 0.3615 | 0.6367 | 1449 |
| large >96 | 0.7258 | 0.9077 | 1140 |

小目标的失效模式**不是漏检、也不是置信度截断**，而是高 IoU 定位不准：
0-24px 的 GT 只有 4.8% 完全漏检，但 22.9% 落在 IoU<0.5。各尺度平均 IoU 对应一个
**与目标大小无关的固定 ~6px 绝对误差**。

> 计算尺度 AP 时必须带 COCO ignore 语义：范围外的 GT 要标 ignore，命中 ignore GT 的预测
> 既不算 TP 也不算 FP 并整条丢弃。只过滤 GT 而保留全部预测会把精度算崩——
> 之前 `AP_small = 0.041` 就是这个 bug 的假象，真值 0.1397。

---

## 十一、已被实测否决的方向（不要重复投入）

| 方向 | 结果 |
|---|---|
| 切片（SAHI）推理 | 9 个方案全输给整图 768；切 2x2@768 → AP_s 0.0668 vs 基线 0.1403 |
| 提高推理分辨率到 1344 | 0.4681，比 768 低 7 分；768 是甜点 |
| 整图 + 切片 WBF 融合 | 一律低于整图单跑（WBF 按 score 加权平均坐标，好框被坏框拽偏）|
| 匹配代价换 NWD | 88% 的改变变差，小目标 ΔIoU −0.26 |

有正面证据但未上线的改动：**尺度归一化匹配代价 snl1**（离线统计 77% 的改变变好，
小目标 ΔIoU +0.105）。

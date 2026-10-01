# 城市场景多模态目标检测 苏大一队

本仓库当前版本只保留复赛线上最高分 **58.903** 对应的 RF-DETR RGB+IR+Depth 方案。分数由队伍在竞赛平台确认，不能等同于独立验证集 mAP。

最高分运行：`full_student_w050_ir_off_depth_x2`。`ir_off` 表示 **IR 持续蒸馏权重为零**；IR backbone、IR 教师初始化和 IR 特征融合仍然保留。

| 项目 | 固定设置 |
| --- | --- |
| 检测器 | RF-DETR 2XL，12 类，RGB+IR+Depth 三分支 |
| 学生结构 | 880×1600；窗口 RGB/IR/Depth = 2/2/4；第 3 层输出；300 queries；OPF 1 层 |
| 教师 | IR 全量训练 38 epoch，EMA epoch37；Depth 全量训练 49 epoch，EMA epoch48 |
| 蒸馏 | 全局 0.5，IR 0，Depth 2；8 epoch 线性预热；稳定阶段有效 Depth 系数 1.0 |
| 学生训练 | 2000 张全量，64 epoch，seed42；双卡，每卡 batch1，梯度累积4 |
| 提交 | epoch63 EMA；复赛 1000 张；置信度 ≥0.25；每图最多100框；无 TTA |
| 最高分提交包 | 1000 个 TXT，共7469框；SHA256 见 `docs/provenance.json` |

原生 RF-DETR 仍注册并计算 5 个 decoder 层；配置 D3 选择第3层预测和前2层辅助输出，不代表物理裁剪为3层。通用 YAML 中的 `backbone/encoder/decoder` 字段保留兼容共享训练器，实际 RF 结构由 `aic/rf_nas/model.py` 和 `nas.force_arch` 控制。

## 安装

Linux、NVIDIA GPU 和 Python 3.10。`requirements.txt` 固定了原训练环境的包版本；PyTorch/CUDA 须与驱动匹配。RF-DETR 2XL 使用 `rfdetr-plus`，使用前请阅读其 Platform Model License。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

预训练文件 `rf-detr-xxlarge.pth` 来自 RF-DETR 官方模型仓库，原文件 MD5 为 `e3204689c1f0280427e4c33e6a2ac6cd`。本仓库不包含数据、预训练文件或竞赛 checkpoint。

## 权重与数据

所有竞赛 checkpoint **只存放本地，不上传 GitHub**。本地归档路径为 `D:/AIC/checkpoints/best_58903`。`docs/checkpoints.json` 记录完整学生、精简 EMA 学生与两个教师的 SHA256。EMA 文件保留原始精度，全部1106个张量与完整 checkpoint 的 EMA 逐一相等；完整文件用于断点恢复。

推理时在 `weights/` 放入 `rf-detr-xxlarge.pth`、`student_epoch63_ema.pt`、`teacher_ir_epoch37.pt` 和 `teacher_depth_epoch48.pt`，或用 `--weights-dir` 指定权重目录。教师用于构建学生初始化，最终学生参数由 checkpoint 完整覆盖。

数据根目录应维持竞赛原有 RGB、IR、Depth 目录及标注布局；扫描与配对逻辑见 `aic/data/dataset.py`。三模态按同名样本配对，训练需要标签，复赛推理不需要标签。

```text
train/                 phase2/
  visible/               visible/
  infrared/              infrared/
  depth/                 depth/
  labels/
```

## 重现训练与推理

只保留三个正式配置：`teacher_ir.yml`、`teacher_depth.yml`、`student_best.yml`。训练顺序为 IR 教师、Depth 教师、学生。将训练得到的教师 epoch37/48 分别复制为权重目录里的上述名称，再训练学生。

```bash
python tools/run_best.py train --stage teacher_ir --data-root /path/to/train --gpus 2,3
python tools/run_best.py train --stage teacher_depth --data-root /path/to/train --gpus 2,3
python tools/run_best.py train --stage student_best --data-root /path/to/train --gpus 2,3
python tools/run_best.py infer --test-root /path/to/phase2 --gpus 0 --output-dir outputs/phase2
```

推理入口固定阈值0.25、最多100框，优先加载 EMA。输出 `predictions/` 和 `submission_epoch63_conf025.zip`。训练入口保存解析后的绝对路径配置，便于复核。

## 依据与报告

- [技术报告](docs/technical_report.docx)：算法、训练协议、过程进度及团队分工。
- `docs/evidence/`：最高分运行的训练日志、推理日志和曲线。
- `docs/provenance.json`：最高分配置与提交包的来源校验。
- `docs/reproduction_check.json`：整理后代码与最高分提交的1000个TXT逐字节一致。
- `docs/checkpoints.json`：本地权重校验清单。
- `docs/environment.original.txt`：原环境版本记录。

当前主分支去除了旧实验配置和自动化队列脚本；历史提交仍保留。共享训练器中保留运行所需的 D-FINE 数据/损失/后处理兼容组件，不表示最高分检测 backbone 是 D-FINE。第三方来源和许可见 [NOTICE](NOTICE.md)。

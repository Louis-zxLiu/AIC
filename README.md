# 城市场景RGB IR Depth三模态目标检测

苏大一队参加AIC2026算法挑战赛的RF-DETR三模态实现。当前版本对应**半决赛线上最高分56.132**：HBS＋正常学习率＋RGB低照增强，使用全量运行的**epoch47 EMA＋原始G6推理**。

团队自编代码采用[MIT许可](LICENSE)；第三方组件及RF-DETR2XL保留各自许可，见[NOTICE](NOTICE.md)。本仓库不包含数据、模型权重、训练环境或结果ZIP。

## 下载模型与材料

**[百度网盘下载](https://pan.baidu.com/s/1N9V1Nkt5HoqH2QnzQSB9HA?pwd=sunh)**，提取码：**sunh**。

从材料根目录`AIC-2026-19640904-苏大一队-半决赛-刘志轩`下载
`AIC-2026-19640904-模型文件/student_epoch47_ema.pt`，放入仓库`weights/`，或用`--checkpoint`指定。

最佳学生权重SHA256：

```text
75679ae9097d3d73028cdb0ef9761d2ab3e4b8057b9e360316c432454ca2959e
```

推理只需要这个学生EMA，**不需要教师、官方预训练或训练标签**。[权重和下载说明](docs/WEIGHTS.md)。

## 模型与结果

| 项目 | 配置 |
|---|---|
| 基座 | RF-DETR2XL，RGB/IR/Depth三条独立特征分支 |
| 固定结构 | RGB W4、IR W2、Depth W4、D4、Q300、OPF1 |
| 输入 | 880×1600，三模态同步几何变换 |
| 教师 | IR38轮，EMA epoch37；Depth49轮，EMA epoch48 |
| 持续蒸馏 | 全局0.5、IR0、Depth2；8轮预热 |
| 学生训练 | 2000张全量、seed42、64轮；每卡batch4×2卡、累积1 |
| 学习率 | AdamW，基础1e-4，warmup2轮，flat-cosine平坦比例0.33 |
| 增强 | RFS、同步尺度裁剪、模态丢弃、RGB低照；HBS两视图检测损失各0.5 |
| 最终权重 | epoch47 EMA，即完成第48轮；1110个EMA状态张量 |
| 推理 | 原始G6、FP32、score≥0.25、全图max100、5位小数 |
| 半决赛结果 | 56.132；1000个TXT，共16286框 |

IR系数0仅关闭IR持续蒸馏，IR教师初始化、输入、backbone和融合仍保留。D4通过原生5层decoder的第4层输出及对应辅助监督实现，完整参数保留。

56.132是队伍确认的线上成绩，不是独立验证集分数。最终版本没有IR局部采样、翻转、C1或额外高分辨率推理。[结果记录](docs/RESULTS.md)、[PDF技术报告](docs/technical_report.pdf)。

## 安装

参考环境为Linux、Python3.10和NVIDIA CUDA GPU。requirements.txt固定归档环境版本，PyTorch/torchvision的CUDA构建需与GPU驱动匹配。RF-DETR2XL使用rfdetr-plus，不能替换为普通RF-DETR配置。

```bash
git clone https://github.com/Louis-zxLiu/AIC.git
cd AIC
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 数据预处理

使用赛题授权数据，按同名stem配对：

```text
data/train/              /path/to/phase3/
  visible/                 visible/
  infrared/                infrared/
  depth/                   depth/
  labels/
```

训练标签每行`class_id cx cy w h`，坐标归一化、类别0–11，背景图使用空TXT。材料网盘内的2000份训练标签已与最高分运行完整核对，不重复上传GitHub。

16位毫米深度与8位近远灰度分别转换，保留有效性、格式和质量，形成5通道；8位深度不重复取距离倒数。预处理在加载时执行，代码见aic/data/。

```bash
python tools/check_data.py --root /path/to/train --training
python tools/check_data.py --root /path/to/phase3
```

## 归档学生推理

```bash
mkdir -p weights
# 将网盘中的student_epoch47_ema.pt放入weights/
python tools/check_artifacts.py --checkpoint weights/student_epoch47_ema.pt
python tools/run_best.py infer --test-root /path/to/phase3 --gpus 0 --output-dir outputs/phase3
python -m aic.eval.submission_validator --pred_dir outputs/phase3/predictions --test_root /path/to/phase3
```

默认按仓库自身位置读取weights/student_epoch47_ema.pt，也可显式指定：

```bash
python tools/run_best.py infer --test-root /path/to/phase3 --gpus 0 --checkpoint /path/to/student_epoch47_ema.pt --output-dir outputs/phase3
```

输出predictions/、submission_epoch47_g6_conf025.zip及complete.json。HBS在推理时旁路，教师和官方预训练加载均跳过，严格加载完整学生EMA后才预测。

G6包含整图与重叠0.2的四个2×2切片。各视图使用同一权重和分辨率，过滤切片内部边缘1.5像素处的截断框；同类IoU0.6一对一匹配，分数加权坐标，簇分数取最大值。仅被切片支持的簇按`0.25+(score-0.25)*0.5`降分，最后执行0.25阈值与全图100框上限。不进行不同模型融合。

## 重新训练

需要官方预训练及两个教师，均保存在产物目录，默认training_weights/。官方下载地址与哈希见[docs/checkpoints.json](docs/checkpoints.json)。

```bash
mkdir -p training_weights
# 将官方rf-detr-xxlarge.pth放入training_weights/
python tools/run_best.py train --stage teacher_ir --data-root /path/to/train --gpus 0,1
python tools/run_best.py train --stage teacher_depth --data-root /path/to/train --gpus 0,1
cp outputs/teacher_ir/epoch37.pt training_weights/teacher_ir_epoch37.pt
cp outputs/teacher_depth/epoch48.pt training_weights/teacher_depth_epoch48.pt
python tools/run_best.py train --stage student --data-root /path/to/train --gpus 0,1
python tools/run_best.py infer --test-root /path/to/phase3 --gpus 0 --checkpoint outputs/student/epoch47.pt --output-dir outputs/retrained_phase3
```

正式学生训练64轮，使用其中epoch47 EMA，保持完整64轮学习率周期。复核本结果时保持结构、配方和推理口径一致。[复现范围与限制](docs/REPRODUCIBILITY.md)。

## 目录和检查

```text
aic/data/                 模态读取、深度语义与同步增强
aic/rf_native/            RF初始化、教师与蒸馏
aic/rf_nas/               固定结构、OPF与HBS
aic/best56132/            便携训练和原始G6推理
configs/                  学生及两个教师配置
tools/                    训练推理、输入与产物检查
third_party/D-FINE/       共享兼容组件及原许可证
docs/                     权重身份、复现记录和技术报告
tests/                    不依赖GPU的回归检查
```

```bash
python -m unittest discover -s tests -v
python tools/check_artifacts.py
python tools/check_artifacts.py --submission outputs/phase3/submission_epoch47_g6_conf025.zip --exact
```

`--exact`比较TXT内容，不受ZIP时间戳或压缩方式影响。跨环境GPU预测可能产生数值差异，重新训练也不保证逐位同分。归档核验确认1110个EMA张量一致，1000张缓存的G6后处理与原提交逐文件、逐字节一致；整理期间未重启GPU训练或完整GPU推理。

## 许可与参考

自编代码MIT不改变第三方模型及数据许可，详见[NOTICE](NOTICE.md)。主要参考：[RF-DETR](https://arxiv.org/abs/2511.09554)、[RDTTrack](https://arxiv.org/abs/2509.24741)、[M²D-LIF](https://arxiv.org/abs/2503.11780)、[SET](https://github.com/HuixinSun/SET)。

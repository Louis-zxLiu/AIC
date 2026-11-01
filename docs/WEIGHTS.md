# 权重下载与身份校验

[百度网盘作品材料](https://pan.baidu.com/s/1N9V1Nkt5HoqH2QnzQSB9HA?pwd=sunh)，提取码：**sunh**。链接由队伍提供；模型文件不存储在GitHub。

## 推理所需文件

在网盘材料中找到：

```text
AIC-2026-19640904-苏大一队-半决赛-刘志轩/
  AIC-2026-19640904-模型文件/
    student_epoch47_ema.pt
```

| 身份 | 数值 |
|---|---|
| 对应成绩 | 半决赛线上56.132 |
| 训练运行 | RF W424/D4＋HBS＋正常学习率＋RGB低照，全量2000张 |
| 权重 | epoch47 EMA，完成第48轮，15072次EMA更新 |
| 状态张量 | 1110，保留原始精度 |
| 文件大小 | 1299635141字节 |
| SHA256 | `75679ae9097d3d73028cdb0ef9761d2ab3e4b8057b9e360316c432454ca2959e` |

文件可放入仓库weights/，或通过--checkpoint指定任何位置：

```bash
python tools/check_artifacts.py --checkpoint weights/student_epoch47_ema.pt
python tools/run_best.py infer --test-root /path/to/phase3 --gpus 0 --checkpoint weights/student_epoch47_ema.pt
```

推理直接构建相同网络并严格加载学生参数，跳过教师和官方预训练初始化。HBS不在推理时修改特征。只下载这一个模型即可预测；完整训练checkpoint、优化器和教师不是推理依赖。

## 重新训练所需文件

重新训练先取得官方RF-DETR2XL预训练，再训练IR/Depth教师。官方预训练链接和SHA256、教师轮数见[checkpoints.json](checkpoints.json)。原训练标签在上述网盘材料“代码与数据”目录，图像使用赛题授权数据。

RF-DETR Plus及2XL遵循自己的Platform Model License；团队MIT许可不替代模型许可。见[NOTICE](../NOTICE.md)。

## 对应的原始结果

网盘内“AIC-2026-19640904-赛题结果文档”目录提供56.132对应的原始ZIP：1000个TXT、16286框、置信度≥0.25。原ZIP SHA256为：

```text
a271eafbe02b4529083127b1664a71d689127c2c163092793bbb98c7a9f540b6
```

可用下面的命令检查TXT内容是否与原始提交一致；它忽略ZIP压缩方式和时间戳：

```bash
python tools/check_artifacts.py --submission /path/to/submission.zip --exact
```

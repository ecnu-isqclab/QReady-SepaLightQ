# 飞机目标分类模块

这个目录用于训练、评估和推理 128x128 飞机裁剪图分类模型。当前默认模型是 EfficientNet-B0，默认数据来自 MAR20 飞机裁剪数据。

## 目录结构

- `configs/efficientnet_b0_mar20_cls.yaml`：默认训练配置。
- `nets/efficientnet_b0_classifier.py`：EfficientNet-B0 分类模型定义。
- `utils/classification_dataloader.py`：数据集、数据增强、txt 标注解析、ImageFolder 扫描和 dataloader 构建。
- `train_classifier.py`：训练入口。
- `eval_classifier.py`：使用已保存 checkpoint 做验证集评估。
- `predict_classifier.py`：对单张图片或图片目录做分类预测。
- `requirements.txt`：最小 Python 依赖。

## 环境依赖

在训练或推理使用的 Python 环境中安装依赖：

```bash
pip install -r aircraft_classification/requirements.txt
```

如果在 GPU 服务器上训练，建议根据本机 CUDA 版本安装匹配的 `torch` 和 `torchvision`。

## 数据格式

默认配置使用裁剪图 txt 清单文件，每一行格式如下：

```text
/abs/path/to/image.jpg x1,y1,x2,y2,class_id
```

分类器只使用最后的 `class_id`。前面的 bbox 字段保留用于追溯裁剪来源，因为输入图片本身已经是裁剪后的目标图。

也支持 ImageFolder 风格的数据目录。需要在配置文件中设置 `train_image_dir` 和 `val_image_dir`：

```text
root/A1/*.jpg
root/A2/*.jpg
...
```

## 训练

```bash
python -m aircraft_classification.train_classifier \
  --config aircraft_classification/configs/efficientnet_b0_mar20_cls.yaml
```

默认输出目录为：

```text
/srv/share/hackathon/aircraft_classification/logs/<experiment_name>/<timestamp>/
```

每次训练会保存以下内容：

- `config.yaml`：本次训练使用的配置副本。
- `train.log`：训练过程日志。
- `metrics.csv`：每个 epoch 的训练/验证 loss、top1、top5 和学习率。
- `per_class_metrics.csv`：每个 epoch 的各类别验证准确率。
- `confusion_matrix_epoch_XXX.csv`：每个 epoch 的验证集混淆矩阵。
- `confusion_matrix_best.csv`：最佳 checkpoint 对应的混淆矩阵。
- `summary.json` 和 `summary.txt`：训练总结，包括最佳 epoch、最佳准确率、各类别表现等。
- `checkpoints/epoch_XXX.pth`、`checkpoints/last.pth`、`checkpoints/best.pth`：模型权重和训练状态。

## 断点续训

```bash
python -m aircraft_classification.train_classifier \
  --config aircraft_classification/configs/efficientnet_b0_mar20_cls.yaml \
  --resume /path/to/checkpoints/last.pth
```

断点续训会恢复：

- 模型权重
- optimizer 状态
- scheduler 状态
- 当前 epoch
- `best_top1`
- `best_epoch`

续训会写入一个新的时间戳目录，并从 `checkpoint_epoch + 1` 继续训练。

## 评估

```bash
python -m aircraft_classification.eval_classifier \
  --config aircraft_classification/configs/efficientnet_b0_mar20_cls.yaml \
  --checkpoint /path/to/checkpoints/best.pth
```

评估脚本会在输出目录中保存：

- `evaluation.json`
- `summary.json`
- `summary.txt`
- `confusion_matrix.csv`

## 预测

单张图片预测：

```bash
python -m aircraft_classification.predict_classifier \
  --config aircraft_classification/configs/efficientnet_b0_mar20_cls.yaml \
  --checkpoint /path/to/checkpoints/best.pth \
  --image /path/to/crop.jpg
```

图片目录批量预测：

```bash
python -m aircraft_classification.predict_classifier \
  --config aircraft_classification/configs/efficientnet_b0_mar20_cls.yaml \
  --checkpoint /path/to/checkpoints/best.pth \
  --image-dir /path/to/crops
```

预测结果会保存为 CSV，包含 top-k 类别和对应概率。

## 当前默认数据

当前配置文件指向的数据目录为：

```text
/srv/share/hackathon/yolov7-pytorch-master/codex_new_code/mar20_hbb_crops_128_letterbox
```

检查时该数据包含：

- 类别数：20
- 训练裁剪图：9427 张
- 验证裁剪图：2345 张

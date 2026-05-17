# QReady-SepaLightQ

QReady-SepaLightQ 是面向 MAR20 遥感飞机数据集的“两阶段”飞机目标识别工程。整体流程先由 Q-Loc 检测模型定位图像中的飞机框，再把每个框裁剪成 128x128 白底图，交给 Q-Rec 分类模型识别 A1-A20 共 20 种飞机类别，最后输出 txt、CSV、VOC XML、可视化图片和评估指标。

## 1. 开发环境与依赖版本

安装依赖示例：

```bash
pip install -r requirements.txt
```

注意：`torch` 和 `torchvision` 建议按本机 CUDA 版本单独安装。若没有可用 GPU，代码会自动回退到 CPU，但完整数据集推理会明显变慢。

## 2. 代码目录结构与文件功能

```text
QReady-SepaLightQ/
  configs/
    Q-Loc/
      setting.py                         # Q-Loc 检测模型当前生效配置
      MAR20_train.txt                    # MAR20 训练列表
      MAR20_val.txt                      # MAR20 验证列表
      MAR20_test.txt                     # MAR20 测试列表
      model_data/
        aircraft_classes.txt             # 检测类别文件，通常为 aircraft
        yolo_anchors.txt                 # YOLO anchors
    Q-Rec/
      efficientnet_b0_qnn_mar20_cls_cpu_finetune.yaml
                                          # Q-Rec 分类模型推理/微调配置
      efficientnet_b0_*.yaml             # 其他分类实验配置
    pipeline/
      final_pipeline.yaml                # 流水线配置记录

  dataset/
    MAR20/
      JPEGImages/                        # 已整理的 MAR20 图像
      Annotations/                       # 已整理的 VOC 格式标注
    MAR20_test/
      MAR20-JPEGImages_test/             # 官方测试图像目录
      MAR20-Annotations_test/
        Horizontal Bounding Boxes/       # 官方测试水平框标签
        Oriented Bounding Boxes/         # 官方测试旋转框标签，本流水线不使用

  models/
    Q-Loc/
      nets/                              # YOLO / QNN 检测网络结构
      qnas/                              # Q-Loc 量子结构搜索相关代码
      qnn_net/                           # QNN 相关网络模块
    Q-Rec/
      efficientnet_b0_qnn_classifier.py  # Q-Rec EfficientNet-B0 + QNN 分类模型
      qnas/                              # Q-Rec 量子结构搜索相关代码

  train/
    Q-Loc/                               # 检测模型训练脚本
    Q-Rec/
      train_classifier.py                # 分类模型训练脚本

  test/
    Q-Loc/
      forward.py                         # 检测模型加载、单图推理、批量推理入口
    Q-Rec/
      predict_classifier.py              # 分类模型预测脚本
      eval_classifier.py                 # 分类模型评估脚本
      plot_confusion_matrix.py           # 混淆矩阵绘制脚本

  pipeline/
    detect_crop_classify.py              # 数据集目录入口：检测 -> 裁剪 -> 分类 -> 评估 -> 输出 XML
    detect_crop_classify_mar20_txt.py    # txt 列表入口：适配 configs/Q-Loc/MAR20_test.txt
    README.md                            # 流水线专项说明

  scripts/
    prepare_data/                        # 数据格式转换、VOC 列表生成、anchor 聚类脚本

  utils/
    Q-Loc/                               # 检测相关工具函数、bbox、dataloader
    Q-Rec/                               # 分类数据加载与 transform
    common/                              # 公共工具目录

  weights/
    Q-Loc/best_epoch_weights.pth         # 最终检测模型权重，约 25 MB
    Q-Rec/best.pth                       # 最终分类模型权重，约 49 MB

  results/
    Q-Loc/                               # 检测单模块结果
    Q-Rec/                               # 分类单模块结果
    pipeline/                            # 完整流水线输出目录
```

两个最终流水线入口的区别：

- `pipeline/detect_crop_classify.py`：直接读取数据集目录，例如 `dataset/MAR20_test`。
- `pipeline/detect_crop_classify_mar20_txt.py`：读取 txt 列表，例如 `configs/Q-Loc/MAR20_test.txt`。

## 3. 一键运行命令与参数说明

### 3.1 对有标签测试集运行完整流水线

适用于：

```text
dataset/MAR20_test/
  MAR20-JPEGImages_test/
  MAR20-Annotations_test/Horizontal Bounding Boxes/
```

运行命令：（我们在比赛跑测试集分数时用此命令）

```bash
cd QReady-SepaLightQ所在位置

  python pipeline/detect_crop_classify.py \
  --dataset-name dataset/MAR20_test \
  --run-name MAR20_test_macro_pipeline \
  --device cuda:0 \
  --classifier-batch-size 32 \
  --progress-interval 50 \
  --dedupe-iou 0.5
```

输出目录：

```text
results/pipeline/MAR20_test_labeled_pipeline/
```

最终结果：/srv/share/hackathon/QReady-SepaLightQ/results/pipeline/MAR20_test_macro_pipeline
Annotations是我们预测出的xml，方便与真实xml对比。
crops_128_white保存的是单独裁出的每个飞机框，是我们做的数据预处理
visualizations保存了30张可视化结果
summary.json说明了我们的各种文件以及路径保存
metrics_summary.json存了评估指标结果
其他为中间文件

该命令会生成检测裁剪结果、分类结果、预测 XML、可视化结果，并根据 `Horizontal Bounding Boxes` 计算 Precision、Recall、F1、mean IoU、mAP。

### 3.2 对 MAR20 txt 列表运行完整流水线

```bash
cd QReady-SepaLightQ所在位置

python pipeline/detect_crop_classify_mar20_txt.py \
  --list-path configs/Q-Loc/MAR20_test.txt \
  --run-name MAR20_test_pipeline_dedupe \
  --device cuda:0 \
  --classifier-batch-size 32 \
  --progress-interval 50 \
  --dedupe-iou 0.5
```

输出目录：

```text
results/pipeline/MAR20_test_pipeline_dedupe/
```

### 3.3 单张或少量图片调试

按图片 id 运行单张图，例如 `2027.jpg`：

```bash
cd QReady-SepaLightQ所在位置

  pipeline/detect_crop_classify.py \
  --dataset-name dataset/MAR20_test \
  --image-id 2027 \
  --run-name debug_2027 \
  --device cuda:0 \
  --classifier-batch-size 32 \
  --progress-interval 1 \
  --dedupe-iou 0.5
```

只跑前 10 张：

```bash
  pipeline/detect_crop_classify.py \
  --dataset-name dataset/MAR20_test \
  --max-images 10 \
  --run-name debug_10_images \
  --device cuda:0 \
  --classifier-batch-size 32 \
  --progress-interval 1 \
  --dedupe-iou 0.5
```

### 3.4 无标签数据集运行方式

无标签数据集必须加 `--skip-evaluation`：

```bash
cd QReady-SepaLightQ所在位置

  pipeline/detect_crop_classify.py \
  --dataset-name path/to/unlabeled_dataset \
  --run-name unlabeled_pipeline \
  --skip-evaluation \
  --device cuda:0 \
  --classifier-batch-size 32 \
  --progress-interval 50 \
  --dedupe-iou 0.5
```

### 3.5 常用参数说明

```text
--dataset-name
```

数据集目录名或路径。支持 `dataset/MAR20_test` 这种相对仓库根目录的路径，也支持绝对路径。目录内图片可放在 `JPEGImages/` 或 `MAR20-JPEGImages_test/`。

```text
--list-path
```

txt 列表路径，仅用于 `detect_crop_classify_mar20_txt.py`。每行包含一张图像路径和若干个真实框。

```text
--run-name
```

输出目录名。结果会写到 `results/pipeline/<run-name>/`。

```text
--device
```

分类模型推理设备，例如 `cuda:0` 或 `cpu`。如果 CUDA 不可用，会自动回退 CPU。

```text
--classifier-batch-size
```

分类模型批量推理大小。默认推荐 `32`，显存充足可尝试 `64` 或 `128`。

```text
--dedupe-iou
```

同一张图内检测框去重阈值。默认推荐 `0.5`。流程会按检测分数从高到低保留框，后续框若和已保留框 IoU 大于等于该阈值，则视为重复框并丢弃。设置为 `0` 可关闭去重。

```text
--min-score
```

检测框最低分数过滤阈值。默认不额外过滤。

```text
--iou-threshold
```

评估 TP 使用的 IoU 阈值，默认 `0.5`。

```text
--score-mode
```

mAP 排序分数来源：

```text
combined    detector_score * classifier_score，默认
detector    只使用检测分数
classifier  只使用分类概率
```

```text
--skip-evaluation
```

不计算指标，适合无标签数据集。

```text
--skip-classification
```

只检测和裁剪，不运行分类模型。调试检测框时可用。

## 4. 输入输出文件格式

### 4.1 输入图像目录格式

标准 VOC 风格：

```text
dataset_name/
  JPEGImages/
    2027.jpg
  Annotations/
    2027.xml
```

MAR20 官方测试集风格：

```text
dataset/MAR20_test/
  MAR20-JPEGImages_test/
    2027.jpg
  MAR20-Annotations_test/
    Horizontal Bounding Boxes/
      2027.xml
    Oriented Bounding Boxes/
      2027.xml
```

当前流水线使用水平框标签：

```text
MAR20-Annotations_test/Horizontal Bounding Boxes/
```

### 4.2 输入 txt 列表格式

`configs/Q-Loc/MAR20_test.txt` 每行格式：

```text
image_path x1,y1,x2,y2,class_id x1,y1,x2,y2,class_id ...
```

示例：

```text
dataset/MAR20/JPEGImages/25.jpg 519,224,630,324,1 559,526,670,634,1
```

坐标含义：

```text
x1,y1 = 左上角
x2,y2 = 右下角
class_id = 0-based 类别 id，例如 0 对应 A1，1 对应 A2
```

### 4.3 输入 XML 标注格式

VOC XML 格式，核心字段如下：

```xml
<annotation>
  <filename>2027.jpg</filename>
  <size>
    <width>800</width>
    <height>800</height>
    <depth>3</depth>
  </size>
  <object>
    <name>A8</name>
    <bndbox>
      <xmin>567</xmin>
      <ymin>18</ymin>
      <xmax>732</xmax>
      <ymax>200</ymax>
    </bndbox>
  </object>
</annotation>
```

类别名为 `A1` 到 `A20`。

### 4.4 输出目录结构

完整流水线输出目录示例：

```text
results/pipeline/MAR20_test_labeled_pipeline/
  crops_128_white/                       # 检测框裁剪出的 128x128 白底 crop
  visualizations/                        # 可视化结果，红框为预测，绿框为 GT
  Annotations/                           # 根据预测结果生成的 VOC XML
  detected_crops_pending_class.txt       # 裁剪图路径 + 检测框
  detected_crops_with_pred_class.txt     # 裁剪图路径 + 检测框 + 预测类别 id
  classification_predictions.csv         # 每个 crop 的 top-k 分类结果
  metrics_summary.json                   # 全局评估指标
  per_class_metrics.csv                  # 每类评估指标
  summary.json                           # 本次运行的输入、输出和参数摘要
```

### 4.5 `detected_crops_pending_class.txt`

分类前结果，每行一个 crop：

```text
crop_image_path x1,y1,x2,y2
```

示例：

```text
results/pipeline/.../crops_128_white/2027_obj000.jpg 567,18,732,200
```

### 4.6 `detected_crops_with_pred_class.txt`

分类后结果，每行一个 crop：

```text
crop_image_path x1,y1,x2,y2,class_id
```

示例：

```text
results/pipeline/.../crops_128_white/2027_obj000.jpg 567,18,732,200,7
```

其中 `class_id` 是 0-based id，`7` 对应 `A8`。

### 4.7 `classification_predictions.csv`

每行一个 crop，字段包括：

```text
image_path
pred_id
pred_class
pred_score
top1_id, top1_class, top1_score
...
top5_id, top5_class, top5_score
```

### 4.8 `Annotations/*.xml`

流水线会自动把预测结果按原始图像聚合成 VOC XML。每张原图一个 XML，同一张图内多个飞机写成多个 `<object>`。

示例：

```xml
<annotation>
  <filename>2027.jpg</filename>
  <source>
    <database>MAR20</database>
  </source>
  <size>
    <width>800</width>
    <height>800</height>
    <depth>3</depth>
  </size>
  <segmented>0</segmented>
  <object>
    <name>A8</name>
    <bndbox>
      <xmin>567</xmin>
      <ymin>18</ymin>
      <xmax>732</xmax>
      <ymax>200</ymax>
    </bndbox>
  </object>
</annotation>
```

### 4.9 `metrics_summary.json`

整体指标是全局 TP、FP、FN 汇总后的 micro-average 口径：

```json
{
  "iou_threshold": 0.5,
  "gt_count": 2932,
  "pred_count": 2961,
  "tp": 2384,
  "fp": 577,
  "fn": 548,
  "precision": 0.8051334008780817,
  "recall": 0.8130968622100955,
  "f1": 0.8090955370778891,
  "IoU": 0.8656369073260053,
  "mean_iou": 0.8656369073260053,
  "mAP": 0.663310152037511
}
```

这里：

```text
precision = tp / (tp + fp)
recall    = tp / (tp + fn)
f1        = 2 * precision * recall / (precision + recall)
```

`mAP` 为按类别 AP 计算后取均值。

## 5. 预期运行时间与硬件要求

### 5.1 推荐硬件

推荐配置：

```text
GPU:    NVIDIA GPU，显存 >= 8 GB
CPU:    8 核以上
内存:   >= 16 GB
磁盘:   至少预留 5 GB 输出空间
```

最低可运行配置：

```text
CPU:    可纯 CPU 运行
内存:   >= 8 GB
磁盘:   至少预留 2 GB 输出空间
```

纯 CPU 模式可以跑通，但检测和分类都会变慢。

### 5.2 已验证数据规模

当前 `dataset/MAR20_test` 完整运行规模：

```text
source_image_count: 500
gt_count:           2932
crop_count:         2961
```

模型权重大小：

```text
Q-Loc: weights/Q-Loc/best_epoch_weights.pth 约 25 MB
Q-Rec: weights/Q-Rec/best.pth               约 49 MB
```

### 5.3 预期运行时间

实际耗时取决于 GPU、CUDA 是否可用、磁盘速度和可视化数量。参考范围：

```text
单张图 smoke test:
  GPU: 数秒内
  CPU: 约 20-40 秒

500 张 MAR20_test 完整流水线:
  GPU: 通常约 5-20 分钟
  CPU: 可能需要 1-3 小时或更久
```

如果只想快速检查路径和输出格式，建议先运行：

```bash
cd /srv/share/hackathon/QReady-SepaLightQ

/home/ivy/miniconda3/envs/yolov7-csdn/bin/python \
  pipeline/detect_crop_classify.py \
  --dataset-name dataset/MAR20_test \
  --max-images 1 \
  --run-name smoke_test \
  --device cuda:0 \
  --classifier-batch-size 32 \
  --progress-interval 1 \
  --dedupe-iou 0.5
```

如果遇到 CUDA 不可用，程序会在日志中显示类似：

```text
CUDA_ERROR_NO_DEVICE
```

此时会回退 CPU 或由底层库给出 CUDA 初始化提示。若希望强制 CPU，可使用：

```bash
--device cpu
```

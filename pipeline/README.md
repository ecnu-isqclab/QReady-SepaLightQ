# Aircraft Detect, Crop, Classify Pipeline

本目录是最终流水线入口，适配当前整理后的仓库：

```text
/srv/share/hackathon/QReady-SepaLightQ
```

流水线读取 `configs/Q-Loc/setting.py` 当前生效的 Q-Loc 检测配置，完成：

1. 对数据集图片检测飞机框。
2. 将每个飞机框裁成一张 `128x128` 白边补齐图片。
3. 调用 Q-Rec 分类模型预测飞机类别。
4. 输出 `x1,y1,x2,y2,class_id` 格式的最终 txt。
5. 有标签时计算 Precision、Recall、F1、IoU、mAP，并在可视化中画预测红框和 GT 绿框。

默认模型文件：

```text
Q-Loc setting: configs/Q-Loc/setting.py
Q-Loc weight:  weights/Q-Loc/best_epoch_weights.pth
Q-Rec config:  configs/Q-Rec/efficientnet_b0_qnn_mar20_cls_cpu_finetune.yaml
Q-Rec weight:  weights/Q-Rec/best.pth
classes:       configs/Q-Loc/model_data/aircraft_classes.txt
anchors:       configs/Q-Loc/model_data/yolo_anchors.txt
```

## 数据集格式

按数据集目录运行时，数据集应放在：

```text
test/Q-Loc/evaluation/testing_data/<DATASET_NAME>/
```

有标签数据集：

```text
<DATASET_NAME>/
  JPEGImages/
    2027.jpg
  Annotations/
    2027.xml
```

无标签数据集：

```text
<DATASET_NAME>/
  JPEGImages/
    2027.jpg
```

也可以直接使用 `configs/Q-Loc/MAR20_test.txt`。该 txt 每行格式是：

```text
图片路径 x1,y1,x2,y2,class_id x1,y1,x2,y2,class_id ...
```

这里的框坐标是左上角和右下角：

```text
x1,y1,x2,y2 = xmin,ymin,xmax,ymax
```

不是中心点和宽高。流水线输出的 `detected_crops_pending_class.txt` 和
`detected_crops_with_pred_class.txt` 也使用这个左上/右下格式。

## 使用命令

先进入仓库根目录：

```bash
cd /srv/share/hackathon/QReady-SepaLightQ
```

### 1. 有标签数据集：完整流水线 + 指标评估

用于有 `Annotations/*.xml` 的数据集：

```bash
/home/ivy/miniconda3/envs/yolov7-csdn/bin/python \
  pipeline/detect_crop_classify.py \
  --dataset-name MAR20_test \
  --device cuda:0 \
  --classifier-batch-size 32
```

输出目录默认是：

```text
results/pipeline/MAR20_test/
```

### 2. 无标签数据集：完整流水线，不计算指标

最终测试集没有 XML 标签时必须加 `--skip-evaluation`：

```bash
/home/ivy/miniconda3/envs/yolov7-csdn/bin/python \
  pipeline/detect_crop_classify.py \
  --dataset-name YOUR_UNLABELED_DATASET \
  --skip-evaluation \
  --device cuda:0 \
  --classifier-batch-size 32
```

无标签模式仍然会输出：

```text
detected_crops_pending_class.txt
detected_crops_with_pred_class.txt
classification_predictions.csv
visualizations/
crops_128_white/
summary.json
```

但是不会输出：

```text
metrics_summary.json
per_class_metrics.csv
```

### 3. 使用 MAR20 txt 列表测试

对 `configs/Q-Loc/MAR20_test.txt` 跑完整流水线并计算 Precision、Recall、F1、IoU、mAP：

```bash
/home/ivy/miniconda3/envs/yolov7-csdn/bin/python \
  pipeline/detect_crop_classify_mar20_txt.py \
  --list-path configs/Q-Loc/MAR20_test.txt \
  --run-name MAR20_test_pipeline \
  --device cuda:0 \
  --classifier-batch-size 32 \
  --progress-interval 50
```

输出目录：

```text
results/pipeline/MAR20_test_pipeline/
```

### 4. 单张图片调试

`--image-id` 后面写图片文件名去掉后缀后的 id，例如 `2027.jpg` 写 `2027`：

```bash
/home/ivy/miniconda3/envs/yolov7-csdn/bin/python \
  pipeline/detect_crop_classify.py \
  --dataset-name MAR20_test \
  --image-id 2027 \
  --run-name test_one_image \
  --device cuda:0 \
  --classifier-batch-size 32 \
  --progress-interval 1
```

使用 txt 列表调试单张图：

```bash
/home/ivy/miniconda3/envs/yolov7-csdn/bin/python \
  pipeline/detect_crop_classify_mar20_txt.py \
  --list-path configs/Q-Loc/MAR20_test.txt \
  --image-id 2027 \
  --run-name test_one_image_from_txt \
  --device cuda:0 \
  --classifier-batch-size 32 \
  --progress-interval 1
```

### 5. 复用已有检测 JSON，不重新跑 YOLO

如果已经有 `result_location_aircraft/.../*.json`，可以直接复用这些检测框，只做裁剪、分类、评估和可视化：

```bash
/home/ivy/miniconda3/envs/yolov7-csdn/bin/python \
  pipeline/detect_crop_classify.py \
  --dataset-name MAR20_test \
  --location-json-dir results/Q-Loc/result_location_aircraft/MAR20_test \
  --run-name MAR20_test_from_location_json \
  --device cuda:0 \
  --classifier-batch-size 32
```

### 6. 常用可选参数

```bash
--run-name NAME
```

指定输出目录名，结果会写到 `results/pipeline/NAME/`。

```bash
--visualize-count 100
```

指定最多保存多少张可视化图片。默认 `30`。

```bash
--device cuda:0
```

指定分类模型使用 GPU。当前默认分类配置来自 CPU finetune 配置文件，所以建议跑预测时显式写上这个参数。

```bash
--classifier-batch-size 32
```

指定裁剪图分类的批量推理大小。显存足够时可尝试 `64` 或 `128`。

```bash
--iou-threshold 0.5
```

指定评估 TP 的 IoU 阈值。默认 `0.5`。

```bash
--skip-evaluation
```

无标签数据集使用，关闭 XML/txt GT 读取和 Precision/Recall/F1/IoU/mAP 计算。

```bash
--skip-classification
```

只生成检测 crop 和 `detected_crops_pending_class.txt`，不做分类。

## 输出文件

每次运行会写到：

```text
results/pipeline/<run-name>/
```

主要输出：

```text
crops_128_white/*.jpg
detected_crops_pending_class.txt
detected_crops_with_pred_class.txt
classification_predictions.csv
metrics_summary.json
per_class_metrics.csv
visualizations/*.jpg
summary.json
```

- `crops_128_white/*.jpg`: each detected aircraft as one 128x128 white-padded crop.
- `detected_crops_pending_class.txt`: crop absolute path plus original-image bbox, no class yet.
- `detected_crops_with_pred_class.txt`: crop absolute path plus original-image `x1,y1,x2,y2,class_id`.
- `classification_predictions.csv`: top-k classifier predictions.
- `metrics_summary.json`: overall Precision, Recall, F1, IoU, mean IoU, and mAP.
- `per_class_metrics.csv`: per-class Precision, Recall, F1, mean IoU, and AP.
- `visualizations/*.jpg`: source images with red prediction boxes and green ground-truth boxes.
- `summary.json`: run metadata.

评估是类别敏感的：预测类别必须和 GT 类别一致，并且预测框和真实框 IoU 达到
`--iou-threshold` 才算 TP。mAP 排序分数默认使用
`detector_score * classifier_score`；如果检测框没有分数，就使用分类概率。

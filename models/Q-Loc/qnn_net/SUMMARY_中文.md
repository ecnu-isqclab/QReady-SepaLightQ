# qnn_net 工作汇总

我阅读了 `quantumenhanced-main.pdf`，它的核心不是单纯把 quantum 层插入 YOLO 特征图，而是：

1. 先把 YOLOv7 做轻量化改造，例如 C2f_bone、WIoU、删除小目标检测头、k-means++ 重新聚类 anchor。
2. 再训练一个 HQNN，用量子旋转门、纠缠结构和测量结果做图像分类。
3. 最后把 DL-YOLO 的分类概率和 HQNN 的分类概率做平均融合，用 HQNN 修正类别判断。

## 做了什么

新增 `hqnn_classifier.py`，做了一个轻量可训练复现版本：

- 数据：从当前 VOC 数据集中读取 XML 标注，并把目标框裁剪出来。
- 类别：把任务归并成 `person / vehicle / other` 三类，贴近现在“识别人、车等目标”的工程。
- 网络：
  - 小型 CNN 提取裁剪图特征。
  - 将特征作为角度编码输入量子层。
  - 量子层使用 `Ry/Rz` 可调旋转门、全连接 CNOT 纠缠、Z 方向测量。
  - 测量结果进入分类层输出三类概率。
- 训练：默认只轻量训练 1 个 epoch，输出权重和训练日志。
- 融合：保留 `fuse_probabilities()`，实现论文里的平均概率融合策略。

另外新增 `hqnn_lite_pure_python.py`：

- 因为当前默认 Python 环境没有 `torch/numpy`，这个脚本只依赖 Python 标准库。
- 它从 VOC XML 标注中读取目标框几何信息，做角度编码和纠缠特征映射，再训练一个轻量分类头。
- 这个版本主要用于“当前环境能跑通训练日志”，不是最终图像分类精度方案。

## 怎么运行

```bash
python qnn_net/hqnn_classifier.py --epochs 1 --max-items 180 --cpu
```

如果当前环境没有 PyTorch，可以先运行纯 Python 版：

```bash
python qnn_net/hqnn_lite_pure_python.py --epochs 5 --max-items 180
```

输出文件：

- `qnn_net/runs/hqnn_classifier.pt`
- `qnn_net/runs/hqnn_train_log.csv`
- `qnn_net/runs/hqnn_train_summary.json`
- `qnn_net/runs/hqnn_lite_model.json`
- `qnn_net/runs/hqnn_lite_train_log.csv`
- `qnn_net/runs/hqnn_lite_summary.json`

## 当前结论

这个目录完成的是论文 HQNN 思路的轻量复现骨架，不是完整复刻论文的电塔底座数据集实验。它已经能在当前工程 VOC 数据上做可运行训练，并把 quantum/classical 融合路线接起来。

后续如果要更接近论文，需要继续做：

- 针对本任务重新定义更精细的类别，而不只是 `person / vehicle / other`。
- 把 YOLO 检测出的候选框裁剪后送入 HQNN，得到真实的分类校正结果。
- 训练后用 `evaluation` 里的统一指标比较：baseline、quantum 插入版、HQNN 融合版。

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]

# ===========================================================================
# 模型索引
# ===========================================================================
#
# 1. quantumoutputp5trueqnn
# 中文简介：
#   只改 YOLOv7 的 P5 输出侧。删除原本的 rep_conv_3，在 P5 侧加入真 QNN gate，
#   训练时只微调量子角度。它是最早的“真 QNN 能否接入 YOLOv7”验证版。
# nets：
#   nets/yolov7_quantumoutp5trueqnn.py
# train：
#   train_yolov7_quantumoutp5trueqnn.py
# logs：
#   logs/yolov7_quantumouyp5trueqnn/
#
# 2. quantump3p4p5trueqnn
# 中文简介：
#   同时改 P3/P4/P5。保留 RepConv 原来的 1x1 分支，删除原来的 3x3 分支，
#   换成“真 QNN gate + 轻量 1x1”分支。后来确认这个版本的问题是：
#   它把负责局部空间建模的 3x3 卷积删得太彻底，QNN 又不足以真正替代它。
# nets：
#   nets/yolov7_quantump3p4p5trueqnn.py
# train：
#   train_quantumyolov7p3p4p5trueqnn.py
# logs：
#   logs/quantumyolov7p3p4p5trueqnn/
#
# 3. quantumnewchanelandconvdown1x1
# 中文简介：
#   把原来的“3x3 + 1x1”RepConv 结构改成“2x2 + 原 1x1 + 新增 QNN 通道”。
#   这版不再让 QNN 独占原 3x3 分支，而是让 2x2 卷积继续负责局部建模，
#   QNN 作为额外辅助通道补充信息。训练时只训练新 2x2 分支和新量子通道。
# nets：
#   nets/yolov7_quantumnewchanelandconvdown1x1.py
# train：
#   train_yolov7_quantumnewchanelandconvdown1x1.py
# logs：
#   logs/yolov7_quantumnewchanelandconvdown1x1/
#
# 4. quantumnewchanel6qubitsanglecoding64output
# 中文简介：
#   在第 3 版结构上升级 QNN 本身。输入先压成 15x2x2=60 维，
#   由 6 qubit 做 5 轮角度重上传编码；输出读取完整 2^6=64 维概率分布，
#   再映射回通道门控。训练时只训练新 2x2 分支和整条量子通道。
# nets：
#   nets/yolov7quantumnewchanel6qubitsanglecoding64output.py
# train：
#   train_yolov7quantumnewchanel6qubitsanglecoding64output.py
# logs：
#   logs/yolov7quantumnewchanel6qubitsanglecoding64output/
#
# 5. quantumnewchanel8qubitsanglecoding256output
# 中文简介：
#   与 6-qubit 版同构，但扩展为 8 qubit。输入压成 20x2x2=80 维，
#   每个 qubit 同样做 5 轮双角度编码；输出读取完整 2^8=256 维概率分布。
#   它用于比较更大 QNN 表达空间是否带来收益，代价是训练更重。
# nets：
#   nets/yolov7quantumnewchanel8qubitsanglecoding256output.py
# train：
#   train_yolov7quantumnewchanel8qubitsanglecoding256output.py
# logs：
#   logs/yolov7quantumnewchanel8qubitsanglecoding256output/
#
# 6. yolov7_mar20_hbb
# 中文简介：
#   用经典 yolo 模型，用比赛数据集 MAR20 训练，用 MAR20_test 测试。
# nets：
#   nets/yolo.py
# train：
#   训练产物目录名为 yolo_mar20_hbb_2026_05_16_10_10_31
# logs：
#   logs/yolo_mar20_hbb_2026_05_16_10_10_31/
#
# 7. yolov7_q6_mar20
# 中文简介：
#   用 6-qubit 量子新增通道版 YOLOv7，在 MAR20 水平框任务上做全模型训练，
#   从之前 BDD100K 上训练好的 q6 权重继续初始化，最终用于 MAR20_test 测试。
# nets：
#   nets/yolov7quantumnewchanel6qubitsanglecoding64output.py
# train：
#   train_yolov7_q6_mar20.py
# logs：
#   logs/yolov7_q6_mar20/
#
# 8. yolov7_mar20_hbb_white_box_finetune
# 中文简介：
#   经典 yolo 练 MAR20，用白色背景抠图增强的模型。
# nets：
#   nets/yolo.py
# train：
#   训练产物目录名为 yolo_mar20_hbb_white_box_finetune_2026_05_16_14_24_20
# logs：
#   logs/yolo_mar20_hbb_white_box_finetune_2026_05_16_14_24_20/
#
# 9. yolo_light_quantumchanel
# 中文简介：
#   基于轻量级 yolo，neck 和分类都有量子通道。
# nets：
#   nets/yolo_light_quantumchanel.py
# train：
#   train_yolo_light_quantumchanel.py
# logs：
#   logs/yolo_light_quantumchanel/
#
# 10. yolo_light_mar20_full_fold1_aircraft
# 中文简介：
#   砍小 yolo 权重但不变框架。
# nets：
#   nets/yolo_light.py
# train：
#   配置文件 configs/yolo_light_mar20_full_fold1_aircraft.yaml
# logs：
#   logs/yolo_light_mar20_full_fold1_aircraft_2026_05_16_17_21_40/
#
# 11. yolo_light_lighterbutnoquantum
# 中文简介：
#   基于轻量级 yolo，把 neck 和分类前的关键 3x3 卷积降成 2x2，
#   但不加入量子通道，用作量子版本的纯经典对照模型。
# nets：
#   nets/yolo_light_lighterbutnoquantum.py
# train：
#   train_yolo_light_lighterbutnoquantum.py
# logs：
#   logs/yolo_light_lighterbutnoquantum/
#
# 12. yolo_light_quantum_minichange
# 中文简介：
#   基于效果较好的 yolo_light_quantumchanel，只把量子读出从 64 维状态概率
#   改成 6 个 Z 测量值，再用小型 MLP 放大回通道门控；只微调量子角度和新读出头。
# nets：
#   nets/yolo_light_quantum_minichange.py
# train：
#   train_yolo_light_quantum_minichange.py
# logs：
#   logs/yolo_light_quantum_minichange/
#
# 13. yolo_light_multiclass_mar20_full_fold1
# 中文简介：
#   轻量级 yolo 的 MAR20 二十分类版本，输出 A1-A20 二十种飞机类别。
# nets：
#   nets/yolo_light_multiclass.py
# train：
#   配置文件 logs/yolo_light_multiclass_mar20_full_fold1_2026_05_16_20_03_37/config.yaml
# logs：
#   logs/yolo_light_multiclass_mar20_full_fold1_2026_05_16_20_03_37/
#


# ===========================================================================
# 当前实际生效的配置
# ===========================================================================



# # Model structure. MODEL_KIND can be "yolov7" or "yolop".
# MODEL_KIND = "yolov7"
# MODEL_BODY = "nets.yolov7_quantumoutp5trueqnn:YoloBodyQuantumOutP5TrueQNN"
# MODEL_PHI = "l"
# MODEL_KWARGS = {
#     "qnn_blocks": 3,
# }

# # Runtime assets.
# WEIGHTS_PATH = ROOT / "logs" / "yolov7_quantumouyp5trueqnn" / "2026_05_15_16_45_05" / "checkpoints" / "best_epoch_weights.pth"
# CLASSES_PATH = ROOT / "model_data" / "yolop_classes.txt"
# ANCHORS_PATH = ROOT / "model_data" / "yolo_anchors.txt"
# ANCHORS_MASK = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

# # Inference parameters.
# INPUT_SHAPE = [640, 640]
# CONFIDENCE = 0.35
# NMS_IOU = 0.3
# LETTERBOX_IMAGE = True
# SEED = 42

# # Dataset used by forward.py. A VOC2007-style directory is expected:
# # DATASET_PATH/JPEGImages and DATASET_PATH/Annotations.
# DATASET_PATH = ROOT / "evaluation" / "testing_data" / "bdd100k_testing"

# # Optional image list for forward.py. If set, the first token of each line is
# # used as an image path or id, and only the stem is kept. If None, all images
# # under DATASET_PATH/JPEGImages are scanned.
# IMAGE_LIST_PATH = None
# IMAGE_LIMIT = None

# # Output for forward.py.
# FORWARD_OUTPUT_DIR = ROOT / "experiment_forward"


# ===========================================================================
# 所有可切换配置版本
# ===========================================================================
#
# 配置 1：quantumoutputp5trueqnn
#
# MODEL_KIND = "yolov7"
# MODEL_BODY = "nets.yolov7_quantumoutp5trueqnn:YoloBodyQuantumOutP5TrueQNN"
# MODEL_PHI = "l"
# MODEL_KWARGS = {
#     "qnn_blocks": 3,
# }
# WEIGHTS_PATH = ROOT / "logs" / "yolov7_quantumouyp5trueqnn" / "2026_05_15_16_45_05" / "checkpoints" / "best_epoch_weights.pth"
# CLASSES_PATH = ROOT / "model_data" / "yolop_classes.txt"
# ANCHORS_PATH = ROOT / "model_data" / "yolo_anchors.txt"
# ANCHORS_MASK = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
# INPUT_SHAPE = [640, 640]
# CONFIDENCE = 0.35
# NMS_IOU = 0.3
# LETTERBOX_IMAGE = True
# DATASET_PATH = ROOT / "evaluation" / "testing_data" / "bdd100k_testing"
# IMAGE_LIST_PATH = None
# IMAGE_LIMIT = None
# FORWARD_OUTPUT_DIR = ROOT / "experiment_forward"
#
#
# 配置 2：quantump3p4p5trueqnn
#
# MODEL_KIND = "yolov7"
# MODEL_BODY = "nets.yolov7_quantump3p4p5trueqnn:YoloBodyQuantumP3P4P5TrueQNN"
# MODEL_PHI = "l"
# MODEL_KWARGS = {
#     "qnn_blocks": 3,
# }
# WEIGHTS_PATH = ROOT / "logs" / "quantumyolov7p3p4p5trueqnn" / "2026_05_15_18_36_30" / "checkpoints" / "best_epoch_weights.pth"
# CLASSES_PATH = ROOT / "model_data" / "yolop_classes.txt"
# ANCHORS_PATH = ROOT / "model_data" / "yolo_anchors.txt"
# ANCHORS_MASK = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
# INPUT_SHAPE = [640, 640]
# CONFIDENCE = 0.35
# NMS_IOU = 0.3
# LETTERBOX_IMAGE = True
# DATASET_PATH = ROOT / "evaluation" / "testing_data" / "bdd100k_testing"
# IMAGE_LIST_PATH = None
# IMAGE_LIMIT = None
# FORWARD_OUTPUT_DIR = ROOT / "experiment_forward"
#
#
# 配置 3：quantumnewchanelandconvdown1x1
#
# MODEL_KIND = "yolov7"
# MODEL_BODY = "nets.yolov7_quantumnewchanelandconvdown1x1:YoloBodyQuantumNewChanelAndConvDown1x1"
# MODEL_PHI = "l"
# MODEL_KWARGS = {}
# WEIGHTS_PATH = ROOT / "logs" / "yolov7_quantumnewchanelandconvdown1x1" / "2026_05_15_19_55_28" / "checkpoints" / "best_epoch_weights.pth"
# CLASSES_PATH = ROOT / "model_data" / "yolop_classes.txt"
# ANCHORS_PATH = ROOT / "model_data" / "yolo_anchors.txt"
# ANCHORS_MASK = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
# INPUT_SHAPE = [640, 640]
# CONFIDENCE = 0.35
# NMS_IOU = 0.3
# LETTERBOX_IMAGE = True
# DATASET_PATH = ROOT / "evaluation" / "testing_data" / "bdd100k_testing"
# IMAGE_LIST_PATH = None
# IMAGE_LIMIT = None
# FORWARD_OUTPUT_DIR = ROOT / "experiment_forward"
#
#
# 配置 4：quantumnewchanel6qubitsanglecoding64output
#
# MODEL_KIND = "yolov7"
# MODEL_BODY = "nets.yolov7quantumnewchanel6qubitsanglecoding64output:YoloBodyQuantumNewChanel6QubitsAngleCoding64Output"
# MODEL_PHI = "l"
# MODEL_KWARGS = {}
# WEIGHTS_PATH = ROOT / "logs" / "yolov7quantumnewchanel6qubitsanglecoding64output" / "2026_05_15_22_38_07" / "checkpoints" / "best_epoch_weights.pth"
# CLASSES_PATH = ROOT / "model_data" / "yolop_classes.txt"
# ANCHORS_PATH = ROOT / "model_data" / "yolo_anchors.txt"
# ANCHORS_MASK = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
# INPUT_SHAPE = [640, 640]
# CONFIDENCE = 0.35
# NMS_IOU = 0.3
# LETTERBOX_IMAGE = True
# DATASET_PATH = ROOT / "evaluation" / "testing_data" / "bdd100k_testing"
# IMAGE_LIST_PATH = None
# IMAGE_LIMIT = None
# FORWARD_OUTPUT_DIR = ROOT / "experiment_forward"
#
#
# 配置 5：quantumnewchanel8qubitsanglecoding256output
#
# MODEL_KIND = "yolov7"
# MODEL_BODY = "nets.yolov7quantumnewchanel8qubitsanglecoding256output:YoloBodyQuantumNewChanel8QubitsAngleCoding256Output"
# MODEL_PHI = "l"
# MODEL_KWARGS = {}
# WEIGHTS_PATH = ROOT / "logs" / "yolov7quantumnewchanel8qubitsanglecoding256output" / "2026_05_15_21_04_12" / "checkpoints" / "best_epoch_weights.pth"
# CLASSES_PATH = ROOT / "model_data" / "yolop_classes.txt"
# ANCHORS_PATH = ROOT / "model_data" / "yolo_anchors.txt"
# ANCHORS_MASK = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
# INPUT_SHAPE = [640, 640]
# CONFIDENCE = 0.35
# NMS_IOU = 0.3
# LETTERBOX_IMAGE = True
# DATASET_PATH = ROOT / "evaluation" / "testing_data" / "bdd100k_testing"
# IMAGE_LIST_PATH = None
# IMAGE_LIMIT = None
# FORWARD_OUTPUT_DIR = ROOT / "experiment_forward"

#
# 配置 6：yolov7_mar20_hbb
# 中文简介：用经典 yolo 模型，用比赛数据集 MAR20 训练，用 MAR20_test 测试。
#
MODEL_KIND = "yolov7"
MODEL_BODY = "nets.yolo_light_quantum_minichange:YoloLightQuantumMiniChangeBody"
MODEL_PHI = "light"
MODEL_KWARGS = {}
WEIGHTS_PATH = REPO_ROOT / "weights" / "Q-Loc" / "best_epoch_weights.pth"
CLASSES_PATH = ROOT / "model_data" / "aircraft_classes.txt"
ANCHORS_PATH = ROOT / "model_data" / "yolo_anchors.txt"
ANCHORS_MASK = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
INPUT_SHAPE = [640, 640]
CONFIDENCE = 0.4
NMS_IOU = 0.3
CLASS_AGNOSTIC_NMS = False
CLASS_AGNOSTIC_NMS_IOU = 0.1
LETTERBOX_IMAGE = True
DATASET_PATH = REPO_ROOT / "test" / "Q-Loc" / "evaluation" / "testing_data" / "MAR20_testingdata"
IMAGE_LIST_PATH = None
IMAGE_LIMIT = None
FORWARD_OUTPUT_DIR = REPO_ROOT / "outputs" / "Q-Loc" / "experiment_forward"

#
# 配置 7：yolov7_q6_mar20
# 中文简介：用 6-qubit 量子新增通道版 YOLOv7，在 MAR20 上全模型训练，用 MAR20_test 测试。
#
# MODEL_KIND = "yolov7"
# MODEL_BODY = "nets.yolov7quantumnewchanel6qubitsanglecoding64output:YoloBodyQuantumNewChanel6QubitsAngleCoding64Output"
# MODEL_PHI = "l"
# MODEL_KWARGS = {}
# WEIGHTS_PATH = ROOT / "logs" / "yolov7_q6_mar20" / "2026_05_16_09_59_42" / "checkpoints" / "best_epoch_weights.pth"
# CLASSES_PATH = ROOT / "model_data" / "mar20_classes.txt"
# ANCHORS_PATH = ROOT / "model_data" / "yolo_anchors.txt"
# ANCHORS_MASK = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
# INPUT_SHAPE = [640, 640]
# CONFIDENCE = 0.03
# NMS_IOU = 0.3
# CLASS_AGNOSTIC_NMS = True
# CLASS_AGNOSTIC_NMS_IOU = 0.1
# LETTERBOX_IMAGE = True
# DATASET_PATH = ROOT / "evaluation" / "testing_data" / "MAR20_testingdata"
# IMAGE_LIST_PATH = None
# IMAGE_LIMIT = None
# FORWARD_OUTPUT_DIR = ROOT / "experiment_forward"
#
#
# 配置 8：yolov7_mar20_hbb_white_box_finetune
# 中文简介：经典 yolo 练 MAR20，用白色背景抠图增强的模型。
#
# MODEL_KIND = "yolov7"
# MODEL_BODY = "nets.yolo:YoloBody"
# MODEL_PHI = "l"
# MODEL_KWARGS = {}
# WEIGHTS_PATH = ROOT / "logs" / "yolo_mar20_hbb_white_box_finetune_2026_05_16_14_24_20" / "checkpoints" / "best_epoch_weights.pth"
# CLASSES_PATH = ROOT / "model_data" / "mar20_classes.txt"
# ANCHORS_PATH = ROOT / "model_data" / "yolo_anchors.txt"
# ANCHORS_MASK = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
# INPUT_SHAPE = [640, 640]
# CONFIDENCE = 0.03
# NMS_IOU = 0.3
# CLASS_AGNOSTIC_NMS = True
# CLASS_AGNOSTIC_NMS_IOU = 0.1
# LETTERBOX_IMAGE = True
# DATASET_PATH = ROOT / "evaluation" / "testing_data" / "MAR20_testingdata"
# IMAGE_LIST_PATH = None
# IMAGE_LIMIT = None
# FORWARD_OUTPUT_DIR = ROOT / "experiment_forward"
#
#
# 配置 9：yolo_light_quantumchanel
# 中文简介：基于轻量级 yolo，neck 和分类都有量子通道。
#
# MODEL_KIND = "yolov7"
# MODEL_BODY = "nets.yolo_light_quantumchanel:YoloLightQuantumChanelBody"
# MODEL_PHI = "light"
# MODEL_KWARGS = {}
# WEIGHTS_PATH = ROOT / "logs" / "yolo_light_quantumchanel" / "2026_05_16_18_02_06" / "checkpoints" / "best_epoch_weights.pth"
# CLASSES_PATH = ROOT / "model_data" / "aircraft_classes.txt"
# ANCHORS_PATH = ROOT / "model_data" / "yolo_anchors.txt"
# ANCHORS_MASK = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
# INPUT_SHAPE = [640, 640]
# CONFIDENCE = 0.4
# NMS_IOU = 0.3
# CLASS_AGNOSTIC_NMS = True
# CLASS_AGNOSTIC_NMS_IOU = 0.1
# LETTERBOX_IMAGE = True
# DATASET_PATH = ROOT / "evaluation" / "testing_data" / "MAR20_test"
# IMAGE_LIST_PATH = None
# IMAGE_LIMIT = None
# FORWARD_OUTPUT_DIR = ROOT / "experiment_forward"

#
# 配置 10：yolo_light_mar20_full_fold1_aircraft
# 中文简介：砍小 yolo 权重但不变框架。
#
# MODEL_KIND = "yolov7"
# MODEL_BODY = "nets.yolo_light:YoloLightBody"
# MODEL_PHI = "light"
# MODEL_KWARGS = {}
# WEIGHTS_PATH = ROOT / "logs" / "yolo_light_mar20_full_fold1_aircraft_2026_05_16_17_21_40" / "checkpoints" / "best_epoch_weights.pth"
# CLASSES_PATH = ROOT / "model_data" / "aircraft_classes.txt"
# ANCHORS_PATH = ROOT / "model_data" / "yolo_anchors.txt"
# ANCHORS_MASK = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
# INPUT_SHAPE = [640, 640]
# CONFIDENCE = 0.15
# NMS_IOU = 0.3
# CLASS_AGNOSTIC_NMS = True
# CLASS_AGNOSTIC_NMS_IOU = 0.1
# LETTERBOX_IMAGE = True
# DATASET_PATH = ROOT / "evaluation" / "testing_data" / "MAR20_test"
# IMAGE_LIST_PATH = None
# IMAGE_LIMIT = None
# FORWARD_OUTPUT_DIR = ROOT / "experiment_forward"

#
# 配置 11：yolo_light_lighterbutnoquantum
# 中文简介：基于轻量级 yolo，把 neck 和分类前的关键 3x3 卷积降成 2x2，但不加入量子通道。
#
# MODEL_KIND = "yolov7"
# MODEL_BODY = "nets.yolo_light_lighterbutnoquantum:YoloLightLighterButNoQuantumBody"
# MODEL_PHI = "light"
# MODEL_KWARGS = {}
# WEIGHTS_PATH = ROOT / "logs" / "yolo_light_lighterbutnoquantum" / "2026_05_16_19_58_42" / "checkpoints" / "best_epoch_weights.pth"
# CLASSES_PATH = ROOT / "model_data" / "aircraft_classes.txt"
# ANCHORS_PATH = ROOT / "model_data" / "yolo_anchors.txt"
# ANCHORS_MASK = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
# INPUT_SHAPE = [640, 640]
# CONFIDENCE = 0.15
# NMS_IOU = 0.3
# CLASS_AGNOSTIC_NMS = True
# CLASS_AGNOSTIC_NMS_IOU = 0.1
# LETTERBOX_IMAGE = True
# DATASET_PATH = ROOT / "evaluation" / "testing_data" / "MAR20_test"
# IMAGE_LIST_PATH = None
# IMAGE_LIMIT = None
# FORWARD_OUTPUT_DIR = ROOT / "experiment_forward"

#
# 配置 12：yolo_light_quantum_minichange
# 中文简介：把 yolo_light_quantumchanel 的 64 维状态概率读出改成 6 个 Z 测量值，再用 MLP 放大回通道门控。
#
# MODEL_KIND = "yolov7"
# MODEL_BODY = "nets.yolo_light_quantum_minichange:YoloLightQuantumMiniChangeBody"
# MODEL_PHI = "light"
# MODEL_KWARGS = {}
# WEIGHTS_PATH = ROOT / "logs" / "yolo_light_quantum_minichange" / "2026_05_16_23_54_59" / "checkpoints" / "best_epoch_weights.pth"
# CLASSES_PATH = ROOT / "model_data" / "aircraft_classes.txt"
# ANCHORS_PATH = ROOT / "model_data" / "yolo_anchors.txt"
# ANCHORS_MASK = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
# INPUT_SHAPE = [640, 640]
# CONFIDENCE = 0.4
# NMS_IOU = 0.3
# CLASS_AGNOSTIC_NMS = True
# CLASS_AGNOSTIC_NMS_IOU = 0.1
# LETTERBOX_IMAGE = True
# DATASET_PATH = ROOT / "evaluation" / "testing_data" / "MAR20_test"
# IMAGE_LIST_PATH = None
# IMAGE_LIMIT = None
# FORWARD_OUTPUT_DIR = ROOT / "experiment_forward"

#
# 配置 13：yolo_light_multiclass_mar20_full_fold1
# 中文简介：轻量级 yolo 的 MAR20 二十分类版本，输出 A1-A20 二十种飞机类别。
#
# MODEL_KIND = "yolov7"
# MODEL_BODY = "nets.yolo_light_multiclass:YoloLightMultiClassBody"
# MODEL_PHI = "light"
# MODEL_KWARGS = {}
# WEIGHTS_PATH = ROOT / "logs" / "yolo_light_multiclass_mar20_full_fold1_2026_05_16_20_03_37" / "checkpoints" / "best_epoch_weights.pth"
# CLASSES_PATH = ROOT / "model_data" / "mar20_classes.txt"
# ANCHORS_PATH = ROOT / "model_data" / "yolo_anchors.txt"
# ANCHORS_MASK = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
# INPUT_SHAPE = [640, 640]
# CONFIDENCE = 0.05
# NMS_IOU = 0.3
# CLASS_AGNOSTIC_NMS = False
# CLASS_AGNOSTIC_NMS_IOU = 0.1
# LETTERBOX_IMAGE = True
# DATASET_PATH = ROOT / "evaluation" / "testing_data" / "MAR20"
# IMAGE_LIST_PATH = None
# IMAGE_LIMIT = None
# FORWARD_OUTPUT_DIR = ROOT / "experiment_forward"
#
#
# 对照配置 A：YOLOv7 BDD100K vehicle 单类 baseline
#
# MODEL_KIND = "yolov7"
# MODEL_BODY = "nets.yolo:YoloBody"
# MODEL_PHI = "l"
# MODEL_KWARGS = {}
# WEIGHTS_PATH = ROOT / "log" / "yolov7forbdd100k" / "2026_05_15_03_23_38" / "checkpoints" / "best_epoch_weights.pth"
# CLASSES_PATH = ROOT / "model_data" / "yolop_classes.txt"
# ANCHORS_PATH = ROOT / "model_data" / "yolo_anchors.txt"
# ANCHORS_MASK = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
# INPUT_SHAPE = [640, 640]
# CONFIDENCE = 0.35
# NMS_IOU = 0.3
# LETTERBOX_IMAGE = True
# DATASET_PATH = ROOT / "evaluation" / "testing_data" / "bdd100k_testing"
# IMAGE_LIST_PATH = None
# IMAGE_LIMIT = None
# FORWARD_OUTPUT_DIR = ROOT / "experiment_forward"
#
#
# 对照配置 B：YOLOP 自动驾驶多任务模型
#
# MODEL_KIND = "yolop"
# MODEL_BODY = "nets.yolop_adapter:YOLOPRuntime"
# MODEL_PHI = "l"
# MODEL_KWARGS = {}
# WEIGHTS_PATH = ROOT / "model_data" / "yolop" / "End-to-end.pth"
# CLASSES_PATH = ROOT / "model_data" / "yolop_classes.txt"
# ANCHORS_PATH = ROOT / "model_data" / "yolo_anchors.txt"
# ANCHORS_MASK = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
# INPUT_SHAPE = [640, 640]
# CONFIDENCE = 0.35
# NMS_IOU = 0.3
# LETTERBOX_IMAGE = True
# DATASET_PATH = ROOT / "evaluation" / "testing_data" / "bdd100k_testing"
# IMAGE_LIST_PATH = None
# IMAGE_LIMIT = None
# FORWARD_OUTPUT_DIR = ROOT / "experiment_forward"
#
#
# 旧版配置：早期 Quantum YOLOv7 快速实验骨架
# 说明：
#   使用 nets/yolov7_quantum.py 和 train_quantumyolov7.py。
#   这套曾使用 PyTorch fallback 代替真 QNN，不作为当前五个主要量子变种之一。
#
# MODEL_KIND = "yolov7"
# MODEL_BODY = "nets.yolov7_quantum:YoloBodyQuantum"
# MODEL_PHI = "l"
# MODEL_KWARGS = {
#     "use_tensorcircuit": False,
#     "quantum_blocks": 5,
# }
# WEIGHTS_PATH = ROOT / "logs" / "quantumyolov7" / "2026_05_15_01_38_05" / "checkpoints" / "best_epoch_weights.pth"
# CLASSES_PATH = ROOT / "model_data" / "yolop_classes.txt"
# ANCHORS_PATH = ROOT / "model_data" / "yolo_anchors.txt"
# ANCHORS_MASK = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
# INPUT_SHAPE = [640, 640]
# CONFIDENCE = 0.35
# NMS_IOU = 0.3
# LETTERBOX_IMAGE = True
# DATASET_PATH = ROOT / "evaluation" / "testing_data" / "bdd100k_testing"
# IMAGE_LIST_PATH = None
# IMAGE_LIMIT = None
# FORWARD_OUTPUT_DIR = ROOT / "experiment_forward"

# Qready-SepaLightQ

This repository contains the final aircraft detection, crop, and recognition pipeline.

## Directory Layout

```text
configs/
  Q-Loc/      # detection model configs and class/anchor files
  Q-Rec/      # recognition model configs
  pipeline/   # final pipeline path/config notes

models/
  Q-Loc/      # YOLO/QNN detection model definitions
  Q-Rec/      # EfficientNet/QNN recognition model definitions

train/
  Q-Loc/      # detection training entry scripts
  Q-Rec/      # recognition training entry scripts

test/
  Q-Loc/      # detection inference/evaluation entry scripts
  Q-Rec/      # recognition inference/evaluation entry scripts

pipeline/     # detect -> crop -> classify final pipeline
scripts/      # data preparation and helper scripts
utils/        # shared utilities copied from the original projects
weights/      # final submitted model weights
results/      # selected metrics and result summaries
```

## Model Roles

- `Q-Loc`: aircraft localization model, adapted from `yolov7-pytorch-master`.
- `Q-Rec`: aircraft recognition model, adapted from `aircraft_classification`.

## Final Weights

```text
weights/Q-Loc/best_epoch_weights.pth
weights/Q-Rec/best.pth
```

The historical training logs and intermediate epoch checkpoints are intentionally not included.


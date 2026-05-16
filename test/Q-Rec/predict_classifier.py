"""Run aircraft crop classifier inference on one image or a folder."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image

from aircraft_classification.train_classifier import build_model, describe_device, load_config, load_checkpoint
from aircraft_classification.utils.classification_dataloader import (
    IMAGE_SUFFIXES,
    build_transforms,
    load_class_names,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict aircraft class for crop images.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("aircraft_classification/configs/efficientnet_b0_mar20_cls.yaml"),
        help="Path to YAML config.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint path.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Single image path.")
    source.add_argument("--image-dir", type=Path, help="Directory of images.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Prediction CSV path.")
    parser.add_argument("--topk", type=int, default=5, help="Number of classes to report.")
    parser.add_argument("--device", default=None, help="Override device, for example cuda or cpu.")
    return parser.parse_args()


def collect_images(image: Path | None, image_dir: Path | None) -> list[Path]:
    if image is not None:
        return [image]
    assert image_dir is not None
    return [path for path in sorted(image_dir.rglob("*")) if path.suffix.lower() in IMAGE_SUFFIXES]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    class_names = load_class_names(cfg["data"]["classes_path"])
    input_shape = tuple(cfg["model"].get("input_shape", [128, 128]))
    transform = build_transforms(input_shape, train=False)

    requested_device = args.device or str(cfg["train"].get("device", "cuda"))
    device = torch.device("cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Device: {describe_device(device)}")

    model = build_model(cfg, num_classes=len(class_names)).to(device)
    load_checkpoint(model, args.checkpoint, strict=True)
    model.eval()

    image_paths = collect_images(args.image, args.image_dir)
    if not image_paths:
        raise ValueError("No images found for prediction.")

    topk = min(max(1, args.topk), len(class_names))
    rows: list[dict[str, str | float]] = []
    with torch.no_grad():
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            probabilities = torch.softmax(model(tensor), dim=1)[0]
            scores, indices = probabilities.topk(topk)
            row: dict[str, str | float] = {
                "image_path": str(image_path),
                "pred_class": class_names[int(indices[0])],
                "pred_score": float(scores[0]),
            }
            for rank, (score, index) in enumerate(zip(scores, indices), start=1):
                row[f"top{rank}_class"] = class_names[int(index)]
                row[f"top{rank}_score"] = float(score)
            rows.append(row)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_csv = args.output_csv or args.checkpoint.parent.parent / f"predictions_{timestamp}.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows[:10]:
        print(f"{row['image_path']} -> {row['pred_class']} ({float(row['pred_score']):.4f})")
    if len(rows) > 10:
        print(f"... {len(rows) - 10} more predictions")
    print(f"Predictions saved to {output_csv}")


if __name__ == "__main__":
    main()

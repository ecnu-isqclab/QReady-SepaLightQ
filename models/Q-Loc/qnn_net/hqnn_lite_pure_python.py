from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = Path(__file__).resolve().parent / "runs"
VEHICLE_CLASSES = {"bicycle", "bus", "car", "motorbike", "train"}
CLASS_NAMES = ["person", "vehicle", "other"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def group_class(name: str) -> str:
    if name == "person":
        return "person"
    if name in VEHICLE_CLASSES:
        return "vehicle"
    return "other"


def softmax(logits: list[float]) -> list[float]:
    max_logit = max(logits)
    exps = [math.exp(value - max_logit) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


def quantum_feature_map(features: list[float]) -> list[float]:
    """Pure-Python HQNN-lite feature map: angle encoding plus pairwise entanglement terms."""
    angles = [math.pi * value for value in features[:6]]
    q_features: list[float] = []
    for angle in angles:
        q_features.append(math.cos(angle))
        q_features.append(math.sin(angle))
    for i in range(len(angles)):
        for j in range(i + 1, len(angles)):
            q_features.append(math.cos(angles[i] + angles[j]))
            q_features.append(math.sin(angles[i] - angles[j]))
    return q_features


def load_samples(max_items: int, seed: int) -> list[tuple[list[float], int]]:
    random.seed(seed)
    split_file = ROOT / "2007_train.txt"
    ann_dir = ROOT / "VOCdevkit" / "VOC2007" / "Annotations"
    image_ids = [Path(line.split()[0]).stem for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    random.shuffle(image_ids)
    samples: list[tuple[list[float], int]] = []
    per_class = {name: 0 for name in CLASS_NAMES}
    per_class_limit = max(1, max_items // len(CLASS_NAMES))
    for image_id in image_ids:
        xml_path = ann_dir / f"{image_id}.xml"
        if not xml_path.exists():
            continue
        root = ET.parse(xml_path).getroot()
        width = float(root.findtext("size/width", "1"))
        height = float(root.findtext("size/height", "1"))
        for obj in root.findall("object"):
            source_class = (obj.findtext("name") or "").strip()
            group = group_class(source_class)
            if per_class[group] >= per_class_limit:
                continue
            bnd = obj.find("bndbox")
            if bnd is None:
                continue
            xmin = float(bnd.findtext("xmin", "0"))
            ymin = float(bnd.findtext("ymin", "0"))
            xmax = float(bnd.findtext("xmax", "0"))
            ymax = float(bnd.findtext("ymax", "0"))
            box_w = max(1.0, xmax - xmin)
            box_h = max(1.0, ymax - ymin)
            area_ratio = min(1.0, box_w * box_h / max(width * height, 1.0))
            aspect = min(4.0, box_w / box_h) / 4.0
            features = [
                ((xmin + xmax) * 0.5) / max(width, 1.0),
                ((ymin + ymax) * 0.5) / max(height, 1.0),
                min(1.0, box_w / max(width, 1.0)),
                min(1.0, box_h / max(height, 1.0)),
                area_ratio,
                aspect,
            ]
            samples.append((quantum_feature_map(features), CLASS_TO_ID[group]))
            per_class[group] += 1
            if len(samples) >= max_items:
                return samples
    return samples


def predict_probs(weights: list[list[float]], bias: list[float], x: list[float]) -> list[float]:
    logits = []
    for class_id in range(len(CLASS_NAMES)):
        logits.append(sum(w * value for w, value in zip(weights[class_id], x)) + bias[class_id])
    return softmax(logits)


def evaluate(samples: list[tuple[list[float], int]], weights: list[list[float]], bias: list[float]) -> dict[str, float]:
    correct = 0
    loss = 0.0
    for x, label in samples:
        probs = predict_probs(weights, bias, x)
        correct += int(max(range(len(probs)), key=lambda idx: probs[idx]) == label)
        loss -= math.log(max(probs[label], 1e-9))
    return {"loss": loss / max(len(samples), 1), "accuracy": correct / max(len(samples), 1)}


def train(args: argparse.Namespace) -> dict[str, object]:
    samples = load_samples(args.max_items, args.seed)
    if len(samples) < 9:
        raise RuntimeError("VOC samples are too few.")
    random.seed(args.seed)
    random.shuffle(samples)
    split = max(1, int(len(samples) * (1.0 - args.val_ratio)))
    train_samples = samples[:split]
    val_samples = samples[split:]
    feature_dim = len(samples[0][0])
    weights = [[random.uniform(-0.01, 0.01) for _ in range(feature_dim)] for _ in CLASS_NAMES]
    bias = [0.0 for _ in CLASS_NAMES]
    history = []
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_samples)
        for x, label in train_samples:
            probs = predict_probs(weights, bias, x)
            for class_id in range(len(CLASS_NAMES)):
                grad = probs[class_id] - (1.0 if class_id == label else 0.0)
                for j, value in enumerate(x):
                    weights[class_id][j] -= args.lr * grad * value
                bias[class_id] -= args.lr * grad
        train_metrics = evaluate(train_samples, weights, bias)
        val_metrics = evaluate(val_samples, weights, bias)
        row = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 5),
            "train_acc": round(train_metrics["accuracy"], 5),
            "val_loss": round(val_metrics["loss"], 5),
            "val_acc": round(val_metrics["accuracy"], 5),
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RUN_DIR / "hqnn_lite_train_log.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    model_path = RUN_DIR / "hqnn_lite_model.json"
    model_payload = {"class_names": CLASS_NAMES, "weights": weights, "bias": bias}
    model_path.write_text(json.dumps(model_payload, ensure_ascii=False), encoding="utf-8")
    label_counts = {name: 0 for name in CLASS_NAMES}
    for _, label in samples:
        label_counts[CLASS_NAMES[label]] += 1
    summary = {
        "task": "Pure-Python HQNN-lite metadata classifier",
        "说明": "当前环境没有 torch/numpy，因此该脚本用 VOC 标注框几何特征做角度编码和纠缠特征，完成一个可运行的轻量 HQNN 思路验证。",
        "samples": len(samples),
        "label_counts": label_counts,
        "feature_dim": feature_dim,
        "epochs": args.epochs,
        "history": history,
        "model": str(model_path),
        "train_log": str(csv_path),
        "wall_time_s": round(time.perf_counter() - started, 3),
    }
    summary_path = RUN_DIR / "hqnn_lite_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="无 torch/numpy 环境下可运行的 HQNN-lite 训练脚本。")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-items", type=int, default=180)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(train(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

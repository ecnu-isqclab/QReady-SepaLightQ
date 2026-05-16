from __future__ import annotations

import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = Path(__file__).resolve().parents[1]
YOLO_ROOT = REPO_ROOT
CLASSIFIER_ROOT = REPO_ROOT
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "pipeline"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class DetectionCrop:
    image_id: str
    source_image: Path
    crop_image: Path
    box: tuple[int, int, int, int]
    score: float | None = None


@dataclass(frozen=True)
class GroundTruthBox:
    image_id: str
    class_name: str
    class_id: int
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class PipelinePrediction:
    image_id: str
    class_name: str
    class_id: int
    box: tuple[int, int, int, int]
    score: float
    crop_image: Path
    detector_score: float | None = None
    classifier_score: float | None = None


def add_import_roots() -> None:
    paths = (
        REPO_ROOT,
        REPO_ROOT / "configs" / "Q-Loc",
        REPO_ROOT / "test" / "Q-Loc",
        REPO_ROOT / "models" / "Q-Loc",
        REPO_ROOT / "utils" / "Q-Loc",
        REPO_ROOT / "train" / "Q-Rec",
        REPO_ROOT / "models" / "Q-Rec",
        REPO_ROOT / "utils" / "Q-Rec" / "utils",
        WORKSPACE_ROOT,
    )
    for path in reversed(paths):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run aircraft sighting, make 128x128 white-padded crops, classify each crop, and write txt results."
    )
    parser.add_argument("--dataset-name", default="MAR20_test", help="Dataset under test/Q-Loc/evaluation/testing_data.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Directory for all generated files.")
    parser.add_argument("--run-name", default=None, help="Optional run directory name.")
    parser.add_argument(
        "--image-id",
        action="append",
        default=None,
        help="Optional image id to process. Can be passed more than once, e.g. --image-id 2027 --image-id 2030.",
    )
    parser.add_argument("--max-images", type=int, default=None, help="Limit processed source images for quick checks.")
    parser.add_argument("--visualize-count", type=int, default=30, help="Number of source images to visualize with class labels.")
    parser.add_argument("--crop-size", type=int, default=128, help="Output crop side length.")
    parser.add_argument("--min-score", type=float, default=None, help="Optional detection score threshold after YOLO output.")
    parser.add_argument(
        "--dedupe-iou",
        type=float,
        default=0.5,
        help="Suppress duplicate detection boxes whose IoU with a higher-scored box is above this value. Set <=0 to disable.",
    )
    parser.add_argument("--progress-interval", type=int, default=50, help="Print progress every N images/crops.")
    parser.add_argument(
        "--location-json-dir",
        type=Path,
        default=None,
        help="Optional existing result_location_aircraft JSON directory. If set, detection is read from JSON files instead of running YOLO.",
    )
    parser.add_argument(
        "--classifier-config",
        type=Path,
        default=CLASSIFIER_ROOT / "configs" / "Q-Rec" / "efficientnet_b0_qnn_mar20_cls_cpu_finetune.yaml",
        help="Classifier YAML config.",
    )
    parser.add_argument(
        "--classifier-checkpoint",
        type=Path,
        default=CLASSIFIER_ROOT / "weights" / "Q-Rec" / "best.pth",
        help="Classifier checkpoint.",
    )
    parser.add_argument("--device", default=None, help="Override classifier device, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument("--classifier-batch-size", type=int, default=32, help="Batch size for crop classification inference.")
    parser.add_argument("--topk", type=int, default=5, help="Top-k classes saved to CSV.")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for precision/recall/F1 and mAP.")
    parser.add_argument(
        "--score-mode",
        choices=("combined", "classifier", "detector"),
        default="combined",
        help="Score used for mAP ranking. combined=detector_score*classifier_score when detector score exists.",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Disable XML ground-truth loading and metric computation for unlabeled datasets.",
    )
    parser.add_argument(
        "--skip-classification",
        action="store_true",
        help="Only write crops and detected_crops_pending_class.txt. Useful in environments without torch.",
    )
    return parser.parse_args()


def resolve_dataset_dir(dataset_name: str) -> Path:
    dataset_dir = YOLO_ROOT / "test" / "Q-Loc" / "evaluation" / "testing_data" / dataset_name
    if (dataset_dir / "JPEGImages").exists():
        return dataset_dir
    voc_dir = dataset_dir / "VOC2007"
    if (voc_dir / "JPEGImages").exists():
        return voc_dir
    raise FileNotFoundError(f"Dataset does not contain JPEGImages: {dataset_dir}")


def find_image_path(image_dir: Path, image_id: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        path = image_dir / f"{image_id}{suffix}"
        if path.exists():
            return path
    return None


def image_paths_for_dataset(dataset_dir: Path, max_images: int | None, image_ids: list[str] | None) -> list[Path]:
    image_dir = dataset_dir / "JPEGImages"
    if image_ids:
        paths = []
        for image_id in image_ids:
            image_path = find_image_path(image_dir, image_id)
            if image_path is None:
                raise FileNotFoundError(f"Missing image id {image_id!r} under {image_dir}")
            paths.append(image_path)
        return paths
    paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    return paths[:max_images] if max_images is not None else paths


def clamp_box(box: Iterable[float], width: int, height: int) -> tuple[int, int, int, int] | None:
    left, top, right, bottom = box
    x1 = max(0, min(width - 1, int(round(left))))
    y1 = max(0, min(height - 1, int(round(top))))
    x2 = max(0, min(width, int(round(right))))
    y2 = max(0, min(height, int(round(bottom))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def crop_letterbox_white(image: Image.Image, box: tuple[int, int, int, int], size: int) -> Image.Image:
    crop = image.crop(box).convert("RGB")
    width, height = crop.size
    scale = min(size / width, size / height)
    resized_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    resized = crop.resize(resized_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    paste_xy = ((size - resized_size[0]) // 2, (size - resized_size[1]) // 2)
    canvas.paste(resized, paste_xy)
    return canvas


def box_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def dedupe_clamped_detections(
    detections: list[tuple[tuple[int, int, int, int], float | None]],
    iou_threshold: float,
) -> list[tuple[tuple[int, int, int, int], float | None]]:
    if iou_threshold <= 0 or len(detections) <= 1:
        return detections

    sorted_detections = sorted(detections, key=lambda item: item[1] if item[1] is not None else 0.0, reverse=True)
    kept: list[tuple[tuple[int, int, int, int], float | None]] = []
    for box, score in sorted_detections:
        if any(box_iou(box, kept_box) >= iou_threshold for kept_box, _ in kept):
            continue
        kept.append((box, score))
    return kept


def parse_location_json(json_path: Path) -> list[tuple[tuple[float, float, float, float], float | None]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    detections = []
    for item in data.get("最终预测框", []):
        corners = item.get("四角坐标", {})
        top_left = corners.get("左上")
        bottom_right = corners.get("右下")
        if not top_left or not bottom_right:
            continue
        detections.append(((top_left[0], top_left[1], bottom_right[0], bottom_right[1]), None))
    return detections


def run_yolo_detections(
    image_paths: list[Path],
    progress_interval: int,
) -> dict[Path, list[tuple[tuple[float, float, float, float], float | None]]]:
    add_import_roots()
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError("当前 Python 环境没有 torch，无法运行 YOLO 检测。请切换到安装了 PyTorch 的环境。") from exc
    import forward as forward_runtime

    model, class_names, bbox_util, device, _, _ = forward_runtime.build_detector()
    detections_by_image: dict[Path, list[tuple[tuple[float, float, float, float], float | None]]] = {}
    total = len(image_paths)
    for index, image_path in enumerate(image_paths, start=1):
        record = forward_runtime.run_detector(model, image_path, class_names, bbox_util, device)
        detections_by_image[image_path] = [
            (tuple(float(value) for value in det["box"]), float(det["score"])) for det in record["detections"]
        ]
        if progress_interval > 0 and (index == 1 or index % progress_interval == 0 or index == total):
            print(
                f"[detect] {index}/{total} images, current={image_path.name}, boxes={len(detections_by_image[image_path])}",
                flush=True,
            )
    return detections_by_image


def load_json_detections(
    image_paths: list[Path],
    location_json_dir: Path,
    progress_interval: int,
) -> dict[Path, list[tuple[tuple[float, float, float, float], float | None]]]:
    detections_by_image = {}
    total = len(image_paths)
    for index, image_path in enumerate(image_paths, start=1):
        json_path = location_json_dir / f"{image_path.stem}.json"
        detections_by_image[image_path] = parse_location_json(json_path) if json_path.exists() else []
        if progress_interval > 0 and (index == 1 or index % progress_interval == 0 or index == total):
            print(
                f"[load-json] {index}/{total} images, current={image_path.name}, boxes={len(detections_by_image[image_path])}",
                flush=True,
            )
    return detections_by_image


def make_crops(
    detections_by_image: dict[Path, list[tuple[tuple[float, float, float, float], float | None]]],
    crops_dir: Path,
    crop_size: int,
    min_score: float | None,
    dedupe_iou: float,
    progress_interval: int,
) -> list[DetectionCrop]:
    crops: list[DetectionCrop] = []
    crops_dir.mkdir(parents=True, exist_ok=True)
    total = len(detections_by_image)
    for image_index, (image_path, detections) in enumerate(detections_by_image.items(), start=1):
        if not detections:
            if progress_interval > 0 and (image_index == 1 or image_index % progress_interval == 0 or image_index == total):
                print(f"[crop] {image_index}/{total} images, current={image_path.name}, total_crops={len(crops)}", flush=True)
            continue
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        clamped_detections: list[tuple[tuple[int, int, int, int], float | None]] = []
        for box_float, score in detections:
            if min_score is not None and score is not None and score < min_score:
                continue
            box = clamp_box(box_float, width, height)
            if box is None:
                continue
            clamped_detections.append((box, score))
        clamped_detections = dedupe_clamped_detections(clamped_detections, dedupe_iou)
        for index, (box, score) in enumerate(clamped_detections):
            crop = crop_letterbox_white(image, box, crop_size)
            crop_path = crops_dir / f"{image_path.stem}_obj{index:03d}.jpg"
            crop.save(crop_path, quality=95)
            crops.append(
                DetectionCrop(
                    image_id=image_path.stem,
                    source_image=image_path.resolve(),
                    crop_image=crop_path.resolve(),
                    box=box,
                    score=score,
                )
            )
        if progress_interval > 0 and (image_index == 1 or image_index % progress_interval == 0 or image_index == total):
            print(f"[crop] {image_index}/{total} images, current={image_path.name}, total_crops={len(crops)}", flush=True)
    return crops


def write_pending_txt(crops: list[DetectionCrop], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{crop.crop_image} {crop.box[0]},{crop.box[1]},{crop.box[2]},{crop.box[3]}" for crop in crops]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_font(size: int = 16):
    font_path = YOLO_ROOT / "model_data" / "simhei.ttf"
    if font_path.exists():
        return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def load_class_names_from_config(config_path: Path) -> list[str]:
    add_import_roots()
    from train_classifier import load_config
    from classification_dataloader import load_class_names

    cfg = load_config(config_path)
    classes_path = Path(cfg["data"]["classes_path"])
    if not classes_path.exists():
        classes_path = REPO_ROOT / "configs" / "Q-Loc" / "model_data" / "aircraft_classes.txt"
    return load_class_names(classes_path)


def load_ground_truths(dataset_dir: Path, image_paths: list[Path], class_names: list[str]) -> dict[str, list[GroundTruthBox]]:
    class_to_id = {class_name: index for index, class_name in enumerate(class_names)}
    annotation_dir = dataset_dir / "Annotations"
    gt_by_image: dict[str, list[GroundTruthBox]] = {}
    for image_path in image_paths:
        image_id = image_path.stem
        gt_by_image[image_id] = []
        annotation_path = annotation_dir / f"{image_id}.xml"
        if not annotation_path.exists():
            continue
        try:
            root = ET.parse(annotation_path).getroot()
        except ET.ParseError:
            continue
        for obj in root.findall("object"):
            if (obj.findtext("difficult") or "0").strip() == "1":
                continue
            class_name = (obj.findtext("name") or "").strip()
            if class_name not in class_to_id:
                continue
            bnd = obj.find("bndbox")
            if bnd is None:
                continue
            box = (
                int(round(float(bnd.findtext("xmin", "0")))),
                int(round(float(bnd.findtext("ymin", "0")))),
                int(round(float(bnd.findtext("xmax", "0")))),
                int(round(float(bnd.findtext("ymax", "0")))),
            )
            gt_by_image[image_id].append(
                GroundTruthBox(
                    image_id=image_id,
                    class_name=class_name,
                    class_id=class_to_id[class_name],
                    box=box,
                )
            )
    return gt_by_image


def load_classifier(config_path: Path, checkpoint_path: Path, device_override: str | None):
    add_import_roots()
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("当前 Python 环境没有 torch，无法运行飞机分类模型。请切换到安装了 PyTorch 的环境。") from exc
    from train_classifier import build_model, load_checkpoint, load_config
    from classification_dataloader import build_transforms, load_class_names

    cfg = load_config(config_path)
    cfg["model"]["module"] = "efficientnet_b0_qnn_classifier"
    classes_path = Path(cfg["data"]["classes_path"])
    if not classes_path.exists():
        classes_path = REPO_ROOT / "configs" / "Q-Loc" / "model_data" / "aircraft_classes.txt"
        cfg["data"]["classes_path"] = str(classes_path)
    class_names = load_class_names(classes_path)
    input_shape = tuple(cfg["model"].get("input_shape", [128, 128]))
    transform = build_transforms(input_shape, train=False)
    requested_device = device_override or str(cfg["train"].get("device", "cuda"))
    use_cuda = requested_device.startswith("cuda") and torch.cuda.is_available()
    device = torch.device(requested_device if use_cuda else "cpu")
    model = build_model(cfg, num_classes=len(class_names)).to(device)
    load_checkpoint(model, checkpoint_path, strict=True)
    model.eval()
    return model, transform, class_names, device, torch


def classify_crops(
    crops: list[DetectionCrop],
    config_path: Path,
    checkpoint_path: Path,
    device_override: str | None,
    batch_size: int,
    topk: int,
    csv_path: Path,
    progress_interval: int,
) -> dict[Path, dict[str, object]]:
    model, transform, class_names, device, torch = load_classifier(config_path, checkpoint_path, device_override)
    topk = min(max(1, topk), len(class_names))
    predictions: dict[Path, dict[str, object]] = {}
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_path", "pred_id", "pred_class", "pred_score"]
    fieldnames += [f"top{rank}_{field}" for rank in range(1, topk + 1) for field in ("id", "class", "score")]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        total = len(crops)
        batch_size = max(1, int(batch_size))
        with torch.no_grad():
            for start in range(0, total, batch_size):
                batch_crops = crops[start : start + batch_size]
                tensors = []
                for crop in batch_crops:
                    image = Image.open(crop.crop_image).convert("RGB")
                    tensors.append(transform(image))
                batch_tensor = torch.stack(tensors, dim=0).to(device)
                batch_probs = torch.softmax(model(batch_tensor), dim=1)
                batch_scores, batch_indices = batch_probs.topk(topk, dim=1)

                for offset, crop in enumerate(batch_crops):
                    scores = batch_scores[offset]
                    indices = batch_indices[offset]
                    pred_id = int(indices[0])
                    row: dict[str, object] = {
                        "image_path": str(crop.crop_image),
                        "pred_id": pred_id,
                        "pred_class": class_names[pred_id],
                        "pred_score": float(scores[0]),
                    }
                    for rank, (score, class_index) in enumerate(zip(scores, indices), start=1):
                        class_id = int(class_index)
                        row[f"top{rank}_id"] = class_id
                        row[f"top{rank}_class"] = class_names[class_id]
                        row[f"top{rank}_score"] = float(score)
                    writer.writerow(row)
                    predictions[crop.crop_image] = row

                done = min(start + len(batch_crops), total)
                if progress_interval > 0 and (done == len(batch_crops) or done % progress_interval == 0 or done == total):
                    print(
                        f"[classify] {done}/{total} crops, batch_size={len(batch_crops)}, device={device}",
                        flush=True,
                    )
    return predictions


def write_final_txt(crops: list[DetectionCrop], predictions: dict[Path, dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for crop in crops:
        pred_id = int(predictions[crop.crop_image]["pred_id"])
        lines.append(f"{crop.crop_image} {crop.box[0]},{crop.box[1]},{crop.box[2]},{crop.box[3]},{pred_id}")
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def add_text(parent: ET.Element, tag: str, text: str | int) -> ET.Element:
    child = ET.SubElement(parent, tag)
    child.text = str(text)
    return child


def indent_xml(element: ET.Element, level: int = 0) -> None:
    indent = "\n" + level * "\t"
    child_indent = "\n" + (level + 1) * "\t"
    children = list(element)
    if children:
        if not element.text or not element.text.strip():
            element.text = child_indent
        for child in children:
            indent_xml(child, level + 1)
        if not children[-1].tail or not children[-1].tail.strip():
            children[-1].tail = indent
    if level and (not element.tail or not element.tail.strip()):
        element.tail = indent


def build_prediction_xml(source_image: Path, image_crops: list[DetectionCrop], predictions: dict[Path, dict[str, object]]) -> ET.Element:
    with Image.open(source_image) as image:
        width, height = image.size
        depth = len(image.getbands())

    root = ET.Element("annotation")
    add_text(root, "filename", source_image.name)

    source = ET.SubElement(root, "source")
    add_text(source, "database", "MAR20")

    size = ET.SubElement(root, "size")
    add_text(size, "width", width)
    add_text(size, "height", height)
    add_text(size, "depth", depth)
    add_text(root, "segmented", 0)

    for crop in image_crops:
        pred = predictions[crop.crop_image]
        obj = ET.SubElement(root, "object")
        add_text(obj, "name", str(pred["pred_class"]))
        bndbox = ET.SubElement(obj, "bndbox")
        x1, y1, x2, y2 = crop.box
        add_text(bndbox, "xmin", x1)
        add_text(bndbox, "ymin", y1)
        add_text(bndbox, "xmax", x2)
        add_text(bndbox, "ymax", y2)

    return root


def write_prediction_annotations(
    crops: list[DetectionCrop],
    predictions: dict[Path, dict[str, object]],
    output_dir: Path,
) -> Path:
    annotation_dir = output_dir / "Annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    for old_xml in annotation_dir.glob("*.xml"):
        old_xml.unlink()

    by_source: dict[Path, list[DetectionCrop]] = {}
    for crop in crops:
        if crop.crop_image in predictions:
            by_source.setdefault(crop.source_image, []).append(crop)

    for source_image, image_crops in by_source.items():
        root = build_prediction_xml(source_image, image_crops, predictions)
        indent_xml(root)
        ET.ElementTree(root).write(annotation_dir / f"{source_image.stem}.xml", encoding="utf-8", xml_declaration=False)
    return annotation_dir


def prediction_score(crop: DetectionCrop, pred: dict[str, object], score_mode: str) -> float:
    classifier_score = float(pred["pred_score"])
    detector_score = crop.score
    if score_mode == "classifier" or detector_score is None:
        return classifier_score
    if score_mode == "detector":
        return float(detector_score)
    return float(detector_score) * classifier_score


def build_pipeline_predictions(
    crops: list[DetectionCrop],
    predictions: dict[Path, dict[str, object]],
    score_mode: str,
) -> list[PipelinePrediction]:
    pipeline_predictions = []
    for crop in crops:
        pred = predictions[crop.crop_image]
        pipeline_predictions.append(
            PipelinePrediction(
                image_id=crop.image_id,
                class_name=str(pred["pred_class"]),
                class_id=int(pred["pred_id"]),
                box=crop.box,
                score=prediction_score(crop, pred, score_mode),
                crop_image=crop.crop_image,
                detector_score=crop.score,
                classifier_score=float(pred["pred_score"]),
            )
        )
    return pipeline_predictions


def box_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def average_precision(recalls: list[float], precisions: list[float]) -> float:
    if not recalls:
        return 0.0
    mrec = [0.0, *recalls, 1.0]
    mpre = [0.0, *precisions, 0.0]
    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])
    ap = 0.0
    for index in range(1, len(mrec)):
        if mrec[index] != mrec[index - 1]:
            ap += (mrec[index] - mrec[index - 1]) * mpre[index]
    return ap


def compute_detection_metrics(
    pipeline_predictions: list[PipelinePrediction],
    gt_by_image: dict[str, list[GroundTruthBox]],
    class_names: list[str],
    iou_threshold: float,
) -> dict[str, object]:
    gt_total_by_class = {class_id: 0 for class_id in range(len(class_names))}
    for gts in gt_by_image.values():
        for gt in gts:
            gt_total_by_class[gt.class_id] += 1

    per_class_rows = []
    total_tp = 0
    total_fp = 0
    total_gt = sum(gt_total_by_class.values())

    for class_id, class_name in enumerate(class_names):
        class_predictions = sorted(
            [pred for pred in pipeline_predictions if pred.class_id == class_id],
            key=lambda item: item.score,
            reverse=True,
        )
        matched: set[tuple[str, int]] = set()
        tp_flags = []
        fp_flags = []
        matched_ious = []

        for pred in class_predictions:
            candidate_gts = [gt for gt in gt_by_image.get(pred.image_id, []) if gt.class_id == class_id]
            best_iou = 0.0
            best_index = -1
            for gt_index, gt in enumerate(candidate_gts):
                match_key = (pred.image_id, gt_index)
                if match_key in matched:
                    continue
                iou = box_iou(pred.box, gt.box)
                if iou > best_iou:
                    best_iou = iou
                    best_index = gt_index
            if best_iou >= iou_threshold and best_index >= 0:
                matched.add((pred.image_id, best_index))
                tp_flags.append(1)
                fp_flags.append(0)
                matched_ious.append(best_iou)
            else:
                tp_flags.append(0)
                fp_flags.append(1)

        tp = sum(tp_flags)
        fp = sum(fp_flags)
        gt_count = gt_total_by_class[class_id]
        fn = max(0, gt_count - tp)
        total_tp += tp
        total_fp += fp

        cumulative_tp = 0
        cumulative_fp = 0
        recalls = []
        precisions = []
        for tp_flag, fp_flag in zip(tp_flags, fp_flags):
            cumulative_tp += tp_flag
            cumulative_fp += fp_flag
            recalls.append(cumulative_tp / gt_count if gt_count else 0.0)
            precisions.append(cumulative_tp / (cumulative_tp + cumulative_fp) if cumulative_tp + cumulative_fp else 0.0)

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / gt_count if gt_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        ap = average_precision(recalls, precisions) if gt_count else None
        mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0
        per_class_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "gt_count": gt_count,
                "pred_count": len(class_predictions),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mean_iou": mean_iou,
                "ap": ap,
            }
        )

    total_fn = max(0, total_gt - total_tp)
    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_recall = total_tp / total_gt if total_gt else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    valid_aps = [row["ap"] for row in per_class_rows if row["ap"] is not None]
    mean_ap = sum(valid_aps) / len(valid_aps) if valid_aps else 0.0
    mean_iou = (
        sum(float(row["mean_iou"]) * int(row["tp"]) for row in per_class_rows if int(row["tp"]) > 0) / total_tp
        if total_tp
        else 0.0
    )

    return {
        "iou_threshold": iou_threshold,
        "gt_count": total_gt,
        "pred_count": len(pipeline_predictions),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": micro_precision,
        "recall": micro_recall,
        "f1": micro_f1,
        "IoU": mean_iou,
        "mean_iou": mean_iou,
        "mAP": mean_ap,
        "per_class": per_class_rows,
    }


def write_metrics(metrics: dict[str, object], summary_path: Path, per_class_path: Path) -> None:
    summary = {key: value for key, value in metrics.items() if key != "per_class"}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = metrics["per_class"]
    if not rows:
        return
    with per_class_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill: tuple[int, int, int]) -> None:
    bbox = draw.textbbox(xy, text, font=font)
    draw.rectangle(bbox, fill=fill)
    draw.text(xy, text, fill=(255, 255, 255), font=font)


def draw_visualizations(
    crops: list[DetectionCrop],
    predictions: dict[Path, dict[str, object]],
    gt_by_image: dict[str, list[GroundTruthBox]] | None,
    output_dir: Path,
    max_images: int,
) -> None:
    if max_images <= 0:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    font = load_font()
    by_source: dict[Path, list[DetectionCrop]] = {}
    for crop in crops:
        by_source.setdefault(crop.source_image, []).append(crop)
    for source_index, (source_image, image_crops) in enumerate(by_source.items()):
        if source_index >= max_images:
            break
        image = Image.open(source_image).convert("RGB")
        draw = ImageDraw.Draw(image)
        if gt_by_image is not None:
            for gt in gt_by_image.get(source_image.stem, []):
                x1, y1, x2, y2 = gt.box
                label = f"GT {gt.class_name} {x1},{y1},{x2},{y2}"
                draw.rectangle((x1, y1, x2, y2), outline=(0, 210, 80), width=3)
                draw_text(draw, (x1, min(image.height - 18, max(0, y2 + 2))), label, font, (0, 90, 30))
        for crop in image_crops:
            x1, y1, x2, y2 = crop.box
            pred = predictions[crop.crop_image]
            label = f"{pred['pred_class']}({pred['pred_id']}) {float(pred['pred_score']):.2f}"
            draw.rectangle((x1, y1, x2, y2), outline=(240, 60, 60), width=3)
            draw_text(draw, (x1, max(0, y1 - 18)), label, font, (0, 0, 0))
        image.save(output_dir / f"{source_image.stem}_classified.jpg", quality=95)


def main() -> None:
    args = parse_args()
    run_name = args.run_name or args.dataset_name
    output_dir = (args.output_root / run_name).resolve()
    crops_dir = output_dir / "crops_128_white"
    pending_txt = output_dir / "detected_crops_pending_class.txt"
    final_txt = output_dir / "detected_crops_with_pred_class.txt"
    csv_path = output_dir / "classification_predictions.csv"
    visualization_dir = output_dir / "visualizations"
    metrics_summary_path = output_dir / "metrics_summary.json"
    per_class_metrics_path = output_dir / "per_class_metrics.csv"
    prediction_annotations_dir = output_dir / "Annotations"
    summary_path = output_dir / "summary.json"

    class_names = None
    gt_by_image: dict[str, list[GroundTruthBox]] = {}
    dataset_dir = resolve_dataset_dir(args.dataset_name)
    image_paths = image_paths_for_dataset(dataset_dir, args.max_images, args.image_id)
    gt_source = str((dataset_dir / "Annotations").resolve())

    if args.location_json_dir:
        detections_by_image = load_json_detections(image_paths, args.location_json_dir, args.progress_interval)
        detection_source = str(args.location_json_dir.resolve())
    else:
        detections_by_image = run_yolo_detections(image_paths, args.progress_interval)
        detection_source = "yolov7 forward_runtime.run_detector using active setting.py"

    crops = make_crops(
        detections_by_image,
        crops_dir,
        args.crop_size,
        args.min_score,
        args.dedupe_iou,
        args.progress_interval,
    )
    write_pending_txt(crops, pending_txt)
    predictions = {}
    if not args.skip_classification and crops:
        predictions = classify_crops(
            crops,
            args.classifier_config,
            args.classifier_checkpoint,
            args.device,
            args.classifier_batch_size,
            args.topk,
            csv_path,
            args.progress_interval,
        )
        write_final_txt(crops, predictions, final_txt)
        prediction_annotations_dir = write_prediction_annotations(crops, predictions, output_dir)
        if class_names is None:
            class_names = load_class_names_from_config(args.classifier_config)
        if args.skip_evaluation:
            metrics_summary_path = None
            per_class_metrics_path = None
            draw_visualizations(
                crops,
                predictions,
                None,
                visualization_dir,
                args.visualize_count,
            )
        else:
            gt_by_image = load_ground_truths(dataset_dir, image_paths, class_names)
            pipeline_predictions = build_pipeline_predictions(crops, predictions, args.score_mode)
            metrics = compute_detection_metrics(pipeline_predictions, gt_by_image, class_names, args.iou_threshold)
            write_metrics(metrics, metrics_summary_path, per_class_metrics_path)
            draw_visualizations(
                crops,
                predictions,
                gt_by_image,
                visualization_dir,
                args.visualize_count,
            )
    elif args.skip_classification:
        final_txt = None
        csv_path = None
        visualization_dir = None
        metrics_summary_path = None
        per_class_metrics_path = None
        prediction_annotations_dir = None
    else:
        write_final_txt(crops, predictions, final_txt)
        prediction_annotations_dir = write_prediction_annotations(crops, predictions, output_dir)

    summary = {
        "dataset_name": args.dataset_name,
        "dataset_dir": str(dataset_dir.resolve()),
        "ground_truth_source": None if args.skip_evaluation else gt_source,
        "detection_source": detection_source,
        "source_image_count": len(image_paths),
        "crop_count": len(crops),
        "box_format": "x1,y1,x2,y2,class_id (top-left and bottom-right corners)",
        "pending_txt": str(pending_txt),
        "final_txt": str(final_txt) if final_txt is not None else None,
        "classification_csv": str(csv_path) if csv_path is not None else None,
        "visualization_dir": str(visualization_dir) if visualization_dir is not None else None,
        "visualize_count": args.visualize_count,
        "metrics_summary": str(metrics_summary_path) if metrics_summary_path is not None else None,
        "per_class_metrics": str(per_class_metrics_path) if per_class_metrics_path is not None else None,
        "prediction_annotations_dir": str(prediction_annotations_dir) if prediction_annotations_dir is not None else None,
        "evaluation_enabled": not args.skip_evaluation and not args.skip_classification,
        "iou_threshold": args.iou_threshold,
        "score_mode": args.score_mode,
        "dedupe_iou": args.dedupe_iou,
        "classifier_config": str(args.classifier_config.resolve()),
        "classifier_checkpoint": str(args.classifier_checkpoint.resolve()),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

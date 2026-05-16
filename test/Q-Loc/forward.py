from __future__ import annotations

import importlib
import inspect
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision.ops import nms
from PIL import Image, ImageDraw, ImageFont

import setting
from utils.utils import cvtColor, get_anchors, get_classes, preprocess_input, resize_image
from utils.utils_bbox import DecodeBox


ROOT = Path(__file__).resolve().parent
FORWARD_OUTPUT_DIR = Path(setting.FORWARD_OUTPUT_DIR)
PREDICTIONS_DIR = FORWARD_OUTPUT_DIR / "predictions"
SUMMARY_JSON = FORWARD_OUTPUT_DIR / "summary.json"
SUMMARY_TXT = FORWARD_OUTPUT_DIR / "summary.txt"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_object_path(object_path: str) -> tuple[str, str]:
    if ":" not in object_path:
        raise ValueError("MODEL_BODY must use 'module.path:ClassName' format, e.g. 'nets.yolo:YoloBody'")
    module_name, object_name = object_path.split(":", 1)
    if not module_name or not object_name:
        raise ValueError(f"Invalid MODEL_BODY: {object_path!r}")
    return module_name, object_name


def load_model_class(object_path: str):
    module_name, object_name = parse_object_path(object_path)
    module = importlib.import_module(module_name)
    model_cls = getattr(module, object_name)
    if not inspect.isclass(model_cls) or not issubclass(model_cls, nn.Module):
        raise TypeError(f"{object_path} must resolve to a torch.nn.Module class")
    return model_cls


def instantiate_model(model_cls, class_count: int):
    kwargs = {
        "anchors_mask": setting.ANCHORS_MASK,
        "num_classes": class_count,
        "phi": setting.MODEL_PHI,
        "pretrained": False,
        **dict(setting.MODEL_KWARGS),
    }
    signature = inspect.signature(model_cls.__init__)
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    if not accepts_kwargs:
        kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return model_cls(**kwargs)


def load_weights_shape_match(model: nn.Module, weights_path: Path, device: torch.device) -> dict[str, object]:
    model_dict = model.state_dict()
    pretrained_dict = torch.load(weights_path, map_location=device)
    matched = {}
    skipped = []
    for key, value in pretrained_dict.items():
        if key in model_dict and tuple(model_dict[key].shape) == tuple(value.shape):
            matched[key] = value
        else:
            skipped.append(key)
    model_dict.update(matched)
    model.load_state_dict(model_dict)
    return {
        "weight_path": str(weights_path),
        "load_policy": "shape_match",
        "loaded_keys": len(matched),
        "skipped_keys": len(skipped),
    }


def build_detector():
    if getattr(setting, "MODEL_KIND", "yolov7") == "yolop":
        from nets.yolop_adapter import build_yolop_runtime

        return build_yolop_runtime(
            weights_path=resolve_path(setting.WEIGHTS_PATH),
            input_shape=setting.INPUT_SHAPE,
            confidence=setting.CONFIDENCE,
            nms_iou=setting.NMS_IOU,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes_path = resolve_path(setting.CLASSES_PATH)
    anchors_path = resolve_path(setting.ANCHORS_PATH)
    weights_path = resolve_path(setting.WEIGHTS_PATH)

    class_names, num_classes = get_classes(str(classes_path))
    anchors, _ = get_anchors(str(anchors_path))
    bbox_util = DecodeBox(
        anchors,
        num_classes,
        tuple(setting.INPUT_SHAPE),
        setting.ANCHORS_MASK,
    )
    model_cls = load_model_class(setting.MODEL_BODY)
    model = instantiate_model(model_cls, len(class_names))
    load_info = load_weights_shape_match(model, weights_path, device)
    model.to(device)
    model = model.fuse().eval() if hasattr(model, "fuse") else model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return model, class_names, bbox_util, device, parameter_count, load_info


def preprocess_pil_image(image: Image.Image) -> torch.Tensor:
    image = cvtColor(image)
    resized = resize_image(image, (setting.INPUT_SHAPE[1], setting.INPUT_SHAPE[0]), setting.LETTERBOX_IMAGE)
    image_array = np.expand_dims(
        np.transpose(preprocess_input(np.array(resized, dtype="float32")), (2, 0, 1)),
        0,
    )
    return torch.from_numpy(image_array).float()


def run_detector(
    model: nn.Module,
    image_path: Path,
    class_names: list[str],
    bbox_util: DecodeBox,
    device: torch.device,
) -> dict[str, object]:
    if getattr(setting, "MODEL_KIND", "yolov7") == "yolop":
        from nets.yolop_adapter import run_yolop_detector

        return run_yolop_detector(model, image_path, class_names)

    image = Image.open(image_path)
    image_shape = np.array(np.shape(image)[0:2])
    image_tensor = preprocess_pil_image(image).to(device)

    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = model(image_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        decoded = bbox_util.decode_box(outputs)
        decoded_tensor = torch.cat(decoded, 1)
        class_conf, _ = torch.max(decoded_tensor[:, :, 5 : 5 + len(class_names)], 2)
        pre_nms_count = int(((decoded_tensor[:, :, 4] * class_conf) >= setting.CONFIDENCE).sum().item())
        results = bbox_util.non_max_suppression(
            decoded_tensor,
            len(class_names),
            setting.INPUT_SHAPE,
            image_shape,
            setting.LETTERBOX_IMAGE,
            conf_thres=setting.CONFIDENCE,
            nms_thres=setting.NMS_IOU,
        )
        if getattr(setting, "CLASS_AGNOSTIC_NMS", False) and results[0] is not None:
            result_tensor = torch.from_numpy(results[0])
            keep = nms(
                result_tensor[:, :4],
                result_tensor[:, 4] * result_tensor[:, 5],
                float(getattr(setting, "CLASS_AGNOSTIC_NMS_IOU", setting.NMS_IOU)),
            )
            results[0] = results[0][keep.cpu().numpy()]
        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()

    detections = []
    if results[0] is not None:
        for row in results[0]:
            top, left, bottom, right = map(float, row[:4])
            score = float(row[4] * row[5])
            class_id = int(row[6])
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_names[class_id],
                    "score": score,
                    "box": [left, top, right, bottom],
                }
            )
        detections.sort(key=lambda item: item["score"], reverse=True)

    return {
        "image_path": str(image_path),
        "detections": detections,
        "forward_ms": (t1 - t0) * 1000.0,
        "postprocess_ms": (t2 - t1) * 1000.0,
        "total_ms": (t2 - t0) * 1000.0,
        "pre_nms_count": pre_nms_count,
        "post_nms_count": len(detections),
    }


def resolve_dataset_dir(dataset_path: str | Path | None = None) -> Path:
    dataset_dir = resolve_path(dataset_path or setting.DATASET_PATH)
    if (dataset_dir / "JPEGImages").exists():
        return dataset_dir
    voc2007_dir = dataset_dir / "VOC2007"
    if (voc2007_dir / "JPEGImages").exists():
        return voc2007_dir
    raise FileNotFoundError(f"Dataset must contain JPEGImages: {dataset_dir}")


def load_forward_image_ids() -> list[str]:
    image_list_path = getattr(setting, "IMAGE_LIST_PATH", None)
    limit = getattr(setting, "IMAGE_LIMIT", None)
    image_ids: list[str] = []
    if image_list_path:
        path = resolve_path(image_list_path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            token = line.strip().split()[0].replace("\\", "/")
            image_ids.append(Path(token).stem)
            if limit is not None and len(image_ids) >= int(limit):
                break
        return image_ids

    image_dir = resolve_dataset_dir() / "JPEGImages"
    image_ids = sorted(path.stem for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if limit is not None:
        image_ids = image_ids[: int(limit)]
    return image_ids


def find_image_path(image_dir: Path, image_id: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = image_dir / f"{image_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def load_font(size: int = 16):
    font_path = ROOT / "model_data" / "simhei.ttf"
    if font_path.exists():
        return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font) -> None:
    bbox = draw.textbbox(xy, text, font=font)
    draw.rectangle(bbox, fill=(0, 0, 0))
    draw.text(xy, text, fill=(255, 255, 255), font=font)


def draw_detections(image_path: Path, detections: list[dict[str, object]], output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = load_font()
    width, height = image.size
    for det in detections:
        left, top, right, bottom = (float(value) for value in det["box"])
        x1 = max(0, min(width - 1, int(round(left))))
        y1 = max(0, min(height - 1, int(round(top))))
        x2 = max(0, min(width - 1, int(round(right))))
        y2 = max(0, min(height - 1, int(round(bottom))))
        draw.rectangle((x1, y1, x2, y2), outline=(240, 60, 60), width=3)
        draw_text(draw, (x1, max(0, y1 - 18)), f"{det['class_name']} {float(det['score']):.2f}", font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)


def draw_forward_record(image_path: Path, record: dict[str, object], output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")

    drivable_mask = record.get("drivable_area_mask")
    lane_mask = record.get("lane_line_mask")
    if drivable_mask is not None or lane_mask is not None:
        base = np.array(image).astype(np.float32)
        overlay = base.copy()
        if drivable_mask is not None:
            overlay[np.asarray(drivable_mask) == 1] = [40, 210, 80]
        if lane_mask is not None:
            overlay[np.asarray(lane_mask) == 1] = [40, 120, 255]
        active = np.zeros(base.shape[:2], dtype=bool)
        if drivable_mask is not None:
            active |= np.asarray(drivable_mask) == 1
        if lane_mask is not None:
            active |= np.asarray(lane_mask) == 1
        base[active] = base[active] * 0.55 + overlay[active] * 0.45
        image = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))

    draw = ImageDraw.Draw(image)
    font = load_font()
    width, height = image.size
    for det in record["detections"]:
        left, top, right, bottom = (float(value) for value in det["box"])
        x1 = max(0, min(width - 1, int(round(left))))
        y1 = max(0, min(height - 1, int(round(top))))
        x2 = max(0, min(width - 1, int(round(right))))
        y2 = max(0, min(height - 1, int(round(bottom))))
        draw.rectangle((x1, y1, x2, y2), outline=(240, 60, 60), width=3)
        draw_text(draw, (x1, max(0, y1 - 18)), f"{det['class_name']} {float(det['score']):.2f}", font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)


def main() -> None:
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    set_seed(int(setting.SEED))
    dataset_dir = resolve_dataset_dir()
    image_dir = dataset_dir / "JPEGImages"
    image_ids = load_forward_image_ids()
    model, class_names, bbox_util, device, parameter_count, load_info = build_detector()

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    for old_file in PREDICTIONS_DIR.glob("*.jpg"):
        old_file.unlink()

    processed = []
    skipped = []
    records = []
    for image_id in image_ids:
        image_path = find_image_path(image_dir, image_id)
        if image_path is None:
            skipped.append({"image_id": image_id, "reason": "missing_image", "expected": str(image_dir / f"{image_id}.*")})
            continue
        record = run_detector(model, image_path, class_names, bbox_util, device)
        records.append(record)
        output_path = PREDICTIONS_DIR / f"{image_id}_predictions.jpg"
        draw_forward_record(image_path, record, output_path)
        processed.append(
            {
                "image_id": image_id,
                "image_path": str(image_path),
                "prediction_visualization": str(output_path),
                "detection_count": len(record["detections"]),
                "total_ms": round(float(record["total_ms"]), 6),
            }
        )

    mean_total_ms = float(np.mean([record["total_ms"] for record in records])) if records else None
    payload = {
        "实验名称": "forward前向传播实验",
        "setting": {
            "model_body": setting.MODEL_BODY,
            "weights_path": str(resolve_path(setting.WEIGHTS_PATH)),
            "classes_path": str(resolve_path(setting.CLASSES_PATH)),
            "anchors_path": str(resolve_path(setting.ANCHORS_PATH)),
            "dataset_path": str(dataset_dir),
            "image_list_path": str(resolve_path(setting.IMAGE_LIST_PATH)) if getattr(setting, "IMAGE_LIST_PATH", None) else None,
            "image_limit": getattr(setting, "IMAGE_LIMIT", None),
            "input_shape": list(setting.INPUT_SHAPE),
            "confidence": setting.CONFIDENCE,
            "nms_iou": setting.NMS_IOU,
        },
        "权重加载": load_info,
        "参数量": parameter_count,
        "请求图像数量": len(image_ids),
        "实际处理图像数量": len(processed),
        "跳过图像数量": len(skipped),
        "平均总耗时毫秒": mean_total_ms,
        "处理汇总": {
            "已处理ID": [item["image_id"] for item in processed],
            "已跳过项目": skipped,
        },
        "逐图结果": processed,
    }

    FORWARD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    SUMMARY_TXT.write_text(
        "\n".join(
            [
                f"实验名称：{payload['实验名称']}",
                f"模型结构：{setting.MODEL_BODY}",
                f"权重：{resolve_path(setting.WEIGHTS_PATH)}",
                f"数据集：{dataset_dir}",
                f"请求图像数量：{len(image_ids)}",
                f"实际处理图像数量：{len(processed)}",
                f"跳过图像数量：{len(skipped)}",
                f"平均总耗时毫秒：{mean_total_ms}",
                f"已处理ID：{', '.join(item['image_id'] for item in processed) if processed else '无'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"predictions dir: {PREDICTIONS_DIR}")
    print(f"summary txt: {SUMMARY_TXT}")
    print(f"summary json: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()

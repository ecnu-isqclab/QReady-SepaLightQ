from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_TXT = PIPELINE_ROOT / "MAR20_test_pipeline_dedupe" / "detected_crops_with_pred_class.txt"
DEFAULT_IMAGE_DIR = REPO_ROOT / "dataset" / "MAR20" / "JPEGImages"
DEFAULT_OUTPUT_DIR = PIPELINE_ROOT / "Annotations"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(frozen=True)
class PredObject:
    class_id: int
    class_name: str
    box: tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert pipeline detected_crops_with_pred_class.txt into one VOC XML annotation per source image."
    )
    parser.add_argument("--input-txt", type=Path, default=DEFAULT_INPUT_TXT, help="Pipeline txt with crop path and x1,y1,x2,y2,class_id.")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR, help="Original image directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for generated XML files.")
    parser.add_argument("--database", default="MAR20", help="Value written to annotation/source/database.")
    parser.add_argument("--class-prefix", default="A", help="Class name prefix. class_id 0 becomes A1, class_id 1 becomes A2.")
    return parser.parse_args()


def image_id_from_crop_path(crop_path: Path) -> str:
    match = re.match(r"(.+)_obj\d+$", crop_path.stem)
    if not match:
        raise ValueError(f"Cannot infer image id from crop filename: {crop_path.name}")
    return match.group(1)


def class_name_from_id(class_id: int, prefix: str) -> str:
    return f"{prefix}{class_id + 1}"


def find_image(image_dir: Path, image_id: str) -> Path:
    for suffix in IMAGE_SUFFIXES:
        image_path = image_dir / f"{image_id}{suffix}"
        if image_path.exists():
            return image_path
    raise FileNotFoundError(f"Missing source image for id {image_id!r} under {image_dir}")


def parse_prediction_line(line: str, class_prefix: str) -> tuple[str, PredObject]:
    parts = line.split()
    if len(parts) != 2:
        raise ValueError(f"Expected '<crop_path> x1,y1,x2,y2,class_id', got: {line}")

    crop_path = Path(parts[0])
    values = parts[1].split(",")
    if len(values) != 5:
        raise ValueError(f"Expected 5 comma-separated box values, got: {parts[1]}")

    x1, y1, x2, y2, class_id = (int(round(float(value))) for value in values)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid box in line: {line}")

    image_id = image_id_from_crop_path(crop_path)
    pred = PredObject(
        class_id=class_id,
        class_name=class_name_from_id(class_id, class_prefix),
        box=(x1, y1, x2, y2),
    )
    return image_id, pred


def load_predictions(input_txt: Path, class_prefix: str) -> dict[str, list[PredObject]]:
    predictions: dict[str, list[PredObject]] = defaultdict(list)
    with input_txt.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                image_id, pred = parse_prediction_line(line, class_prefix)
            except Exception as exc:
                raise ValueError(f"{input_txt}:{line_number}: {exc}") from exc
            predictions[image_id].append(pred)
    return dict(predictions)


def add_text(parent: ET.Element, tag: str, text: str | int) -> ET.Element:
    child = ET.SubElement(parent, tag)
    child.text = str(text)
    return child


def build_xml(image_path: Path, objects: list[PredObject], database: str) -> ET.Element:
    with Image.open(image_path) as image:
        width, height = image.size
        depth = len(image.getbands())

    root = ET.Element("annotation")
    add_text(root, "filename", image_path.name)

    source = ET.SubElement(root, "source")
    add_text(source, "database", database)

    size = ET.SubElement(root, "size")
    add_text(size, "width", width)
    add_text(size, "height", height)
    add_text(size, "depth", depth)
    add_text(root, "segmented", 0)

    for pred in objects:
        obj = ET.SubElement(root, "object")
        add_text(obj, "name", pred.class_name)
        bndbox = ET.SubElement(obj, "bndbox")
        x1, y1, x2, y2 = pred.box
        add_text(bndbox, "xmin", x1)
        add_text(bndbox, "ymin", y1)
        add_text(bndbox, "xmax", x2)
        add_text(bndbox, "ymax", y2)

    return root


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


def write_xml(path: Path, root: ET.Element) -> None:
    indent_xml(root)
    tree = ET.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=False)


def main() -> None:
    args = parse_args()
    predictions = load_predictions(args.input_txt, args.class_prefix)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for image_id, objects in sorted(predictions.items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0]):
        image_path = find_image(args.image_dir, image_id)
        root = build_xml(image_path, objects, args.database)
        write_xml(args.output_dir / f"{image_id}.xml", root)
        written += 1

    print(f"input_txt: {args.input_txt.resolve()}")
    print(f"image_dir: {args.image_dir.resolve()}")
    print(f"output_dir: {args.output_dir.resolve()}")
    print(f"xml_files: {written}")
    print(f"objects: {sum(len(items) for items in predictions.values())}")


if __name__ == "__main__":
    main()

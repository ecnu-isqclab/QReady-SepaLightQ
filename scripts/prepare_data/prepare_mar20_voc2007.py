from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MAR20_ROOT = PROJECT_ROOT / "evaluation" / "testing_data" / "MAR20"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "MAR20devkit" / "MAR20"
DEFAULT_CLASSES_PATH = PROJECT_ROOT / "model_data" / "mar20_classes.txt"
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "MAR20_train.txt"
DEFAULT_VAL_PATH = PROJECT_ROOT / "MAR20_val.txt"
DEFAULT_TEST_PATH = PROJECT_ROOT / "MAR20_test.txt"


MAR20_CLASSES = [f"A{i}" for i in range(1, 21)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert MAR20 horizontal bounding-box XML annotations into the "
            "VOC-style dataset layout and txt files used by this YOLOv7 repository."
        )
    )
    parser.add_argument("--mar20-root", type=Path, default=DEFAULT_MAR20_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--classes-path", type=Path, default=DEFAULT_CLASSES_PATH)
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val-path", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--keep-difficult",
        action="store_true",
        help="Keep objects marked difficult=1. By default they are skipped, matching voc_annotation.py.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images into MAR20devkit/MAR20/JPEGImages. By default, symlinks are created.",
    )
    return parser.parse_args()


def natural_image_id_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.stem
    except ValueError:
        return 10**12, path.stem


def read_int(node: ET.Element | None, name: str) -> int | None:
    if node is None:
        return None
    child = node.find(name)
    if child is None or child.text is None:
        return None
    return int(float(child.text.strip()))


def clip_box(
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    xmin = max(0, min(width - 1, xmin))
    ymin = max(0, min(height - 1, ymin))
    xmax = max(0, min(width, xmax))
    ymax = max(0, min(height, ymax))
    if xmax <= xmin or ymax <= ymin:
        return None
    return xmin, ymin, xmax, ymax


def sanitize_annotation(
    source_xml: Path,
    output_xml: Path,
    image_filename: str,
    class_to_id: dict[str, int],
    keep_difficult: bool,
) -> tuple[int, dict[str, int]]:
    root = ET.parse(source_xml).getroot()

    folder = root.find("folder")
    if folder is None:
        folder = ET.SubElement(root, "folder")
    folder.text = "JPEGImages"

    filename = root.find("filename")
    if filename is None:
        filename = ET.SubElement(root, "filename")
    filename.text = image_filename

    path = root.find("path")
    if path is not None:
        path.text = str((output_xml.parents[1] / "JPEGImages" / image_filename).resolve())

    size = root.find("size")
    width = read_int(size, "width")
    height = read_int(size, "height")
    if width is None or height is None:
        raise ValueError(f"Missing image size in {source_xml}")

    kept_objects: list[ET.Element] = []
    stats = {"objects": 0, "kept": 0, "difficult": 0, "unknown": 0, "invalid": 0}
    for obj in root.findall("object"):
        stats["objects"] += 1
        name = (obj.findtext("name") or "").strip()
        difficult = int((obj.findtext("difficult") or "0").strip())
        if difficult == 1 and not keep_difficult:
            stats["difficult"] += 1
            continue
        if name not in class_to_id:
            stats["unknown"] += 1
            continue

        bndbox = obj.find("bndbox")
        xmin = read_int(bndbox, "xmin")
        ymin = read_int(bndbox, "ymin")
        xmax = read_int(bndbox, "xmax")
        ymax = read_int(bndbox, "ymax")
        if None in (xmin, ymin, xmax, ymax):
            stats["invalid"] += 1
            continue

        clipped = clip_box(xmin, ymin, xmax, ymax, width, height)
        if clipped is None:
            stats["invalid"] += 1
            continue
        for tag, value in zip(("xmin", "ymin", "xmax", "ymax"), clipped):
            node = bndbox.find(tag)
            if node is not None:
                node.text = str(value)
        kept_objects.append(obj)
        stats["kept"] += 1

    for obj in root.findall("object"):
        root.remove(obj)
    for obj in kept_objects:
        root.append(obj)

    if stats["kept"] == 0:
        return 0, stats

    output_xml.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    if hasattr(ET, "indent"):
        ET.indent(tree, space="\t")
    else:
        indent_xml(root)
    tree.write(output_xml, encoding="utf-8", xml_declaration=False)
    return stats["kept"], stats


def convert_xml_to_line(
    image_path: Path,
    annotation_path: Path,
    class_to_id: dict[str, int],
    keep_difficult: bool,
) -> str | None:
    root = ET.parse(annotation_path).getroot()
    boxes: list[str] = []
    for obj in root.iter("object"):
        difficult = int((obj.findtext("difficult") or "0").strip())
        if difficult == 1 and not keep_difficult:
            continue

        name = (obj.findtext("name") or "").strip()
        if name not in class_to_id:
            continue

        bndbox = obj.find("bndbox")
        xmin = read_int(bndbox, "xmin")
        ymin = read_int(bndbox, "ymin")
        xmax = read_int(bndbox, "xmax")
        ymax = read_int(bndbox, "ymax")
        if None in (xmin, ymin, xmax, ymax):
            continue

        boxes.append(",".join([str(xmin), str(ymin), str(xmax), str(ymax), str(class_to_id[name])]))

    if not boxes:
        return None
    return " ".join([str(image_path.absolute()), *boxes])


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def link_or_copy_image(source: Path, target: Path, copy_images: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return
    if copy_images:
        shutil.copy2(source, target)
    else:
        target.symlink_to(source.resolve())


def main() -> None:
    args = parse_args()
    image_dir = args.mar20_root / "MAR20-JPEGImages_train"
    annotation_dir = args.mar20_root / "MAR20-Annotations_train" / "Horizontal Bounding Boxes"
    output_image_dir = args.output_root / "JPEGImages"
    output_annotation_dir = args.output_root / "Annotations"
    imagesets_dir = args.output_root / "ImageSets" / "Main"

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not annotation_dir.is_dir():
        raise FileNotFoundError(f"HBB annotation directory not found: {annotation_dir}")
    if not 0 <= args.val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1")
    if not 0 <= args.test_ratio < 1:
        raise ValueError("--test-ratio must be between 0 and 1")
    if args.val_ratio + args.test_ratio >= 1:
        raise ValueError("--val-ratio + --test-ratio must be less than 1")

    class_to_id = {name: index for index, name in enumerate(MAR20_CLASSES)}
    image_paths = sorted(image_dir.glob("*.jpg"), key=natural_image_id_key)

    image_ids: list[str] = []
    total_stats = {"objects": 0, "kept": 0, "difficult": 0, "unknown": 0, "invalid": 0}
    missing_annotations = 0
    empty_images = 0
    class_counts = [0 for _ in MAR20_CLASSES]

    for image_path in image_paths:
        annotation_path = annotation_dir / f"{image_path.stem}.xml"
        if not annotation_path.exists():
            missing_annotations += 1
            continue

        output_image_path = output_image_dir / image_path.name
        output_annotation_path = output_annotation_dir / f"{image_path.stem}.xml"
        kept_count, stats = sanitize_annotation(
            source_xml=annotation_path,
            output_xml=output_annotation_path,
            image_filename=image_path.name,
            class_to_id=class_to_id,
            keep_difficult=args.keep_difficult,
        )
        for key, value in stats.items():
            total_stats[key] += value
        if kept_count == 0:
            empty_images += 1
            continue

        link_or_copy_image(image_path, output_image_path, args.copy_images)
        line = convert_xml_to_line(
            image_path=output_image_path,
            annotation_path=output_annotation_path,
            class_to_id=class_to_id,
            keep_difficult=args.keep_difficult,
        )
        if line is None:
            empty_images += 1
            continue
        for box in line.split()[1:]:
            class_counts[int(box.rsplit(",", 1)[1])] += 1
        image_ids.append(image_path.stem)

    rng = random.Random(args.seed)
    rng.shuffle(image_ids)
    test_count = int(round(len(image_ids) * args.test_ratio))
    val_count = int(round(len(image_ids) * args.val_ratio))
    test_ids = sorted(image_ids[:test_count], key=lambda item: natural_image_id_key(Path(item)))
    val_ids = sorted(image_ids[test_count:test_count + val_count], key=lambda item: natural_image_id_key(Path(item)))
    train_ids = sorted(image_ids[test_count + val_count:], key=lambda item: natural_image_id_key(Path(item)))
    trainval_ids = sorted([*train_ids, *val_ids], key=lambda item: natural_image_id_key(Path(item)))

    def make_annotation_lines(ids: list[str]) -> list[str]:
        lines: list[str] = []
        for image_id in ids:
            line = convert_xml_to_line(
                image_path=output_image_dir / f"{image_id}.jpg",
                annotation_path=output_annotation_dir / f"{image_id}.xml",
                class_to_id=class_to_id,
                keep_difficult=args.keep_difficult,
            )
            if line is not None:
                lines.append(line)
        return lines

    write_lines(args.classes_path, MAR20_CLASSES)
    write_lines(args.train_path, make_annotation_lines(train_ids))
    write_lines(args.val_path, make_annotation_lines(val_ids))
    write_lines(args.test_path, make_annotation_lines(test_ids))
    write_lines(imagesets_dir / "train.txt", train_ids)
    write_lines(imagesets_dir / "val.txt", val_ids)
    write_lines(imagesets_dir / "test.txt", test_ids)
    write_lines(imagesets_dir / "trainval.txt", trainval_ids)

    print(f"images found: {len(image_paths)}")
    print(f"samples kept: {len(image_ids)}")
    print(f"output root: {args.output_root}")
    print(f"train samples: {len(train_ids)} -> {args.train_path}")
    print(f"val samples: {len(val_ids)} -> {args.val_path}")
    print(f"test samples: {len(test_ids)} -> {args.test_path}")
    print(f"classes: {args.classes_path}")
    print(f"objects kept: {total_stats['kept']} / {total_stats['objects']}")
    print(f"skipped difficult: {total_stats['difficult']}")
    print(f"skipped unknown class: {total_stats['unknown']}")
    print(f"skipped invalid box: {total_stats['invalid']}")
    print(f"missing annotations: {missing_annotations}")
    print(f"images without usable boxes: {empty_images}")
    print("class counts:")
    for name, count in zip(MAR20_CLASSES, class_counts):
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()

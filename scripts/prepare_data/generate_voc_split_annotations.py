import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def get_classes(classes_path):
    return [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def convert_annotation(annotation_path, classes):
    root = ET.parse(annotation_path).getroot()
    boxes = []

    for obj in root.iter("object"):
        difficult = obj.findtext("difficult", default="0")
        cls = obj.findtext("name")
        if cls not in classes or int(difficult) == 1:
            continue

        cls_id = classes.index(cls)
        xmlbox = obj.find("bndbox")
        box = (
            int(float(xmlbox.findtext("xmin"))),
            int(float(xmlbox.findtext("ymin"))),
            int(float(xmlbox.findtext("xmax"))),
            int(float(xmlbox.findtext("ymax"))),
            cls_id,
        )
        boxes.append(",".join(str(value) for value in box))

    return boxes


def generate_split(voc_root, year, split, classes, output_path):
    split_path = voc_root / f"VOC{year}" / "ImageSets" / "Main" / f"{split}.txt"
    image_dir = voc_root / f"VOC{year}" / "JPEGImages"
    annotation_dir = voc_root / f"VOC{year}" / "Annotations"

    image_ids = split_path.read_text(encoding="utf-8").strip().split()
    kept = 0
    skipped_missing = 0
    skipped_empty = 0

    with output_path.open("w", encoding="utf-8") as output:
        for image_id in image_ids:
            image_path = image_dir / f"{image_id}.jpg"
            annotation_path = annotation_dir / f"{image_id}.xml"

            if not image_path.exists() or not annotation_path.exists():
                skipped_missing += 1
                continue

            boxes = convert_annotation(annotation_path, classes)
            if not boxes:
                skipped_empty += 1
                continue

            output.write(" ".join([str(image_path), *boxes]) + "\n")
            kept += 1

    return {
        "split": split,
        "source_ids": len(image_ids),
        "kept": kept,
        "skipped_missing": skipped_missing,
        "skipped_empty": skipped_empty,
        "output": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate YOLO training annotation txt files from existing VOC ImageSets splits."
    )
    parser.add_argument("--voc-root", default="VOCdevkit", help="Path to VOCdevkit.")
    parser.add_argument("--year", default="2007", help="VOC year, e.g. 2007.")
    parser.add_argument("--classes-path", default="model_data/voc_classes.txt")
    parser.add_argument("--output-prefix", default="VOC2007")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    voc_root = resolve_path(args.voc_root)
    classes = get_classes(resolve_path(args.classes_path))

    for split in args.splits:
        output_path = ROOT / f"{args.output_prefix}_{split}.txt"
        stats = generate_split(voc_root, args.year, split, classes, output_path)
        print(
            "{split}: kept {kept}/{source_ids}, skipped_missing={skipped_missing}, "
            "skipped_empty={skipped_empty}, output={output}".format(**stats)
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

from detect_crop_classify import (
    CLASSIFIER_ROOT,
    DEFAULT_OUTPUT_ROOT,
    REPO_ROOT,
    YOLO_ROOT,
    GroundTruthBox,
    build_pipeline_predictions,
    classify_crops,
    compute_detection_metrics,
    draw_visualizations,
    load_class_names_from_config,
    load_json_detections,
    make_crops,
    run_yolo_detections,
    write_final_txt,
    write_metrics,
    write_pending_txt,
    write_prediction_annotations,
)


DEFAULT_LIST_PATH = YOLO_ROOT / "configs" / "Q-Loc" / "MAR20_test.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run detect-crop-classify pipeline on configs/Q-Loc/MAR20_test.txt."
    )
    parser.add_argument("--list-path", type=Path, default=DEFAULT_LIST_PATH, help="MAR20 txt list path.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Directory for all generated files.")
    parser.add_argument("--run-name", default="MAR20devkit_test_pipeline", help="Output run directory name.")
    parser.add_argument(
        "--image-id",
        action="append",
        default=None,
        help="Optional image id to process. Can be passed more than once, e.g. --image-id 25 --image-id 48.",
    )
    parser.add_argument("--max-images", type=int, default=None, help="Limit source images for quick checks.")
    parser.add_argument("--visualize-count", type=int, default=30, help="Number of source images to visualize.")
    parser.add_argument("--crop-size", type=int, default=128, help="Output crop side length.")
    parser.add_argument("--min-score", type=float, default=None, help="Optional detection score threshold.")
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
        help="Optional existing result_location_aircraft JSON directory. If set, detection is read from JSON files.",
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
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for TP and mAP.")
    parser.add_argument(
        "--score-mode",
        choices=("combined", "classifier", "detector"),
        default="combined",
        help="Score used for mAP ranking.",
    )
    parser.add_argument("--skip-evaluation", action="store_true", help="Disable metric computation.")
    parser.add_argument("--skip-classification", action="store_true", help="Only write crops and pending txt.")
    return parser.parse_args()


def load_mar20_txt(
    list_path: Path,
    class_names: list[str],
    max_images: int | None,
    image_ids: list[str] | None,
) -> tuple[list[Path], dict[str, list[GroundTruthBox]]]:
    image_paths: list[Path] = []
    gt_by_image: dict[str, list[GroundTruthBox]] = {}
    wanted_ids = set(image_ids or [])

    with list_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            image_path = Path(parts[0])
            if not image_path.is_absolute():
                image_path = (list_path.parent / image_path).resolve()
            image_id = image_path.stem
            if wanted_ids and image_id not in wanted_ids:
                continue

            image_paths.append(image_path)
            gt_by_image[image_id] = []
            for target in parts[1:]:
                values = target.split(",")
                if len(values) < 5:
                    continue
                x1, y1, x2, y2 = (int(round(float(value))) for value in values[:4])
                class_id = int(values[4])
                if class_id < 0 or class_id >= len(class_names):
                    continue
                gt_by_image[image_id].append(
                    GroundTruthBox(
                        image_id=image_id,
                        class_name=class_names[class_id],
                        class_id=class_id,
                        box=(x1, y1, x2, y2),
                    )
                )

            if max_images is not None and len(image_paths) >= max_images:
                break

    if not image_paths:
        raise ValueError(f"No images selected from {list_path}")
    return image_paths, gt_by_image


def main() -> None:
    args = parse_args()
    output_dir = (args.output_root / args.run_name).resolve()
    crops_dir = output_dir / "crops_128_white"
    pending_txt = output_dir / "detected_crops_pending_class.txt"
    final_txt = output_dir / "detected_crops_with_pred_class.txt"
    csv_path = output_dir / "classification_predictions.csv"
    visualization_dir = output_dir / "visualizations"
    metrics_summary_path = output_dir / "metrics_summary.json"
    per_class_metrics_path = output_dir / "per_class_metrics.csv"
    prediction_annotations_dir = output_dir / "Annotations"
    summary_path = output_dir / "summary.json"

    class_names = load_class_names_from_config(args.classifier_config)
    image_paths, gt_by_image = load_mar20_txt(args.list_path.resolve(), class_names, args.max_images, args.image_id)

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

        if args.skip_evaluation:
            metrics_summary_path = None
            per_class_metrics_path = None
            draw_visualizations(crops, predictions, None, visualization_dir, args.visualize_count)
        else:
            pipeline_predictions = build_pipeline_predictions(crops, predictions, args.score_mode)
            metrics = compute_detection_metrics(pipeline_predictions, gt_by_image, class_names, args.iou_threshold)
            write_metrics(metrics, metrics_summary_path, per_class_metrics_path)
            draw_visualizations(crops, predictions, gt_by_image, visualization_dir, args.visualize_count)
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
        "list_path": str(args.list_path.resolve()),
        "ground_truth_source": None if args.skip_evaluation else str(args.list_path.resolve()),
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

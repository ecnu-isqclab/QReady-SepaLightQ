"""Evaluate an aircraft crop classifier checkpoint."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from aircraft_classification.train_classifier import (
    build_model,
    describe_device,
    load_config,
    load_checkpoint,
    run_one_epoch,
    save_confusion_matrix,
    save_summary,
)
from aircraft_classification.utils.classification_dataloader import (
    build_dataloader,
    build_dataset,
    load_class_names,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an aircraft classifier checkpoint.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("aircraft_classification/configs/efficientnet_b0_mar20_cls.yaml"),
        help="Path to YAML config.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint path. Overrides eval.checkpoint in YAML.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for evaluation outputs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override validation batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader workers.")
    parser.add_argument("--device", default=None, help="Override device, for example cuda or cpu.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train_cfg = cfg["train"]
    eval_cfg = cfg.get("eval", {})
    class_names = load_class_names(cfg["data"]["classes_path"])
    input_shape = tuple(cfg["model"].get("input_shape", [128, 128]))

    checkpoint_path = args.checkpoint or eval_cfg.get("checkpoint")
    if checkpoint_path is None:
        raise ValueError("Checkpoint path is required. Pass --checkpoint or set eval.checkpoint in the YAML config.")
    checkpoint_path = Path(checkpoint_path)

    val_dataset = build_dataset(
        list_path=cfg["data"].get("val_list"),
        image_dir=cfg["data"].get("val_image_dir"),
        class_names=class_names,
        input_shape=input_shape,
        train=False,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=args.batch_size or int(eval_cfg.get("batch_size", train_cfg.get("batch_size", 64))),
        num_workers=(
            args.num_workers
            if args.num_workers is not None
            else int(eval_cfg.get("num_workers", train_cfg.get("num_workers", 4)))
        ),
        train=False,
    )

    requested_device = args.device or str(eval_cfg.get("device", train_cfg.get("device", "cuda")))
    device = torch.device("cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Device: {describe_device(device)}")

    model = build_model(cfg, num_classes=len(class_names)).to(device)
    load_checkpoint(model, checkpoint_path, strict=bool(eval_cfg.get("load_strict", True)))

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_classes = checkpoint.get("class_names")
    if checkpoint_classes is not None and list(checkpoint_classes) != class_names:
        raise ValueError("Checkpoint class_names do not match the config classes file.")

    criterion = nn.CrossEntropyLoss(label_smoothing=float(train_cfg.get("label_smoothing", 0.0)))
    metrics = run_one_epoch(
        model,
        val_loader,
        criterion,
        device,
        optimizer=None,
        scaler=None,
        fp16=False,
        num_classes=len(class_names),
    )

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_dir = args.output_dir or eval_cfg.get("output_dir") or checkpoint_path.parent.parent / f"eval_{timestamp}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix(output_dir / "confusion_matrix.csv", metrics["confusion_matrix"], class_names)

    summary_metrics = {
        "loss": metrics["loss"],
        "top1": metrics["top1"],
        "top5": metrics["top5"],
    }
    save_summary(
        output_dir,
        best_epoch=int(checkpoint.get("epoch", 0)),
        best_metrics=summary_metrics,
        best_confusion_matrix=metrics["confusion_matrix"],
        last_epoch=int(checkpoint.get("epoch", 0)),
        last_metrics=summary_metrics,
        class_names=class_names,
        train_samples=0,
        val_samples=len(val_dataset),
        checkpoint_dir=checkpoint_path.parent,
    )

    with (output_dir / "evaluation.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "val_samples": len(val_dataset),
                "val_loss": metrics["loss"],
                "val_top1": metrics["top1"],
                "val_top5": metrics["top5"],
                "output_dir": str(output_dir),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(f"val_loss={metrics['loss']:.4f} val_top1={metrics['top1']:.2f} val_top5={metrics['top5']:.2f}")
    print(f"Evaluation outputs saved to {output_dir}")


if __name__ == "__main__":
    main()

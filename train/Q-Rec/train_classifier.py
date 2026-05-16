"""Train an aircraft crop classifier from a YAML config."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import importlib
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from classification_dataloader import (
    build_dataloader,
    build_dataset,
    load_class_names,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 on MAR20 aircraft crops.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("aircraft_classification/configs/efficientnet_b0_mar20_cls.yaml"),
        help="Path to YAML config.",
    )
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume from.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config file: {path}")
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def import_model(module_name: str):
    module = importlib.import_module(module_name)
    if not hasattr(module, "MODEL_CLASS"):
        raise AttributeError(f"{module_name} must expose MODEL_CLASS")
    return module.MODEL_CLASS


def build_model(cfg: dict[str, Any], num_classes: int) -> nn.Module:
    model_cfg = cfg["model"]
    model_cls = import_model(model_cfg["module"])
    kwargs = dict(model_cfg)
    kwargs.pop("module", None)
    kwargs["num_classes"] = int(kwargs.get("num_classes", num_classes))
    if kwargs["num_classes"] != num_classes:
        raise ValueError(f"Config num_classes={kwargs['num_classes']} but classes file has {num_classes}")
    return model_cls(**kwargs)


def filter_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    include_prefixes: list[str] | None = None,
    exclude_prefixes: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Filter checkpoint tensors by key prefix before loading."""
    if not include_prefixes and not exclude_prefixes:
        return state_dict

    filtered = {}
    for key, value in state_dict.items():
        if include_prefixes and not any(key.startswith(prefix) for prefix in include_prefixes):
            continue
        if exclude_prefixes and any(key.startswith(prefix) for prefix in exclude_prefixes):
            continue
        filtered[key] = value
    return filtered


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    strict: bool = False,
    include_prefixes: list[str] | None = None,
    exclude_prefixes: list[str] | None = None,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    state_dict = filter_state_dict(
        state_dict,
        include_prefixes=include_prefixes,
        exclude_prefixes=exclude_prefixes,
    )
    if include_prefixes or exclude_prefixes:
        print(f"Filtered checkpoint tensors: {len(state_dict)}")
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")


def resume_checkpoint(
    checkpoint_path: Path,
    *,
    model: nn.Module,
    optimizer,
    scheduler,
    strict: bool = True,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")

    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint


def build_optimizer(cfg: dict[str, Any], model: nn.Module):
    optim_cfg = cfg["optimizer"]
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optim_type = str(optim_cfg.get("type", "adamw")).lower()
    lr = float(optim_cfg.get("lr", 3e-4))
    weight_decay = float(optim_cfg.get("weight_decay", 1e-4))
    if optim_type == "sgd":
        return SGD(params, lr=lr, momentum=float(optim_cfg.get("momentum", 0.9)), weight_decay=weight_decay)
    if optim_type == "adamw":
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer type: {optim_type}")


def build_scheduler(cfg: dict[str, Any], optimizer, epochs: int):
    scheduler_cfg = cfg.get("scheduler", {})
    scheduler_type = str(scheduler_cfg.get("type", "cosine")).lower()
    if scheduler_type == "none":
        return None
    if scheduler_type == "step":
        return StepLR(
            optimizer,
            step_size=int(scheduler_cfg.get("step_size", max(1, epochs // 3))),
            gamma=float(scheduler_cfg.get("gamma", 0.1)),
        )
    if scheduler_type == "cosine":
        base_lr = optimizer.param_groups[0]["lr"]
        min_lr_ratio = float(scheduler_cfg.get("min_lr_ratio", 0.01))
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=base_lr * min_lr_ratio)
    raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...] = (1, 5)) -> list[torch.Tensor]:
    maxk = min(max(topk), output.size(1))
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    values = []
    for k in topk:
        k = min(k, output.size(1))
        correct_k = correct[:k].reshape(-1).float().sum(0)
        values.append(correct_k.mul_(100.0 / target.size(0)))
    return values


def run_one_epoch(
    model: nn.Module,
    dataloader,
    criterion,
    device: torch.device,
    optimizer=None,
    scaler: GradScaler | None = None,
    fp16: bool = False,
    num_classes: int | None = None,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)

    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0
    per_class_correct = None
    per_class_total = None
    confusion_matrix = None
    if num_classes is not None:
        per_class_correct = torch.zeros(num_classes, dtype=torch.long)
        per_class_total = torch.zeros(num_classes, dtype=torch.long)
        confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with autocast(enabled=fp16):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if train:
                if scaler is not None and fp16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        batch_size = labels.size(0)
        top1, top5 = accuracy(outputs.detach(), labels, topk=(1, 5))
        if per_class_correct is not None and per_class_total is not None:
            predictions = outputs.detach().argmax(dim=1)
            correct = predictions.eq(labels)
            for class_id in range(num_classes):
                mask = labels == class_id
                per_class_total[class_id] += mask.sum().cpu()
                per_class_correct[class_id] += correct[mask].sum().cpu()
            if confusion_matrix is not None:
                for target_id, pred_id in zip(labels.detach().cpu(), predictions.detach().cpu()):
                    confusion_matrix[int(target_id), int(pred_id)] += 1
        total_loss += loss.item() * batch_size
        total_top1 += top1.item() * batch_size
        total_top5 += top5.item() * batch_size
        total_samples += batch_size

    metrics = {
        "loss": total_loss / max(total_samples, 1),
        "top1": total_top1 / max(total_samples, 1),
        "top5": total_top5 / max(total_samples, 1),
    }
    if per_class_correct is not None and per_class_total is not None:
        metrics["per_class_correct"] = per_class_correct.tolist()
        metrics["per_class_total"] = per_class_total.tolist()
    if confusion_matrix is not None:
        metrics["confusion_matrix"] = confusion_matrix.tolist()
    return metrics


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    best_top1: float,
    class_names: list[str],
    best_epoch: int = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_top1": best_top1,
            "best_epoch": best_epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "class_names": class_names,
        },
        path,
    )


def append_metrics(path: Path, row: dict[str, float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_text_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def describe_device(device: torch.device) -> str:
    cuda_available = torch.cuda.is_available()
    parts = [f"requested/actual device: {device}", f"cuda_available: {cuda_available}"]
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(index)
        props = torch.cuda.get_device_properties(index)
        total_gb = props.total_memory / (1024**3)
        parts.append(f"gpu_index: {index}")
        parts.append(f"gpu_name: {name}")
        parts.append(f"gpu_total_memory_gb: {total_gb:.2f}")
    return " | ".join(parts)


def resolve_device(requested_device: str) -> torch.device:
    """Resolve config device strings such as cpu, cuda, cuda:0, or cuda:1."""
    requested_device = requested_device.strip().lower()
    if requested_device.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(requested_device)
        print(f"CUDA requested as {requested_device!r}, but CUDA is not available. Falling back to CPU.")
    return torch.device("cpu")


def append_per_class_metrics(
    path: Path,
    *,
    epoch: int,
    class_names: list[str],
    correct: list[int],
    total: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        fieldnames = ["epoch", "class_id", "class_name", "correct", "total", "accuracy"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for class_id, class_name in enumerate(class_names):
            class_total = int(total[class_id])
            class_correct = int(correct[class_id])
            class_acc = 0.0 if class_total == 0 else 100.0 * class_correct / class_total
            writer.writerow(
                {
                    "epoch": epoch,
                    "class_id": class_id,
                    "class_name": class_name,
                    "correct": class_correct,
                    "total": class_total,
                    "accuracy": class_acc,
                }
            )


def save_confusion_matrix(path: Path, matrix: list[list[int]], class_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted", *class_names])
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name, *row])


def save_config_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def summarize_per_class(
    matrix: list[list[int]],
    class_names: list[str],
    top_k: int = 5,
) -> tuple[float, list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    class_rows: list[dict[str, float | int | str]] = []
    for class_id, class_name in enumerate(class_names):
        total = int(sum(matrix[class_id]))
        correct = int(matrix[class_id][class_id])
        accuracy_value = 0.0 if total == 0 else 100.0 * correct / total
        class_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "correct": correct,
                "total": total,
                "accuracy": accuracy_value,
            }
        )
    valid_rows = [row for row in class_rows if int(row["total"]) > 0]
    macro_accuracy = 0.0
    if valid_rows:
        macro_accuracy = sum(float(row["accuracy"]) for row in valid_rows) / len(valid_rows)
    worst_classes = sorted(valid_rows, key=lambda item: float(item["accuracy"]))[:top_k]
    best_classes = sorted(valid_rows, key=lambda item: float(item["accuracy"]), reverse=True)[:top_k]
    return macro_accuracy, worst_classes, best_classes


def summarize_model_size(model: nn.Module) -> dict[str, int]:
    """Return parameter and state-dict size statistics for experiment summaries."""
    stats = {
        "model_params": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_params": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "state_dict_tensors": len(model.state_dict()),
        "state_dict_numel": sum(tensor.numel() for tensor in model.state_dict().values()),
    }

    backbone = getattr(model, "backbone", None)
    features = getattr(backbone, "features", None)
    classifier = getattr(backbone, "classifier", None)
    if features is not None:
        stats["features_params"] = sum(parameter.numel() for parameter in features.parameters())
    if classifier is not None:
        stats["classifier_params"] = sum(parameter.numel() for parameter in classifier.parameters())
    return stats


def save_summary(
    run_dir: Path,
    *,
    model: nn.Module,
    best_epoch: int,
    best_metrics: dict[str, float],
    best_confusion_matrix: list[list[int]],
    last_epoch: int,
    last_metrics: dict[str, float],
    class_names: list[str],
    train_samples: int,
    val_samples: int,
    checkpoint_dir: Path,
) -> None:
    macro_accuracy, worst_classes, best_classes = summarize_per_class(best_confusion_matrix, class_names)
    model_size = summarize_model_size(model)
    summary = {
        "num_classes": len(class_names),
        **model_size,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "best_epoch": best_epoch,
        "best_val_top1": best_metrics["top1"],
        "best_val_top5": best_metrics["top5"],
        "best_val_loss": best_metrics["loss"],
        "last_epoch": last_epoch,
        "last_val_top1": last_metrics["top1"],
        "last_val_top5": last_metrics["top5"],
        "last_val_loss": last_metrics["loss"],
        "macro_class_accuracy": macro_accuracy,
        "worst_classes": worst_classes,
        "best_classes": best_classes,
        "best_checkpoint": str(checkpoint_dir / "best.pth"),
        "last_checkpoint": str(checkpoint_dir / "last.pth"),
    }

    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    lines = [
        f"num_classes: {summary['num_classes']}",
        f"model_params: {summary['model_params']}",
        f"trainable_params: {summary['trainable_params']}",
        f"features_params: {summary.get('features_params', 0)}",
        f"classifier_params: {summary.get('classifier_params', 0)}",
        f"state_dict_tensors: {summary['state_dict_tensors']}",
        f"state_dict_numel: {summary['state_dict_numel']}",
        f"train_samples: {summary['train_samples']}",
        f"val_samples: {summary['val_samples']}",
        f"best_epoch: {summary['best_epoch']}",
        f"best_val_top1: {summary['best_val_top1']:.2f}",
        f"best_val_top5: {summary['best_val_top5']:.2f}",
        f"best_val_loss: {summary['best_val_loss']:.4f}",
        f"last_epoch: {summary['last_epoch']}",
        f"last_val_top1: {summary['last_val_top1']:.2f}",
        f"last_val_top5: {summary['last_val_top5']:.2f}",
        f"last_val_loss: {summary['last_val_loss']:.4f}",
        f"macro_class_accuracy: {summary['macro_class_accuracy']:.2f}",
        f"best_checkpoint: {summary['best_checkpoint']}",
        f"last_checkpoint: {summary['last_checkpoint']}",
        "worst_classes:",
    ]
    for row in worst_classes:
        lines.append(
            f"  {row['class_name']} "
            f"(id={row['class_id']}): {float(row['accuracy']):.2f} "
            f"({row['correct']}/{row['total']})"
        )
    lines.append("best_classes:")
    for row in best_classes:
        lines.append(
            f"  {row['class_name']} "
            f"(id={row['class_id']}): {float(row['accuracy']):.2f} "
            f"({row['correct']}/{row['total']})"
        )
    with (run_dir / "summary.txt").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train_cfg = cfg["train"]

    seed_everything(int(train_cfg.get("seed", 11)))
    class_names = load_class_names(cfg["data"]["classes_path"])
    input_shape = tuple(cfg["model"].get("input_shape", [128, 128]))

    train_dataset = build_dataset(
        list_path=cfg["data"].get("train_list"),
        image_dir=cfg["data"].get("train_image_dir"),
        class_names=class_names,
        input_shape=input_shape,
        train=True,
    )
    val_dataset = build_dataset(
        list_path=cfg["data"].get("val_list"),
        image_dir=cfg["data"].get("val_image_dir"),
        class_names=class_names,
        input_shape=input_shape,
        train=False,
    )

    batch_size = int(train_cfg.get("batch_size", 64))
    num_workers = int(train_cfg.get("num_workers", 4))
    train_loader = build_dataloader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        train=True,
        use_weighted_sampler=bool(cfg["data"].get("use_weighted_sampler", False)),
        num_classes=len(class_names),
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        train=False,
    )

    requested_device = str(train_cfg.get("device", "cuda"))
    device = resolve_device(requested_device)
    fp16 = bool(train_cfg.get("fp16", False)) and device.type == "cuda"

    model = build_model(cfg, num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(train_cfg.get("label_smoothing", 0.0)))
    optimizer = build_optimizer(cfg, model)
    epochs = int(train_cfg.get("epochs", 50))
    scheduler = build_scheduler(cfg, optimizer, epochs)
    scaler = GradScaler(enabled=fp16)
    start_epoch = 1
    best_top1 = -math.inf
    best_epoch = 0

    weights_path = cfg.get("weights", {}).get("path")
    if weights_path:
        weights_cfg = cfg.get("weights", {})
        load_checkpoint(
            model,
            Path(weights_path),
            strict=bool(weights_cfg.get("load_strict", False)),
            include_prefixes=weights_cfg.get("include_prefixes"),
            exclude_prefixes=weights_cfg.get("exclude_prefixes"),
        )
    if args.resume:
        resume_state = resume_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            strict=True,
        )
        start_epoch = int(resume_state.get("epoch", 0)) + 1
        best_top1 = float(resume_state.get("best_top1", -math.inf))
        best_epoch = int(resume_state.get("best_epoch", 0))
        if start_epoch > epochs:
            raise ValueError(
                f"Resume checkpoint is already at epoch {start_epoch - 1}, "
                f"but config train.epochs is {epochs}."
            )

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    save_root = Path(train_cfg.get("save_dir", "aircraft_classification/logs"))
    run_dir = save_root / str(train_cfg.get("experiment_name", "efficientnet_b0_cls")) / timestamp
    checkpoint_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    per_class_metrics_path = run_dir / "per_class_metrics.csv"
    log_path = run_dir / "train.log"
    save_config_copy(args.config, run_dir / "config.yaml")

    header_lines = [
        f"Classes: {len(class_names)}",
        f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}",
        f"Device: {describe_device(device)} | fp16: {fp16}",
        f"Run dir: {run_dir}",
    ]
    if args.resume:
        header_lines.append(f"Resume from: {args.resume} | start_epoch: {start_epoch}")
    for line in header_lines:
        print(line)
        append_text_log(log_path, line)

    best_metrics = None
    best_confusion_matrix = None
    last_val_metrics = None
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs + 1):
        train_metrics = run_one_epoch(model, train_loader, criterion, device, optimizer, scaler, fp16)
        val_metrics = run_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            scaler=None,
            fp16=fp16,
            num_classes=len(class_names),
        )
        if scheduler is not None:
            scheduler.step()

        epoch_message = (
            f"Epoch {epoch:03d}/{epochs:03d} "
            f"train_loss={train_metrics['loss']:.4f} train_top1={train_metrics['top1']:.2f} "
            f"val_loss={val_metrics['loss']:.4f} val_top1={val_metrics['top1']:.2f} "
            f"val_top5={val_metrics['top5']:.2f}"
        )
        print(epoch_message)
        append_text_log(log_path, epoch_message)

        current_lr = optimizer.param_groups[0]["lr"]
        append_metrics(
            metrics_path,
            {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": train_metrics["loss"],
                "train_top1": train_metrics["top1"],
                "train_top5": train_metrics["top5"],
                "val_loss": val_metrics["loss"],
                "val_top1": val_metrics["top1"],
                "val_top5": val_metrics["top5"],
                "best_top1": max(best_top1, val_metrics["top1"]),
            },
        )
        append_per_class_metrics(
            per_class_metrics_path,
            epoch=epoch,
            class_names=class_names,
            correct=val_metrics["per_class_correct"],
            total=val_metrics["per_class_total"],
        )
        save_confusion_matrix(
            run_dir / f"confusion_matrix_epoch_{epoch:03d}.csv",
            val_metrics["confusion_matrix"],
            class_names,
        )

        if val_metrics["top1"] > best_top1:
            best_top1 = val_metrics["top1"]
            best_epoch = epoch
            best_metrics = {
                "loss": val_metrics["loss"],
                "top1": val_metrics["top1"],
                "top5": val_metrics["top5"],
            }
            best_confusion_matrix = val_metrics["confusion_matrix"]
            save_checkpoint(
                checkpoint_dir / "best.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                best_top1,
                class_names,
                best_epoch,
            )
            save_confusion_matrix(run_dir / "confusion_matrix_best.csv", val_metrics["confusion_matrix"], class_names)
        save_checkpoint(
            checkpoint_dir / f"epoch_{epoch:03d}.pth",
            model,
            optimizer,
            scheduler,
            epoch,
            best_top1,
            class_names,
            best_epoch,
        )
        save_checkpoint(checkpoint_dir / "last.pth", model, optimizer, scheduler, epoch, best_top1, class_names, best_epoch)
        last_val_metrics = {
            "loss": val_metrics["loss"],
            "top1": val_metrics["top1"],
            "top5": val_metrics["top5"],
        }
        last_epoch = epoch

    final_message = f"Best val top1: {best_top1:.2f}"
    print(final_message)
    append_text_log(log_path, final_message)
    if best_metrics is not None and best_confusion_matrix is not None and last_val_metrics is not None:
        save_summary(
            run_dir,
            model=model,
            best_epoch=best_epoch,
            best_metrics=best_metrics,
            best_confusion_matrix=best_confusion_matrix,
            last_epoch=last_epoch,
            last_metrics=last_val_metrics,
            class_names=class_names,
            train_samples=len(train_dataset),
            val_samples=len(val_dataset),
            checkpoint_dir=checkpoint_dir,
        )


if __name__ == "__main__":
    main()

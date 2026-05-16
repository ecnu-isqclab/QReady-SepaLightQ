from __future__ import annotations

import argparse
import datetime
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from nets.yolo_light_quantum_minichange import (
    TrueQNN6QubitsAngleCoding6ZOutput,
    YoloLightQuantumMiniChangeBody,
)
from nets.yolo_training import YOLOLoss
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import get_anchors, seed_everything, worker_init_fn


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "MAR20devkit" / "MAR20"
DEFAULT_PRETRAINED = (
    ROOT
    / "logs"
    / "yolo_light_quantumchanel"
    / "2026_05_16_18_02_06"
    / "checkpoints"
    / "best_epoch_weights.pth"
)
DEFAULT_ANCHORS = ROOT / "model_data" / "yolo_anchors.txt"
DEFAULT_SAVE_ROOT = ROOT / "logs" / "yolo_light_quantum_minichange"
ANCHORS_MASK = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune yolo_light_quantum_minichange on MAR20.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--anchors-path", type=Path, default=DEFAULT_ANCHORS)
    parser.add_argument("--save-root", type=Path, default=DEFAULT_SAVE_ROOT)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--input-shape", type=int, nargs=2, default=[640, 640])
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cuda", action="store_true")
    return parser.parse_args()


def load_annotation_line(dataset_dir: Path, image_id: str) -> str | None:
    image_path = dataset_dir / "JPEGImages" / f"{image_id}.jpg"
    annotation_path = dataset_dir / "Annotations" / f"{image_id}.xml"
    if not image_path.exists() or not annotation_path.exists():
        return None
    try:
        root = ET.parse(annotation_path).getroot()
    except ET.ParseError:
        return None
    boxes = []
    for obj in root.findall("object"):
        if (obj.findtext("difficult") or "0").strip() == "1":
            continue
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        xmin = int(float(bndbox.findtext("xmin", "0")))
        ymin = int(float(bndbox.findtext("ymin", "0")))
        xmax = int(float(bndbox.findtext("xmax", "0")))
        ymax = int(float(bndbox.findtext("ymax", "0")))
        if xmax > xmin and ymax > ymin:
            boxes.append(f"{xmin},{ymin},{xmax},{ymax},0")
    return " ".join([str(image_path), *boxes]) if boxes else None


def build_annotation_lines(dataset_dir: Path) -> tuple[list[str], list[str]]:
    annotation_ids = sorted(path.stem for path in (dataset_dir / "Annotations").glob("*.xml"))
    lines, skipped = [], []
    for image_id in annotation_ids:
        line = load_annotation_line(dataset_dir, image_id)
        if line is None:
            skipped.append(image_id)
        else:
            lines.append(line)
    if not lines:
        raise RuntimeError(f"No usable MAR20 annotations found in {dataset_dir}")
    print(f"[data] kept={len(lines)} skipped={len(skipped)}")
    return lines, skipped


def load_shape_matched_weights(model: torch.nn.Module, weight_path: Path) -> dict[str, object]:
    checkpoint = torch.load(weight_path, map_location="cpu")
    model_dict = model.state_dict()
    matched = {
        key: value
        for key, value in checkpoint.items()
        if key in model_dict and tuple(model_dict[key].shape) == tuple(value.shape)
    }
    skipped = sorted(key for key in checkpoint if key not in matched)
    model_dict.update(matched)
    model.load_state_dict(model_dict)
    return {"weight_path": str(weight_path), "loaded_keys": len(matched), "skipped_keys": len(skipped)}


def freeze_except_quantum_angles_and_readout(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    trainable = []
    for module in model.modules():
        if isinstance(module, TrueQNN6QubitsAngleCoding6ZOutput):
            module.q_angles.requires_grad = True
            trainable.append(module.q_angles)
            for parameter in module.readout.parameters():
                parameter.requires_grad = True
                trainable.append(parameter)
    return trainable


def make_run_dir(save_root: Path) -> Path:
    run_dir = save_root / datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def train_one_epoch(model, yolo_loss, optimizer, dataloader, device, epoch: int, total_epochs: int) -> float:
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}", mininterval=0.3)
    for iteration, (images, targets) in enumerate(pbar, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = yolo_loss(model(images), targets, images)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        pbar.set_postfix(loss=total_loss / iteration, lr=optimizer.param_groups[0]["lr"])
    return total_loss / max(1, len(dataloader))


@torch.no_grad()
def validate_one_epoch(model, yolo_loss, dataloader, device) -> float:
    model.eval()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="Validation", mininterval=0.3)
    for iteration, (images, targets) in enumerate(pbar, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        loss = yolo_loss(model(images), targets, images)
        total_loss += float(loss.item())
        pbar.set_postfix(val_loss=total_loss / iteration)
    return total_loss / max(1, len(dataloader))


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda:0" if cuda else "cpu")
    if cuda:
        torch.backends.cudnn.benchmark = True
    print(f"[device] {device}")
    print(f"[seed] {args.seed}")

    lines, skipped_ids = build_annotation_lines(args.dataset)
    val_count = max(1, int(len(lines) * args.val_ratio))
    train_lines, val_lines = random_split(
        lines,
        [len(lines) - val_count, val_count],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_lines, val_lines = list(train_lines), list(val_lines)
    print(f"[data] train={len(train_lines)} val={len(val_lines)} classes=1")

    anchors, _ = get_anchors(str(args.anchors_path))
    model = YoloLightQuantumMiniChangeBody(ANCHORS_MASK, num_classes=1, phi="light", pretrained=False)
    load_info = load_shape_matched_weights(model, args.pretrained) if args.pretrained.exists() else None
    trainable_params = freeze_except_quantum_angles_and_readout(model)
    print(f"[weights] {load_info}")
    print(f"[train] trainable scalar params: {sum(parameter.numel() for parameter in trainable_params)}")

    model = model.to(device)
    yolo_loss = YOLOLoss(anchors, 1, tuple(args.input_shape), ANCHORS_MASK)
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    train_dataset = YoloDataset(train_lines, tuple(args.input_shape), 1, anchors, ANCHORS_MASK, epoch_length=args.epochs, mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=True, special_aug_ratio=0)
    val_dataset = YoloDataset(val_lines, tuple(args.input_shape), 1, anchors, ANCHORS_MASK, epoch_length=args.epochs, mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False, special_aug_ratio=0)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=cuda, drop_last=True, collate_fn=yolo_dataset_collate, worker_init_fn=lambda worker_id: worker_init_fn(worker_id, 0, args.seed))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=cuda, drop_last=True, collate_fn=yolo_dataset_collate, worker_init_fn=lambda worker_id: worker_init_fn(worker_id, 0, args.seed))

    run_dir = make_run_dir(args.save_root)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "loss_history.txt"
    metadata = {
        "scheme": "yolo_light_quantumchanel mini-change: 64-state readout replaced by six Z measurements and MLP readout",
        "dataset": str(args.dataset),
        "class_names": ["aircraft"],
        "usable_samples": len(lines),
        "skipped_ids": skipped_ids,
        "val_ratio": args.val_ratio,
        "train_size": len(train_lines),
        "val_size": len(val_lines),
        "epochs": args.epochs,
        "checkpoint_every": args.checkpoint_every,
        "seed": args.seed,
        "pretrained": str(args.pretrained),
        "load_info": load_info,
        "trainable_scope": "only q_angles and new readout MLP",
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {run_dir}")

    best_val = None
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, yolo_loss, optimizer, train_loader, device, epoch, args.epochs)
        val_loss = validate_one_epoch(model, yolo_loss, val_loader, device)
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{epoch}\t{train_loss:.8f}\t{val_loss:.8f}\n")
        torch.save(model.state_dict(), checkpoint_dir / "last_epoch_weights.pth")
        if epoch % args.checkpoint_every == 0:
            torch.save(model.state_dict(), checkpoint_dir / f"epoch_{epoch:03d}_weights.pth")
        if best_val is None or val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), checkpoint_dir / "best_epoch_weights.pth")
        print(f"[epoch {epoch:03d}] train_loss={train_loss:.6f} val_loss={val_loss:.6f} best_val={best_val:.6f}")


if __name__ == "__main__":
    main()

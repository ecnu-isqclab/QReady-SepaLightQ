from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = Path(__file__).resolve().parent / "runs"
VEHICLE_CLASSES = {"bicycle", "bus", "car", "motorbike", "train"}
CLASS_NAMES = ["person", "vehicle", "other"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def voc_name_to_group(name: str) -> str:
    if name == "person":
        return "person"
    if name in VEHICLE_CLASSES:
        return "vehicle"
    return "other"


@dataclass(frozen=True)
class CropItem:
    image_path: Path
    box: tuple[int, int, int, int]
    label: int
    source_class: str


def collect_voc_crops(max_items: int, split_file: Path = ROOT / "2007_train.txt") -> list[CropItem]:
    image_dir = ROOT / "VOCdevkit" / "VOC2007" / "JPEGImages"
    ann_dir = ROOT / "VOCdevkit" / "VOC2007" / "Annotations"
    image_ids = [Path(line.split()[0]).stem for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    random.shuffle(image_ids)
    items: list[CropItem] = []
    per_group_count = {name: 0 for name in CLASS_NAMES}
    per_group_limit = max(1, max_items // len(CLASS_NAMES))
    for image_id in image_ids:
        xml_path = ann_dir / f"{image_id}.xml"
        image_path = image_dir / f"{image_id}.jpg"
        if not xml_path.exists() or not image_path.exists():
            continue
        root = ET.parse(xml_path).getroot()
        for obj in root.findall("object"):
            source_class = (obj.findtext("name") or "").strip()
            group = voc_name_to_group(source_class)
            if per_group_count[group] >= per_group_limit:
                continue
            bnd = obj.find("bndbox")
            if bnd is None:
                continue
            box = (
                max(0, int(float(bnd.findtext("xmin", "0")))),
                max(0, int(float(bnd.findtext("ymin", "0")))),
                max(1, int(float(bnd.findtext("xmax", "1")))),
                max(1, int(float(bnd.findtext("ymax", "1")))),
            )
            items.append(CropItem(image_path, box, CLASS_TO_ID[group], source_class))
            per_group_count[group] += 1
            if len(items) >= max_items:
                return items
    return items


class VOCCropDataset(Dataset):
    def __init__(self, items: list[CropItem], image_size: int = 64):
        self.items = items
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.items[index]
        image = Image.open(item.image_path).convert("RGB")
        crop = image.crop(item.box).resize((self.image_size, self.image_size), Image.BILINEAR)
        tensor = torch.tensor(list(crop.getdata()), dtype=torch.float32).view(self.image_size, self.image_size, 3)
        tensor = tensor.permute(2, 0, 1) / 255.0
        tensor = (tensor - 0.5) / 0.5
        return tensor, torch.tensor(item.label, dtype=torch.long)


class TinyCNNEncoder(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(48, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.fc(self.net(x).flatten(1)))


def _complex_gate(real_gate: torch.Tensor, device: torch.device) -> torch.Tensor:
    return real_gate.to(device=device, dtype=torch.complex64)


class QuantumCircuitLayer(nn.Module):
    """Small state-vector HQNN layer: angle encoding, Ry/Rz ansatz, all-connected CNOTs, Z measurements."""

    def __init__(self, n_qubits: int = 4, depth: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        self.theta_y = nn.Parameter(0.05 * torch.randn(depth, n_qubits))
        self.theta_z = nn.Parameter(0.05 * torch.randn(depth, n_qubits))
        self.register_buffer("z_signs", self._make_z_signs(n_qubits), persistent=False)

    @staticmethod
    def _make_z_signs(n_qubits: int) -> torch.Tensor:
        values = []
        for qubit in range(n_qubits):
            signs = []
            for basis in range(2**n_qubits):
                bit = (basis >> (n_qubits - qubit - 1)) & 1
                signs.append(1.0 if bit == 0 else -1.0)
            values.append(signs)
        return torch.tensor(values, dtype=torch.float32)

    def _apply_single_qubit(self, state: torch.Tensor, gate: torch.Tensor, qubit: int) -> torch.Tensor:
        batch = state.size(0)
        left = 2**qubit
        right = 2 ** (self.n_qubits - qubit - 1)
        state_view = state.view(batch, left, 2, right)
        out = torch.einsum("ab,blcr->blar", gate, state_view)
        return out.reshape(batch, 2**self.n_qubits)

    def _apply_cnot(self, state: torch.Tensor, control: int, target: int) -> torch.Tensor:
        indices = list(range(2**self.n_qubits))
        for basis in range(2**self.n_qubits):
            control_bit = (basis >> (self.n_qubits - control - 1)) & 1
            if control_bit:
                indices[basis] = basis ^ (1 << (self.n_qubits - target - 1))
        return state[:, indices]

    def _ry(self, angle: torch.Tensor, device: torch.device) -> torch.Tensor:
        c = torch.cos(angle / 2)
        s = torch.sin(angle / 2)
        return torch.stack([torch.stack([c, -s]), torch.stack([s, c])]).to(device=device, dtype=torch.complex64)

    def _rz(self, angle: torch.Tensor, device: torch.device) -> torch.Tensor:
        minus = torch.exp(-0.5j * angle.to(dtype=torch.complex64))
        plus = torch.exp(0.5j * angle.to(dtype=torch.complex64))
        zero = torch.zeros((), dtype=torch.complex64, device=device)
        return torch.stack([torch.stack([minus, zero]), torch.stack([zero, plus])])

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        batch = angles.size(0)
        device = angles.device
        state = torch.zeros(batch, 2**self.n_qubits, dtype=torch.complex64, device=device)
        state[:, 0] = 1.0 + 0.0j

        for qubit in range(self.n_qubits):
            state = self._apply_single_qubit(state, self._ry(math.pi * angles[:, qubit], device), qubit)

        for layer in range(self.depth):
            for qubit in range(self.n_qubits):
                state = self._apply_single_qubit(state, self._ry(self.theta_y[layer, qubit], device), qubit)
                state = self._apply_single_qubit(state, self._rz(self.theta_z[layer, qubit], device), qubit)
            for control in range(self.n_qubits):
                for target in range(control + 1, self.n_qubits):
                    state = self._apply_cnot(state, control, target)

        probs = state.abs().pow(2)
        signs = self.z_signs.to(device=device)
        return probs @ signs.t()


class HQNNClassifier(nn.Module):
    def __init__(self, n_qubits: int = 4, depth: int = 2, num_classes: int = 3):
        super().__init__()
        self.encoder = TinyCNNEncoder(out_dim=n_qubits)
        self.quantum = QuantumCircuitLayer(n_qubits=n_qubits, depth=depth)
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = self.encoder(x)
        q_features = self.quantum(angles)
        return self.classifier(q_features)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss_sum += float(loss) * labels.numel()
            correct += int((logits.argmax(1) == labels).sum())
            total += labels.numel()
    return {"loss": loss_sum / max(total, 1), "accuracy": correct / max(total, 1)}


def train(args: argparse.Namespace) -> dict[str, object]:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    items = collect_voc_crops(args.max_items)
    dataset = VOCCropDataset(items, image_size=args.image_size)
    if len(dataset) < 6:
        raise RuntimeError("VOC crop samples are too few for train/val split.")
    val_size = max(3, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = HQNNClassifier(n_qubits=args.n_qubits, depth=args.depth, num_classes=len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)
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
    csv_path = RUN_DIR / "hqnn_train_log.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    weights_path = RUN_DIR / "hqnn_classifier.pt"
    torch.save(model.state_dict(), weights_path)
    label_counts = {name: 0 for name in CLASS_NAMES}
    for item in items:
        label_counts[CLASS_NAMES[item.label]] += 1
    payload = {
        "task": "VOC crop classification with paper-inspired HQNN",
        "class_names": CLASS_NAMES,
        "label_counts": label_counts,
        "samples": len(items),
        "device": str(device),
        "epochs": args.epochs,
        "n_qubits": args.n_qubits,
        "depth": args.depth,
        "history": history,
        "weights": str(weights_path),
        "train_log": str(csv_path),
        "wall_time_s": round(time.perf_counter() - started, 3),
        "paper_mapping": "对应论文 HQNN：角度编码 + Ry/Rz 可调量子门 + 全连接纠缠 + Z 测量 + 分类层；可与 YOLO 分类概率做平均融合。",
    }
    summary_path = RUN_DIR / "hqnn_train_summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def fuse_probabilities(yolo_probs: torch.Tensor, hqnn_probs: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """Average-fusion strategy from the paper: alpha*YOLO + (1-alpha)*HQNN."""
    if yolo_probs.shape != hqnn_probs.shape:
        raise ValueError("probability tensors must have the same shape")
    return alpha * yolo_probs + (1.0 - alpha) * hqnn_probs


def main() -> None:
    parser = argparse.ArgumentParser(description="轻量复现 quantumenhanced-main.pdf 里的 HQNN 分类校正模块。")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-items", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    payload = train(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

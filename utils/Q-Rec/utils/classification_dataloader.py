"""Dataset and dataloader helpers for aircraft crop classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ClassificationSample:
    image_path: Path
    label: int


def load_class_names(classes_path: str | Path) -> list[str]:
    path = Path(classes_path)
    with path.open("r", encoding="utf-8") as handle:
        class_names = [line.strip() for line in handle if line.strip()]
    if not class_names:
        raise ValueError(f"No class names found in {path}")
    return class_names


def parse_crop_list(list_path: str | Path) -> list[ClassificationSample]:
    """Parse crop annotation txt.

    Expected row format:
        /abs/path/to/image.jpg x1,y1,x2,y2,class_id

    The bbox fields are kept in the file for traceability but are not needed by
    the classifier because each image is already a 128x128 crop.

    The label is read directly from the final comma-separated value.
    For example, ``1_obj000_A2.jpg 4,0,123,128,1`` has class id ``1``.
    """
    path = Path(list_path)
    samples: list[ClassificationSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Invalid row in {path}:{line_number}: {raw_line!r}")
            image_path = Path(parts[0])
            target_parts = parts[1].split(",")
            if len(target_parts) < 5:
                raise ValueError(f"Missing class id in {path}:{line_number}: {raw_line!r}")
            label = int(target_parts[-1])
            samples.append(ClassificationSample(image_path=image_path, label=label))
    if not samples:
        raise ValueError(f"No samples found in {path}")
    return samples


def scan_image_folder(root: str | Path, class_names: Iterable[str]) -> list[ClassificationSample]:
    """Scan an ImageFolder-style split directory.

    Expected layout:
        root/A1/*.jpg
        root/A2/*.jpg
        ...
    """
    root_path = Path(root)
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    samples: list[ClassificationSample] = []
    for class_name, class_id in class_to_id.items():
        class_dir = root_path / class_name
        if not class_dir.is_dir():
            continue
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.suffix.lower() in IMAGE_SUFFIXES:
                samples.append(ClassificationSample(image_path=image_path, label=class_id))
    if not samples:
        raise ValueError(f"No images found under {root_path}")
    return samples


class AircraftCropDataset(Dataset):
    """Dataset returning ``image_tensor, class_id`` for aircraft crops."""

    def __init__(
        self,
        samples: list[ClassificationSample],
        transform: Callable | None = None,
        num_classes: int | None = None,
    ) -> None:
        self.samples = samples
        self.transform = transform
        self.num_classes = num_classes
        if num_classes is not None:
            for sample in self.samples:
                if sample.label < 0 or sample.label >= num_classes:
                    raise ValueError(
                        f"Label {sample.label} is outside valid range [0, {num_classes}) "
                        f"for image {sample.image_path}"
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, sample.label

    def class_counts(self, num_classes: int) -> list[int]:
        counts = [0 for _ in range(num_classes)]
        for sample in self.samples:
            counts[sample.label] += 1
        return counts


def build_transforms(input_shape: tuple[int, int], train: bool) -> transforms.Compose:
    height, width = input_shape
    if train:
        return transforms.Compose(
            [
                transforms.Resize((height, width)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.15,
                            contrast=0.15,
                            saturation=0.10,
                            hue=0.02,
                        )
                    ],
                    p=0.5,
                ),
                transforms.RandomAffine(
                    degrees=8,
                    translate=(0.04, 0.04),
                    scale=(0.92, 1.08),
                    fill=0,
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_dataset(
    *,
    list_path: str | Path | None,
    image_dir: str | Path | None,
    class_names: list[str],
    input_shape: tuple[int, int],
    train: bool,
) -> AircraftCropDataset:
    if list_path:
        samples = parse_crop_list(list_path)
    elif image_dir:
        samples = scan_image_folder(image_dir, class_names)
    else:
        raise ValueError("Either list_path or image_dir must be provided.")
    return AircraftCropDataset(
        samples=samples,
        transform=build_transforms(input_shape, train=train),
        num_classes=len(class_names),
    )


def build_dataloader(
    dataset: AircraftCropDataset,
    *,
    batch_size: int,
    num_workers: int,
    train: bool,
    use_weighted_sampler: bool = False,
    num_classes: int | None = None,
) -> DataLoader:
    sampler = None
    shuffle = train
    if train and use_weighted_sampler:
        if num_classes is None:
            raise ValueError("num_classes is required when use_weighted_sampler=True")
        counts = dataset.class_counts(num_classes)
        weights_by_class = [0.0 if count == 0 else 1.0 / count for count in counts]
        sample_weights = [weights_by_class[sample.label] for sample in dataset.samples]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
    )

import ast
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


CLASSES_8 = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
]


@dataclass(frozen=True)
class Cxr8WsodPaths:
    image_dir: Path
    split_csv: Path
    prior_dir: Path


def _parse_labels_cell(cell: str) -> List[str]:
    labels = ast.literal_eval(cell)
    if not isinstance(labels, list):
        raise ValueError(f"Expected labels cell to be a list literal, got: {cell!r}")
    labels = [str(x) for x in labels]

    # Normalize "No Finding" to empty for C=8 setting.
    if "No Finding" in labels:
        labels = [x for x in labels if x != "No Finding"]
    return labels


def _filename_to_image_id(file_name: str) -> int:
    """
    Convert e.g. '00000002_000.png' -> integer id that is stable and unique
    for common CXR8 naming conventions.
    """
    stem = Path(file_name).stem
    parts = stem.split("_")
    try:
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            return int(parts[0]) * 1000 + int(parts[1])
        if parts[0].isdigit():
            return int(parts[0])
    except Exception:
        pass
    return abs(hash(stem)) % (2**31)


class Cxr8WsodDataset(Dataset):
    def __init__(
        self,
        paths: Cxr8WsodPaths,
        *,
        fixed_size: int = 512,
        prior_hw: Tuple[int, int] = (36, 36),
        hflip_p: float = 0.5,
        is_train: bool = True,
        return_pil: bool = False,
        exclude_files: Optional[Sequence[str]] = None,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
        class_names: Sequence[str] = tuple(CLASSES_8),
    ) -> None:
        self.image_dir = paths.image_dir
        self.split_csv = paths.split_csv
        self.prior_dir = paths.prior_dir

        self.fixed_size = int(fixed_size)
        self.prior_hw = (int(prior_hw[0]), int(prior_hw[1]))
        self.hflip_p = float(hflip_p)
        self.is_train = bool(is_train)
        self.return_pil = bool(return_pil)
        self.exclude_files = set(exclude_files or [])
        self.mean = tuple(float(x) for x in mean)
        self.std = tuple(float(x) for x in std)

        self.class_names = list(class_names)
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        self.samples: List[Tuple[str, List[str]]] = []
        self._load_split()

    def _load_split(self) -> None:
        if not self.split_csv.exists():
            raise FileNotFoundError(f"Split CSV not found: {self.split_csv}")

        with self.split_csv.open("r", newline="") as f:
            reader = csv.DictReader(f)
            if "file_name" not in (reader.fieldnames or []) or "labels" not in (reader.fieldnames or []):
                raise ValueError(f"Unexpected columns in {self.split_csv}: {reader.fieldnames}")

            skipped = 0
            for row in reader:
                file_name = row["file_name"].strip()
                if file_name in self.exclude_files:
                    skipped += 1
                    continue
                labels = _parse_labels_cell(row["labels"])
                self.samples.append((file_name, labels))

        if not self.samples:
            raise ValueError(f"No samples found in split CSV: {self.split_csv}")
        if skipped:
            # Lightweight notice for leakage-safe training.
            print(f"[Cxr8WsodDataset] excluded {skipped} files from {self.split_csv.name}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_priors(self, file_name: str) -> torch.Tensor:
        prior_path = self.prior_dir / (Path(file_name).stem + ".npz")
        if not prior_path.exists():
            raise FileNotFoundError(f"Missing prior file: {prior_path}")

        with np.load(prior_path) as x:
            cardiac = x["cardiac"]
            pulmonary = x["pulmonary"]

        if cardiac.shape != self.prior_hw or pulmonary.shape != self.prior_hw:
            raise ValueError(
                f"Unexpected prior shapes in {prior_path}: cardiac={cardiac.shape}, pulmonary={pulmonary.shape}, "
                f"expected={self.prior_hw}"
            )

        cardiac_t = torch.from_numpy(cardiac.astype(np.float32))
        pulmonary_t = torch.from_numpy(pulmonary.astype(np.float32))
        agnostic_t = torch.zeros_like(cardiac_t)
        pri = torch.stack([cardiac_t, pulmonary_t, agnostic_t], dim=0)
        return pri

    def _encode_labels(self, labels: List[str]) -> torch.Tensor:
        y = torch.zeros((len(self.class_names),), dtype=torch.float32)
        for lb in labels:
            if lb not in self.class_to_idx:
                # Ignore labels outside the 8 CXR8 classes
                continue
            y[self.class_to_idx[lb]] = 1.0
        return y

    def __getitem__(self, idx: int):
        file_name, labels = self.samples[idx]
        img_path = self.image_dir / file_name
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        if (not self.return_pil) and self.fixed_size and self.fixed_size > 0:
            image = TF.resize(image, [self.fixed_size, self.fixed_size], interpolation=Image.BILINEAR)

        priors = self._load_priors(file_name)

        do_hflip = self.is_train and (random.random() < self.hflip_p)
        if do_hflip:
            image = TF.hflip(image)
            priors = priors.flip(-1)  # flip width on (3,Hp,Wp)

        if self.return_pil:
            image_out = image
        else:
            image_t = TF.to_tensor(image)
            image_t = TF.normalize(image_t, mean=self.mean, std=self.std)
            image_out = image_t

        target = {
            "image_id": torch.tensor([_filename_to_image_id(file_name)], dtype=torch.int64),
            "file_name": file_name,
            "img_labels": self._encode_labels(labels),
            "priors": priors,
        }
        return image_out, target

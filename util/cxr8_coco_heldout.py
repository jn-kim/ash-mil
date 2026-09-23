import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset


class CxrCocoHeldout(Dataset):
    """
    Minimal COCO-format dataset wrapper for CXR8 held-out bbox annotations.

    Returns:
      - image: PIL.Image (RGB)
      - priors: FloatTensor (3,H',W') from cxas_mask npz (cardiac/pulmonary/agnostic)
      - meta: {image_id, file_name, size=(H,W), anns=[...]}
    """

    def __init__(self, ann_path: Path, image_dir: Path, prior_dir: Path) -> None:
        self.ann_path = Path(ann_path)
        self.image_dir = Path(image_dir)
        self.prior_dir = Path(prior_dir)

        d = json.loads(self.ann_path.read_text())
        self.images = d["images"]
        self.annotations = d["annotations"]
        self.categories = d.get("categories", [])

        self.anns_by_img: Dict[int, List[dict]] = {}
        for a in self.annotations:
            self.anns_by_img.setdefault(int(a["image_id"]), []).append(a)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        import numpy as np
        from PIL import Image

        im = self.images[idx]
        file_name = str(im["file_name"])
        img_path = self.image_dir / file_name
        prior_path = self.prior_dir / (Path(file_name).stem + ".npz")

        img = Image.open(img_path).convert("RGB")
        with np.load(prior_path) as x:
            cardiac = x["cardiac"].astype("float32")
            pulmonary = x["pulmonary"].astype("float32")
        priors = np.stack([cardiac, pulmonary, np.zeros_like(cardiac)], axis=0)
        priors_t = torch.from_numpy(priors)

        meta = {
            "image_id": int(im["id"]),
            "file_name": file_name,
            "size": (int(im["height"]), int(im["width"])),
            "anns": self.anns_by_img.get(int(im["id"]), []),
        }
        return img, priors_t, meta


def collate_cxr_coco_heldout(batch):
    images, priors, meta = zip(*batch)
    return list(images), torch.stack(list(priors), dim=0), list(meta)

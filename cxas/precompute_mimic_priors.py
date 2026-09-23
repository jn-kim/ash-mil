from __future__ import annotations

import argparse
import os
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from tqdm import tqdm


@dataclass(frozen=True)
class PriorConfig:
    model_name: str
    gpus: str
    batch_size: int
    num_workers: int
    sigma: float
    token_hw: Tuple[int, int]
    cardiac_ids: Tuple[int, ...]
    pulmonary_ids: Tuple[int, ...]
    overwrite: bool

    ann: str
    mimic_files_root: str
    mimic_metadata_csv: str


def write_meta(output_dir: Path, cfg: PriorConfig) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ashmil_cxas_priors_meta.json").write_text(json.dumps(asdict(cfg), indent=2))


def _dicom_ids_from_ann(ann_path: Path) -> List[str]:
    d = json.loads(ann_path.read_text())
    if "images" not in d:
        raise ValueError(f"COCO json missing 'images': {ann_path}")
    return sorted({Path(im["file_name"]).stem for im in d["images"] if im.get("file_name")})


def _build_dicom_to_jpg_path(
    dicom_ids: Iterable[str],
    *,
    metadata_csv: Path,
    files_root: Path,
) -> Tuple[Dict[str, Path], List[str]]:
    want = set(str(d).strip() for d in dicom_ids if str(d).strip())
    out: Dict[str, Path] = {}
    if not want:
        return out, []

    with metadata_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            did = (row.get("dicom_id") or "").strip()
            if not did or did not in want:
                continue
            subj = (row.get("subject_id") or "").strip()
            study = (row.get("study_id") or "").strip()
            if not subj or not study:
                continue
            prefix = subj[:2]
            rel = Path(f"p{prefix}") / f"p{subj}" / f"s{study}" / f"{did}.jpg"
            out[did] = files_root / rel
            if len(out) == len(want):
                break

    missing = sorted([d for d in want if d not in out])
    return out, missing


class MimicJpgDataset:
    def __init__(self, dicom_ids: Sequence[str], dicom_to_path: Dict[str, Path]):
        from cxas.file_io import FileLoader

        self.dicom_ids = list(dicom_ids)
        self.dicom_to_path = dict(dicom_to_path)
        # Load on CPU: each dataloader worker would otherwise create its own CUDA context.
        self.loader = FileLoader(gpus="cpu")

    def __len__(self) -> int:
        return len(self.dicom_ids)

    def __getitem__(self, idx: int):
        did = self.dicom_ids[idx]
        p = self.dicom_to_path[did]
        d = self.loader.load_file(str(p))
        # d["data"]: (1,3,512,512)
        return {"data": d["data"][0], "dicom_id": did, "filename": str(p)}


def collate_fn(batch):
    import torch

    return {
        "data": torch.stack([b["data"] for b in batch], dim=0),
        "dicom_id": [b["dicom_id"] for b in batch],
        "filename": [b["filename"] for b in batch],
    }


def gaussian_kernel_1d(sigma: float, device, dtype):
    import torch

    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    k = int(max(3, 2 * round(3 * sigma) + 1))  # ~6*sigma, odd
    coords = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return g


def gaussian_blur_2d(x_bhw, sigma: float):
    import torch.nn.functional as F

    if sigma <= 0:
        return x_bhw

    g = gaussian_kernel_1d(sigma, x_bhw.device, x_bhw.dtype)
    k = int(g.numel())
    pad = k // 2
    x = x_bhw[:, None]  # (B,1,H,W)
    # Horizontal
    x = F.pad(x, (pad, pad, 0, 0), mode="replicate")
    x = F.conv2d(x, g.view(1, 1, 1, k))
    # Vertical
    x = F.pad(x, (0, 0, pad, pad), mode="replicate")
    x = F.conv2d(x, g.view(1, 1, k, 1))
    return x[:, 0]


def main() -> None:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    ap = argparse.ArgumentParser(
        description="Precompute ASH-MIL priors for MIMIC heldout (CXAS soft union + blur + token resize)."
    )
    ap.add_argument(
        "--ann",
        type=str,
        default="data/mimic/annotations/221v2hiqualnihsplit.json",
        help="COCO json used to define the heldout image set.",
    )
    ap.add_argument("--output_dir", type=str, default="/path/to/datasets/mimic-cxr-jpg/cxas_mask")
    ap.add_argument("--mimic_files_root", type=str, default="/path/to/datasets/mimic-cxr-jpg/files")
    ap.add_argument("--mimic_metadata_csv", type=str, default="/path/to/datasets/mimic-cxr-jpg/mimic-cxr-2.0.0-metadata.csv")
    ap.add_argument("--gpus", default="0", help="GPU id string, e.g. '0' or 'cpu'.")
    ap.add_argument("--model_name", default="UNet_ResNet50_default", help="CXAS model name.")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=6.0, help="Gaussian blur sigma (pixels at 512x512).")
    ap.add_argument("--token_hw", type=int, nargs=2, default=(36, 36), help="Token grid H W (default: 36x36).")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    ann_path = Path(args.ann)
    if not ann_path.exists():
        raise FileNotFoundError(f"--ann not found: {ann_path}")
    files_root = Path(args.mimic_files_root)
    meta_csv = Path(args.mimic_metadata_csv)
    if not files_root.exists():
        raise FileNotFoundError(f"--mimic_files_root not found: {files_root}")
    if not meta_csv.exists():
        raise FileNotFoundError(f"--mimic_metadata_csv not found: {meta_csv}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    token_hw = (int(args.token_hw[0]), int(args.token_hw[1]))
    cardiac_ids = (121, 115, 116, 117, 127)
    pulmonary_ids = (134,)

    cfg = PriorConfig(
        model_name=str(args.model_name),
        gpus=str(args.gpus),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        sigma=float(args.sigma),
        token_hw=token_hw,
        cardiac_ids=cardiac_ids,
        pulmonary_ids=pulmonary_ids,
        overwrite=bool(args.overwrite),
        ann=str(ann_path),
        mimic_files_root=str(files_root),
        mimic_metadata_csv=str(meta_csv),
    )
    write_meta(output_dir, cfg)

    dicom_ids = _dicom_ids_from_ann(ann_path)
    if not dicom_ids:
        raise ValueError(f"No dicom_ids found in {ann_path}")

    dicom_to_path, missing = _build_dicom_to_jpg_path(dicom_ids, metadata_csv=meta_csv, files_root=files_root)
    if missing:
        raise FileNotFoundError(f"Failed to map {len(missing)}/{len(dicom_ids)} dicom_ids via metadata. First 10: {missing[:10]}")
    bad = [d for d, p in dicom_to_path.items() if not p.exists()]
    if bad:
        raise FileNotFoundError(f"Mapped jpg paths missing on disk for {len(bad)} dicom_ids. First 10: {bad[:10]}")

    # Skip already done outputs unless overwriting.
    pending = []
    for did in dicom_ids:
        out_p = output_dir / f"{did}.npz"
        if out_p.exists() and not cfg.overwrite:
            continue
        pending.append(did)
    if not pending:
        print("No pending images to process (all priors exist).")
        return

    # Avoid importing cxas.segmentor.CXAS (radiomics not required).
    try:
        from cxas.models import get_model
    except ModuleNotFoundError as exc:
        if exc.name not in {"radiomics", "SimpleITK"}:
            raise
        # `import cxas` pulls in cxas.segmentor -> pyradiomics, which we do not use.
        # Stub it out so only the segmentation model is needed.
        import sys
        import types

        for name in ("radiomics", "radiomics.featureextractor", "SimpleITK"):
            mod = types.ModuleType(name)
            mod.featureextractor = types.ModuleType("radiomics.featureextractor")
            sys.modules.setdefault(name, mod)
        from cxas.models import get_model

    torch.backends.cudnn.benchmark = True
    model = get_model(cfg.model_name, gpus=cfg.gpus)
    model.eval()
    device = next(model.parameters()).device

    dataset = MimicJpgDataset(pending, dicom_to_path)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=("cpu" not in cfg.gpus),
        collate_fn=collate_fn,
    )

    tok_h, tok_w = cfg.token_hw
    cardiac_idx = torch.as_tensor(list(cfg.cardiac_ids), device=device)
    pulmonary_idx = torch.as_tensor(list(cfg.pulmonary_ids), device=device)

    pbar = tqdm(loader, desc="MIMIC CXAS priors", unit="batch")
    with torch.inference_mode():
        for batch in pbar:
            x = batch["data"].to(device, non_blocking=True)
            out = model({"data": x, "filename": batch["filename"]})
            logits = out["logits"]  # (B,159,512,512)

            cardiac = logits.index_select(1, cardiac_idx).sigmoid().amax(dim=1)  # (B,512,512)
            pulmonary = logits.index_select(1, pulmonary_idx).sigmoid().amax(dim=1)  # (B,512,512)

            cardiac = gaussian_blur_2d(cardiac, cfg.sigma)
            pulmonary = gaussian_blur_2d(pulmonary, cfg.sigma)

            cardiac_tok = F.interpolate(cardiac[:, None], size=(tok_h, tok_w), mode="bilinear", align_corners=False)[:, 0]
            pulmonary_tok = F.interpolate(pulmonary[:, None], size=(tok_h, tok_w), mode="bilinear", align_corners=False)[:, 0]

            cardiac_np = cardiac_tok.detach().float().cpu().numpy().astype(np.float16)
            pulmonary_np = pulmonary_tok.detach().float().cpu().numpy().astype(np.float16)

            for i, did in enumerate(batch["dicom_id"]):
                out_p = output_dir / f"{did}.npz"
                out_p.parent.mkdir(parents=True, exist_ok=True)
                # Write to a temp file then rename so an interrupted run never leaves a corrupt .npz behind.
                tmp_p = out_p.with_suffix(".npz.tmp")
                with open(tmp_p, "wb") as fh:
                    np.savez_compressed(fh, cardiac=cardiac_np[i], pulmonary=pulmonary_np[i])
                os.replace(tmp_p, out_p)


if __name__ == "__main__":
    main()

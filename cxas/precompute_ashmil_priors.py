from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from tqdm import tqdm


@dataclass(frozen=True)
class PriorConfig:
    model_name: str
    gpus: str
    batch_size: int
    num_workers: int
    sigma: float
    patch_size: int
    token_hw: Tuple[int, int]
    cardiac_ids: Tuple[int, ...]
    pulmonary_ids: Tuple[int, ...]
    overwrite: bool
    shard_by_prefix: int


class CxrPngDataset:
    def __init__(self, image_paths: Sequence[Path]):
        from cxas.file_io import FileLoader

        self.image_paths = list(image_paths)
        # Load on CPU: each dataloader worker would otherwise create its own CUDA context.
        self.loader = FileLoader(gpus="cpu")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        p = self.image_paths[idx]
        d = self.loader.load_file(str(p))
        # d["data"]: (1,3,512,512)
        return {
            "data": d["data"][0],
            "filename": str(p),
        }


def collate_fn(batch):
    import torch

    return {
        "data": torch.stack([b["data"] for b in batch], dim=0),
        "filename": [b["filename"] for b in batch],
    }


def gaussian_kernel_1d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    import torch

    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    k = int(max(3, 2 * round(3 * sigma) + 1))  # ~6*sigma, odd
    coords = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return g


def gaussian_blur_2d(x_bhw: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    x_bhw: (B,H,W) float tensor.
    Returns: (B,H,W) blurred tensor.
    """
    import torch.nn.functional as F

    if sigma <= 0:
        return x_bhw

    b, h, w = x_bhw.shape
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


def iter_pngs(input_dir: Path) -> Iterable[Path]:
    yield from input_dir.rglob("*.png")


def out_path_for(image_path: Path, input_dir: Path, output_dir: Path, shard_by_prefix: int) -> Path:
    # Keep relative structure if possible.
    try:
        rel = image_path.relative_to(input_dir)
        base = (output_dir / rel).with_suffix(".npz")
    except Exception:
        base = output_dir / (image_path.stem + ".npz")

    if shard_by_prefix > 0:
        stem = base.stem
        prefix = stem[:shard_by_prefix] if len(stem) >= shard_by_prefix else stem
        return base.parent / prefix / base.name
    return base


def write_meta(output_dir: Path, cfg: PriorConfig) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "ashmil_cxas_priors_meta.json"
    # Always refresh so the folder stays self-describing after a re-run.
    meta_path.write_text(json.dumps(asdict(cfg), indent=2))


def main() -> None:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(
        description="Precompute ASH-MIL anatomy priors from CXAS logits (soft union + Gaussian blur + token resize).",
    )
    parser.add_argument("--input_dir", required=True, help="Root directory containing CXR PNG files.")
    parser.add_argument(
        "--output_dir",
        default="/path/to/datasets/CXR8/cxas_mask",
    )
    parser.add_argument("--gpus", default="0", help="GPU id string, e.g. '0' or 'cpu'.")
    parser.add_argument("--model_name", default="UNet_ResNet50_default", help="CXAS model name.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--sigma", type=float, default=6.0, help="Gaussian blur sigma (pixels at 512x512).")
    parser.add_argument("--patch_size", type=int, default=14, help="Backbone patch size to derive token grid.")
    parser.add_argument(
        "--token_hw",
        type=int,
        nargs=2,
        default=(36, 36),
        help="Explicit token grid H W (overrides patch_size). Default matches RAD-DINO at 512x512 input (36x36).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--shard_by_prefix",
        type=int,
        default=0,
        help="Shard outputs into subfolders by filename prefix length (default: 0, disabled).",
    )
    parser.add_argument(
        "--cardiac_ids",
        type=int,
        nargs="+",
        default=[121, 115, 116, 117, 127],
        help="CXAS label ids to union for the cardiac prior.",
    )
    parser.add_argument(
        "--pulmonary_ids",
        type=int,
        nargs="+",
        default=[134],
        help="CXAS label ids to union for the pulmonary prior.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"--input_dir does not exist: {input_dir}")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be >= 1")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be >= 0")

    if args.token_hw is None:
        # CXAS inference is at 512x512.
        tok_h = 512 // int(args.patch_size)
        tok_w = 512 // int(args.patch_size)
        token_hw = (tok_h, tok_w)
    else:
        token_hw = (int(args.token_hw[0]), int(args.token_hw[1]))

    if not args.cardiac_ids:
        raise ValueError("--cardiac_ids must contain at least one id")
    if not args.pulmonary_ids:
        raise ValueError("--pulmonary_ids must contain at least one id")

    cfg = PriorConfig(
        model_name=args.model_name,
        gpus=str(args.gpus),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        sigma=float(args.sigma),
        patch_size=int(args.patch_size),
        token_hw=token_hw,
        cardiac_ids=tuple(int(x) for x in args.cardiac_ids),
        pulmonary_ids=tuple(int(x) for x in args.pulmonary_ids),
        overwrite=bool(args.overwrite),
        shard_by_prefix=int(args.shard_by_prefix),
    )
    write_meta(output_dir, cfg)

    # Collect image paths and optionally filter by existing outputs for resume.
    print(f"[1/3] Scanning PNGs under {input_dir} ...", flush=True)
    all_paths = sorted(iter_pngs(input_dir))
    print(f"      found {len(all_paths)} images", flush=True)

    pending_paths: List[Path] = []
    for p in tqdm(all_paths, desc="      checking existing priors", unit="img", leave=False):
        out_p = out_path_for(p, input_dir, output_dir, cfg.shard_by_prefix)
        if out_p.exists() and not cfg.overwrite:
            continue
        pending_paths.append(p)
    print(f"      {len(all_paths) - len(pending_paths)} already done, {len(pending_paths)} to process", flush=True)

    if not pending_paths:
        print("No pending images to process (all priors exist).")
        return

    # Avoid importing cxas.segmentor.CXAS to keep dependencies minimal (radiomics is not required).
    print(f"[2/3] Loading CXAS model '{cfg.model_name}' (downloads weights on first run) ...", flush=True)
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
    print(f"[3/3] Computing priors -> {output_dir}", flush=True)

    device = next(model.parameters()).device
    dataset = CxrPngDataset(pending_paths)
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

    pbar = tqdm(loader, desc="CXAS priors", unit="batch", total=len(loader), dynamic_ncols=True)
    with torch.inference_mode():
        for batch in pbar:
            x = batch["data"].to(device, non_blocking=True)
            out = model({"data": x, "filename": batch["filename"]})
            logits = out["logits"]  # (B,159,512,512)

            # Compute soft union without accumulation in overlaps: max over anatomy channels.
            cardiac = logits.index_select(1, cardiac_idx).sigmoid().amax(dim=1)  # (B,512,512)
            pulmonary = logits.index_select(1, pulmonary_idx).sigmoid().amax(dim=1)  # (B,512,512)

            cardiac = gaussian_blur_2d(cardiac, cfg.sigma)
            pulmonary = gaussian_blur_2d(pulmonary, cfg.sigma)

            cardiac_tok = F.interpolate(
                cardiac[:, None], size=(tok_h, tok_w), mode="bilinear", align_corners=False
            )[:, 0]
            pulmonary_tok = F.interpolate(
                pulmonary[:, None], size=(tok_h, tok_w), mode="bilinear", align_corners=False
            )[:, 0]

            cardiac_np = cardiac_tok.detach().float().cpu().numpy().astype(np.float16)
            pulmonary_np = pulmonary_tok.detach().float().cpu().numpy().astype(np.float16)

            for i, fname in enumerate(batch["filename"]):
                img_path = Path(fname)
                out_p = out_path_for(img_path, input_dir, output_dir, cfg.shard_by_prefix)
                out_p.parent.mkdir(parents=True, exist_ok=True)
                # Write to a temp file then rename so an interrupted run never leaves a corrupt .npz behind.
                tmp_p = out_p.with_suffix(".npz.tmp")
                with open(tmp_p, "wb") as fh:
                    np.savez_compressed(fh, cardiac=cardiac_np[i], pulmonary=pulmonary_np[i])
                os.replace(tmp_p, out_p)


if __name__ == "__main__":
    main()

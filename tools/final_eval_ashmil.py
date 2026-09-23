import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import csv
import json
import torch
from torch.utils.data import DataLoader


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_repo_on_path() -> None:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _parse_iou_thrs(s: str) -> str:
    toks = [t.strip() for t in str(s).split(",") if t.strip()]
    if not toks:
        raise ValueError("Empty --iou_thrs")
    for t in toks:
        float(t)
    return ",".join(toks)


def main() -> None:
    _ensure_repo_on_path()

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument(
        "--eval_dirname",
        type=str,
        default="final_eval",
        help="Subdirectory under output_dir where evaluation CSVs are written.",
    )

    ap.add_argument("--nih_ann", type=str, default="data/cxr8/coco_det/annotations/instances_test2017.json")
    ap.add_argument("--nih_image_dir", type=str, default="/path/to/datasets/CXR8/images")
    ap.add_argument("--nih_prior_dir", type=str, default="/path/to/datasets/CXR8/cxas_mask")

    ap.add_argument("--mimic_ann", type=str, default="data/mimic/annotations/221v2hiqualnihsplit.json")
    ap.add_argument("--mimic_prior_dir", type=str, default="/path/to/datasets/mimic-cxr-jpg/cxas_mask")
    ap.add_argument("--mimic_files_root", type=str, default="/path/to/datasets/mimic-cxr-jpg/files")
    ap.add_argument(
        "--mimic_metadata_csv",
        type=str,
        default="/path/to/datasets/mimic-cxr-jpg/mimic-cxr-2.0.0-metadata.csv",
    )
    ap.add_argument("--skip_mimic", action="store_true")

    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--score_thresh", type=float, default=0.0)
    ap.add_argument("--mass_thresh", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--min_pixels", type=int, default=16)
    ap.add_argument("--topk_per_class", type=int, default=200)
    ap.add_argument("--iou_thrs", type=str, default="0.1,0.2,0.3,0.4,0.5")
    args = ap.parse_args()

    iou_thrs = _parse_iou_thrs(args.iou_thrs)

    out_dir = Path(args.output_dir)
    eval_root = out_dir / str(args.eval_dirname)
    nih_out = eval_root / "nih"
    mimic_out = eval_root / "mimic"
    nih_out.mkdir(parents=True, exist_ok=True)
    mimic_out.mkdir(parents=True, exist_ok=True)

    from pycocotools.coco import COCO
    from models.ashmil import ASHMIL, ASHMILConfig
    from models.postprocess import postprocess_ashmil_detections
    from util.cxr8_coco_heldout import CxrCocoHeldout, collate_cxr_coco_heldout as collate_coco
    from util.detailed_eval import (
        compute_detailed_localization_report,
        parse_iou_thresholds,
        with_required_iou_thresholds,
        write_classwise_metric_csv,
    )

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    num_queries = int(ckpt.get("args", {}).get("num_queries", 100) or 100)

    cfg = ASHMILConfig(num_classes=8, num_queries=num_queries)
    model = ASHMIL(cfg, freeze_backbone=True).to(device).eval()
    model.load_state_dict(ckpt["model"], strict=True)

    coco_gt = COCO(str(args.nih_ann))
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    name_to_id = {c["name"]: c["id"] for c in cats}
    model_class_names = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax"]
    class_to_catid: List[int] = []
    for n in model_class_names:
        if n in name_to_id:
            class_to_catid.append(int(name_to_id[n]))
        elif n == "Infiltration" and "Infiltrate" in name_to_id:
            class_to_catid.append(int(name_to_id["Infiltrate"]))
        else:
            raise ValueError(f"Missing category mapping for {n!r} in {list(name_to_id.keys())}")

    ds = CxrCocoHeldout(Path(args.nih_ann), Path(args.nih_image_dir), Path(args.nih_prior_dir))
    dl = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=collate_coco,
        drop_last=False,
        persistent_workers=(int(args.num_workers) > 0),
        prefetch_factor=2 if int(args.num_workers) > 0 else None,
    )

    dets = []
    with torch.no_grad():
        for images, priors, meta in dl:
            priors = priors.to(device, non_blocking=True)
            out = model(images, priors=priors, return_attn=True)
            sizes = [m["size"] for m in meta]
            preds = postprocess_ashmil_detections(
                out,
                score_thresh=float(args.score_thresh),
                mass_thresh=float(args.mass_thresh),
                tau=float(args.tau),
                min_pixels=int(args.min_pixels),
                topk_per_class=int(args.topk_per_class),
                image_sizes=sizes,
            )
            for m, r in zip(meta, preds):
                img_id = int(m["image_id"])
                boxes = r["boxes"].detach().cpu().tolist()
                scores = r["scores"].detach().cpu().tolist()
                labels = r["labels"].detach().cpu().tolist()
                for bx, sc, li in zip(boxes, scores, labels):
                    cat_id = int(class_to_catid[int(li)])
                    x1, y1, x2, y2 = bx
                    w = max(0.0, x2 - x1)
                    h = max(0.0, y2 - y1)
                    dets.append({"image_id": img_id, "category_id": cat_id, "bbox": [x1, y1, w, h], "score": float(sc)})

    coco_dt = coco_gt.loadRes(dets) if dets else coco_gt.loadRes([])
    iou_list = with_required_iou_thresholds(parse_iou_thresholds(iou_thrs))
    report = compute_detailed_localization_report(coco_gt, coco_dt, iou_list)
    write_classwise_metric_csv(report, nih_out / "nih_evaluation_result_ap.csv", epoch=0, metric_name="ap")
    write_classwise_metric_csv(report, nih_out / "nih_evaluation_result_loc_acc.csv", epoch=0, metric_name="loc_acc")
    print(f"[NIH] wrote: {nih_out}")

    # ---- MIMIC eval ----
    if args.skip_mimic:
        print("[MIMIC] skipped")
        return

    from PIL import Image
    from pycocotools.cocoeval import COCOeval

    MODEL_CLASSES_8 = [
        "Atelectasis",
        "Cardiomegaly",
        "Effusion",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pneumonia",
        "Pneumothorax",
    ]
    MIMIC_CATS = ["pneumonia", "pneumothorax"]
    MIMIC_TO_MODEL_INDEX = {"pneumonia": 6, "pneumothorax": 7}

    def _dicom_id_from_file_name(file_name: str) -> str:
        return Path(file_name).stem

    def _build_dicom_to_jpg_path(
        dicom_ids: List[str],
        *,
        metadata_csv: Path,
        files_root: Path,
    ) -> Tuple[Dict[str, Path], List[str]]:
        want = set(dicom_ids)
        out: Dict[str, Path] = {}

        with metadata_csv.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                did = row.get("dicom_id")
                if not did or did not in want:
                    continue
                subj = row.get("subject_id")
                study = row.get("study_id")
                if not subj or not study:
                    continue
                prefix = str(subj)[:2]
                rel = Path(f"p{prefix}") / f"p{subj}" / f"s{study}" / f"{did}.jpg"
                out[did] = files_root / rel
                if len(out) == len(want):
                    break

        missing: List[str] = sorted([d for d in want if d not in out])
        return out, missing

    class _MimicHeldout(torch.utils.data.Dataset):
        def __init__(
            self,
            coco_dict: dict,
            *,
            dicom_to_path: dict,
            prior_dir: Optional[Path],
            prior_hw: Tuple[int, int] = (36, 36),
        ) -> None:
            self.images = list(coco_dict["images"])
            self.images.sort(key=lambda x: int(x.get("id", 0)))
            self.dicom_to_path = dicom_to_path
            self.prior_dir = prior_dir
            self.prior_hw = prior_hw

        def __len__(self) -> int:
            return len(self.images)

        def __getitem__(self, idx: int):
            import numpy as np

            im = self.images[idx]
            image_id = int(im["id"])
            file_name = str(im.get("file_name", ""))
            did = _dicom_id_from_file_name(file_name)
            img_path = self.dicom_to_path[did]

            img = Image.open(img_path).convert("RGB")
            w, h = img.size

            if self.prior_dir is not None:
                prior_path = self.prior_dir / f"{did}.npz"
                with np.load(prior_path) as x:
                    cardiac = x["cardiac"].astype("float32")
                    pulmonary = x["pulmonary"].astype("float32")
                priors = np.stack([cardiac, pulmonary, np.zeros_like(cardiac)], axis=0)
                priors_t = torch.from_numpy(priors)
            else:
                ph, pw = self.prior_hw
                priors_t = torch.zeros((3, ph, pw), dtype=torch.float32)

            meta = {
                "image_id": image_id,
                "file_name": file_name,
                "dicom_id": did,
                "img_path": str(img_path),
                "size": (h, w),
            }
            return img, priors_t, meta

    def _collate(batch):
        images, priors, meta = zip(*batch)
        priors_t = torch.stack(list(priors), dim=0)
        return list(images), priors_t, list(meta)

    mimic_ann = Path(args.mimic_ann)
    if mimic_ann.suffix.lower() != ".json":
        raise ValueError(f"--mimic_ann must be a COCO-format .json file, got: {mimic_ann}")
    coco_dict_m = json.loads(mimic_ann.read_text())
    coco_gt_m = COCO(str(mimic_ann))

    dicom_ids = [_dicom_id_from_file_name(im["file_name"]) for im in coco_dict_m["images"]]
    dicom_to_path, missing = _build_dicom_to_jpg_path(
        dicom_ids,
        metadata_csv=Path(args.mimic_metadata_csv),
        files_root=Path(args.mimic_files_root),
    )
    if missing:
        raise FileNotFoundError(f"Failed to map {len(missing)}/{len(set(dicom_ids))} dicom_ids. First 10: {missing[:10]}")
    bad = [d for d, p in dicom_to_path.items() if not Path(p).exists()]
    if bad:
        raise FileNotFoundError(f"Mapped paths missing on disk for {len(bad)} dicom_ids. First 10: {bad[:10]}")

    prior_dir = Path(args.mimic_prior_dir) if str(args.mimic_prior_dir).strip() else None
    ds_m = _MimicHeldout(coco_dict_m, dicom_to_path=dicom_to_path, prior_dir=prior_dir, prior_hw=(36, 36))
    dl_m = DataLoader(
        ds_m,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=_collate,
        drop_last=False,
        persistent_workers=(int(args.num_workers) > 0),
        prefetch_factor=2 if int(args.num_workers) > 0 else None,
    )

    cat_ids_m = sorted(coco_gt_m.getCatIds())
    cats_m = coco_gt_m.loadCats(cat_ids_m)
    name_to_catid_m = {c["name"].lower(): int(c["id"]) for c in cats_m}
    eval_catids_m = [name_to_catid_m[n] for n in MIMIC_CATS if n in name_to_catid_m]
    if len(eval_catids_m) != 2:
        raise ValueError(f"Expected 2 categories {MIMIC_CATS}, got {list(name_to_catid_m.keys())}")

    dets_m = []
    with torch.no_grad():
        for images, priors, meta in dl_m:
            priors = priors.to(device, non_blocking=True)
            out = model(images, priors=priors, return_attn=True)
            sizes = [m["size"] for m in meta]
            preds = postprocess_ashmil_detections(
                out,
                score_thresh=float(args.score_thresh),
                mass_thresh=float(args.mass_thresh),
                tau=float(args.tau),
                min_pixels=int(args.min_pixels),
                topk_per_class=int(args.topk_per_class),
                image_sizes=sizes,
            )
            for m, r in zip(meta, preds):
                img_id = int(m["image_id"])
                boxes = r["boxes"].detach().cpu().tolist()
                scores = r["scores"].detach().cpu().tolist()
                labels = r["labels"].detach().cpu().tolist()
                for bx, sc, li in zip(boxes, scores, labels):
                    li = int(li)
                    if li not in (MIMIC_TO_MODEL_INDEX["pneumonia"], MIMIC_TO_MODEL_INDEX["pneumothorax"]):
                        continue
                    name = MODEL_CLASSES_8[li].lower()
                    if name not in name_to_catid_m:
                        continue
                    cat_id = name_to_catid_m[name]
                    x1, y1, x2, y2 = bx
                    w = max(0.0, x2 - x1)
                    h = max(0.0, y2 - y1)
                    dets_m.append(
                        {"image_id": img_id, "category_id": int(cat_id), "bbox": [float(x1), float(y1), float(w), float(h)], "score": float(sc)}
                    )

    coco_dt_m = coco_gt_m.loadRes(dets_m) if dets_m else coco_gt_m.loadRes([])
    iou_list_m = with_required_iou_thresholds(parse_iou_thresholds(iou_thrs))
    report_m = compute_detailed_localization_report(coco_gt_m, coco_dt_m, iou_list_m)
    write_classwise_metric_csv(report_m, mimic_out / "mimic_evaluation_result_ap.csv", epoch=0, metric_name="ap")
    write_classwise_metric_csv(report_m, mimic_out / "mimic_evaluation_result_loc_acc.csv", epoch=0, metric_name="loc_acc")

    ev = COCOeval(coco_gt_m, coco_dt_m, iouType="bbox")
    ev.params.catIds = eval_catids_m
    ev.params.iouThrs = [float(x) for x in str(iou_thrs).split(",")]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    print(f"[MIMIC] wrote: {mimic_out}")


if __name__ == "__main__":
    main()

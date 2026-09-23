#!/usr/bin/env python3
import argparse
import random
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import util.misc as utils
from datasets.cxr8_wsod import Cxr8WsodDataset, Cxr8WsodPaths
from datasets.cxr8_split import prepare_auto_split
from engine import evaluate_ashmil, evaluate_ashmil_collect, train_one_epoch_ashmil
from models.ashmil import ASHMIL, ASHMILConfig


def collate_ashmil(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def get_args_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--split_mode", type=str, default="csv", choices=["csv", "auto"])
    p.add_argument("--split_seed", type=int, default=42, help="Seed for --split_mode auto.")
    p.add_argument("--split_ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1), metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument(
        "--data_entry_csv",
        type=str,
        default="",
        help="NIH Data_Entry_2017.csv for --split_mode auto. Default: <image_dir>/../Data_Entry_2017.csv",
    )
    p.add_argument(
        "--split_dir",
        type=str,
        default="",
        help="csv mode: directory with split CSVs (default data/cxr8/cxr8_split). "
        "auto mode: where generated CSVs are written (default <output_dir>/cxr8_split).",
    )
    p.add_argument("--image_dir", type=str, default="/path/to/datasets/CXR8/images")
    p.add_argument("--prior_dir", type=str, default="/path/to/datasets/CXR8/cxas_mask")
    p.add_argument("--output_dir", type=str, default="outputs/ashmil")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from.")

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lr_drop", type=int, default=40)
    p.add_argument("--clip_max_norm", type=float, default=0.1)
    p.add_argument("--print_freq", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--cls_val_every", type=int, default=0, help="If >0, compute classification metrics every N epochs.")
    p.add_argument(
        "--cls_report_dirname",
        type=str,
        default="cls_val",
        help="Subdirectory name under output_dir where classification-val artifacts are written.",
    )

    p.add_argument(
        "--loc_val_ann",
        type=str,
        default="data/cxr8/coco_det/annotations/instances_val2017.json",
        help="COCO-format annotation JSON for localization validation (val432). Its images are always excluded from training. "
        "Set to '' to disable and select checkpoints by val loss instead.",
    )
    p.add_argument("--loc_val_every", type=int, default=1, help="Run localization validation every N epochs.")
    p.add_argument(
        "--loc_iou_thrs",
        type=str,
        default="0.1,0.2,0.3,0.4,0.5",
        help="Comma-separated IoU thresholds for localization validation.",
    )
    p.add_argument("--loc_score_thresh", type=float, default=0.0)
    p.add_argument("--loc_mass_thresh", type=float, default=0.1)
    p.add_argument("--loc_topk_per_class", type=int, default=200)
    p.add_argument(
        "--loc_tau",
        type=float,
        default=0.8,
        help="Top-mass ratio for evidence-map binarization during localization validation.",
    )
    p.add_argument(
        "--loc_min_pixels",
        type=int,
        default=16,
        help="Discard connected components smaller than this area (in image pixels).",
    )
    p.add_argument(
        "--exclude_ann",
        type=str,
        action="append",
        default=None,
        help="COCO-format annotation JSON; all its image file_names will be excluded from WSOD training/val splits. "
        "Can be set multiple times. Default: instances_test2017.json (test448).",
    )

    p.add_argument(
        "--early_stop",
        action="store_true",
        default=True,
        help="Enable early stopping during training (default: enabled). Use --no_early_stop to disable.",
    )
    p.add_argument("--no_early_stop", dest="early_stop", action="store_false", help="Disable early stopping.")
    p.add_argument(
        "--early_stop_patience",
        type=int,
        default=5,
        help="Stop after this many consecutive epochs without improvement.",
    )
    p.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=1e-4,
        help="Minimum improvement to reset patience (metric-dependent).",
    )

    p.add_argument("--final_eval", action="store_true", help="Run final NIH + MIMIC evaluation after training ends.")
    p.add_argument(
        "--final_nih_ann",
        type=str,
        default="data/cxr8/coco_det/annotations/instances_test2017.json",
        help="NIH COCO-format annotation JSON for final evaluation (default: test448).",
    )
    p.add_argument(
        "--final_mimic_ann",
        type=str,
        default="data/mimic/annotations/221v2hiqualnihsplit.json",
        help="MIMIC heldout COCO json for final evaluation.",
    )
    p.add_argument(
        "--final_mimic_files_root",
        type=str,
        default="",
        help="Root of MIMIC-CXR-JPG `files/` directory for final evaluation. If empty, MIMIC eval is skipped.",
    )
    p.add_argument(
        "--final_mimic_metadata_csv",
        type=str,
        default="",
        help="Path to MIMIC-CXR-JPG metadata CSV for final evaluation. If empty, MIMIC eval is skipped.",
    )
    p.add_argument(
        "--final_mimic_prior_dir",
        type=str,
        default="",
        help="For ASH-MIL on MIMIC: dir containing <dicom_id>.npz priors. If empty, uses zero priors.",
    )
    p.add_argument("--final_eval_batch_size", type=int, default=8, help="Batch size for final evaluation.")
    p.add_argument("--final_eval_num_workers", type=int, default=4, help="Num workers for final evaluation.")
    p.add_argument(
        "--final_eval_dirname",
        type=str,
        default="final_eval",
        help="Subdirectory name under output_dir where final evaluation artifacts are written.",
    )
    p.add_argument(
        "--final_cls",
        action="store_true",
        help="Also compute CXR8 image-level classification metrics (AUROC/AP) after training (writes CSVs).",
    )

    p.add_argument("--num_queries", type=int, default=100)
    p.add_argument(
        "--freeze_backbone",
        action="store_true",
        default=True,
        help="Freeze backbone parameters (default: enabled). Use --no_freeze_backbone to train the backbone.",
    )
    p.add_argument(
        "--no_freeze_backbone",
        dest="freeze_backbone",
        action="store_false",
        help="Unfreeze backbone parameters (finetune).",
    )
    p.add_argument("--hflip_p", type=float, default=0.5)
    p.add_argument("--prior_hw", type=int, nargs=2, default=(36, 36))
    return p


def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    torch.backends.cudnn.benchmark = True

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    cfg = ASHMILConfig(num_classes=8, num_queries=args.num_queries)
    model = ASHMIL(cfg, freeze_backbone=bool(args.freeze_backbone)).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop, gamma=0.1)

    start_epoch = 0
    # Select best checkpoint by val loc mAP (higher) if available, else by val loss (lower).
    best_score = -float("inf") if args.loc_val_ann else float("inf")
    best_epoch = -1
    no_improve = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "lr_scheduler" in ckpt:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1
        if args.loc_val_ann and "loc_stats" in ckpt and isinstance(ckpt["loc_stats"], dict):
            if "loc_map_10_50" in ckpt["loc_stats"]:
                best_score = float(ckpt["loc_stats"]["loc_map_10_50"])
        elif "val_stats" in ckpt and isinstance(ckpt["val_stats"], dict) and "loss" in ckpt["val_stats"]:
            best_score = float(ckpt["val_stats"]["loss"])

    if args.exclude_ann is None:
        args.exclude_ann = ["data/cxr8/coco_det/annotations/instances_test2017.json"]
    if args.split_mode == "auto":
        data_entry_csv = Path(args.data_entry_csv) if args.data_entry_csv else Path(args.image_dir).resolve().parent / "Data_Entry_2017.csv"
        split_dir = Path(args.split_dir) if args.split_dir else out_dir / "cxr8_split"
        if utils.is_main_process():
            prepare_auto_split(
                data_entry_csv,
                split_dir,
                seed=int(args.split_seed),
                ratios=tuple(float(r) for r in args.split_ratios),
            )
        if args.distributed:
            torch.distributed.barrier()
    else:
        split_dir = Path(args.split_dir) if args.split_dir else Path("data/cxr8/cxr8_split")
    args.split_dir = str(split_dir)
    exclude_files = set()
    # Localization-validation images are never used for training.
    if args.loc_val_ann:
        args.exclude_ann = list(args.exclude_ann) + [args.loc_val_ann]

    for ann in args.exclude_ann:
        ann_p = Path(ann)
        if not ann_p.exists():
            raise FileNotFoundError(f"--exclude_ann not found: {ann_p}")
        d = json.loads(ann_p.read_text())
        for im in d.get("images", []):
            exclude_files.add(str(im["file_name"]))

    paths_train = Cxr8WsodPaths(
        image_dir=Path(args.image_dir),
        split_csv=split_dir / "train_list_detection_811_train.csv",
        prior_dir=Path(args.prior_dir),
    )
    paths_val = Cxr8WsodPaths(
        image_dir=Path(args.image_dir),
        split_csv=split_dir / "train_list_detection_811_val.csv",
        prior_dir=Path(args.prior_dir),
    )

    ds_train = Cxr8WsodDataset(
        paths_train,
        fixed_size=0,
        prior_hw=tuple(args.prior_hw),
        hflip_p=args.hflip_p,
        is_train=True,
        return_pil=True,
        exclude_files=sorted(exclude_files),
    )
    ds_val = Cxr8WsodDataset(
        paths_val,
        fixed_size=0,
        prior_hw=tuple(args.prior_hw),
        hflip_p=0.0,
        is_train=False,
        return_pil=True,
        exclude_files=sorted(exclude_files),
    )

    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(ds_train)
        sampler_val = torch.utils.data.DistributedSampler(ds_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(ds_train)
        sampler_val = torch.utils.data.SequentialSampler(ds_val)

    data_loader_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        sampler=sampler_train,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
        collate_fn=collate_ashmil,
        drop_last=True,
    )
    data_loader_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        sampler=sampler_val,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
        collate_fn=collate_ashmil,
        drop_last=False,
    )

    def run_loc_val() -> dict:
        loc_stats = {}
        from util.cxr8_coco_heldout import CxrCocoHeldout, collate_cxr_coco_heldout as collate_coco
        from models.postprocess import postprocess_ashmil_detections
        from pycocotools.coco import COCO

        from util.detailed_eval import (
            compute_detailed_localization_report,
            flatten_overall_metrics,
            parse_iou_thresholds,
            with_required_iou_thresholds,
        )

        ann_path = Path(args.loc_val_ann)
        coco = COCO(str(ann_path))
        loc_ds = CxrCocoHeldout(ann_path, Path(args.image_dir), Path(args.prior_dir))
        loc_dl = DataLoader(
            loc_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_coco,
            drop_last=False,
            persistent_workers=(args.num_workers > 0),
            prefetch_factor=2 if args.num_workers > 0 else None,
        )

        # class index -> COCO category_id (Infiltration vs Infiltrate naming)
        cats = coco.loadCats(coco.getCatIds())
        name_to_id = {c["name"]: c["id"] for c in cats}
        model_class_names = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax"]
        class_to_catid = []
        for n in model_class_names:
            if n in name_to_id:
                class_to_catid.append(name_to_id[n])
            elif n == "Infiltration" and "Infiltrate" in name_to_id:
                class_to_catid.append(name_to_id["Infiltrate"])
            else:
                raise ValueError(f"Missing category mapping for {n!r} in {list(name_to_id.keys())}")

        dets = []
        model.eval()
        with torch.no_grad():
            for images, priors, meta in loc_dl:
                priors = priors.to(device, non_blocking=True)
                out = model(images, priors=priors, return_attn=True)
                sizes = [m["size"] for m in meta]
                preds = postprocess_ashmil_detections(
                    out,
                    score_thresh=args.loc_score_thresh,
                    mass_thresh=float(args.loc_mass_thresh),
                    topk_per_class=args.loc_topk_per_class,
                    tau=args.loc_tau,
                    min_pixels=args.loc_min_pixels,
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
                        dets.append(
                            {"image_id": img_id, "category_id": cat_id, "bbox": [x1, y1, w, h], "score": float(sc)}
                        )

        try:
            coco_dt = coco.loadRes(dets)
        except Exception:
            coco_dt = coco.loadRes([])
        iou_thrs = with_required_iou_thresholds(parse_iou_thresholds(args.loc_iou_thrs))
        detailed_report = compute_detailed_localization_report(coco, coco_dt, iou_thrs)

        loc_flat = flatten_overall_metrics(detailed_report["overall"])
        for k, v in loc_flat.items():
            loc_stats[k] = v

        ap_keys_10_50 = ["0.10", "0.20", "0.30", "0.40", "0.50"]
        ap_vals = []
        for kk in ap_keys_10_50:
            if kk in detailed_report["overall"] and detailed_report["overall"][kk]["ap"] == detailed_report["overall"][kk]["ap"]:
                ap_vals.append(float(detailed_report["overall"][kk]["ap"]))
        if ap_vals:
            loc_stats["loc_map_10_50"] = float(sum(ap_vals) / len(ap_vals))

        model.train()
        return loc_stats

    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch_ashmil(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            max_norm=args.clip_max_norm,
            print_freq=args.print_freq,
        )
        lr_scheduler.step()

        val_stats = evaluate_ashmil(model, data_loader_val, device, print_freq=args.print_freq)
        val_loss = float(val_stats.get("loss", float("inf")))

        cls_stats = {}
        if args.cls_val_every and int(args.cls_val_every) > 0 and (epoch % int(args.cls_val_every) == 0):
            try:
                from sklearn.metrics import average_precision_score, roc_auc_score
            except Exception as exc:
                if utils.is_main_process():
                    print(f"[ClsEval] skipped (install scikit-learn): {exc}")
            else:
                _, y_score, y_true = evaluate_ashmil_collect(model, data_loader_val, device, print_freq=args.print_freq)
                y_score_np = y_score.numpy()
                y_true_np = y_true.numpy()
                class_names = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax"]
                per = []
                aurocs = []
                auprcs = []
                for ci, name in enumerate(class_names):
                    yt = y_true_np[:, ci]
                    ys = y_score_np[:, ci]
                    try:
                        auc = float(roc_auc_score(yt, ys))
                    except Exception:
                        auc = float("nan")
                    try:
                        auprc = float(average_precision_score(yt, ys))
                    except Exception:
                        auprc = float("nan")
                    per.append((name, auc, auprc))
                    if auc == auc:
                        aurocs.append(auc)
                    if auprc == auprc:
                        auprcs.append(auprc)

                cls_stats["cls_auroc_mean"] = float(sum(aurocs) / len(aurocs)) if aurocs else float("nan")
                cls_stats["cls_auprc_mean"] = float(sum(auprcs) / len(auprcs)) if auprcs else float("nan")
                for name, auc, auprc in per:
                    cls_stats[f"cls_auroc_{name}"] = auc
                    cls_stats[f"cls_auprc_{name}"] = auprc

                if utils.is_main_process():
                    import csv as _csv
                    cls_dir = out_dir / str(args.cls_report_dirname)
                    cls_dir.mkdir(parents=True, exist_ok=True)
                    csv_path = cls_dir / "cxr8_val_classification.csv"
                    fieldnames = ["epoch", "class_name", "auroc", "auprc"]
                    with csv_path.open("w", newline="") as f:
                        w = _csv.DictWriter(f, fieldnames=fieldnames)
                        w.writeheader()
                        for name, auc, auprc in per:
                            w.writerow(
                                {
                                    "epoch": int(epoch),
                                    "class_name": name,
                                    "auroc": "" if auc != auc else f"{auc:.4f}",
                                    "auprc": "" if auprc != auprc else f"{auprc:.4f}",
                                }
                            )
                        w.writerow(
                            {
                                "epoch": int(epoch),
                                "class_name": "MEAN",
                                "auroc": "" if cls_stats["cls_auroc_mean"] != cls_stats["cls_auroc_mean"] else f"{cls_stats['cls_auroc_mean']:.4f}",
                                "auprc": "" if cls_stats["cls_auprc_mean"] != cls_stats["cls_auprc_mean"] else f"{cls_stats['cls_auprc_mean']:.4f}",
                            }
                        )
                    (cls_dir / "summary.json").write_text(json.dumps(cls_stats, indent=2))

        loc_stats = {}
        if args.loc_val_ann and (args.loc_val_every > 0) and (epoch % args.loc_val_every == 0):
            loc_stats = run_loc_val()

        improved = False
        if args.loc_val_ann:
            if "loc_map_10_50" in loc_stats:
                metric = float(loc_stats["loc_map_10_50"])
                improved = metric > (best_score + float(args.early_stop_min_delta))
                if improved:
                    best_score = metric
                    best_epoch = epoch
                    no_improve = 0
                else:
                    no_improve += 1
        else:
            metric = float(val_loss)
            improved = metric < (best_score - float(args.early_stop_min_delta))
            if improved:
                best_score = metric
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "train_stats": train_stats,
            "val_stats": val_stats,
            "loc_stats": loc_stats,
            "best_epoch": best_epoch,
            "best_score": best_score,
        }
        torch.save(ckpt, out_dir / "checkpoint_last.pth")
        if improved:
            torch.save(ckpt, out_dir / "checkpoint_best.pth")

        if args.early_stop and args.early_stop_patience > 0 and no_improve >= int(args.early_stop_patience):
            if utils.is_main_process():
                sel = "loc_map_10_50" if args.loc_val_ann else "val_loss"
                print(f"[EarlyStop] epoch={epoch} patience={args.early_stop_patience} no_improve={no_improve} (best {sel}={best_score:.6f} at epoch {best_epoch})")
            break

    if args.final_eval and utils.is_main_process():
        eval_dir = out_dir / str(args.final_eval_dirname)
        eval_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = out_dir / "checkpoint_best.pth"
        if not ckpt_path.exists():
            ckpt_path = out_dir / "checkpoint_last.pth"
        print(f"[FinalEval] ckpt={ckpt_path}")

        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
            model.load_state_dict(ckpt["model"], strict=True)
            model.eval()
            eval_epoch = int(ckpt.get("best_epoch", ckpt.get("epoch", 0)))
        except Exception as exc:
            print(f"[FinalEval] Failed to load checkpoint for evaluation: {exc}")
            return

        try:
            # Free GPU memory before the evaluator subprocess loads its own model.
            try:
                model.to("cpu")
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            cmd = [
                sys.executable,
                str((Path(__file__).resolve().parent / "tools" / "final_eval_ashmil.py").resolve()),
                "--ckpt",
                str(ckpt_path),
                "--output_dir",
                str(out_dir),
                "--eval_dirname",
                str(args.final_eval_dirname),
                "--nih_ann",
                str(args.final_nih_ann),
                "--nih_image_dir",
                str(args.image_dir),
                "--nih_prior_dir",
                str(args.prior_dir),
                "--mimic_ann",
                str(args.final_mimic_ann),
                "--mimic_prior_dir",
                str(args.final_mimic_prior_dir),
                "--mimic_files_root",
                str(args.final_mimic_files_root),
                "--mimic_metadata_csv",
                str(args.final_mimic_metadata_csv),
                "--device",
                str(args.device),
                "--batch_size",
                str(int(args.final_eval_batch_size)),
                "--num_workers",
                str(int(args.final_eval_num_workers)),
                "--score_thresh",
                str(float(args.loc_score_thresh)),
                "--mass_thresh",
                str(float(args.loc_mass_thresh)),
                "--tau",
                str(float(args.loc_tau)),
                "--min_pixels",
                str(int(args.loc_min_pixels)),
                "--topk_per_class",
                str(int(args.loc_topk_per_class)),
                "--iou_thrs",
                "0.1,0.2,0.3,0.4,0.5",
            ]
            if not str(args.final_mimic_files_root).strip() or not str(args.final_mimic_metadata_csv).strip():
                cmd += ["--skip_mimic"]
            subprocess.run(cmd, check=True)
            print(f"[FinalEval] done (eval_dir={eval_dir})")
        except Exception as exc:
            print(f"[FinalEval] failed: {exc}")

        if args.final_cls:
            try:
                from sklearn.metrics import average_precision_score, roc_auc_score
            except Exception as exc:
                print(f"[FinalEval][CLS] skipped (install scikit-learn): {exc}")
            else:
                from datasets.cxr8_wsod import CLASSES_8

                cls_out = eval_dir / "cxr8_cls"
                cls_out.mkdir(parents=True, exist_ok=True)

                def _run_split(split_name: str, csv_name: str) -> tuple[dict, list]:
                    paths = Cxr8WsodPaths(
                        image_dir=Path(args.image_dir),
                        split_csv=Path(args.split_dir) / csv_name,
                        prior_dir=Path(args.prior_dir),
                    )
                    ds = Cxr8WsodDataset(paths, fixed_size=0, prior_hw=tuple(args.prior_hw), hflip_p=0.0, is_train=False, return_pil=True)
                    dl = DataLoader(
                        ds,
                        batch_size=int(args.final_eval_batch_size),
                        shuffle=False,
                        num_workers=int(args.final_eval_num_workers),
                        pin_memory=True,
                        persistent_workers=(int(args.final_eval_num_workers) > 0),
                        prefetch_factor=2 if int(args.final_eval_num_workers) > 0 else None,
                        collate_fn=collate_ashmil,
                        drop_last=False,
                    )

                    model.to(device)
                    model.eval()
                    _, y_score, y_true = evaluate_ashmil_collect(model, dl, device, print_freq=args.print_freq)
                    y_score_np = y_score.numpy()
                    y_true_np = y_true.numpy()

                    per = []
                    aurocs = []
                    auprcs = []
                    for ci, name in enumerate(list(CLASSES_8)):
                        yt = y_true_np[:, ci]
                        ys = y_score_np[:, ci]
                        try:
                            auc = float(roc_auc_score(yt, ys))
                        except Exception:
                            auc = float("nan")
                        try:
                            auprc = float(average_precision_score(yt, ys))
                        except Exception:
                            auprc = float("nan")
                        per.append((name, auc, auprc))
                        if auc == auc:
                            aurocs.append(auc)
                        if auprc == auprc:
                            auprcs.append(auprc)

                    out = {
                        "split": split_name,
                        "auroc_mean": float(sum(aurocs) / len(aurocs)) if aurocs else float("nan"),
                        "auprc_mean": float(sum(auprcs) / len(auprcs)) if auprcs else float("nan"),
                    }

                    import csv as _csv
                    csv_path = cls_out / f"cxr8_{split_name}_classification.csv"
                    fieldnames = ["class_name", "auroc", "auprc"]
                    with csv_path.open("w", newline="") as f:
                        w = _csv.DictWriter(f, fieldnames=fieldnames)
                        w.writeheader()
                        for name, auc, auprc in per:
                            w.writerow(
                                {
                                    "class_name": name,
                                    "auroc": "" if auc != auc else f"{auc:.4f}",
                                    "auprc": "" if auprc != auprc else f"{auprc:.4f}",
                                }
                            )
                        w.writerow(
                            {
                                "class_name": "MEAN",
                                "auroc": "" if out["auroc_mean"] != out["auroc_mean"] else f"{out['auroc_mean']:.4f}",
                                "auprc": "" if out["auprc_mean"] != out["auprc_mean"] else f"{out['auprc_mean']:.4f}",
                            }
                        )
                    (cls_out / f"cxr8_{split_name}_summary.json").write_text(json.dumps(out, indent=2))
                    return out, per

                val_sum, val_per = _run_split("val", "train_list_detection_811_val.csv")
                test_sum, test_per = _run_split("test", "train_list_detection_811_test.csv")
                (cls_out / "cxr8_classification_summary.json").write_text(json.dumps({"val": val_sum, "test": test_sum}, indent=2))

                import csv as _csv
                root_csv = out_dir / "classification_result.csv"
                fieldnames = ["epoch", "split", "class_name", "auroc", "auprc"]
                with root_csv.open("w", newline="") as f:
                    w = _csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader()
                    for name, auc, auprc in test_per:
                        w.writerow(
                            {
                                "epoch": int(eval_epoch),
                                "split": "test",
                                "class_name": name,
                                "auroc": "" if auc != auc else f"{auc:.4f}",
                                "auprc": "" if auprc != auprc else f"{auprc:.4f}",
                            }
                        )
                    w.writerow(
                        {
                            "epoch": int(eval_epoch),
                            "split": "test",
                            "class_name": "MEAN",
                            "auroc": "" if test_sum["auroc_mean"] != test_sum["auroc_mean"] else f"{test_sum['auroc_mean']:.4f}",
                            "auprc": "" if test_sum["auprc_mean"] != test_sum["auprc_mean"] else f"{test_sum['auprc_mean']:.4f}",
                        }
                    )

                model.to("cpu")
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser("ASH-MIL trainer", parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

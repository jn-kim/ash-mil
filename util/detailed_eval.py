import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from pycocotools.cocoeval import COCOeval

MAP_10_50_IOU_KEYS = ["0.10", "0.20", "0.30", "0.40", "0.50"]

DETAILED_EVAL_PROTOCOL = "detr"
DETAILED_EVAL_SCORE_THR = 0.0
DETAILED_EVAL_TOP_K = 0
DETAILED_EVAL_POSITIVE_ONLY = False


def parse_iou_thresholds(thresholds: str) -> List[float]:
    values = []
    for token in thresholds.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("No valid IoU thresholds were provided.")
    return sorted(set(values))


def format_iou_thr(thr: float) -> str:
    return f"{thr:.2f}"


def with_required_iou_thresholds(iou_thrs: List[float]) -> List[float]:
    required = [0.1, 0.2, 0.3, 0.4, 0.5]  # always report mAP@0.1:0.5
    return sorted(set(iou_thrs + required))


def to_xyxy(box_xywh: List[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = box_xywh
    return x, y, x + w, y + h


def box_iou_xyxy(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def safe_div(n: float, d: float) -> float:
    if d <= 0:
        return float("nan")
    return n / d


def nanmean(values: List[float]) -> float:
    filtered = [v for v in values if not math.isnan(v)]
    if not filtered:
        return float("nan")
    return float(sum(filtered) / len(filtered))


def metric_float(v, ndigits: int = 4):
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (int, float, np.floating)):
        return round(float(v), ndigits)
    return v


def build_gt_dt_indices(
    coco_gt,
    coco_dt,
    cat_ids: List[int],
    score_thr: float,
    dt_top_k: int,
    positive_only: bool,
):
    cat_set = set(cat_ids)
    gt_boxes = defaultdict(lambda: defaultdict(list))  # cid -> image_id -> [xyxy]
    dt_boxes = defaultdict(lambda: defaultdict(list))  # cid -> image_id -> [(score, xyxy)]
    pos_images = defaultdict(set)  # cid -> {image_id}

    for ann in coco_gt.dataset.get("annotations", []):
        cid = ann["category_id"]
        if cid not in cat_set:
            continue
        img_id = ann["image_id"]
        gt_boxes[cid][img_id].append(to_xyxy(ann["bbox"]))
        pos_images[cid].add(img_id)

    for ann in coco_dt.dataset.get("annotations", []):
        cid = ann["category_id"]
        if cid not in cat_set:
            continue
        score = ann.get("score", 0.0)
        if score < score_thr:
            continue
        img_id = ann["image_id"]
        dt_boxes[cid][img_id].append((score, to_xyxy(ann["bbox"])))

    for cid in cat_ids:
        for img_id in list(dt_boxes[cid].keys()):
            dt_boxes[cid][img_id].sort(key=lambda x: x[0], reverse=True)
            if dt_top_k > 0:
                dt_boxes[cid][img_id] = dt_boxes[cid][img_id][:dt_top_k]
            if positive_only and img_id not in pos_images[cid]:
                del dt_boxes[cid][img_id]

    return gt_boxes, dt_boxes, pos_images


def compute_ap_tables(
    coco_gt,
    coco_dt,
    cat_ids: List[int],
    iou_thrs: List[float],
):
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.catIds = cat_ids
    coco_eval.params.iouThrs = np.array(iou_thrs, dtype=np.float64)
    coco_eval.evaluate()
    coco_eval.accumulate()

    precision_tensor = coco_eval.eval["precision"]  # [T, R, K, A, M]
    ap_by_thr = {}
    for t, thr in enumerate(iou_thrs):
        thr_key = format_iou_thr(thr)
        class_ap = {}
        class_ap_values = []
        for k, cid in enumerate(cat_ids):
            p = precision_tensor[t, :, k, 0, 2]  # area=all, maxDets=100
            p = p[p > -1]
            ap = float(np.mean(p)) if p.size else float("nan")
            class_ap[cid] = ap
            class_ap_values.append(ap)
        ap_by_thr[thr_key] = {
            "class_ap": class_ap,
            "overall_ap_macro": nanmean(class_ap_values),
        }
    return ap_by_thr


def compute_locacc(
    coco_gt,
    coco_dt,
    cat_ids: List[int],
    iou_thrs: List[float],
    score_thr: float,
    dt_top_k: int,
    positive_only: bool,
):
    gt_boxes, dt_boxes, pos_images = build_gt_dt_indices(
        coco_gt, coco_dt, cat_ids, score_thr=score_thr, dt_top_k=dt_top_k, positive_only=positive_only
    )

    results = {}
    for thr in iou_thrs:
        thr_key = format_iou_thr(thr)
        per_class = {}

        for cid in cat_ids:
            image_ids = set(gt_boxes[cid].keys()) if positive_only else (
                set(gt_boxes[cid].keys()) | set(dt_boxes[cid].keys())
            )
            positive_images = len(pos_images[cid])

            localized_images = 0
            for image_id in pos_images[cid]:
                gt = gt_boxes[cid].get(image_id, [])
                dt = [b for _, b in dt_boxes[cid].get(image_id, [])]
                found = False
                for pred in dt:
                    for tgt in gt:
                        if box_iou_xyxy(pred, tgt) >= thr:
                            found = True
                            break
                    if found:
                        break
                if found:
                    localized_images += 1

            loc_acc = safe_div(localized_images, positive_images)

            per_class[cid] = {
                "positive_images": positive_images,
                "localized_images": localized_images,
                "loc_acc": loc_acc,
            }

        macro_loc_acc = nanmean([per_class[c]["loc_acc"] for c in cat_ids])

        results[thr_key] = {
            "class_metrics": per_class,
            "overall": {
                "loc_acc": macro_loc_acc,
            },
        }
    return results


def compute_detailed_localization_report(
    coco_gt,
    coco_dt,
    iou_thrs: List[float],
):
    score_thr = DETAILED_EVAL_SCORE_THR
    dt_top_k = DETAILED_EVAL_TOP_K
    positive_only = DETAILED_EVAL_POSITIVE_ONLY

    cat_ids = sorted(coco_gt.getCatIds())
    cat_info = coco_gt.loadCats(cat_ids)
    cat_name = {c["id"]: c["name"] for c in cat_info}

    ap_tables = compute_ap_tables(coco_gt, coco_dt, cat_ids, iou_thrs)
    loc_tables = compute_locacc(
        coco_gt, coco_dt, cat_ids, iou_thrs, score_thr, dt_top_k, positive_only
    )

    overall = {}
    classwise = {}

    for cid in cat_ids:
        cname = cat_name.get(cid, str(cid))
        classwise[cname] = {"category_id": cid, "metrics": {}}

    for thr in iou_thrs:
        k = format_iou_thr(thr)
        overall[k] = {
            "ap": ap_tables[k]["overall_ap_macro"],
            **loc_tables[k]["overall"],
        }
        for cid in cat_ids:
            cname = cat_name.get(cid, str(cid))
            classwise[cname]["metrics"][k] = {
                "ap": ap_tables[k]["class_ap"][cid],
                "loc_acc": loc_tables[k]["class_metrics"][cid]["loc_acc"],
            }

    return {
        "iou_thresholds": [format_iou_thr(t) for t in iou_thrs],
        "score_threshold": score_thr,
        "protocol": DETAILED_EVAL_PROTOCOL,
        "top_k_per_class_per_image": dt_top_k,
        "positive_only": positive_only,
        "overall": overall,
        "classwise": classwise,
    }


def flatten_overall_metrics(overall_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    flat = {}
    for thr_key, m in overall_metrics.items():
        prefix = f"iou_{thr_key.replace('.', 'p')}"
        for metric_name, value in m.items():
            flat[f"{prefix}_{metric_name}"] = metric_float(value)
    return flat


def _get_metric(metrics_by_thr: Dict[str, Dict[str, float]], thr_key: str, metric_key: str) -> float:
    if thr_key not in metrics_by_thr:
        return float("nan")
    return float(metrics_by_thr[thr_key].get(metric_key, float("nan")))


def _mean_metric(metrics_by_thr: Dict[str, Dict[str, float]], thr_keys: List[str], metric_key: str) -> float:
    vals = []
    for k in thr_keys:
        v = _get_metric(metrics_by_thr, k, metric_key)
        if not math.isnan(v):
            vals.append(v)
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def write_classwise_metric_csv(report: Dict, csv_path: Path, *, epoch: int, metric_name: str) -> None:
    """One overall row + one row per class. AP in percent, loc_acc as a ratio."""
    if metric_name not in {"ap", "loc_acc"}:
        raise ValueError(f"Unsupported metric_name: {metric_name}")

    fieldnames = [
        "epoch",
        "scope",
        "class_name",
        "category_id",
        "metric_iou_10",
        "metric_iou_20",
        "metric_iou_30",
        "metric_iou_40",
        "metric_iou_50",
        "mean_iou_10_50",
    ]

    def _fmt(x: float) -> str:
        if x != x or math.isnan(x):
            return ""
        if metric_name == "ap":
            return f"{(float(x) * 100.0):.2f}"
        return f"{float(x):.4f}"

    def _row(scope: str, class_name: str, category_id: int, metrics_by_thr: Dict[str, Dict[str, float]]) -> dict:
        return {
            "epoch": str(int(epoch)),
            "scope": scope,
            "class_name": class_name,
            "category_id": str(int(category_id)),
            "metric_iou_10": _fmt(_get_metric(metrics_by_thr, "0.10", metric_name)),
            "metric_iou_20": _fmt(_get_metric(metrics_by_thr, "0.20", metric_name)),
            "metric_iou_30": _fmt(_get_metric(metrics_by_thr, "0.30", metric_name)),
            "metric_iou_40": _fmt(_get_metric(metrics_by_thr, "0.40", metric_name)),
            "metric_iou_50": _fmt(_get_metric(metrics_by_thr, "0.50", metric_name)),
            "mean_iou_10_50": _fmt(_mean_metric(metrics_by_thr, MAP_10_50_IOU_KEYS, metric_name)),
        }

    rows = [_row("overall", "ALL", -1, report["overall"])]
    classwise = report.get("classwise", {})
    for cname in sorted(classwise.keys()):
        item = classwise[cname]
        rows.append(_row("class", str(cname), int(item.get("category_id", -1)), item.get("metrics", {})))

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

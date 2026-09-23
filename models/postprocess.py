from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage


def postprocess_ashmil_detections(
    outputs: Dict[str, torch.Tensor],
    *,
    score_thresh: float = 0.0,
    mass_thresh: float = 0.0,
    tau: float = 0.8,
    topk_per_class: int = 100,
    min_pixels: int = 1,
    image_sizes: Optional[List[Tuple[int, int]]] = None,
) -> List[Dict[str, torch.Tensor]]:
    """
    Evidence map -> top-mass threshold -> connected components -> boxes.
    Returns per image: boxes (N,4) xyxy in pixels, scores (N,), labels (N,).
    """
    if "cross_attn" not in outputs:
        raise ValueError("outputs must include cross attention weights. Call model(..., return_attn=True).")

    if image_sizes is None:
        raise ValueError("image_sizes is required.")

    if not (0.0 < float(tau) < 1.0):
        raise ValueError(f"tau must be in (0,1), got {tau}")
    if int(min_pixels) < 1:
        raise ValueError(f"min_pixels must be >= 1, got {min_pixels}")
    if float(mass_thresh) < 0.0:
        raise ValueError(f"mass_thresh must be >= 0, got {mass_thresh}")

    alpha = outputs["alpha"]  # (B,P,C)
    a_query = outputs["a_query"]  # (B,P,Nq,C)
    y = outputs["y"]  # (B,C)
    attn = outputs["cross_attn"]  # (B,P,Nq,T)

    token_hw_t = outputs["token_hw"]
    token_h, token_w = int(token_hw_t[0].item()), int(token_hw_t[1].item())
    t = token_h * token_w
    if attn.shape[-1] != t:
        raise ValueError(f"cross_attn has T={attn.shape[-1]} but token_hw implies T={t}")

    # Evidence map per class: H^c(t) = sum_p alpha_p^c * sum_k a_{p,k}^c * W_{p,k}(t)
    h_bct = torch.einsum("bpc,bpnc,bpnt->bct", alpha, a_query, attn)
    h_bchw = h_bct.reshape(h_bct.shape[0], h_bct.shape[1], token_h, token_w)

    b, c = y.shape
    results: List[Dict[str, torch.Tensor]] = []
    for bi in range(b):
        boxes_list: List[List[float]] = []
        scores_list: List[float] = []
        labels_list: List[int] = []

        h_map = h_bchw[bi].detach().float().cpu().numpy()
        y_b = y[bi].detach().float().cpu().numpy()

        for ci in range(c):
            class_boxes: List[List[float]] = []
            class_scores: List[float] = []

            e_tok = h_map[ci]
            h_img, w_img = image_sizes[bi]
            e_t = torch.from_numpy(e_tok)[None, None, ...]
            e = (
                F.interpolate(e_t, size=(int(h_img), int(w_img)), mode="bilinear", align_corners=False)
                .squeeze(0)
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )
            h_cc, w_cc = int(h_img), int(w_img)

            total = float(e.sum())
            if not (total > 0.0):
                continue

            # Smallest pixel set whose cumulative mass >= tau
            flat = e.reshape(-1)
            order = np.argsort(flat)[::-1]
            cumsum = np.cumsum(flat[order])
            target = float(tau) * float(total)
            k = int(np.searchsorted(cumsum, target, side="left"))
            k = min(max(k, 0), len(order) - 1)
            theta = float(flat[order[k]])
            mask = e >= theta

            lbl, num = ndimage.label(mask.astype(np.uint8), structure=np.ones((3, 3), dtype=np.uint8))
            if num <= 0:
                continue

            comp_sums = ndimage.sum(e, lbl, index=list(range(1, num + 1)))
            slices = ndimage.find_objects(lbl)
            for idx, sl in enumerate(slices, start=1):
                if sl is None:
                    continue
                y0, y1 = int(sl[0].start), int(sl[0].stop)
                x0, x1 = int(sl[1].start), int(sl[1].stop)
                if (y1 - y0) * (x1 - x0) < int(min_pixels):
                    continue

                mass_abs = float(comp_sums[idx - 1]) if idx - 1 < len(comp_sums) else float(0.0)
                mass = mass_abs / float(total)
                if mass < float(mass_thresh):
                    continue

                # score = y * mass
                score = float(y_b[ci]) * mass
                if score < float(score_thresh):
                    continue

                bx = [
                    max(0.0, min(1.0, x0 / w_cc)),
                    max(0.0, min(1.0, y0 / h_cc)),
                    max(0.0, min(1.0, x1 / w_cc)),
                    max(0.0, min(1.0, y1 / h_cc)),
                ]
                class_boxes.append(bx)
                class_scores.append(score)

            if topk_per_class > 0 and len(class_scores) > int(topk_per_class):
                order = np.argsort(np.asarray(class_scores, dtype=np.float32))[::-1][: int(topk_per_class)]
                class_boxes = [class_boxes[i] for i in order]
                class_scores = [class_scores[i] for i in order]

            boxes_list.extend(class_boxes)
            scores_list.extend(class_scores)
            labels_list.extend([int(ci)] * len(class_scores))

        if boxes_list:
            boxes = torch.tensor(boxes_list, dtype=torch.float32)
            scores = torch.tensor(scores_list, dtype=torch.float32)
            labels = torch.tensor(labels_list, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            scores = torch.zeros((0,), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        if image_sizes is not None:
            h_img, w_img = image_sizes[bi]
            scale = torch.tensor([w_img, h_img, w_img, h_img], dtype=boxes.dtype)
            boxes = boxes * scale

        results.append({"boxes": boxes, "scores": scores, "labels": labels})

    return results

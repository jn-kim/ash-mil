import math
import sys
from typing import Iterable, Tuple

import torch

import util.misc as utils
from models.ashmil import ashmil_loss_bce


def _to_device_batch(images, targets, device: torch.device):
    priors = torch.stack([t["priors"] for t in targets], dim=0).to(device, non_blocking=True)
    labels = torch.stack([t["img_labels"] for t in targets], dim=0).to(device, non_blocking=True)
    return images, priors, labels


def train_one_epoch_ashmil(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    *,
    max_norm: float = 0.0,
    print_freq: int = 50,
) -> dict:
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    for step, (images, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images, priors, labels = _to_device_batch(images, targets, device)

        outputs = model(images, priors=priors, return_attn=False)
        loss = ashmil_loss_bce(outputs, labels)

        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if max_norm and max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        with torch.no_grad():
            alpha = outputs.get("alpha", None)
            if alpha is not None:
                alpha_branch = alpha.mean(dim=(0, 2)).detach().float().cpu()
                for p, v in enumerate(alpha_branch.tolist()):
                    metric_logger.update(**{f"alpha_b{p}": v})

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_ashmil(
    model: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    *,
    print_freq: int = 50,
) -> dict:
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Val:"

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images, priors, labels = _to_device_batch(images, targets, device)
        outputs = model(images, priors=priors)
        loss = ashmil_loss_bce(outputs, labels)
        metric_logger.update(loss=float(loss.item()))

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_ashmil_collect(
    model: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    *,
    print_freq: int = 50,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Val:"

    ys = []
    ts = []
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images, priors, labels = _to_device_batch(images, targets, device)
        outputs = model(images, priors=priors)
        loss = ashmil_loss_bce(outputs, labels)
        metric_logger.update(loss=float(loss.item()))

        ys.append(outputs["y"].detach().float().cpu())
        ts.append(labels.detach().float().cpu())

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    y_score = torch.cat(ys, dim=0) if ys else torch.zeros((0, 0), dtype=torch.float32)
    y_true = torch.cat(ts, dim=0) if ts else torch.zeros((0, 0), dtype=torch.float32)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, y_score, y_true

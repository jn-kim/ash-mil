from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .rad_dino import RadDinoTokenBackbone


@dataclass(frozen=True)
class ASHMILConfig:
    num_classes: int = 8
    num_queries: int = 100
    num_branches: int = 3  # cardiac, pulmonary, agnostic
    d_model: int = 768
    nhead: int = 8
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
        return x


class AnatomyBiasedDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.nhead = nhead
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    @staticmethod
    def _with_pos(x: torch.Tensor, pos: Optional[torch.Tensor]) -> torch.Tensor:
        return x if pos is None else (x + pos)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        *,
        query_pos: Optional[torch.Tensor] = None,
        memory_bias: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ):
        q = k = self._with_pos(tgt, query_pos)
        tgt2, _ = self.self_attn(q, k, value=tgt, need_weights=False)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        q = self._with_pos(tgt, query_pos)
        k = memory
        v = memory

        attn_mask = None
        if memory_bias is not None:
            if memory_bias.ndim != 3:
                raise ValueError(f"memory_bias must be (B,1,T) or (B,Nq,T), got {tuple(memory_bias.shape)}")
            b, nq = tgt.shape[0], tgt.shape[1]
            if memory_bias.shape[1] == 1:
                bias = memory_bias.expand(b, nq, memory_bias.shape[2])
            else:
                bias = memory_bias
            attn_mask = bias.unsqueeze(1).repeat(1, self.nhead, 1, 1).flatten(0, 1)
            attn_mask = attn_mask.to(dtype=q.dtype, device=q.device)

        if return_attn:
            tgt2, attn = self.cross_attn(
                q, k, v, attn_mask=attn_mask, need_weights=True, average_attn_weights=True
            )
        else:
            tgt2, attn = self.cross_attn(q, k, v, attn_mask=attn_mask, need_weights=False)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        if return_attn:
            return tgt, attn
        return tgt


class AnatomyBiasedDecoder(nn.Module):
    def __init__(self, cfg: ASHMILConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                AnatomyBiasedDecoderLayer(
                    d_model=cfg.d_model,
                    nhead=cfg.nhead,
                    dim_feedforward=cfg.dim_feedforward,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.num_decoder_layers)
            ]
        )
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        *,
        memory: torch.Tensor,  # (B,T,D)
        query_embed: torch.Tensor,  # (Nq,D)
        memory_bias: Optional[torch.Tensor] = None,  # (B,1,T) or (B,Nq,T)
        return_intermediate: bool = False,
        return_attn: bool = False,
    ) -> torch.Tensor:
        b = memory.shape[0]
        nq = query_embed.shape[0]
        tgt = torch.zeros((b, nq, query_embed.shape[1]), device=memory.device, dtype=memory.dtype)
        query_pos = query_embed.unsqueeze(0).expand(b, -1, -1)

        inter = []
        out = tgt
        last_attn = None
        for layer in self.layers:
            if return_attn:
                out, last_attn = layer(out, memory, query_pos=query_pos, memory_bias=memory_bias, return_attn=True)
            else:
                out = layer(out, memory, query_pos=query_pos, memory_bias=memory_bias, return_attn=False)
            if return_intermediate:
                inter.append(self.norm(out))
        out = self.norm(out)
        if return_intermediate:
            return torch.stack(inter, dim=0)  # (L,B,Nq,D)
        if return_attn:
            if last_attn is None:
                raise RuntimeError("return_attn=True but no attention weights were produced.")
            return out, last_attn  # (B,Nq,D), (B,Nq,T)
        return out  # (B,Nq,D)


class HierarchicalMIL(nn.Module):
    def __init__(self, cfg: ASHMILConfig) -> None:
        super().__init__()
        c, d = cfg.num_classes, cfg.d_model
        self.w = nn.Parameter(torch.randn(c, d) * 0.02)
        self.v = nn.Parameter(torch.randn(c, d) * 0.02)
        self.u = nn.Parameter(torch.randn(c, d) * 0.02)

    def forward(self, h_bpnqd: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, p, nq, d = h_bpnqd.shape
        c = self.w.shape[0]
        h = h_bpnqd

        scores = torch.einsum("bpnd,cd->bpnc", h, self.w)
        a_query = torch.softmax(scores, dim=2)

        z = torch.einsum("bpnc,bpnd->bpcd", a_query, h)

        y_branch_logits = torch.einsum("bpcd,cd->bpc", z, self.v)
        y_branch = torch.sigmoid(y_branch_logits)

        alpha_logits = torch.einsum("bpcd,cd->bpc", z, self.u)
        alpha = torch.softmax(alpha_logits, dim=1)

        y = (alpha * y_branch).sum(dim=1)

        return {"y": y, "y_branch": y_branch, "alpha": alpha, "a_query": a_query, "z": z}


class ASHMIL(nn.Module):
    def __init__(self, cfg: ASHMILConfig, *, freeze_backbone: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = RadDinoTokenBackbone(freeze=freeze_backbone)

        self.query_embed = nn.ModuleList(
            [nn.Embedding(cfg.num_queries, cfg.d_model) for _ in range(cfg.num_branches)]
        )
        self.decoders = nn.ModuleList([AnatomyBiasedDecoder(cfg) for _ in range(cfg.num_branches)])

        self.mlp_cls = MLP(cfg.d_model, cfg.d_model, cfg.num_classes, 3)
        self.mlp_reg = MLP(cfg.d_model, cfg.d_model, 4, 3)

        self.mil = HierarchicalMIL(cfg)

        self.lambda_p = nn.Parameter(torch.tensor([1.25, 1.25, 1.25], dtype=torch.float32))

    def forward(
        self,
        images_or_pixel_values,
        *,
        priors: torch.Tensor,
        return_intermediate: bool = False,
        return_attn: bool = False,
    ) -> Dict[str, torch.Tensor]:
        patch_tokens, token_hw, cls_token = self.backbone(images_or_pixel_values)  # (B,T,D), (H',W')
        b, t, d = patch_tokens.shape

        if priors.ndim != 4 or priors.shape[0] != b or priors.shape[1] != self.cfg.num_branches:
            raise ValueError(f"priors must be (B,3,H,W), got {tuple(priors.shape)}")
        if priors.shape[2] * priors.shape[3] != t:
            raise ValueError(
                f"priors H*W must match tokens T. got priors={tuple(priors.shape[2:])} -> {priors.shape[2]*priors.shape[3]}, T={t}"
            )

        hs = []
        q_logits = []
        q_boxes = []
        attn_list = []
        for p in range(self.cfg.num_branches):
            prior_p = priors[:, p].flatten(1)  # (B,T)
            bias = (self.lambda_p[p].to(prior_p.dtype) * prior_p).unsqueeze(1)  # (B,1,T)

            qemb = self.query_embed[p].weight  # (Nq,D)
            dec_out = self.decoders[p](
                memory=patch_tokens,
                query_embed=qemb,
                memory_bias=bias,
                return_intermediate=return_intermediate,
                return_attn=return_attn,
            )
            if return_attn:
                h, attn = dec_out
                attn_list.append(attn)  # (B,Nq,T)
            else:
                h = dec_out
            # h: (B,Nq,D) or (L,B,Nq,D)
            if return_intermediate:
                h_last = h[-1]
            else:
                h_last = h

            hs.append(h_last)
            q_logits.append(torch.sigmoid(self.mlp_cls(h_last)))  # (B,Nq,C)
            q_boxes.append(torch.sigmoid(self.mlp_reg(h_last)))   # (B,Nq,4)

        h_bpnqd = torch.stack(hs, dim=1)  # (B,P,Nq,D)
        q_logits = torch.stack(q_logits, dim=1)  # (B,P,Nq,C)
        q_boxes = torch.stack(q_boxes, dim=1)    # (B,P,Nq,4)

        mil_out = self.mil(h_bpnqd)

        out = {
            "token_hw": torch.tensor(token_hw, device=patch_tokens.device),
            "cls_token": cls_token,
            "h": h_bpnqd,
            "query_scores": q_logits,
            "query_boxes": q_boxes,
            **mil_out,
        }
        if return_attn:
            out["cross_attn"] = torch.stack(attn_list, dim=1)  # (B,P,Nq,T)
        return out


def ashmil_loss_bce(outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
    y = outputs["y"]
    return F.binary_cross_entropy(y, targets)


def ashmil_detection_scores(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    alpha = outputs["alpha"]
    y_branch = outputs["y_branch"]
    a_query = outputs["a_query"]
    return a_query * (alpha * y_branch).unsqueeze(2)


def attn_to_boxes_cxcywh(attn_bpnt: torch.Tensor, token_hw: Tuple[int, int], *, topk_ratio: float = 0.15) -> torch.Tensor:
    h, w = int(token_hw[0]), int(token_hw[1])
    b, p, nq, t = attn_bpnt.shape
    if h * w != t:
        raise ValueError(f"token_hw={token_hw} implies T={h*w}, but got T={t}")

    k = max(1, int(round(topk_ratio * t)))
    topk = torch.topk(attn_bpnt, k=k, dim=-1).indices
    ys = (topk // w).to(torch.int64)
    xs = (topk % w).to(torch.int64)

    x0 = xs.min(dim=-1).values
    x1 = xs.max(dim=-1).values
    y0 = ys.min(dim=-1).values
    y1 = ys.max(dim=-1).values

    x0f = x0.to(torch.float32) / w
    y0f = y0.to(torch.float32) / h
    x1f = (x1.to(torch.float32) + 1.0) / w
    y1f = (y1.to(torch.float32) + 1.0) / h

    cx = (x0f + x1f) * 0.5
    cy = (y0f + y1f) * 0.5
    bw = (x1f - x0f).clamp(min=1e-6)
    bh = (y1f - y0f).clamp(min=1e-6)
    return torch.stack([cx, cy, bw, bh], dim=-1).to(attn_bpnt.device)
